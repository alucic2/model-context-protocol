#!/usr/bin/env python3
"""
LLM Service for intelligent query understanding
- Converts natural language queries to structured search criteria
- Provides semantic understanding of user intent
- Supports multiple LLM backends
"""

import json
import re
import asyncio
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import httpx

# Optional: repair malformed JSON from LLM responses (e.g. missing commas)
try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    json_repair = None

# Optional Gemini import
try:
    import google.generativeai as genai
    # Check version and required attributes
    try:
        # Try to get version info
        version_info = None
        if hasattr(genai, '__version__'):
            version_info = genai.__version__
        elif hasattr(genai, 'version'):
            version_info = getattr(genai.version, '__version__', 'unknown')
        else:
            try:
                import pkg_resources
                version_info = pkg_resources.get_distribution('google-generativeai').version
            except:
                version_info = 'unknown'
        if version_info:
            print(f"🧠 google-generativeai version: {version_info}")
    except Exception as e:
        print(f"🧠 Could not determine google-generativeai version: {e}")
    
    # Check if GenerativeModel is available (version check)
    if hasattr(genai, 'GenerativeModel'):
        GEMINI_AVAILABLE = True
        print(f"✅ google-generativeai has GenerativeModel attribute")
    else:
        print(f"❌ google-generativeai installed but GenerativeModel not available.")
        print(f"   This usually means the package version is too old (< 0.2.0)")
        available_attrs = [attr for attr in dir(genai) if not attr.startswith('_')]
        print(f"   Available attributes (first 15): {available_attrs[:15]}")
        print(f"   Solution:")
        print(f"   1. Upgrade pip: pip install --upgrade pip")
        print(f"   2. Upgrade package: pip install --upgrade --force-reinstall google-generativeai")
        print(f"   3. If that fails, check Python version (need >= 3.8): python --version")
        print(f"   4. Try: pip install --upgrade pip setuptools wheel")
        print(f"   Current version 0.1.0rc1 is too old - need >= 0.3.0 for GenerativeModel")
        GEMINI_AVAILABLE = False
        genai = None
except ImportError as e:
    print(f"⚠️  google-generativeai not installed: {e}")
    print(f"   Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False
    genai = None

# Shared instructions so the LLM resolves synonyms and normalization (reduces need for code injections)
LLM_SPECIES_AND_NORMALIZATION_RULES = """
SPECIES SYNONYMS – use the EXACT value from available_filters:
- "rabbit", "rabbits", "cottontail", "cottontails", "white cottontail", "white cottontails" → use the species in available_filters that contains "cottontail" (e.g. eastern_cottontail). Pick that exact filter value; do not use "rabbit" or "white cottontail" as the species value.
- "groundhog", "groundhogs", "whistlepig" → use "woodchuck" if it appears in available_filters (groundhog and woodchuck are the same animal).
- "ladybug", "ladybugs", "lady bug", "ladybird" → use a species/common-name value containing "lady beetle" or "ladybird" if it appears in available_filters.
- For partial pest common names like "golden beetle", prefer a longer available common name containing the adjective and pest type (e.g. "golden tortoise beetle") if present.
- "crow"/"crows" → use "american_crow" if it appears in available_filters.
- "american bear", "american bears", "black bear", "black bears", "american black bear", "american black bears" → use "american_black_bear" if it appears in available_filters.
- "celery" → use "celery" if it appears in available_filters (crop/plant dataset). Do not use only "celery leaftier" or "celery looper" when "celery" itself is available.
- For goat queries: "tail up", "tail down", "with tail up", "tail up" → add tail_positions: ["up"] or ["down"] (only for goat/goat_2; use exact value that appears in available filters if present).
- NEVER put action words in species. Words like "eating", "feeding", "walking", "standing", "running", "sleeping" are ACTIONS only → put them in "action", never in "species". Example: "rabbit eating" → species: [eastern_cottontail], action: [foraging]; "cat sleeping" → species: [cat] or [domestic_cat], action: [sleeping]; "dog eating" → species: [dog], action: [eating] or [feeding]. Do NOT output "pin" or "cat, pin" for "cat sleeping"—"sleeping" is the action. Do NOT output species "eating" for "dog eating"—"eating" is the action.

TIME NORMALIZATION – use canonical values that match available_filters:
- Canonical time values are ALWAYS valid: day, night, dawn, dusk, evening. Use them even if raw metadata strings differ (e.g. "Nighttime" in data → time: ["night"]).
- "night time", "nighttime", "at night", "during night", "Nighttime" → time: ["night"]
- "day time", "daytime", "during day" → time: ["day"]
- "dawn", "sunrise" → time: ["dawn"]
- "dusk", "sunset" → time: ["dusk"]
- "evening", "twilight", "late afternoon" → time: ["evening"]

SPECIES DATASET NAMES – wildlife, livestock, domestic, and plant species use dataset names as filter values (e.g. coyote, bobcat, raspberry). These are valid even when the species list is abbreviated for pests. For pests, use dataset names or pest-type words (moth, beetle) when mentioned.
"""


@dataclass
class QueryUnderstanding:
    """Structured understanding of a natural language query"""
    intent: str  # What the user wants to find
    entities: List[str]  # Key entities mentioned
    filters: Dict[str, Any]  # Structured filters
    confidence: float  # Confidence in the understanding
    reasoning: str  # Why this understanding was chosen
    description_query: Optional[str] = None  # Phrase in the style of image descriptions; used to match/rank by description field

class LLMService:
    """Service for LLM-powered query understanding - supports Azure OpenAI, OpenAI, and Gemini"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5-mini-2",
        provider: str = "auto",
        azure_endpoint: str = None,
        azure_api_key: str = None,
        azure_deployment: str = None,
        azure_api_version: str = None,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider  # "openai", "gemini", or "auto"
        
        # Initialize OpenAI/Azure settings
        self.openai_api_key = api_key
        self.openai_model = model
        self.openai_available = False
        self.is_azure = False
        self.is_project_key = False
        self.is_azure_openai_v1 = False  # True when endpoint is .../openai/v1 (model in body, not in URL)
        
        # Azure OpenAI (from args or env)
        self.azure_endpoint = (azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")).strip().rstrip("/")
        self.azure_api_key = (azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")).strip()
        if not self.azure_api_key and api_key:
            self.azure_api_key = api_key  # fallback to OPENAI_API_KEY for Azure key
        self.azure_deployment = (azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or "gpt-5-mini-2").strip()
        self.azure_api_version = (azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")).strip()
        
        # Prefer Azure when endpoint and key are set
        if self.azure_endpoint and self.azure_api_key:
            self.is_azure = True
            ep = self.azure_endpoint.lower().rstrip("/")
            # OpenAI-compatible Azure API (e.g. .../openai/v1): POST .../chat/completions with model=deployment in body
            if "/openai/v1" in ep or ep.endswith("/openai/v1"):
                self.is_azure_openai_v1 = True
                self.base_url = f"{self.azure_endpoint.rstrip('/')}/chat/completions"
            else:
                self.is_azure_openai_v1 = False
                self.base_url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_api_version}"
            self.openai_available = True
            print(f"🧠 Azure OpenAI: endpoint={self.azure_endpoint[:60]}..., deployment={self.azure_deployment}")
        elif api_key and api_key.startswith("sk-proj-"):
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.is_project_key = True
            self.openai_available = True
            print(f"🧠 OpenAI: Project API key detected (length: {len(api_key)})")
        elif api_key and api_key.startswith("sk-"):
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.openai_available = True
            print(f"🧠 OpenAI: Personal API key detected (length: {len(api_key)})")
        else:
            self.base_url = "https://api.openai.com/v1/chat/completions"
            print(f"🧠 OpenAI: No API key provided (set OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT+AZURE_OPENAI_API_KEY)")
        
        # Initialize Gemini settings
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_available = False
        self.gemini_model_obj = None
        
        # Debug: Print environment variable status
        print(f"🧠 Gemini initialization check:")
        print(f"   GOOGLE_API_KEY from env: {'SET' if self.gemini_api_key else 'NOT SET'}")
        if self.gemini_api_key:
            print(f"   GOOGLE_API_KEY length: {len(self.gemini_api_key)}")
            print(f"   GOOGLE_API_KEY starts with: {self.gemini_api_key[:10]}...")
        
        # Check Gemini availability
        if self.gemini_api_key:
            # Try SDK first if available
            if GEMINI_AVAILABLE and hasattr(genai, 'GenerativeModel'):
                try:
                    print(f"🧠 Gemini: Attempting to configure SDK with API key (length: {len(self.gemini_api_key)})")
                    genai.configure(api_key=self.gemini_api_key)
                    print(f"🧠 Gemini: Creating model object: {self.gemini_model}")
                    self.gemini_model_obj = genai.GenerativeModel(self.gemini_model)
                    self.gemini_available = True
                    print(f"✅ Gemini: SDK initialized successfully, using {self.gemini_model}")
                except Exception as e:
                    print(f"⚠️  Gemini SDK initialization failed: {e}")
                    print(f"   Will use REST API instead (works with Python 3.8)")
                    self.gemini_available = True  # Still available via REST API
                    self.gemini_model_obj = None
            else:
                # SDK not available, but we can use REST API
                print(f"🧠 Gemini: SDK not available (GenerativeModel missing)")
                print(f"   Will use REST API instead (works with Python 3.8)")
                self.gemini_available = True  # Available via REST API
                self.gemini_model_obj = None
        elif GEMINI_AVAILABLE:
            print(f"🧠 Gemini: SDK available but no GOOGLE_API_KEY set")
        else:
            print(f"🧠 Gemini: SDK not installed, but REST API can be used if GOOGLE_API_KEY is set")
        
        # Determine which provider to use
        if provider == "auto":
            # Prefer OpenAI first when both are available (user can set LLM_PROVIDER=gemini to prefer Gemini)
            if self.openai_available:
                self.provider = "openai"
                print(f"🧠 Auto-selected: OpenAI (prefer OpenAI; set LLM_PROVIDER=gemini to prefer Gemini)")
            elif self.gemini_available:
                self.provider = "gemini"
                print(f"🧠 Auto-selected: Gemini (OpenAI not available)")
            else:
                self.provider = None
        else:
            self.provider = provider
        
        if self.provider:
            print(f"🧠 Using provider: {self.provider}")
            print(f"   OpenAI available: {self.openai_available}")
            print(f"   Gemini available: {self.gemini_available}")
            if self.provider == "gemini" and self.openai_available:
                print(
                    f"   💡 LLM_PROVIDER=gemini — OpenAI/Azure ({self.azure_deployment if self.is_azure else self.model}) "
                    f"is configured but skipped. Set LLM_PROVIDER=auto to prefer OpenAI first."
                )
        else:
            print(f"⚠️  No LLM provider available - will use metadata-based fallback")
        
        # Define the search schema for the LLM
        self.search_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the user is looking for"
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key entities mentioned in the query"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories to search in (e.g., wildlife, domestic_animal, livestock, plants, pests)"
                        },
                        "species": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific species mentioned"
                        },
                        "time": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Time of day (dawn, day, dusk, night)"
                        },
                        "season": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Season (spring, summer, fall, winter)"
                        },
                        "action": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Actions or behaviors (walking, eating, sleeping)"
                        },
                        "scene": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Environment or scene (forest, field, water)"
                        },
                        "weather": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Weather conditions"
                        },
                        "plant_state": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Plant/fruit state (ripe, unripe, blooming, fruiting, mixed)"
                        }
                    }
                },
                "description_query": {
                    "type": "string",
                    "description": "A short phrase (5-20 words) describing the desired image in the style of image descriptions, e.g. 'close-up of ripe raspberries on a bush with green leaves'. Used to match and rank results by the description field."
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the understanding (0-1)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the understanding"
                }
            },
            "required": ["intent", "entities", "filters", "confidence", "reasoning"]
        }
    
    async def understand_query(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Convert natural language query to structured understanding"""
        # Try providers in order: OpenAI first (when auto or openai), then Gemini if needed, then OpenAI as fallback if Gemini failed, then metadata-based
        print(f"🧠 understand_query called:")
        print(f"   Query: '{query}'")
        print(f"   Provider: {self.provider}")
        print(f"   OpenAI available: {self.openai_available}")
        print(f"   Gemini available: {self.gemini_available}")
        print(f"   Gemini model object: {self.gemini_model_obj is not None}")
        
        # Try OpenAI first when provider is "openai" or "auto" (auto now prefers OpenAI)
        openai_should_try = (self.provider == "openai") or (self.provider == "auto" and self.openai_available)
        
        if openai_should_try:
            try:
                print(f"🧠 Attempting OpenAI understanding...")
                result = await self._openai_understanding(query, available_filters)
                print(f"✅ OpenAI understanding successful")
                return result
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a quota/billing error
                if "quota" in error_str or "429" in error_str or "insufficient" in error_str:
                    print(f"⚠️  OpenAI quota exceeded - trying Gemini...")
                    # Fall through to try Gemini
                else:
                    print(f"❌ OpenAI error: {e} - trying Gemini...")
                    # Fall through to try Gemini
        
        # Try Gemini if:
        # 1. Provider is "gemini"
        # 2. Provider is "auto" and Gemini is available (after OpenAI was tried and failed or skipped)
        # 3. OpenAI was tried but failed (e.g. quota)
        gemini_should_try = (self.provider == "gemini") or (self.provider == "auto" and self.gemini_available) or (openai_should_try and self.gemini_available)
        
        if gemini_should_try and self.gemini_available:
            try:
                print(f"🧠 Attempting Gemini understanding...")
                print(f"   Provider: {self.provider}")
                print(f"   Gemini model: {self.gemini_model}")
                print(f"   Gemini API key present: {bool(self.gemini_api_key)}")
                print(f"   Gemini model object: {self.gemini_model_obj is not None}")
                # If model object is None, _gemini_understanding will automatically use REST API
                result = await self._gemini_understanding(query, available_filters)
                print(f"✅ Gemini understanding successful (confidence: {result.confidence})")
                return result
            except Exception as e:
                print(f"❌ Gemini error: {e} - trying OpenAI before metadata fallback...")
                import traceback
                traceback.print_exc()
                # Try OpenAI if we haven't already (e.g. provider was gemini and Gemini failed)
                if self.openai_available and not openai_should_try:
                    try:
                        print(f"🧠 Attempting OpenAI understanding (fallback after Gemini failed)...")
                        result = await self._openai_understanding(query, available_filters, fallback_after_gemini=True)
                        print(f"✅ OpenAI understanding successful (fallback)")
                        return result
                    except Exception as e2:
                        print(f"❌ OpenAI fallback error: {e2} - falling back to metadata-based...")
                # Fall through to metadata-based
        else:
            print(f"⚠️  Gemini not being tried:")
            print(f"   - gemini_should_try: {gemini_should_try}")
            print(f"   - gemini_available: {self.gemini_available}")
            print(f"   - provider: {self.provider}")
            print(f"   - GEMINI_AVAILABLE: {GEMINI_AVAILABLE}")
            print(f"   - gemini_api_key set: {bool(self.gemini_api_key)}")
            print(f"   - gemini_model_obj initialized: {self.gemini_model_obj is not None}")
        
        # Fall back to metadata-based understanding (LLM errored or returned invalid response)
        print("⚠️  Using metadata-based query understanding (LLM unavailable or returned invalid response)")
        print(f"   Available filters: {list(available_filters.keys()) if available_filters else 'none'}")
        if available_filters and "species" in available_filters:
            print(f"   Available species: {available_filters['species'][:10]}...")  # Show first 10
        return self._metadata_based_understanding(query, available_filters)

    def _openai_chat_payloads(self, messages: List[Dict[str, str]], max_output: int = 500) -> List[Dict[str, Any]]:
        """Build one or more request bodies compatible with gpt-5-mini / Azure (max_completion_tokens, no temperature)."""
        model_name = (self.azure_deployment if self.is_azure else self.model or "").lower()
        gpt5 = "gpt-5" in model_name
        out_tokens = max(max_output, 64)
        payloads: List[Dict[str, Any]] = []
        if self.is_azure or gpt5:
            payloads.append({"messages": messages, "max_completion_tokens": out_tokens})
            if not gpt5:
                payloads.append({"messages": messages, "max_completion_tokens": out_tokens, "temperature": 0.1})
        payloads.append({"messages": messages, "max_tokens": out_tokens, "temperature": 0.1})
        for body in payloads:
            if not self.is_azure:
                body["model"] = self.model
            elif getattr(self, "is_azure_openai_v1", False):
                body["model"] = self.azure_deployment
        return payloads
    
    async def _openai_understanding(self, query: str, available_filters: Dict[str, List[str]] = None, *, fallback_after_gemini: bool = False) -> QueryUnderstanding:
        """Use OpenAI API for query understanding. When fallback_after_gemini=True, 429 is retried once with short delay then we give up so caller can use metadata-based understanding."""
        system_prompt = f"""You are an expert at understanding natural language queries about agricultural and wildlife datasets.

Your task is to convert user queries into structured search criteria.

CRITICAL FIRST STEP: Categorize the query immediately as one of:
- "pest" (insects, diseases, harmful organisms)
- "wildlife" (wild mammals and birds, e.g. bobcat, coyote, deer — not dogs/cats/goats/chickens)
- "domestic_animal" (dogs, cats — companion animals)
- "livestock" (goats, chickens, cattle, pigs, sheep)
- "animal" (any animal: wildlife + domestic_animal + livestock)
- "plant" (crops, fruits, vegetables, vegetation)

This categorization helps narrow the search space and improves performance. Always set the "category" filter first.

Available filter options:
{json.dumps(available_filters, indent=2) if available_filters else "All filters available"}

Return a JSON response matching this schema:
{json.dumps(self.search_schema, indent=2)}

Examples:
- "bobcat" → {{"intent": "find bobcat images", "entities": ["bobcat"], "filters": {{"species": ["bobcat"]}}, "confidence": 0.9, "reasoning": "Direct species mention"}}
- "bobcat at night" → {{"intent": "find bobcat images at night", "entities": ["bobcat", "night"], "filters": {{"species": ["bobcat"], "time": ["night"]}}, "confidence": 0.95, "reasoning": "Species and time specification"}}
- "coyote looking at the camera" → {{"intent": "find coyote images looking at camera", "entities": ["coyote", "looking at camera"], "filters": {{"species": ["coyote"], "action": ["alert"]}}, "confidence": 0.9, "reasoning": "Species and action specification - looking at camera maps to alert"}}
- "crows" → {{"intent": "find crow images", "entities": ["crow"], "filters": {{"species": ["american_crow"]}}, "confidence": 0.9, "reasoning": "Crow query maps to american_crow species"}}
- "rabbit eating" → {{"intent": "find rabbit images eating", "entities": ["rabbit", "eating"], "filters": {{"species": ["eastern_cottontail"], "action": ["foraging"]}}, "confidence": 0.95, "reasoning": "Rabbit/cottontail synonym: use eastern_cottontail from filters. 'Eating' is action only → foraging"}}
- "cat sleeping" → {{"intent": "find cat images sleeping", "entities": ["cat", "sleeping"], "filters": {{"species": ["cat"], "action": ["sleeping"]}}, "confidence": 0.95, "reasoning": "Species is cat; sleeping is action only (never put sleeping or pin in species)"}}
- "dog eating" → {{"intent": "find dog images eating", "entities": ["dog", "eating"], "filters": {{"species": ["dog"], "action": ["eating"]}}, "confidence": 0.95, "reasoning": "Species is dog; eating is action only (never put eating in species)"}}
- "white cottontail" → {{"intent": "find white cottontail images", "entities": ["white cottontail"], "filters": {{"species": ["eastern_cottontail"]}}, "confidence": 0.95, "reasoning": "White cottontail maps to eastern_cottontail (same as rabbit) from available filters"}}
- "horse at night" or "horse at night time" → {{"intent": "find horse images at night", "entities": ["horse", "night"], "filters": {{"species": ["horse"], "time": ["night"]}}, "confidence": 0.95, "reasoning": "Species and time; use canonical time 'night' not 'Nighttime'"}}
- "animals in summer forest" → {{"intent": "find wildlife in summer forest environment", "entities": ["animals", "summer", "forest"], "filters": {{"category": ["wildlife"], "season": ["summer"], "scene": ["forest"]}}, "confidence": 0.8, "reasoning": "Combined environmental and seasonal criteria"}}
- "goats in the field" → {{"intent": "find goat images in field environment", "entities": ["goat", "field"], "filters": {{"species": ["goat"], "scene": ["field"]}}, "confidence": 0.9, "reasoning": "Species and scene specification - 'in the field' indicates field scene"}}
- "bird on a tree" → {{"intent": "find bird images in tree", "entities": ["bird", "tree"], "filters": {{"species": ["american_crow"], "scene": ["tree"]}}, "confidence": 0.9, "reasoning": "Subject is bird; 'on a tree' is scene, not species"}}
- "predators hunting at dawn" → {{"intent": "find hunting predators at dawn", "entities": ["predators", "hunting", "dawn"], "filters": {{"action": ["hunting"], "time": ["dawn"]}}, "confidence": 0.85, "reasoning": "Behavior and time specification"}}
- "raspberry ripe" → {{"intent": "find ripe raspberry images", "entities": ["raspberry", "ripe"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "description_query": "close-up of ripe raspberries on a bush with green leaves", "confidence": 0.95, "reasoning": "Species and ripeness specification - 'ripe' maps to plant_state, not action"}}
- "raspberry red" → {{"intent": "find red/ripe raspberry images", "entities": ["raspberry", "red"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "description_query": "close-up of red ripe raspberries and green foliage", "confidence": 0.9, "reasoning": "Species and color specification - 'red' for berries indicates ripe, maps to plant_state"}}

DESCRIPTION_QUERY (important for ranking):
- Always provide "description_query" when the user describes what they want to see. It should be a short phrase (5-20 words) in the style of image descriptions in our dataset.
- Style: start with "close-up of" or "image of", include the subject (e.g. raspberry, bobcat), key visual traits (ripe, red, green leaves, at night), and setting if relevant.
- Examples: "close-up of ripe raspberries on a bush with green leaves", "bobcat at night in forest", "coyote looking at camera in field".
- This phrase is matched against each image's description field to rank results so the best-matching images appear first.

IMPORTANT:
{LLM_SPECIES_AND_NORMALIZATION_RULES}
- Extract species names EXACTLY as they appear in available filters (e.g., "bobcat", "coyote", "american_crow", "american_black_bear", "strawberry", "raspberry", "eastern_cottontail")
- "crow" or "crows" should map to "american_crow" if that's the available species name. "american bear", "black bear", or "american black bear" should map to "american_black_bear" if available.
- ONLY extract species that are in the available_filters list - if a species is not in available_filters, DO NOT extract it as a species filter
- If the query mentions a species that is NOT in available_filters, return an empty species filter and explain in reasoning that the species is not available
- Map action keywords to canonical action names:
  * "feeding", "eating" → "foraging"
  * "sleeping", "resting" → "sleeping" or "resting"
  * "looking at camera", "looking at the camera", "staring at camera", "facing camera", "looking toward camera" → "alert"
  * "walking", "moving" → "walking" or "moving"
- Extract scene keywords from phrases like "in the field" → scene: ["field"], "in forest" → scene: ["forest"], "in garden" → scene: ["garden"], "on a tree" → scene: ["tree"]. For "X on a tree" (e.g. "bird on a tree") the subject is X (species), not tree — use species: [X], scene: ["tree"].
- Scene keywords: field, forest, water, mountain, garden, farm, meadow, indoor, outdoor, tree
- For PLANT/FRUIT queries (raspberry, strawberry, etc.), map ripeness/color descriptors to plant_state, NOT action:
  * "ripe", "red", "mature" → plant_state: ["ripe"] (for fruits/berries)
  * "unripe", "green", "immature" → plant_state: ["unripe"] (for fruits/berries)
  * "raspberry ripe" → species: ["raspberry"], plant_state: ["ripe"] (NOT action!)
  * "raspberry red" → species: ["raspberry"], plant_state: ["ripe"] (red = ripe for berries)
- Only use filters that are explicitly mentioned in the query
- Do NOT infer or add filters that are not mentioned
- Species names must match exactly (case-insensitive) with available filter values

Query: "{query}"
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Prepare headers: Azure uses api-key, OpenAI uses Authorization Bearer
        headers = {"Content-Type": "application/json"}
        if self.is_azure:
            headers["api-key"] = self.azure_api_key
            print(f"🧠 Using Azure OpenAI (api-key) authentication")
        elif self.is_project_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"🧠 Using project API key authentication")
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"🧠 Using personal API key authentication")
        
        # Azure uses deployment in URL (legacy) or model in body (OpenAI-compatible /openai/v1); OpenAI expects "model" in body.
        payloads = self._openai_chat_payloads(messages, max_output=500)
        
        max_retries = 1 if fallback_after_gemini else 3
        last_error = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = None
                    for body in payloads:
                        response = await client.post(
                            self.base_url,
                            headers=headers,
                            json=body,
                            timeout=90.0
                        )
                        if response.status_code == 400 and len(payloads) > 1:
                            continue
                        break
                    if response is None:
                        raise ValueError("OpenAI API request failed: no response")
                    
                    print(f"🧠 OpenAI API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        try:
                            # Parse the JSON response
                            parsed = json.loads(content)
                            parsed.setdefault("description_query", None)
                            understanding = QueryUnderstanding(**parsed)
                            
                            # Validate that we got real LLM understanding (not empty)
                            if not understanding.filters and not understanding.entities:
                                raise ValueError("LLM returned empty understanding")
                            
                            return understanding
                        except Exception as e:
                            print(f"❌ Failed to parse LLM response: {e}")
                            print(f"🧠 Raw response content: {content}")
                            raise ValueError(f"Failed to parse LLM response: {e}")
                    elif response.status_code == 429:
                        # Rate limit: retry after delay (Azure often suggests 60 seconds)
                        last_error = ValueError(f"OpenAI API error: 429 - {response.text}")
                        retry_after = 10 if fallback_after_gemini else 65
                        try:
                            hint = response.json()
                            if "retry" in str(hint).lower() and "second" in str(hint).lower():
                                import re
                                m = re.search(r"(\d+)\s*second", response.text, re.I)
                                if m:
                                    retry_after = min(120, max(10 if fallback_after_gemini else 65, int(m.group(1)) + 5))
                        except Exception:
                            pass
                        if attempt < max_retries - 1:
                            print(f"   ⏳ Rate limited (429). Retrying after {retry_after} seconds (attempt {attempt + 1}/{max_retries})...")
                            await asyncio.sleep(retry_after)
                            continue
                        if fallback_after_gemini:
                            print(f"   ⚠️  429 on OpenAI fallback — giving up after one retry; will use metadata-based understanding.")
                        print(f"❌ OpenAI API error: {response.status_code}")
                        print(f"🧠 Error response: {response.text}")
                        raise last_error
                    else:
                        print(f"❌ OpenAI API error: {response.status_code}")
                        print(f"🧠 Error response: {response.text}")
                        if response.status_code == 404 and self.is_azure:
                            print(f"   💡 Azure 404 = resource not found (not an auth problem). Check: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT (must match your deployment name in Azure portal), AZURE_OPENAI_API_VERSION")
                        raise ValueError(f"OpenAI API error: {response.status_code} - {response.text}")
            except ValueError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"   ⏳ Request failed: {e}. Retrying in 5 seconds (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(5)
                    continue
                print(f"❌ OpenAI API request failed: {e}")
                raise
        
        if last_error:
            raise last_error
    
    async def _gemini_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Use Google Gemini API for query understanding"""
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        # Try using the REST API directly if GenerativeModel is not available
        if not GEMINI_AVAILABLE or not hasattr(genai, 'GenerativeModel'):
            # Fallback to REST API
            return await self._gemini_rest_api_understanding(query, available_filters)
        
        if not self.gemini_model_obj:
            # Try to reinitialize if model object is None
            print(f"⚠️  Gemini model object is None, attempting to reinitialize...")
            print(f"   API key present: {bool(self.gemini_api_key)}")
            print(f"   Model name: {self.gemini_model}")
            try:
                if not self.gemini_api_key:
                    raise ValueError("GOOGLE_API_KEY not set - cannot reinitialize")
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model_obj = genai.GenerativeModel(self.gemini_model)
                self.gemini_available = True
                print(f"✅ Gemini model object reinitialized successfully")
            except Exception as e:
                print(f"❌ Failed to reinitialize Gemini model: {e}")
                print(f"   Falling back to REST API...")
                # Fallback to REST API
                return await self._gemini_rest_api_understanding(query, available_filters)
        
        system_prompt = f"""You are an expert at understanding natural language queries about agricultural and wildlife datasets.

Your task is to convert user queries into structured search criteria.

Available filter options:
{json.dumps(available_filters, indent=2) if available_filters else "All filters available"}

Return a JSON response matching this schema:
{json.dumps(self.search_schema, indent=2)}

Examples:
- "bobcat" → {{"intent": "find bobcat images", "entities": ["bobcat"], "filters": {{"species": ["bobcat"]}}, "confidence": 0.9, "reasoning": "Direct species mention"}}
- "bobcat at night" → {{"intent": "find bobcat images at night", "entities": ["bobcat", "night"], "filters": {{"species": ["bobcat"], "time": ["night"]}}, "confidence": 0.95, "reasoning": "Species and time specification"}}
- "coyote looking at the camera" → {{"intent": "find coyote images looking at camera", "entities": ["coyote", "looking at camera"], "filters": {{"species": ["coyote"], "action": ["alert"]}}, "confidence": 0.9, "reasoning": "Species and action specification - looking at camera maps to alert"}}
- "crows" → {{"intent": "find crow images", "entities": ["crow"], "filters": {{"species": ["american_crow"]}}, "confidence": 0.9, "reasoning": "Crow query maps to american_crow species"}}
- "rabbit eating" → {{"intent": "find rabbit images eating", "entities": ["rabbit", "eating"], "filters": {{"species": ["eastern_cottontail"], "action": ["foraging"]}}, "confidence": 0.95, "reasoning": "Rabbit/cottontail synonym: use eastern_cottontail from filters. 'Eating' is action only → foraging"}}
- "cat sleeping" → {{"intent": "find cat images sleeping", "entities": ["cat", "sleeping"], "filters": {{"species": ["cat"], "action": ["sleeping"]}}, "confidence": 0.95, "reasoning": "Species is cat; sleeping is action only (never put sleeping or pin in species)"}}
- "dog eating" → {{"intent": "find dog images eating", "entities": ["dog", "eating"], "filters": {{"species": ["dog"], "action": ["eating"]}}, "confidence": 0.95, "reasoning": "Species is dog; eating is action only (never put eating in species)"}}
- "white cottontail" → {{"intent": "find white cottontail images", "entities": ["white cottontail"], "filters": {{"species": ["eastern_cottontail"]}}, "confidence": 0.95, "reasoning": "White cottontail maps to eastern_cottontail (same as rabbit) from available filters"}}
- "horse at night" or "horse at night time" → {{"intent": "find horse images at night", "entities": ["horse", "night"], "filters": {{"species": ["horse"], "time": ["night"]}}, "confidence": 0.95, "reasoning": "Species and time; use canonical time 'night' not 'Nighttime'"}}
- "animals in summer forest" → {{"intent": "find wildlife in summer forest environment", "entities": ["animals", "summer", "forest"], "filters": {{"category": ["wildlife"], "season": ["summer"], "scene": ["forest"]}}, "confidence": 0.8, "reasoning": "Combined environmental and seasonal criteria"}}
- "goats in the field" → {{"intent": "find goat images in field environment", "entities": ["goat", "field"], "filters": {{"species": ["goat"], "scene": ["field"]}}, "confidence": 0.9, "reasoning": "Species and scene specification - 'in the field' indicates field scene"}}
- "bird on a tree" → {{"intent": "find bird images in tree", "entities": ["bird", "tree"], "filters": {{"species": ["american_crow"], "scene": ["tree"]}}, "confidence": 0.9, "reasoning": "Subject is bird; 'on a tree' is scene, not species"}}
- "predators hunting at dawn" → {{"intent": "find hunting predators at dawn", "entities": ["predators", "hunting", "dawn"], "filters": {{"action": ["hunting"], "time": ["dawn"]}}, "confidence": 0.85, "reasoning": "Behavior and time specification"}}
- "raspberry ripe" → {{"intent": "find ripe raspberry images", "entities": ["raspberry", "ripe"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and ripeness specification - 'ripe' maps to plant_state, not action"}}
- "raspberry red" → {{"intent": "find red/ripe raspberry images", "entities": ["raspberry", "red"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.9, "reasoning": "Species and color specification - 'red' for berries indicates ripe, maps to plant_state"}}
- "raspberries that can be eaten" → {{"intent": "find edible/ripe raspberry images", "entities": ["raspberry", "edible"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and edibility specification - 'can be eaten' for fruits/berries means ripe, maps to plant_state"}}
- "edible raspberries" → {{"intent": "find edible/ripe raspberry images", "entities": ["raspberry", "edible"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and edibility specification - 'edible' for fruits/berries means ripe, maps to plant_state"}}
- "red raspberry" → {{"intent": "find red/ripe raspberry images", "entities": ["raspberry", "red"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "description_query": "close-up of red ripe raspberries and green foliage", "confidence": 0.9, "reasoning": "Species and color specification - 'red' for berries indicates ripe, maps to plant_state"}}

DESCRIPTION_QUERY: Always provide "description_query" when the user describes what they want to see: a short phrase (5-20 words) in the style of image descriptions (e.g. "close-up of ripe raspberries on a bush with green leaves", "bobcat at night in forest"). Used to match and rank results by the image description field.

IMPORTANT:
{LLM_SPECIES_AND_NORMALIZATION_RULES}
- Extract species names EXACTLY as they appear in available filters (e.g., "bobcat", "coyote", "american_crow", "american_black_bear", "strawberry", "raspberry", "eastern_cottontail")
- "crow" or "crows" should map to "american_crow" if that's the available species name. "american bear", "black bear", or "american black bear" should map to "american_black_bear" if available.
- ONLY extract species that are in the available_filters list - if a species is not in available_filters, DO NOT extract it as a species filter
- If the query mentions a species that is NOT in available_filters, return an empty species filter and explain in reasoning that the species is not available
- For PLANT/FRUIT queries (raspberry, strawberry, etc.), map ripeness/color/edibility descriptors to plant_state, NOT action:
  * "ripe", "red", "mature" → plant_state: ["ripe"] (for fruits/berries)
  * "unripe", "green", "immature" → plant_state: ["unripe"] (for fruits/berries)
  * "edible", "can be eaten", "ready to eat", "ready for eating" → plant_state: ["ripe"] (for fruits/berries - edible means ripe)
  * "raspberry ripe" → species: ["raspberry"], plant_state: ["ripe"] (NOT action!)
  * "raspberry red" → species: ["raspberry"], plant_state: ["ripe"] (red = ripe for berries)
  * "raspberries that can be eaten" → species: ["raspberry"], plant_state: ["ripe"] (edible = ripe for fruits)
  * "edible raspberries" → species: ["raspberry"], plant_state: ["ripe"] (edible = ripe for fruits)
- For edibility queries about fruits/berries, you MUST infer plant_state: ["ripe"] - this is a semantic mapping, not arbitrary inference
- Only use filters that are explicitly mentioned in the query OR semantically implied (edibility → ripeness for fruits)
- Species names must match exactly (case-insensitive) with available filter values
- Return ONLY valid JSON, no markdown formatting or code blocks

Query: "{query}"
"""
        
        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nReturn the JSON response:"
        
        try:
            # Use Gemini to generate response (run in thread since it's sync)
            def call_gemini():
                return self.gemini_model_obj.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 500,
                    }
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, call_gemini)
            
            content = response.text.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to extract JSON if response contains extra text
            # Look for JSON object boundaries
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            # Parse the JSON response
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Try to fix common JSON issues
                # Remove trailing commas before closing braces/brackets
                import re
                content_fixed = re.sub(r',(\s*[}\]])', r'\1', content)
                try:
                    parsed = json.loads(content_fixed)
                except:
                    # Last resort: show more context for debugging
                    print(f"🧠 Failed to parse JSON. Content (first 500 chars): {content[:500]}")
                    raise json_err
            
            parsed.setdefault("description_query", None)
            understanding = QueryUnderstanding(**parsed)
            
            # Validate that we got real LLM understanding
            if not understanding.filters and not understanding.entities:
                raise ValueError("Gemini returned empty understanding")
            
            print(f"🧠 Gemini understanding successful (confidence: {understanding.confidence})")
            return understanding
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Gemini response: {e}")
            print(f"🧠 Raw response content: {content[:200]}...")
            raise ValueError(f"Failed to parse Gemini response: {e}")
        except Exception as e:
            print(f"❌ Gemini API request failed: {e}")
            raise
    
    async def _gemini_rest_api_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Use Gemini REST API directly (works with Python 3.8 and old SDK versions)"""
        system_prompt = f"""You are an expert at understanding natural language queries about agricultural and wildlife datasets.

Your task is to convert user queries into structured search criteria.

Available filter options:
{json.dumps(available_filters, indent=2) if available_filters else "All filters available"}

Return a JSON response matching this schema:
{json.dumps(self.search_schema, indent=2)}

Examples:
- "bobcat" → {{"intent": "find bobcat images", "entities": ["bobcat"], "filters": {{"species": ["bobcat"]}}, "confidence": 0.9, "reasoning": "Direct species mention"}}
- "bobcat at night" → {{"intent": "find bobcat images at night", "entities": ["bobcat", "night"], "filters": {{"species": ["bobcat"], "time": ["night"]}}, "confidence": 0.95, "reasoning": "Species and time specification"}}
- "pigs feeding" → {{"intent": "find pig images feeding", "entities": ["pig", "feeding"], "filters": {{"species": ["pig"], "action": ["foraging"]}}, "confidence": 0.9, "reasoning": "Species and action specification - feeding maps to foraging"}}
- "coyote looking at the camera" → {{"intent": "find coyote images looking at camera", "entities": ["coyote", "looking at camera"], "filters": {{"species": ["coyote"], "action": ["alert"]}}, "confidence": 0.9, "reasoning": "Species and action specification - looking at camera maps to alert"}}
- "raspberry ripe" → {{"intent": "find ripe raspberry images", "entities": ["raspberry", "ripe"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and ripeness specification - 'ripe' maps to plant_state, not action"}}
- "raspberry red" → {{"intent": "find red/ripe raspberry images", "entities": ["raspberry", "red"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.9, "reasoning": "Species and color specification - 'red' for berries indicates ripe, maps to plant_state"}}
- "raspberries that can be eaten" → {{"intent": "find edible/ripe raspberry images", "entities": ["raspberry", "edible"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and edibility specification - 'can be eaten' for fruits/berries means ripe, maps to plant_state"}}
- "edible raspberries" → {{"intent": "find edible/ripe raspberry images", "entities": ["raspberry", "edible"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "confidence": 0.95, "reasoning": "Species and edibility specification - 'edible' for fruits/berries means ripe, maps to plant_state"}}
- "red raspberry" → {{"intent": "find red/ripe raspberry images", "entities": ["raspberry", "red"], "filters": {{"species": ["raspberry"], "plant_state": ["ripe"]}}, "description_query": "close-up of red ripe raspberries and green foliage", "confidence": 0.9, "reasoning": "Species and color specification - 'red' for berries indicates ripe, maps to plant_state"}}

DESCRIPTION_QUERY: Always provide "description_query" when the user describes what they want to see: a short phrase (5-20 words) in the style of image descriptions (e.g. "close-up of ripe raspberries on a bush with green leaves", "bobcat at night in forest"). Used to match and rank results by the image description field.

IMPORTANT:
{LLM_SPECIES_AND_NORMALIZATION_RULES}
- Extract species names EXACTLY as they appear in available filters (e.g., "bobcat", "coyote", "american_crow", "american_black_bear", "strawberry", "raspberry", "eastern_cottontail")
- "crow" or "crows" should map to "american_crow" if that's the available species name. "american bear", "black bear", or "american black bear" should map to "american_black_bear" if available.
- ONLY extract species that are in the available_filters list - if a species is not in available_filters, DO NOT extract it as a species filter
- If the query mentions a species that is NOT in available_filters, return an empty species filter and explain in reasoning that the species is not available
- Map action keywords to canonical action names:
  * "feeding", "eating" → "foraging"
  * "sleeping", "resting" → "sleeping" or "resting"
  * "looking at camera", "looking at the camera", "staring at camera", "facing camera", "looking toward camera" → "alert"
  * "walking", "moving" → "walking" or "moving"
- Extract scene keywords from phrases like "in the field" → scene: ["field"], "in forest" → scene: ["forest"], "in garden" → scene: ["garden"], "on a tree" → scene: ["tree"]. For "X on a tree" (e.g. "bird on a tree") the subject is X (species), not tree — use species: [X], scene: ["tree"].
- Scene keywords: field, forest, water, mountain, garden, farm, meadow, indoor, outdoor, tree
- For PLANT/FRUIT queries (raspberry, strawberry, etc.), map ripeness/color/edibility descriptors to plant_state, NOT action:
  * "ripe", "red", "mature" → plant_state: ["ripe"] (for fruits/berries)
  * "unripe", "green", "immature" → plant_state: ["unripe"] (for fruits/berries)
  * "edible", "can be eaten", "ready to eat", "ready for eating" → plant_state: ["ripe"] (for fruits/berries - edible means ripe)
  * "raspberry ripe" → species: ["raspberry"], plant_state: ["ripe"] (NOT action!)
  * "raspberry red" → species: ["raspberry"], plant_state: ["ripe"] (red = ripe for berries)
  * "raspberries that can be eaten" → species: ["raspberry"], plant_state: ["ripe"] (edible = ripe for fruits)
  * "edible raspberries" → species: ["raspberry"], plant_state: ["ripe"] (edible = ripe for fruits)
- For edibility queries about fruits/berries, you MUST infer plant_state: ["ripe"] - this is a semantic mapping, not arbitrary inference
- Only use filters that are explicitly mentioned in the query OR semantically implied (edibility → ripeness for fruits)
- Species names must match exactly (case-insensitive) with available filter values
- Return ONLY valid JSON, no markdown formatting or code blocks
- Keep the JSON compact: use a single line if possible and do not put newline characters inside string values (avoids truncation)

Query: "{query}"
"""

        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nReturn the JSON response:"
        
        # Use Gemini REST API v1
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        params = {
            "key": self.gemini_api_key
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, params=params, json=payload, timeout=30.0)
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"❌ Gemini REST API error: {response.status_code}")
                    print(f"   Response: {error_text}")
                    raise ValueError(f"Gemini API error: {response.status_code} - {error_text}")
                
                result = response.json()
                
                # Extract text from response
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            content = parts[0]["text"].strip()
                        else:
                            raise ValueError("No text in Gemini response")
                    else:
                        raise ValueError("No content in Gemini response candidate")
                else:
                    raise ValueError("No candidates in Gemini response")
                
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # Try to extract JSON if response contains extra text
                # Look for JSON object boundaries
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    content = content[json_start:json_end]
                
                # Parse the JSON response
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as json_err:
                    import re
                    content_fixed = re.sub(r',(\s*[}\]])', r'\1', content)
                    parsed = None
                    try:
                        parsed = json.loads(content_fixed)
                    except json.JSONDecodeError:
                        pass
                    # Optional: use json_repair for robust fix of missing commas, trailing commas, etc.
                    if parsed is None and JSON_REPAIR_AVAILABLE and json_repair is not None:
                        try:
                            parsed = json_repair.loads(content_fixed)
                        except Exception:
                            pass
                    # Repair missing comma between tokens (e.g. "key": "value" "key2" -> add comma before "key2")
                    if parsed is None and "Expecting ',' delimiter" in str(json_err):
                        # Fix adjacent quoted strings with only whitespace (missing comma)
                        content_comma = re.sub(r'"(\s+)"', r'", "', content_fixed)
                        try:
                            parsed = json.loads(content_comma)
                        except json.JSONDecodeError:
                            pass
                        # Try inserting comma at error position (use original content; position is from first parse)
                        if parsed is None:
                            pos_match = re.search(r'char\s+(\d+)', str(json_err))
                            if pos_match:
                                pos = int(pos_match.group(1))
                                for src in (content, content_fixed):
                                    if parsed is not None:
                                        break
                                    for insert_at in (pos, max(0, pos - 1), max(0, pos - 2), min(len(src), pos + 1)):
                                        if 0 <= insert_at <= len(src):
                                            candidate = src[:insert_at] + ',' + src[insert_at:]
                                            try:
                                                parsed = json.loads(candidate)
                                                break
                                            except json.JSONDecodeError:
                                                continue
                    # Repair truncated JSON (Gemini sometimes stops mid-response)
                    if parsed is None and ("Unterminated" in str(json_err) or "Expecting" in str(json_err)):
                        # Option A: truncate at error position if reported (e.g. "char 60")
                        pos_match = re.search(r'char\s+(\d+)', str(json_err))
                        if pos_match:
                            pos = int(pos_match.group(1))
                            truncated = content_fixed[:pos].rstrip().rstrip(',').rstrip()
                            if truncated.startswith('{') and not truncated.endswith('}'):
                                truncated += '}'
                                try:
                                    parsed = json.loads(truncated)
                                except json.JSONDecodeError:
                                    parsed = None
                                if parsed is not None:
                                    parsed.setdefault("intent", query[:100] if query else "search")
                                    parsed.setdefault("entities", [])
                                    parsed.setdefault("filters", {})
                                    parsed.setdefault("confidence", 0.7)
                                    parsed.setdefault("reasoning", "Response was truncated; partial parse used.")
                                    parsed.setdefault("description_query", None)
                        # Option B: remove incomplete trailing key/value and close
                        if parsed is None:
                            repair = content_fixed.strip()
                            if repair and repair.startswith('{') and not repair.rstrip().endswith('}'):
                                repair = re.sub(r',\s*"[^"]*$', '', repair)
                                repair = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', repair)
                                repair = repair.rstrip().rstrip(',').rstrip()
                                if repair.startswith('{') and not repair.endswith('}'):
                                    repair += '}'
                                try:
                                    parsed = json.loads(repair)
                                except json.JSONDecodeError:
                                    parsed = None
                            if parsed is not None:
                                parsed.setdefault("intent", query[:100] if query else "search")
                                parsed.setdefault("entities", [])
                                parsed.setdefault("filters", {})
                                parsed.setdefault("confidence", 0.7)
                                parsed.setdefault("reasoning", "Response was truncated; partial parse used.")
                                parsed.setdefault("description_query", None)
                    if parsed is None:
                        print(f"🧠 Failed to parse JSON. Content (first 500 chars): {content[:500]}")
                        raise json_err
                
                parsed.setdefault("description_query", None)
                understanding = QueryUnderstanding(**parsed)
                
                # Validate that we got real LLM understanding
                if not understanding.filters and not understanding.entities:
                    raise ValueError("Gemini returned empty understanding")
                
                print(f"🧠 Gemini REST API understanding successful (confidence: {understanding.confidence})")
                return understanding
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Gemini REST API response: {e}")
            print(f"🧠 Raw response content: {content[:200] if 'content' in locals() else 'N/A'}...")
            raise ValueError(f"Failed to parse Gemini response: {e}")
        except Exception as e:
            print(f"❌ Gemini REST API request failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _metadata_based_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Metadata-based query understanding using actual MCP metadata values"""
        import re
        query_lower = query.lower()
        
        entities = []
        filters = {
            "category": [],
            "species": [],
            "time": [],
            "season": [],
            "action": [],
            "scene": [],
            "weather": [],
            "plant_state": []
        }
        
        if not available_filters:
            available_filters = {}
        
        # Match query against actual available filter values from MCP metadata
        # This ensures we only match what actually exists in the data
        
        # Match species from available filters (MCP uses "species" key)
        # Species are extracted from both "species" field and "collection" field
        species_to_check = []
        if "species" in available_filters:
            species_to_check.extend(available_filters["species"])
            print(f"   🔍 Available species from filters: {available_filters['species'][:10]}...")  # Show first 10
        
        # Also check if collections are available (they often contain species names)
        if "collections" in available_filters:
            print(f"   🔍 Available collections: {available_filters['collections'][:10]}...")  # Show first 10
            for coll in available_filters["collections"]:
                # Extract base species name from collection (e.g., "bobcat_001" -> "bobcat", "Red_fox" -> "red")
                coll_base = coll.split("_")[0].split("-")[0].strip().lower()
                # Normalize: remove underscores, convert to lowercase
                coll_normalized = coll_base.replace("_", "").replace("-", "").lower()
                if coll_normalized not in [s.lower().replace("_", "").replace("-", "") for s in species_to_check]:
                    species_to_check.append(coll_normalized)
        
        print(f"   🔍 Total species to check: {len(species_to_check)}")
        print(f"   🔍 Query: '{query_lower}'")
        
        # Common species names to check even if not in available_filters
        # This helps when the species exists in data but wasn't extracted to filters
        # Include both singular and plural forms, and variations
        # Rabbit, cottontail, and white cottontail all map to eastern_cottontail (never white_cottontail as filter value)
        # Pest type words (beetle, butterfly, wasp, etc.) match via common_names so "show me beetles" finds pest images
        common_species = [
            "bobcat", "bobcats",
            "cat", "cats", "dog", "dogs",  # domestic cat/dog for "cat eating", "dog eating"
            "bird", "birds",  # for "bird on a tree" etc.
            "coyote", "coyotes",
            "deer",
            "fox", "foxes", "red_fox", "redfox", "red_foxes",
            "crow", "crows", "american_crow", "american_crows",
            "american bear", "american bears", "black bear", "black bears", "american black bear", "american_black_bear",
            "strawberry", "strawberries",
            "raspberry", "raspberries",
            "chicken", "chickens",
            "goat", "goats",
            "carrot", "carrots",
            "pig", "pigs",
            "rabbit", "rabbits", "cottontail", "cottontails", "eastern_cottontail",
            "white_cottontail", "white_cottontails",  # query trigger only; resolved to eastern_cottontail
            "opossum", "opossums", "oppossum", "oppossums", "virginia_opossum",
            # Pest types (matched via common_names in MCP data)
            "beetle", "beetles", "butterfly", "butterflies", "moth", "moths",
            "wasp", "wasps", "bee", "bees", "ant", "ants", "fly", "flies",
            "grasshopper", "grasshoppers", "dragonfly", "dragonflies",
            "spider", "spiders", "stink bug", "stink bugs", "true bug", "bugs", "insect", "insects",
        ]
        
        # Add common species to check if they're in the query
        # Handle pluralization and variations
        query_normalized = query_lower.replace("_", "").replace("-", "")
        for common in common_species:
            common_normalized = common.replace("_", "").replace("-", "").lower()
            # Check if the normalized common species is in the normalized query
            # Also check if query contains the base word (e.g., "strawberries" contains "strawberry")
            base_word = common_normalized.rstrip('s')  # Remove trailing 's' for plural matching
            if (common_normalized in query_normalized or 
                (len(base_word) >= 4 and base_word in query_normalized) or
                query_normalized in common_normalized):
                # Check if it's not already in species_to_check
                already_in_list = any(
                    s.replace("_", "").replace("-", "").lower() == common_normalized 
                    for s in species_to_check
                )
                if not already_in_list:
                    species_to_check.append(common)
                    print(f"   🔍 Added common species to check: {common} (matched from query)")
        
        # Match species in query - try multiple matching strategies
        # Important: We need to match the query to actual filter values in available_filters
        species_matched = False
        query_normalized = query_lower.replace("_", "").replace("-", "").replace(" ", "")
        query_word_count = len(query_lower.split())
        
        # EXACT MATCH FIRST: For single-word queries, if query (or singular form) is exactly a species/dataset name, use it.
        # This guarantees "carrots" -> "carrot" (crop) and never "carrot seed moth" (pest), even with 3500+ species in the list.
        if "species" in available_filters and query_word_count <= 1:
            query_stem = query_lower.strip()
            candidates = [query_stem]
            if query_stem.endswith("s") and len(query_stem) > 1:
                candidates.append(query_stem[:-1])  # carrots -> carrot
            for c in candidates:
                for s in available_filters["species"]:
                    if s.lower().strip() == c:
                        entities.append(s)
                        filters["species"].append(s)
                        print(f"   ✅ Matched species (exact): '{s}' from query '{query_lower}'")
                        species_matched = True
                        break
                if species_matched:
                    break
        
        # Words that are actions, not species - never match these as species (e.g. "eating" in "rabbit eating")
        action_words = {
            "eating", "feeding", "foraging", "standing", "walking", "running", "sitting", "sleeping",
            "resting", "moving", "alert", "hunting", "perching", "flying", "blooming", "fruiting",
            "growing", "mature", "stretching", "reaching", "consuming", "lowering", "facing", "engaging",
        }
        # Stopwords / nonsense that must never be matched as species (from action descriptions leaking into filter extraction)
        species_stopwords = {
            "the", "and", "or", "at", "in", "on", "to", "for", "of", "with", "within", "from", "an", "a",
            "possibly", "appears", "stationary", "looking", "staring", "camera", "enclosure", "animal",
            "toward", "directly", "king", "pea", "tan", "thin",
        }
        # First, try to match against actual available filter values (unless we already exact-matched)
        # Single-word query: prefer SHORT/one-word species. Multi-word: prefer more words/longest first.
        if "species" in available_filters and not species_matched:
            def sort_key(s):
                word_count = s.count('_') + s.count('-') + s.count(' ') + 1
                if query_word_count <= 1:
                    # Prefer fewer words, then shorter: "carrot" before "carrot seed moth"
                    return (word_count, len(s), s.lower())
                # Prefer more words, then longest: "red_leaf" before "red"
                return (-word_count, -len(s), s.lower())
            sorted_species = sorted(available_filters["species"], key=sort_key)
            
            for filter_species in sorted_species:
                filter_species_lower = filter_species.lower().strip()
                # Skip filter values that are action verbs (e.g. "eating" in "rabbit eating")
                if filter_species_lower in action_words:
                    continue
                # Skip stopwords / nonsense that leaked from action descriptions into species filters
                if filter_species_lower in species_stopwords:
                    continue
                # "bird on a tree" → species is bird, not tree/trees; skip matching "tree"/"trees" as species when query has "bird"
                if filter_species_lower in ("tree", "trees") and "bird" in query_lower:
                    continue
                filter_species_normalized = filter_species_lower.replace("_", "").replace("-", "").replace(" ", "")
                
                # Replace underscores/hyphens with spaces for word boundary matching
                species_words = filter_species_lower.replace("_", " ").replace("-", " ")
                
                # Strategy 1: Handle plurals FIRST with word boundaries (e.g., "raspberries" -> "raspberry")
                # This needs to come first so plurals are normalized before simple word matches
                base_filter = filter_species_normalized.rstrip('s')
                if len(base_filter) >= 4:  # Only check plurals for words with 4+ chars (avoid "red" -> "re")
                    base_species_words = species_words.rstrip('s').rstrip(' ')
                    if base_species_words and len(base_species_words) >= 4:
                        # Match plural forms: handle both regular (s) and irregular (ies, es) plurals
                        # Pattern: match "raspberry", "raspberrys", "raspberries", "raspberryes"
                        # Use word boundary at start, allow 's', 'ies', 'es' at end, then word boundary
                        escaped_base = re.escape(base_species_words)
                        # Match: word boundary + base + (s|ies|es) + word boundary
                        plural_patterns = [
                            r'\b' + escaped_base + r'(?:ies|es|s)\b',  # Plural forms
                            r'\b' + escaped_base + r'\b',  # Singular form
                        ]
                        for pattern in plural_patterns:
                            if re.search(pattern, query_lower):
                                # Store the canonical (singular) form from the filter
                                entities.append(filter_species)
                                filters["species"].append(filter_species)
                                print(f"   ✅ Matched species from available filters (plural): {filter_species} from query '{query_lower}' (pattern: {pattern})")
                                species_matched = True
                                break
                        if species_matched:
                            break
                
                # Strategy 2: Exact word boundary matching (handles "raspberry" in "red raspberries", "red leaf" → "red_leaf")
                # But skip very short words (3 chars or less) if there are longer potential matches
                # This prevents "red" from matching in "red raspberries" before "raspberry" is checked
                if len(filter_species_normalized) >= 4 or len(sorted_species) == 1:
                    pattern = r'\b' + re.escape(species_words) + r'\b'
                    match = re.search(pattern, query_lower)
                    if match:
                        # Prefer exact matches: if query matches species_words exactly, use it
                        # This ensures "red leaf" matches "red_leaf" rather than "red_leaf_lettuce"
                        matched_text = match.group()
                        # Check if this is an exact match (all words in species_words are in query)
                        species_word_list = species_words.split()
                        if len(species_word_list) == 1 or all(word in query_lower for word in species_word_list):
                            entities.append(filter_species)
                            filters["species"].append(filter_species)
                            print(f"   ✅ Matched species from available filters (word boundary): {filter_species} from query '{query_lower}'")
                            species_matched = True
                            break
                
                # Strategy 3: For very short species names (3 chars), only match if it's the ONLY word in query
                # or if it's a compound like "red_fox" or "red_leaf"
                if len(filter_species_normalized) <= 3:
                    # Only match short names if they're compound (contain underscore) or if query is just that word
                    if "_" in filter_species or filter_species_lower == query_lower.strip():
                        pattern = r'\b' + re.escape(species_words) + r'\b'
                        if re.search(pattern, query_lower):
                            entities.append(filter_species)
                            filters["species"].append(filter_species)
                            print(f"   ✅ Matched species from available filters (short/compound): {filter_species} from query '{query_lower}'")
                            species_matched = True
                            break
                
                # Strategy 3: Normalized matching (fallback for compound names like "red_fox")
                # Only use if the species is longer than 4 chars to avoid matching "red" in "red raspberries"
                # When query is single-word, do NOT match when query is only a substring of a multi-word species
                # (e.g. "carrots" -> "carrot" crop, not "carrot seed moth" pest)
                if len(filter_species_normalized) >= 5:
                    filter_word_count = filter_species.count('_') + filter_species.count('-') + filter_species.count(' ') + 1
                    query_is_single = query_word_count <= 1
                    if filter_species_normalized in query_normalized:
                        pass  # query contains full species, ok
                    elif query_normalized in filter_species_normalized:
                        if query_is_single and filter_word_count > 1:
                            continue  # skip: single-word query must not match multi-word species as substring
                    else:
                        continue
                    entities.append(filter_species)
                    filters["species"].append(filter_species)
                    print(f"   ✅ Matched species from available filters (normalized): {filter_species} from query")
                    species_matched = True
                    break
        
        # If no match in available filters, try common species list
        # Sort by length (longest first) to prioritize specific matches
        if not species_matched:
            species_to_check_sorted = sorted(species_to_check, key=lambda x: (-len(x), x.lower()))
            for species in species_to_check_sorted:
                species_lower = species.lower().strip()
                species_normalized = species_lower.replace("_", "").replace("-", "").replace(" ", "")
                
                # Strategy 1: Word boundary matching (exact word match)
                # Handle both "red fox" and "red_fox" / "redfox"
                # But normalize plurals: if "raspberries" matches, store "raspberry" if it exists in available_filters
                pattern = r'\b' + re.escape(species_lower.replace("_", " ").replace("-", " ")) + r'\b'
                if re.search(pattern, query_lower):
                    # Normalize plurals: if species is plural (ends in 's', 'ies', 'es'), try to find singular in available_filters
                    matched_species = species
                    if species.endswith('ies'):
                        singular = species[:-3] + 'y'  # "raspberries" -> "raspberry"
                        if available_filters and "species" in available_filters:
                            if singular in available_filters["species"]:
                                matched_species = singular
                    elif species.endswith('es') and len(species) > 2:
                        singular = species[:-2]  # "foxes" -> "fox"
                        if available_filters and "species" in available_filters:
                            if singular in available_filters["species"]:
                                matched_species = singular
                    elif species.endswith('s') and len(species) > 1:
                        singular = species[:-1]  # "bobcats" -> "bobcat"
                        if available_filters and "species" in available_filters:
                            if singular in available_filters["species"]:
                                matched_species = singular
                    # Rabbit/cottontail/white cottontail synonym: map to eastern_cottontail (never white_cottontail)
                    if matched_species.lower() in ("rabbit", "rabbits", "cottontail", "cottontails", "white_cottontail", "white_cottontails") and available_filters and "species" in available_filters:
                        cottontail_canonical = next(
                            (s for s in available_filters["species"] if s and s.lower() == "eastern_cottontail"),
                            next((s for s in available_filters["species"] if "cottontail" in s.lower()), None)
                        )
                        if cottontail_canonical:
                            matched_species = cottontail_canonical
                            print(f"   ✅ Mapped rabbit/cottontail/white cottontail to species: {matched_species}")
                    # Opossum/oppossum (common misspelling) → virginia_opossum when available
                    if matched_species.lower() in ("opossum", "opossums", "oppossum", "oppossums") and available_filters and "species" in available_filters:
                        opossum_canonical = next((s for s in available_filters["species"] if "opossum" in s.lower()), None)
                        if opossum_canonical:
                            matched_species = opossum_canonical
                            print(f"   ✅ Mapped opossum/oppossum to species: {matched_species}")
                    # Bird/birds → american_crow (or first available bird-like species) when available
                    if matched_species.lower() in ("bird", "birds") and available_filters and "species" in available_filters:
                        bird_canonical = next((s for s in available_filters["species"] if s and s.lower() == "american_crow"), None) or next(
                            (s for s in available_filters["species"] if "crow" in s.lower() or "bird" in s.lower()), None)
                        if bird_canonical:
                            matched_species = bird_canonical
                            print(f"   ✅ Mapped bird/birds to species: {bird_canonical}")
                    # American bear / black bear → american_black_bear when available
                    if matched_species.lower().replace(" ", "_") in ("american_bear", "american_bears", "black_bear", "black_bears", "american_black_bear") and available_filters and "species" in available_filters:
                        bear_canonical = next((s for s in available_filters["species"] if s and s.lower() == "american_black_bear"), None)
                        if bear_canonical:
                            matched_species = bear_canonical
                            print(f"   ✅ Mapped american bear/black bear to species: {bear_canonical}")
                    
                    entities.append(matched_species)
                    filters["species"].append(matched_species)
                    print(f"   ✅ Matched species (word boundary): {matched_species} from query (normalized from {species})")
                    species_matched = True
                    break
                
                # Strategy 2: Normalized matching (handles underscores, hyphens, spaces)
                # This handles "red fox" matching "red_fox" or "redfox"
                # For short names (<=5 chars) require whole-word match so "ring" does not match inside "during"
                if species_normalized in query_normalized and len(species_normalized) >= 3:
                    if len(species_normalized) <= 5:
                        pattern = r"\b" + re.escape(species_lower.replace("_", " ").replace("-", " ")) + r"\b"
                        if not re.search(pattern, query_lower):
                            pass  # skip: short species not as whole word (e.g. "ring" in "during")
                        else:
                            pass  # fall through to resolution
                    if len(species_normalized) > 5 or re.search(r"\b" + re.escape(species_lower.replace("_", " ").replace("-", " ")) + r"\b", query_lower):
                        # Make sure it's not a partial match (e.g., "cat" in "bobcat" should match "bobcat" not just "cat")
                        if len(species_normalized) >= 4 or species_normalized in ["fox", "cat", "pig", "rabbit", "cottontail", "whitecottontail", "opossum", "oppossum"]:
                            resolved = species
                            if species.lower() in ("rabbit", "rabbits", "cottontail", "cottontails", "white_cottontail", "white_cottontails") and available_filters and "species" in available_filters:
                                cottontail_canonical = next((s for s in available_filters["species"] if s and s.lower() == "eastern_cottontail"), None) or next((s for s in available_filters["species"] if "cottontail" in s.lower()), None)
                                if cottontail_canonical:
                                    resolved = cottontail_canonical
                            if species.lower() in ("opossum", "opossums", "oppossum", "oppossums") and available_filters and "species" in available_filters:
                                opossum_canonical = next((s for s in available_filters["species"] if "opossum" in s.lower()), None)
                                if opossum_canonical:
                                    resolved = opossum_canonical
                            if species.lower().replace(" ", "_") in ("american_bear", "american_bears", "black_bear", "black_bears") and available_filters and "species" in available_filters:
                                bear_canonical = next((s for s in available_filters["species"] if s and s.lower() == "american_black_bear"), None)
                                if bear_canonical:
                                    resolved = bear_canonical
                            entities.append(resolved)
                            filters["species"].append(resolved)
                            print(f"   ✅ Matched species (normalized): {resolved} from query")
                            species_matched = True
                            break
                
                # Strategy 3: Handle plurals (e.g., "strawberries" -> "strawberry", "pigs" -> "pig")
                base_species = species_normalized.rstrip('s')
                base_query = query_normalized.rstrip('s')
                if len(base_species) >= 4 and (base_species == base_query or base_species in base_query):
                    resolved = species
                    if species.lower() in ("rabbit", "rabbits", "cottontail", "cottontails", "white_cottontail", "white_cottontails") and available_filters and "species" in available_filters:
                        cottontail_canonical = next((s for s in available_filters["species"] if s and s.lower() == "eastern_cottontail"), None) or next((s for s in available_filters["species"] if "cottontail" in s.lower()), None)
                        if cottontail_canonical:
                            resolved = cottontail_canonical
                    if species.lower().replace(" ", "_") in ("american_bear", "american_bears", "black_bear", "black_bears") and available_filters and "species" in available_filters:
                        bear_canonical = next((s for s in available_filters["species"] if s and s.lower() == "american_black_bear"), None)
                        if bear_canonical:
                            resolved = bear_canonical
                    entities.append(resolved)
                    filters["species"].append(resolved)
                    print(f"   ✅ Matched species (plural): {resolved} from query")
                    species_matched = True
                    break
        
        if not species_matched:
            print(f"   ⚠️  No species matched from query '{query_lower}'")
            print(f"   Available species to check were: {species_to_check[:20]}...")  # Show first 20
        
        # Match time from query using canonical keywords only (adapter matches by concept: day/daytime, night, etc.)
        # Do NOT add every dataset "time" value that contains the keyword (e.g. hundreds of "Daytime (11:40 AM)...")
        time_keywords = ["night", "day", "dawn", "dusk", "morning", "afternoon", "evening"]
        # Map query phrases to canonical filter value (adapter uses these to set is_day, is_night, etc.)
        time_phrase_to_canonical = {
            "daytime": "day", "day time": "day", "during day": "day",
            "nighttime": "night", "night time": "night", "at night": "night", "during night": "night",
            "sunrise": "dawn", "sunset": "dusk", "twilight": "evening", "late afternoon": "evening",
        }
        time_matched = set()
        for keyword in time_keywords:
            if keyword in query_lower:
                time_matched.add(keyword)
        for phrase, canonical in time_phrase_to_canonical.items():
            if phrase in query_lower:
                time_matched.add(canonical)
        for t in sorted(time_matched):
            if t not in filters["time"]:
                entities.append(t)
                filters["time"].append(t)
        
        # Match weather from query using canonical keywords (adapter matches by substring in metadata.weather)
        weather_keywords = ["sunny", "rainy", "clear", "cloudy", "overcast", "foggy", "snow", "snowy", "dry", "wet"]
        weather_phrase_to_canonical = {
            "clear sky": "clear", "clear skies": "clear", "blue sky": "clear",
            "raining": "rainy", "rain": "rainy", "sun": "sunny", "sunshine": "sunny",
        }
        weather_matched = set()
        for keyword in weather_keywords:
            if keyword in query_lower:
                weather_matched.add(keyword)
        for phrase, canonical in weather_phrase_to_canonical.items():
            if phrase in query_lower:
                weather_matched.add(canonical)
        for w in sorted(weather_matched):
            if w not in filters["weather"]:
                entities.append(w)
                filters["weather"].append(w)
        
        # Match season from available filters (MCP uses "seasons" key, filter uses "season")
        if "seasons" in available_filters:
            season_keywords = ["spring", "summer", "fall", "winter", "autumn"]
            for season_val in available_filters["seasons"]:
                season_lower = season_val.lower()
                for keyword in season_keywords:
                    if keyword in query_lower and keyword in season_lower:
                        entities.append(season_val)
                        filters["season"].append(season_val)
                        break
        
        # Match action from available filters (MCP uses "actions" key, filter uses "action")
        # Handle variations like "sleeping" -> "sleep", "feeding" -> "feed"/"foraging", "resting" -> "rest"
        # When available_filters has long descriptive strings (e.g. "Animal appears stationary, possibly standing or walking..."),
        # we store the canonical keyword (e.g. "walking") so the adapter can match items by keyword.
        action_keyword_map = {
            "sleeping": ["sleep", "sleeping", "rest", "resting"],
            "feeding": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
            "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
            "resting": ["rest", "resting", "sleep", "sleeping"],
            "walking": ["walk", "walking", "moving"],
            "hunting": ["hunt", "hunting"],
            "alert": ["alert", "alerts", "watch", "watching", "looking at camera", "looking at the camera",
                     "staring at camera", "staring at the camera", "facing camera", "facing the camera",
                     "looking toward camera", "looking toward the camera", "staring toward camera",
                     "staring toward the camera", "facing toward camera", "facing toward the camera",
                     "looking directly at camera", "looking directly at the camera", "staring directly at camera",
                     "staring directly at the camera", "facing directly at camera", "facing directly at the camera"],
            "moving": ["move", "moving", "walk", "walking"],
            "running": ["run", "running", "moving"],
            "perching": ["perch", "perching", "sitting", "sit"],
            "flying": ["fly", "flying"],
            "blooming": ["bloom", "blooming", "flowering", "flower"],
            "fruiting": ["fruit", "fruiting"],
            "growing": ["grow", "growing"],
            "mature": ["mature", "matured", "ripe"]
        }
        # Use canonical keyword for filter when the dataset action value is a long sentence (adapter matches by keyword)
        def _action_filter_val(raw_val: str, canonical_keyword: Optional[str] = None) -> str:
            if len(raw_val) <= 40:
                return raw_val
            if canonical_keyword:
                return canonical_keyword
            for kw, variations in action_keyword_map.items():
                if kw in raw_val.lower() or any(v in raw_val.lower() for v in variations):
                    return kw
            return raw_val

        action_matched = False
        if "actions" in available_filters:
            print(f"   🔍 Available actions: {available_filters['actions'][:10]}...")
            for action_val in available_filters["actions"]:
                action_lower = action_val.lower().strip()
                
                # Check keyword variations FIRST - this ensures we map query keywords to actual filter values
                # (e.g., "feeding" in query -> "foraging" in filters)
                # Check keyword variations - check if query keyword matches any variation of the filter action
                # IMPORTANT: We want to match query keywords (like "feeding") to actual filter values (like "foraging")
                for keyword, variations in action_keyword_map.items():
                    # First check if any variation (including multi-word phrases) is in the query
                    # This handles cases like "looking at the camera" -> "alert"
                    for v in variations:
                        if v in query_lower:
                            # Check if this variation maps to the current filter action
                            if action_lower == keyword or action_lower in variations:
                                entities.append(action_val)
                                filters["action"].append(_action_filter_val(action_val, keyword))
                                print(f"   ✅ Matched action (variation in query): {_action_filter_val(action_val, keyword)} from query phrase '{v}'")
                                action_matched = True
                                break
                    if action_matched:
                        break
                    
                    # Check if the query contains the keyword (e.g., "feeding" in query)
                    if keyword in query_lower:
                        # Check if the filter action (e.g., "foraging") is in the variations list for this keyword
                        # This means "feeding" in query should match "foraging" in filters
                        if action_lower in variations:
                            entities.append(action_val)
                            filters["action"].append(_action_filter_val(action_val, keyword))
                            print(f"   ✅ Matched action (variation): {_action_filter_val(action_val, keyword)} from query keyword '{keyword}'")
                            action_matched = True
                            break
                        # Also check if any variation in the list matches the filter value
                        for v in variations:
                            if v == action_lower or (len(v) >= 3 and (v in action_lower or action_lower in v)):
                                entities.append(action_val)
                                filters["action"].append(_action_filter_val(action_val, keyword))
                                print(f"   ✅ Matched action (variation): {_action_filter_val(action_val, keyword)} from query keyword '{keyword}' (via '{v}')")
                                action_matched = True
                                break
                        if action_matched:
                            break
                    # Also check reverse: if filter action is a keyword, check if query contains its variations
                    if action_lower == keyword:
                        # Check if any variation of this action is in the query
                        if any(v in query_lower for v in variations):
                            entities.append(action_val)
                            filters["action"].append(_action_filter_val(action_val, keyword))
                            print(f"   ✅ Matched action (reverse variation): {_action_filter_val(action_val, keyword)} from query")
                            action_matched = True
                            break
                    if action_matched:
                        break
                
                if action_matched:
                    break
                
                # Also check if query contains base form (e.g., "sleep" matches "sleeping", "feed" matches "feeding"/"foraging")
                action_base = action_lower.rstrip('ing').rstrip('ed')
                query_base = query_lower.rstrip('ing').rstrip('ed')
                if (action_base in query_lower and len(action_base) >= 3) or (query_base in action_lower and len(query_base) >= 3):
                    entities.append(action_val)
                    filters["action"].append(_action_filter_val(action_val))  # no keyword; helper extracts from long text
                    print(f"   ✅ Matched action (base form): {_action_filter_val(action_val)} from query (base: {action_base} vs {query_base})")
                    action_matched = True
                    break
        
        if not action_matched and "actions" in available_filters:
            print(f"   ⚠️  No action matched from query '{query_lower}'")
            print(f"   Available actions were: {available_filters['actions'][:10]}...")
        
        # Match category from available filters (MCP uses "categories" key, filter uses "category")
        if "categories" in available_filters:
            for category_val in available_filters["categories"]:
                category_lower = category_val.lower()
                if category_lower in query_lower:
                    entities.append(category_val)
                    filters["category"].append(category_val)
        
        # Match plant_state from available filters (MCP uses "plant_states" key, filter uses "plant_state")
        # Common plant state keywords: green, ripe, unripe, mature, immature, blooming, fruiting, etc.
        plant_state_keywords = {
            "green": ["green", "unripe", "immature", "young"],
            "ripe": ["ripe", "mature", "ready", "edible", "can be eaten", "ready to eat"],
            "unripe": ["unripe", "green", "immature"],
            "mature": ["mature", "ripe", "ready", "edible"],
            "blooming": ["blooming", "flowering", "bloom", "flower"],
            "fruiting": ["fruiting", "fruits", "berries"],
            "growing": ["growing", "developing"]
        }
        
        plant_state_matched = False
        
        # SEMANTIC UNDERSTANDING: Map color descriptors and edibility to ripeness states when used with fruits/berries
        # This handles queries like "red raspberries" → should prioritize ripe/mature fruits
        # Also handles "raspberries that can be eaten" → should map to ripe
        fruit_berry_species = ["raspberry", "strawberry", "blueberry", "blackberry", "cherry", 
                              "apple", "tomato", "grape", "berry", "berries", "fruit", "fruits"]
        
        # Check if we have a fruit/berry species in the filters OR in the query
        matched_fruit_berry = False
        if filters.get("species"):
            for species in filters["species"]:
                species_lower = species.lower()
                # Check if any part of the species name matches fruit/berry keywords
                if any(fb in species_lower for fb in fruit_berry_species):
                    matched_fruit_berry = True
                    break
        
        # Also check if query mentions fruit/berry even if not in filters yet
        if not matched_fruit_berry:
            if any(fb in query_lower for fb in fruit_berry_species):
                matched_fruit_berry = True
        
        # If we have a fruit/berry species, handle semantic mappings
        if matched_fruit_berry:
            # EDIBILITY MAPPING: Map edibility concepts to ripeness states
            # "edible", "can be eaten", "ready to eat" → ripe (for fruits/berries)
            edibility_keywords = {
                "edible": ["ripe", "mature"],
                "can be eaten": ["ripe", "mature"],
                "can eat": ["ripe", "mature"],
                "ready to eat": ["ripe", "mature"],
                "ready for eating": ["ripe", "mature"],
                "ready for harvest": ["ripe", "mature"],
                "harvestable": ["ripe", "mature"],
            }
            
            # Check for edibility keywords in the query
            for edibility_phrase, ripeness_states in edibility_keywords.items():
                if edibility_phrase in query_lower:
                    # Map to available plant_state filters if they exist
                    if "plant_states" in available_filters:
                        # Find matching plant_state values (e.g., "ripe", "mature")
                        for ripeness in ripeness_states:
                            for plant_state_val in available_filters["plant_states"]:
                                plant_state_lower = plant_state_val.lower().strip()
                                if ripeness == plant_state_lower or ripeness in plant_state_keywords.get(plant_state_lower, []):
                                    if plant_state_val not in filters["plant_state"]:
                                        filters["plant_state"].append(plant_state_val)
                                        entities.append(f"{edibility_phrase} (→ {plant_state_val})")
                                        print(f"   ✅ Mapped edibility '{edibility_phrase}' + fruit/berry → plant_state: {plant_state_val}")
                                        plant_state_matched = True
                    else:
                        # If no available_filters, add the ripeness states directly
                        for ripeness in ripeness_states:
                            if ripeness not in filters["plant_state"]:
                                filters["plant_state"].append(ripeness)
                                entities.append(f"{edibility_phrase} (→ {ripeness})")
                                print(f"   ✅ Mapped edibility '{edibility_phrase}' + fruit/berry → plant_state: {ripeness}")
                                plant_state_matched = True
                    break  # Only match one edibility phrase
            
            # COLOR MAPPING: Map color descriptors to ripeness states
            # Color to ripeness mappings for fruits/berries
            color_ripeness_map = {
                "red": ["ripe", "mature"],  # Red fruits/berries are typically ripe
                "dark red": ["ripe", "mature"],
                "deep red": ["ripe", "mature"],
                "yellow": ["ripe", "mature"],  # Yellow fruits are typically ripe
                "orange": ["ripe", "mature"],  # Orange fruits are typically ripe
                "blue": ["ripe", "mature"],  # Blueberries are ripe when blue
                "purple": ["ripe", "mature"],  # Purple fruits are typically ripe
                "green": ["unripe", "immature"],  # Green fruits are typically unripe
                "pale": ["unripe", "immature"],
                "light": ["unripe", "immature"],
            }
            
            # Check for color descriptors in the query
            for color, ripeness_states in color_ripeness_map.items():
                if color in query_lower:
                    # Map to available plant_state filters if they exist
                    if "plant_states" in available_filters:
                        # Find matching plant_state values (e.g., "ripe", "mature", "unripe")
                        for ripeness in ripeness_states:
                            for plant_state_val in available_filters["plant_states"]:
                                plant_state_lower = plant_state_val.lower().strip()
                                if ripeness == plant_state_lower or ripeness in plant_state_keywords.get(plant_state_lower, []):
                                    if plant_state_val not in filters["plant_state"]:
                                        filters["plant_state"].append(plant_state_val)
                                        entities.append(f"{color} (→ {plant_state_val})")
                                        print(f"   ✅ Mapped color descriptor '{color}' + fruit/berry → plant_state: {plant_state_val}")
                                        plant_state_matched = True
                    else:
                        # If no available_filters, add the ripeness states directly
                        for ripeness in ripeness_states:
                            if ripeness not in filters["plant_state"]:
                                filters["plant_state"].append(ripeness)
                                entities.append(f"{color} (→ {ripeness})")
                                print(f"   ✅ Mapped color descriptor '{color}' + fruit/berry → plant_state: {ripeness}")
                                plant_state_matched = True
                    break  # Only match one color
        
        # First, try to match plant_state keywords directly from query (even if not in available_filters)
        # This handles cases where plant_states aren't extracted but the query mentions them
        for keyword, variations in plant_state_keywords.items():
            if keyword in query_lower:
                # Add the keyword itself as the plant_state filter
                filters["plant_state"].append(keyword)
                entities.append(keyword)
                print(f"   ✅ Matched plant state (keyword): {keyword} from query")
                plant_state_matched = True
                break
        
        # Then, try to match against available_filters if they exist
        if "plant_states" in available_filters and not plant_state_matched:
            print(f"   🔍 Available plant states: {available_filters['plant_states'][:10]}...")
            for plant_state_val in available_filters["plant_states"]:
                plant_state_lower = plant_state_val.lower().strip()
                
                # Direct match
                if plant_state_lower in query_lower:
                    entities.append(plant_state_val)
                    filters["plant_state"].append(plant_state_val)
                    print(f"   ✅ Matched plant state (direct): {plant_state_val} from query")
                    plant_state_matched = True
                    break
                
                # Check keyword variations
                for keyword, variations in plant_state_keywords.items():
                    if keyword in query_lower:
                        # Check if the filter value matches any variation
                        if plant_state_lower in variations or any(v in plant_state_lower for v in variations):
                            entities.append(plant_state_val)
                            filters["plant_state"].append(plant_state_val)
                            print(f"   ✅ Matched plant state (variation): {plant_state_val} from query keyword '{keyword}'")
                            plant_state_matched = True
                            break
                    # Reverse: if filter value is a keyword, check if query contains its variations
                    if plant_state_lower == keyword:
                        if any(v in query_lower for v in variations):
                            entities.append(plant_state_val)
                            filters["plant_state"].append(plant_state_val)
                            print(f"   ✅ Matched plant state (reverse variation): {plant_state_val} from query")
                            plant_state_matched = True
                            break
                    if plant_state_matched:
                        break
                if plant_state_matched:
                    break
        
        if not plant_state_matched:
            print(f"   ⚠️  No plant state matched from query '{query_lower}'")
            if "plant_states" in available_filters:
                print(f"   Available plant states were: {available_filters['plant_states'][:10]}...")
        
        # Match scene from query (e.g. "on a tree", "in the field", "forest")
        scene_keywords = ["forest", "field", "water", "mountain", "garden", "farm", "meadow", "indoor", "outdoor", "tree"]
        scene_phrases = {
            "field": ["in the field", "in field", "on field", "field"],
            "forest": ["in forest", "in the forest", "forest"],
            "tree": ["on a tree", "on tree", "in a tree", "in tree", "in the tree"],
            "garden": ["in garden", "in the garden", "garden"],
            "farm": ["on farm", "on the farm", "farm"],
            "meadow": ["in meadow", "in the meadow", "meadow"],
            "indoor": ["indoors", "indoor", "inside"],
            "outdoor": ["outdoors", "outdoor", "outside"]
        }
        for scene, phrases in scene_phrases.items():
            for phrase in phrases:
                if phrase in query_lower and scene not in filters["scene"]:
                    filters["scene"].append(scene)
                    if scene not in entities:
                        entities.append(scene)
                    print(f"   ✅ Matched scene (phrase): {scene} from '{phrase}'")
                    break
        for scene in scene_keywords:
            if scene in query_lower and scene not in filters["scene"]:
                filters["scene"].append(scene)
                if scene not in entities:
                    entities.append(scene)
                print(f"   ✅ Matched scene (keyword): {scene}")
        
        # "bird on a tree" → species = bird, scene = tree (not species = trees)
        if "bird" in query_lower and ("tree" in query_lower or "on a tree" in query_lower):
            species_is_tree_only = not filters.get("species") or all(
                s.lower().strip() in ("tree", "trees") or (len(s) < 15 and "tree" in s.lower() and "crow" not in s.lower() and "bird" not in s.lower())
                for s in filters["species"]
            )
            if species_is_tree_only and available_filters and "species" in available_filters:
                bird_species = next((s for s in available_filters["species"] if s and s.lower() == "american_crow"), None) or next(
                    (s for s in available_filters["species"] if "crow" in s.lower() or "bird" in s.lower()), None)
                if bird_species:
                    filters["species"] = [bird_species]
                    if "tree" not in filters["scene"]:
                        filters["scene"].append("tree")
                    print(f"   ✅ Metadata fix: \"bird on a tree\" → species [{bird_species}], scene [tree]")
        
        # Never leave action words in species (e.g. "cat eating" parsed as species ["eating"] by substring match)
        action_only_words = {"eating", "feeding", "foraging"}
        if filters.get("species"):
            species_cleaned = [s for s in filters["species"] if s and s.lower().strip().replace(" ", "_") not in action_words]
            if len(species_cleaned) < len(filters["species"]):
                filters["species"] = species_cleaned
                print(f"   ✅ Removed action words from species; species now: {filters['species']}")
            # If the only "species" was an action word and query is "cat eating" / "dog eating", set species and action correctly
            if not filters["species"] and (action_only_words & set(query_lower.split())):
                if "dog" in query_lower:
                    filters["species"] = ["dog"]
                    action_word = next((w for w in action_only_words if w in query_lower), "eating")
                    if action_word not in filters["action"]:
                        filters["action"].append(action_word)
                    print(f"   ✅ Metadata fix: 'X eating' → species ['dog'], action [{action_word}]")
                elif "cat" in query_lower:
                    filters["species"] = ["cat"]
                    action_word = next((w for w in action_only_words if w in query_lower), "eating")
                    if action_word not in filters["action"]:
                        filters["action"].append(action_word)
                    print(f"   ✅ Metadata fix: 'X eating' → species ['cat'], action [{action_word}]")
        
        # Determine intent
        if filters["species"]:
            intent = f"find {', '.join(filters['species'])} images"
            if filters["plant_state"]:
                intent += f" that are {', '.join(filters['plant_state'])}"
            if filters["action"]:
                intent += f" {', '.join(filters['action'])}"
            if filters["scene"]:
                intent += f" in {', '.join(filters['scene'])}"
            if filters["time"]:
                intent += f" at {', '.join(filters['time'])}"
            if filters["season"]:
                intent += f" in {', '.join(filters['season'])}"
        elif filters["time"] or filters["season"]:
            intent = f"find images from {', '.join(filters['time'] + filters['season'])}"
        else:
            intent = f"search for {query}"
        
        # Confidence based on how many filters we matched
        # Higher confidence for more specific queries (species + action/time/season/plant_state)
        confidence = 0.7 if filters["species"] else 0.5
        if filters["species"] and (filters["time"] or filters["season"] or filters["action"] or filters["plant_state"]):
            confidence = 0.8
        if filters["species"] and filters["action"] and (filters["time"] or filters["season"]):
            confidence = 0.85
        if filters["species"] and filters["plant_state"]:
            confidence = 0.8  # Species + plant_state is specific
        
        return QueryUnderstanding(
            intent=intent,
            entities=entities,
            filters=filters,
            confidence=confidence,
            reasoning=f"Metadata-based matching using actual MCP filter values. Matched: {list(filters.keys())}"
        )
    
    def _rule_based_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Fallback rule-based query understanding"""
        query_lower = query.lower()
        
        # Simple keyword matching
        entities = []
        filters = {
            "category": [],
            "species": [],
            "time": [],
            "season": [],
            "action": [],
            "scene": [],
            "weather": [],
            "plant_state": []
        }
        
        # Extract species - check for exact matches first to avoid false positives
        species_keywords = ["bobcat", "coyote", "deer", "fox", "chicken", "pig", "goat", "carrot", "strawberry", "raspberry", "crow", "crows", "american_crow"]
        for species in species_keywords:
            # Use word boundaries to avoid partial matches (e.g., "bobcat" not matching "bobcat123")
            if re.search(r'\b' + re.escape(species) + r'\b', query_lower):
                entities.append(species)
                filters["species"].append(species)
        
        # Extract time
        time_keywords = ["dawn", "day", "dusk", "night", "morning", "afternoon", "evening"]
        for time in time_keywords:
            if time in query_lower:
                entities.append(time)
                filters["time"].append(time)
        
        # Extract season
        season_keywords = ["spring", "summer", "fall", "winter", "autumn"]
        for season in season_keywords:
            if season in query_lower:
                entities.append(season)
                filters["season"].append(season)
        
        # Extract weather (sunny, rainy, clear sky, cloudy, etc.)
        weather_keywords = ["sunny", "rainy", "clear", "cloudy", "overcast", "foggy", "snow", "snowy"]
        weather_phrases = {"clear sky": "clear", "clear skies": "clear", "blue sky": "clear", "raining": "rainy", "rain": "rainy"}
        for phrase, canonical in weather_phrases.items():
            if phrase in query_lower and canonical not in filters["weather"]:
                entities.append(canonical)
                filters["weather"].append(canonical)
        for w in weather_keywords:
            if w in query_lower and w not in filters["weather"]:
                entities.append(w)
                filters["weather"].append(w)
        
        # Extract actions
        action_keywords = ["walking", "hunting", "eating", "sleeping", "running", "standing"]
        for action in action_keywords:
            if action in query_lower:
                entities.append(action)
                filters["action"].append(action)
        
        # Extract scenes (including phrases like "in the field", "in forest", etc.)
        scene_keywords = ["forest", "field", "water", "mountain", "garden", "farm", "meadow", "indoor", "outdoor"]
        scene_phrases = {
            "field": ["in the field", "in field", "on field", "field"],
            "forest": ["in forest", "in the forest", "forest"],
            "garden": ["in garden", "in the garden", "garden"],
            "farm": ["on farm", "on the farm", "farm"],
            "meadow": ["in meadow", "in the meadow", "meadow"],
            "indoor": ["indoors", "indoor", "inside"],
            "outdoor": ["outdoors", "outdoor", "outside"]
        }
        
        # Check for scene phrases first (more specific)
        for scene, phrases in scene_phrases.items():
            for phrase in phrases:
                if phrase in query_lower:
                    if scene not in filters["scene"]:
                        entities.append(scene)
                        filters["scene"].append(scene)
                        print(f"   ✅ Matched scene (phrase): {scene} from '{phrase}'")
                    break
        
        # Also check for direct scene keywords
        for scene in scene_keywords:
            if scene in query_lower and scene not in filters["scene"]:
                entities.append(scene)
                filters["scene"].append(scene)
                print(f"   ✅ Matched scene (keyword): {scene}")
        
        # Determine intent
        if filters["species"]:
            intent = f"find {', '.join(filters['species'])} images"
        elif filters["time"] or filters["season"]:
            intent = f"find images from {', '.join(filters['time'] + filters['season'])}"
        else:
            intent = f"search for {query}"
        
        return QueryUnderstanding(
            intent=intent,
            entities=entities,
            filters=filters,
            confidence=0.6 if entities else 0.3,
            reasoning="Rule-based keyword matching"
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        return [
            "gpt-5-mini-2",
            "gpt-5-mini",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-sonnet",
            "claude-3-opus"
        ]
    
    def is_available(self) -> bool:
        """Check if LLM service is available (OpenAI or Gemini)"""
        return self.openai_available or self.gemini_available

