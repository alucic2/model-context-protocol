#!/usr/bin/env python3
"""
Core MCP Server
- Handles MCP protocol and tool registration
- Extensible tool system for adding new functionalities
- Core service that can be consumed by other applications
"""

import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union


def _metadata_str(val: Any) -> str:
    """Normalize a metadata field value to a string (item metadata can have list values e.g. action)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(x).strip() for x in val if x is not None and str(x).strip())
    return str(val)


def _canonical_time_from_text(time_text: str) -> set:
    """Map raw time metadata to canonical buckets used by the LLM and search filters."""
    if not time_text:
        return set()
    time_info = str(time_text).lower()
    buckets = set()
    if "night" in time_info or "dark" in time_info:
        buckets.add("night")
    if "day" in time_info or "morning" in time_info or "afternoon" in time_info:
        buckets.add("day")
    if "dawn" in time_info or "sunrise" in time_info:
        buckets.add("dawn")
    if "dusk" in time_info or "sunset" in time_info:
        buckets.add("dusk")
    if "evening" in time_info or "twilight" in time_info or "late afternoon" in time_info:
        buckets.add("evening")
    return buckets


_TIME_QUERY_PHRASES = {
    "night": ("at night", "nighttime", "night time", "during night", "nocturnal"),
    "day": ("daytime", "day time", "during day", "in daylight"),
    "dawn": ("at dawn", "sunrise"),
    "dusk": ("at dusk", "sunset"),
    "evening": ("evening", "twilight", "late afternoon"),
}


# Common-name synonyms → a term that appears in our dataset names. Lets queries like "groundhog" resolve to
# the "woodchuck" dataset (and not be flagged as "not in catalog"). Keys are lowercase; values are matched
# as a segment/substring of dataset names.
_SPECIES_SYNONYMS = {
    "groundhog": "woodchuck",
    "groundhogs": "woodchuck",
    "whistlepig": "woodchuck",
    "whistle-pig": "woodchuck",
    "possum": "opossum",
    "possums": "opossum",
    "rabbit": "cottontail",
    "rabbits": "cottontail",
    "bunny": "cottontail",
    "bunnies": "cottontail",
    "raccoon": "raccoon",
    "racoon": "raccoon",
    "buck": "deer",
    "doe": "deer",
    "fawn": "deer",
}

# Generic insect/pest group words. These are too broad to index as a specific common name (a single one
# would map to thousands of datasets), so the common-name index skips them when they appear alone.
_PEST_TYPE_WORDS_SET = {
    "beetle", "beetles", "butterfly", "butterflies", "moth", "moths", "wasp", "wasps",
    "bee", "bees", "ant", "ants", "fly", "flies", "grasshopper", "grasshoppers",
    "dragonfly", "dragonflies", "spider", "spiders", "bug", "bugs", "insect", "insects",
    "weevil", "weevils", "aphid", "aphids", "caterpillar", "caterpillars", "midge", "midges",
}

# Larval / life-stage group nouns users search as a single word (e.g. "caterpillar", "cutworm").
# These also appear in _PEST_TYPE_WORDS_SET but should still resolve via head-noun common-name lookup
# (last word of "saddleback caterpillar", "black cutworm", etc.). cutworm/maggot are not in the pest-type
# set; caterpillar is — which is why bare "caterpillar" failed while cutworm/maggot worked.
_HEAD_NOUN_SEARCH_WORDS = {
    "caterpillar", "caterpillars", "cutworm", "cutworms", "maggot", "maggots",
    "armyworm", "armyworms", "hornworm", "hornworms", "borer", "borers",
    "looper", "loopers", "grub", "grubs", "webworm", "webworms", "sawfly", "sawflies",
}

# User-facing pest common-name aliases that may not be present literally in metadata.
# These are expanded only for common-name lookup, so they do not broaden generic pest-type searches.
_COMMON_NAME_QUERY_ALIASES = {
    "ladybug": ("lady beetle", "ladybird beetle", "ladybird"),
    "ladybugs": ("lady beetle", "lady beetles", "ladybird beetles", "ladybirds"),
    "lady bug": ("lady beetle", "ladybird beetle"),
    "lady bugs": ("lady beetles", "ladybird beetles"),
    "ladybird": ("lady beetle", "ladybird beetle"),
    "ladybirds": ("lady beetles", "ladybird beetles"),
    # Common misspellings of "caterpillar" so the search still finds it.
    "caterpiller": ("caterpillar",),
    "caterpillers": ("caterpillars",),
    "caterpiler": ("caterpillar",),
    "catterpillar": ("caterpillar",),
    "catepillar": ("caterpillar",),
    # Spaced spellings of "whitefly" / "mealybug" so they hit the (single-token) common-name index.
    "white fly": ("whitefly",),
    "white flies": ("whitefly",),
    "meal bug": ("mealybug",),
    "meal bugs": ("mealybug",),
}


# Generic color/quality adjectives. On their own (e.g. "white" + "fly") they are too broad to drive a
# partial pest common-name match, so the resolver requires an exact phrase match in that case.
_GENERIC_ADJECTIVE_WORDS = {
    "white", "black", "brown", "gray", "grey", "red", "green", "blue", "yellow", "orange",
    "golden", "gold", "silver", "common", "giant", "large", "small", "little", "great",
    "lesser", "spotted", "striped", "banded", "dark", "pale", "tiny",
}


# Map behavior words in a query to a canonical action value. Used to inject an action filter when the
# LLM omits one, so e.g. "squirrel eating" filters via the synonym-aware action matcher (eating→foraging)
# instead of requiring the literal word "eating" in the description.
_ACTION_QUERY_MAP = {
    "foraging": ("eating", "eat", "feeding", "feed", "foraging", "forage", "grazing", "graze", "nibbling", "nibble"),
    "sleeping": ("sleeping", "sleep", "resting", "rest"),
    "walking": ("walking", "walk", "moving", "move"),
    "running": ("running", "run"),
    "standing": ("standing", "stand"),
    "sitting": ("sitting", "sit"),
    "alert": ("alert", "looking", "staring", "facing", "watching", "watch"),
    "hunting": ("hunting", "hunt"),
    "flying": ("flying", "fly"),
    "perching": ("perching", "perch", "perched"),
    "drinking": ("drinking", "drink"),
    "climbing": ("climbing", "climb"),
    "jumping": ("jumping", "jump"),
    "swimming": ("swimming", "swim"),
}


# Words that describe an attribute/behavior/setting rather than a species subject.
# Used to tell "unknown species" queries (e.g. "hedgehog standing") apart from
# attribute-only queries (e.g. "sleeping at night") so we can return a clear message.
_NON_SUBJECT_WORDS = {
    # actions / behavior
    "walking", "walk", "standing", "stand", "sitting", "sit", "eating", "eat", "feeding", "feed",
    "foraging", "forage", "sleeping", "sleep", "resting", "rest", "running", "run", "moving", "move",
    "hunting", "hunt", "alert", "perching", "perch", "flying", "fly", "watching", "watch",
    "looking", "look", "staring", "stare", "facing", "face", "grazing", "graze", "drinking", "drink",
    "jumping", "jump", "climbing", "climb", "swimming", "swim", "lying", "crouching", "playing", "play",
    "posing", "pose",
    # time
    "day", "night", "dawn", "dusk", "evening", "morning", "afternoon", "daytime", "nighttime",
    "twilight", "sunrise", "sunset", "noon", "midnight", "nocturnal",
    # season
    "spring", "summer", "fall", "autumn", "winter",
    # scene / setting
    "field", "forest", "water", "mountain", "garden", "farm", "meadow", "indoor", "outdoor", "tree",
    "trees", "snow", "grass", "road", "sky", "barn", "pen", "enclosure", "cage", "bush", "leaf", "leaves",
    "foliage", "background",
    # plant state / descriptors
    "ripe", "unripe", "mature", "immature", "blooming", "flowering", "fruiting", "green", "red",
    "edible", "ready",
    # generic group / category terms (let these fall through to category browse / all-search, not "unknown species")
    "animal", "animals", "wildlife", "creature", "creatures", "species", "pest", "pests", "plant",
    "plants", "crop", "crops", "livestock", "fruit", "fruits", "vegetable", "vegetables", "insect", "insects",
    # generic / stopwords
    "the", "and", "for", "with", "from", "near", "during", "under", "over", "image", "images",
    "picture", "pictures", "photo", "photos", "show", "showing", "find", "finding", "search", "some",
    "any", "all", "that", "this", "are", "being", "taken", "captured", "camera", "trail",
}


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import our modular components
from models import SearchRequest, SearchResponse, InferenceRequest, InferenceResult, DatasetType
from tool_registry import ToolRegistry
from dataset_registry import DatasetRegistry, Dataset
from dataset_adapter import _item_canonical_action, _action_filter_conflicts, _description_indicates_awake_or_observing, _description_has_ripe_phrase
from model_registry import ModelRegistry
from config import MCP_CONFIG, MCP_PROTOCOL_CONFIG, BASE_DIR, LLM_CONFIG, IMAGES_DIR, IMAGES_TRY_SPECIES_FIRST

# Optional imports
try:
    from llm_service import LLMService, QueryUnderstanding
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMService = None
    from dataclasses import dataclass, field
    @dataclass
    class QueryUnderstanding:
        intent: str = ""
        entities: List = field(default_factory=list)
        filters: Dict = field(default_factory=dict)
        confidence: float = 0.0
        reasoning: str = ""
        description_query: Optional[str] = None

# IMPORTANT: This import happens at module load time
# Make sure croissant_crawler.py is in the same directory as this file
print("=" * 60)
print("🔍 CHECKING CROISSANT CRAWLER IMPORT...")
print("=" * 60)
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Current file directory: {current_dir}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Python path (first 3): {sys.path[:3]}")
    
    crawler_file = os.path.join(current_dir, "croissant_crawler.py")
    print(f"📄 Looking for: {crawler_file}")
    print(f"📄 File exists: {os.path.exists(crawler_file)}")
    
    if not os.path.exists(crawler_file):
        # Try current directory
        crawler_file = "croissant_crawler.py"
        print(f"📄 Trying current dir: {crawler_file}")
        print(f"📄 File exists: {os.path.exists(crawler_file)}")
    
    print(f"🔍 Attempting import...")
    from croissant_crawler import CroissantCrawler
    CROISSANT_CRAWLER_AVAILABLE = True
    print("=" * 60)
    print("✅ SUCCESS: Croissant crawler imported successfully!")
    print(f"   CroissantCrawler class: {CroissantCrawler}")
    print("=" * 60)
except ImportError as e:
    CROISSANT_CRAWLER_AVAILABLE = False
    CroissantCrawler = None
    print("=" * 60)
    print(f"❌ FAILED: Croissant crawler import error (ImportError)")
    print(f"   Error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)
except Exception as e:
    CROISSANT_CRAWLER_AVAILABLE = False
    CroissantCrawler = None
    print("=" * 60)
    print(f"❌ FAILED: Croissant crawler import error (Other)")
    print(f"   Error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)

class MCPServer:
    """Core MCP Server that manages tools and provides MCP protocol endpoints"""
    
    def __init__(self):
        self.name = MCP_PROTOCOL_CONFIG["server_name"]
        self.version = MCP_PROTOCOL_CONFIG["server_version"]
        self.description = MCP_PROTOCOL_CONFIG["server_description"]
        
        print(f"🚀 Initializing {self.name} v{self.version}")
        print(f"📁 Base directory: {BASE_DIR}")
        print(f"📁 Looking for MCP data files in: {BASE_DIR}")
        
        # Check what MCP files exist
        mcp_files = list(BASE_DIR.glob("*_mcp_data.json"))
        print(f"🔍 Found {len(mcp_files)} MCP data files:")
        for f in mcp_files:
            print(f"   - {f.name}")
        
        # Initialize registries
        print("🔧 Initializing tool registry...")
        self.tool_registry = ToolRegistry()
        
        print("📁 Initializing dataset registry...")
        self.dataset_registry = DatasetRegistry()
        # Lazily-built index mapping a common-name phrase (e.g. "painted lady") → dataset names
        # (e.g. ["Vanessa_cardui"]) so common-name queries resolve to scientific-name datasets.
        self._common_name_index: Optional[Dict[str, List[str]]] = None
        
        print("🤖 Initializing model registry...")
        self.model_registry = ModelRegistry()
        
        # Initialize LLM service
        print("🧠 Initializing LLM service...")
        # Check environment variables (Azure takes precedence over OpenAI when set)
        import os
        google_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        print(f"🧠 Environment check:")
        print(f"   AZURE_OPENAI_ENDPOINT: {'SET' if azure_endpoint else 'NOT SET'}")
        print(f"   AZURE_OPENAI_API_KEY: {'SET' if azure_key else 'NOT SET'}")
        print(f"   OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
        print(f"   GOOGLE_API_KEY: {'SET' if google_key else 'NOT SET'}")
        if google_key:
            print(f"   GOOGLE_API_KEY length: {len(google_key)}")
        
        if LLM_AVAILABLE and LLMService:
            self.llm_service = LLMService(
                api_key=LLM_CONFIG.get("api_key"),
                model=LLM_CONFIG.get("model"),
                provider=LLM_CONFIG.get("provider", "auto"),
                azure_endpoint=LLM_CONFIG.get("azure_endpoint") or None,
                azure_api_key=LLM_CONFIG.get("azure_api_key") or None,
                azure_deployment=LLM_CONFIG.get("azure_deployment") or None,
                azure_api_version=LLM_CONFIG.get("azure_api_version") or None,
            )
            print(f"🧠 LLM service: {'enabled' if self.llm_service.is_available() else 'disabled (fallback to rules)'}")
            if self.llm_service:
                print(f"   OpenAI available: {self.llm_service.openai_available}")
                print(f"   Gemini available: {self.llm_service.gemini_available}")
                print(f"   Provider: {self.llm_service.provider}")
        else:
            self.llm_service = None
            print("🧠 LLM service: not available (module not found)")
        
        # Setup FastAPI app
        print("🌐 Setting up FastAPI app...")
        self.app = FastAPI(title=self.name, version=self.version)
        self._setup_middleware()
        self._setup_routes()
        self._register_default_tools()
        
        print(f"✅ Initialized {self.name} v{self.version}")
        print(f"📊 Summary:")
        print(f"   - Tools: {len(self.tool_registry.get_all_tools())}")
        print(f"   - Datasets: {len(self.dataset_registry.get_all_datasets())}")
        print(f"   - Models: {len(self.model_registry.get_all_models())}")
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup MCP protocol routes and API endpoints"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "server": self.name,
                "version": self.version,
                "raspberry_full_results": True,  # Present only in build that returns all raspberry images for "raspberries" query
                "datasets_loaded": len(self.dataset_registry.datasets),
                "tools_available": len(self.tool_registry.get_all_tools()),
                "models_available": len(self.model_registry.get_all_models())
            }
        
        # MCP Protocol endpoints
        @self.app.get("/mcp")
        async def mcp_info():
            """MCP protocol information"""
            return {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "capabilities": {
                    "resources": self._get_resources(),
                    "tools": self._get_tools()
                },
                "endpoints": {
                    "mcp_base": "/mcp",
                    "health": "/health",
                    "tools": "/mcp/tools",
                    "resources": "/mcp/resources",
                    "datasets": "/api/datasets",
                    "models": "/api/models"
                }
            }
        
        # MCP Tools
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List all available tools"""
            return {
                "tools": self.tool_registry.get_all_tools(),
                "total": len(self.tool_registry.get_all_tools())
            }
        
        @self.app.get("/mcp/tools/{tool_name}")
        async def execute_tool_get(tool_name: str, request: Request):
            """Execute a tool via GET (for tools that don't require input)"""
            # Check if tool exists
            if tool_name not in self.tool_registry.tools:
                available_tools = list(self.tool_registry.tools.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}. Use POST /mcp/tools/{tool_name} with JSON body for tools that require input."
                )
            
            # For GET requests, use empty body (only works for tools that don't require input)
            body = {}
            
            # Check if tool requires input by looking at its schema
            tool = self.tool_registry.get_tool(tool_name)
            if tool and tool.input_schema.get("properties"):
                # Tool has required properties - suggest using POST
                required = tool.input_schema.get("required", [])
                if required:
                    raise HTTPException(
                        status_code=405,
                        detail=f"Tool '{tool_name}' requires input parameters: {', '.join(required)}. Please use POST /mcp/tools/{tool_name} with a JSON body."
                    )
            
            print(f"🔧 Executing tool via GET: {tool_name}")
            print(f"   Input data: {body}")
            
            try:
                result = await self.tool_registry.execute_tool(tool_name, body)
                print(f"✅ Tool {tool_name} executed successfully")
                return result
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, 
                    detail=f"Tool execution failed: {str(e)}. Try using POST /mcp/tools/{tool_name} with a JSON body."
                )
        
        @self.app.post("/mcp/tools/{tool_name}")
        async def execute_tool(tool_name: str, request: Request):
            """Execute a specific tool"""
            try:
                # Check if tool exists first
                if tool_name not in self.tool_registry.tools:
                    available_tools = list(self.tool_registry.tools.keys())
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
                    )
                
                # Handle empty body or missing JSON
                try:
                    body = await request.json()
                except Exception as json_error:
                    # If no JSON body, use empty dict (some tools don't need input)
                    body = {}
                    print(f"⚠️  No JSON body provided, using empty dict: {json_error}")
                
                print(f"🔧 Executing tool: {tool_name}")
                print(f"   Input data: {body}")
                
                result = await self.tool_registry.execute_tool(tool_name, body)
                print(f"✅ Tool {tool_name} executed successfully")
                return result
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except ValueError as e:
                # Tool not found or validation error
                print(f"❌ Tool execution error (ValueError): {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                # Other errors
                print(f"❌ Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
        
        # OPTIMIZATION: Cache directory listing to avoid repeated iterations
        _image_dir_cache = {}
        _image_dir_cache_time = {}
        _image_dir_cache_ttl = 60  # Cache for 60 seconds
        
        def _get_cached_dir_listing(images_dir: Path) -> List[Path]:
            """Get cached directory listing to avoid repeated iterations"""
            import time
            current_time = time.time()
            
            dir_str = str(images_dir)
            if dir_str in _image_dir_cache:
                cache_time = _image_dir_cache_time.get(dir_str, 0)
                if current_time - cache_time < _image_dir_cache_ttl:
                    return _image_dir_cache[dir_str]
            
            # Cache miss - refresh cache
            try:
                listing = list(images_dir.iterdir())
                _image_dir_cache[dir_str] = listing
                _image_dir_cache_time[dir_str] = current_time
                return listing
            except Exception as e:
                print(f"⚠️  Error listing directory {images_dir}: {e}")
                return []
        
        # Image serving endpoint
        @self.app.get("/images/{filename:path}")
        async def serve_image(filename: str):
            """Serve images: order by IMAGES_TRY_SPECIES_FIRST (Taiga = subdirs only → species subdir first)."""
            try:
                from pathlib import Path
                from fastapi.responses import FileResponse
                
                images_dir = Path(IMAGES_DIR)
                filename_no_ext = Path(filename).stem
                image_path_flat = images_dir / filename
                try:
                    images_dir_resolved = images_dir.resolve()
                    image_path_flat_resolved = (images_dir_resolved / filename).resolve()
                except OSError:
                    images_dir_resolved = images_dir
                    image_path_flat_resolved = image_path_flat
                flat_to_try = image_path_flat_resolved if image_path_flat_resolved != image_path_flat else image_path_flat
                images_base = images_dir_resolved
                
                def _try_species():
                    if "_" not in filename_no_ext:
                        return None
                    # Try dataset-name subdir first (e.g. domestic_cat for domestic_cat_001), then first segment
                    import re
                    subdir_candidates = []
                    if re.search(r"_\d+$", filename_no_ext):
                        subdir_candidates.append(re.sub(r"_\d+$", "", filename_no_ext))  # wild_turkey_001 -> wild_turkey
                    elif "_" in filename_no_ext:
                        subdir_candidates.append(filename_no_ext)  # wild_turkey (no number) -> wild_turkey
                    subdir_candidates.append(filename_no_ext.split("_")[0])
                    # Case-insensitive subdir match first (e.g. dir is Daktulosphaira_vitifoliae, we have daktulosphaira_vitifoliae)
                    try:
                        for candidate in subdir_candidates:
                            if not candidate:
                                continue
                            target_lower = candidate.lower().replace(" ", "_")
                            for child in images_base.iterdir():  # fresh list, not cache, so we see actual dirs
                                if not child.is_dir():
                                    continue
                                if child.name.lower().replace(" ", "_") == target_lower:
                                    sub_path = child / filename
                                    if sub_path.exists() and sub_path.is_file():
                                        return FileResponse(str(sub_path))
                                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                        sub_path = child / f"{filename_no_ext}{ext}"
                                        if sub_path.exists() and sub_path.is_file():
                                            return FileResponse(str(sub_path))
                                    # Prefixed filename (e.g. goat_Luzignan-20160310_140237): try without prefix in subdir (goat/Luzignan-20160310_140237.jpg)
                                    if filename_no_ext.startswith(candidate + "_"):
                                        suffix = filename_no_ext[len(candidate) + 1:]
                                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                            sub_path = child / f"{suffix}{ext}"
                                            if sub_path.exists() and sub_path.is_file():
                                                return FileResponse(str(sub_path))
                                    # Fallback: try numeric suffix only (e.g. wild_turkey/252.jpg)
                                    num_suffix = re.search(r"(\d+)$", filename_no_ext)
                                    if num_suffix:
                                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                            sub_path = child / f"{num_suffix.group(1)}{ext}"
                                            if sub_path.exists() and sub_path.is_file():
                                                return FileResponse(str(sub_path))
                            break
                    except Exception:
                        pass
                    for subdir in subdir_candidates:
                        if not subdir:
                            continue
                        species_dir = images_base / subdir
                        sub_path = species_dir / filename
                        if sub_path.exists() and sub_path.is_file():
                            return FileResponse(str(sub_path))
                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                            sub_path = species_dir / f"{filename_no_ext}{ext}"
                            if sub_path.exists() and sub_path.is_file():
                                return FileResponse(str(sub_path))
                        # Prefixed filename (e.g. goat_Luzignan-...): try without prefix in this subdir
                        if "_" in filename_no_ext and filename_no_ext.startswith(subdir + "_"):
                            suffix = filename_no_ext[len(subdir) + 1:]
                            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                sub_path = species_dir / f"{suffix}{ext}"
                                if sub_path.exists() and sub_path.is_file():
                                    return FileResponse(str(sub_path))
                        # Fallback: try numeric suffix only in this subdir (e.g. wild_turkey/252.jpg)
                        num_suffix = re.search(r"(\d+)$", filename_no_ext)
                        if num_suffix:
                            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                sub_path = species_dir / f"{num_suffix.group(1)}{ext}"
                                if sub_path.exists() and sub_path.is_file():
                                    return FileResponse(str(sub_path))
                    return None
                
                if IMAGES_TRY_SPECIES_FIRST:
                    r = _try_species()
                    if r is not None:
                        return r
                    if flat_to_try.exists() and flat_to_try.is_file():
                        return FileResponse(str(flat_to_try))
                else:
                    if flat_to_try.exists() and flat_to_try.is_file():
                        return FileResponse(str(flat_to_try))
                    r = _try_species()
                    if r is not None:
                        return r
                filename_lower = filename.lower()
                dir_listing = _get_cached_dir_listing(images_base)
                for item in dir_listing:
                    if item.is_file() and item.name.lower() == filename_lower:
                        return FileResponse(str(item))
                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                    potential_file = images_base / f"{filename_no_ext}{ext}"
                    if potential_file.exists() and potential_file.is_file():
                        return FileResponse(str(potential_file))
                
                print(f"❌ MCP Server: Image not found: {filename}")
                print(f"   Flat path resolved: {image_path_flat_resolved} (exists={image_path_flat_resolved.exists()})")
                prefix = filename_no_ext.split("_")[0] if "_" in filename_no_ext else filename_no_ext
                try:
                    same_prefix = [p.name for p in _get_cached_dir_listing(images_base) if p.is_file() and p.name.lower().startswith(prefix.lower() + "_")]
                    if same_prefix:
                        print(f"   Files with prefix '{prefix}_': {same_prefix[:10]}{'...' if len(same_prefix) > 10 else ''}")
                    else:
                        print(f"   No files with prefix '{prefix}_' in images dir.")
                except Exception:
                    pass
                raise HTTPException(status_code=404, detail=f"Image {filename} not found")
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ MCP Server: Error serving image: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")
        
        # MCP Resources
        @self.app.get("/mcp/resources")
        async def list_resources():
            """List all available resources"""
            return {
                "resources": self._get_resources(),
                "total": len(self._get_resources())
            }
        
        @self.app.get("/mcp/resources/{resource_name}")
        async def get_resource(resource_name: str):
            """Get a specific resource"""
            resources = self._get_resources()
            if resource_name not in resources:
                raise HTTPException(status_code=404, detail=f"Resource {resource_name} not found")
            return resources[resource_name]
        
        # Dataset API - single list endpoint so UI gets name + type for each dataset
        @self.app.get("/api/datasets/{dataset_name}")
        async def get_dataset(dataset_name: str):
            """Get specific dataset information"""
            dataset = self.dataset_registry.get_dataset(dataset_name)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
            return dataset
        
        @self.app.get("/api/datasets/{dataset_name}/images")
        async def get_dataset_images(dataset_name: str, limit: int = 100, offset: int = 0):
            """Get images from a specific dataset"""
            images = self.dataset_registry.get_images(dataset_name)
            if not images:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found or has no images")
            
            total = len(images)
            paginated_images = images[offset:offset + limit]
            
            return {
                "dataset": dataset_name,
                "images": paginated_images,
                "total_count": total,
                "limit": limit,
                "offset": offset
            }
        
        # Model API
        @self.app.get("/api/models")
        async def list_models():
            """List all available models"""
            # Get all models and convert to serializable format
            all_models = self.model_registry.get_all_models()
            models_dict = {}
            for name, model_info in all_models.items():
                models_dict[name] = {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            return {
                "models": models_dict,
                "total": len(models_dict)
            }
        
        @self.app.get("/api/models/{model_name}")
        async def get_model(model_name: str):
            """Get specific model information"""
            from models import ModelInfo
            
            model = self.model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            # Convert Model to ModelInfo (removes non-serializable handler)
            model_info = ModelInfo(
                name=model.name,
                type=model.type,
                description=model.description,
                version=model.version,
                supported_datasets=model.supported_datasets,
                parameters=model.parameters,
                metadata=model.metadata
            )
            
            # Convert to dict for JSON serialization
            return {
                "name": model_info.name,
                "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                "description": model_info.description,
                "version": model_info.version,
                "supported_datasets": model_info.supported_datasets,
                "parameters": model_info.parameters,
                "metadata": model_info.metadata
            }
        
        # Search API
        @self.app.post("/api/search")
        async def search_images(request: Request):
            """Search across all datasets"""
            try:
                body = await request.json()
                search_request = SearchRequest(**body)
                
                # This will be implemented to search across all datasets
                # For now, return a placeholder
                return {
                    "message": "Search functionality will be implemented",
                    "request": body
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Datasets API
        @self.app.get("/api/datasets")
        async def get_datasets():
            """Get all available datasets (all mcp_json names so UI dropdown is complete)."""
            try:
                datasets = []
                seen = set()
                for dataset_name, dataset in self.dataset_registry.datasets.items():
                    seen.add(dataset_name)
                    images = self.dataset_registry.get_images(dataset_name)
                    dataset_info = {
                        "name": dataset_name,
                        "description": dataset.description,
                        "type": dataset.type.value,
                        "image_count": len(images),
                        "collections": list(dataset.collections) if isinstance(dataset.collections, list) else list(dataset.collections.keys()),
                        "filters": self._extract_dataset_filters(dataset)
                    }
                    datasets.append(dataset_info)
                # Include all names from mcp_json filenames so dropdown lists every possible dataset
                for name in self.dataset_registry.get_all_mcp_dataset_names():
                    if name not in seen:
                        seen.add(name)
                        # Infer type from name so category filter works even when dataset didn't load
                        inferred = self.dataset_registry.infer_type_from_name(name)
                        datasets.append({
                            "name": name,
                            "description": "(dataset not loaded)",
                            "type": inferred,
                            "image_count": 0,
                            "collections": [],
                            "filters": {}
                        })
                datasets.sort(key=lambda d: (d["name"].lower(), d["name"]))
                return {"datasets": datasets}
            except Exception as e:
                print(f"❌ Error getting datasets: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/reload-datasets")
        async def reload_datasets():
            """Reload dataset registry from disk (re-read MCP JSON files). Use after updating JSON. Code changes require server restart."""
            try:
                out = self.dataset_registry.reload_datasets()
                print(f"🔄 Reloaded datasets: {out.get('datasets_loaded', 0)}")
                return out
            except Exception as e:
                print(f"❌ Error reloading datasets: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        # Inference API
        @self.app.post("/api/inference")
        async def run_inference(request: Request):
            """Run model inference"""
            try:
                body = await request.json()
                inference_request = InferenceRequest(**body)
                
                result = await self.model_registry.run_inference(inference_request)
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def _get_resources(self) -> Dict[str, Any]:
        """Get available resources"""
        return {
            "mcp://aifarms.org/resources/core": {
                "name": "Core Server Resources",
                "description": "Core MCP server resources and capabilities",
                "schema": {
                    "type": "object",
                    "properties": {
                        "server_info": {"type": "object"},
                        "available_tools": {"type": "array"},
                        "available_datasets": {"type": "array"},
                        "available_models": {"type": "array"}
                    }
                }
            },
            "mcp://aifarms.org/resources/datasets": {
                "name": "Available Datasets",
                "description": "List of all available datasets and their schemas",
                "schema": {
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "array", "items": {"type": "object"}}
                    }
                }
            },
            "mcp://aifarms.org/resources/models": {
                "name": "Available Models",
                "description": "List of all available ML models and their capabilities",
                "schema": {
                    "type": "object",
                    "properties": {
                        "models": {"type": "array", "items": {"type": "object"}}
                    }
                }
            }
        }
    
    def _get_tools(self) -> Dict[str, Any]:
        """Get available tools from registry"""
        tools = {}
        for tool_name, tool_info in self.tool_registry.get_all_tools().items():
            tools[f"mcp://aifarms.org/tools/{tool_name}"] = tool_info
        return tools
    
    def _register_default_tools(self):
        """Register default tools with the server"""
        # Search tool
        self.tool_registry.register_tool(
            name="search_images",
            description="Search for images across all datasets using natural language queries and filters",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "dataset": {"type": "string", "description": "Specific dataset to search in"},
                    "filters": {"type": "object", "description": "Search filters"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0}
                }
            },
            handler=self._search_tool_handler,
            tags=["search", "images", "datasets"]
        )
        
        # Inference tool
        self.tool_registry.register_tool(
            name="run_inference",
            description="Run ML model inference on images",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Dataset to run inference on"},
                    "model_name": {"type": "string", "description": "Model to use for inference"},
                    "image_ids": {"type": "array", "items": {"type": "string"}, "description": "Image IDs to process"},
                    "parameters": {"type": "object", "description": "Additional model parameters"}
                }
            },
            handler=self._inference_tool_handler,
            tags=["inference", "ml", "models"]
        )
        
        # LLM-powered search tool
        self.tool_registry.register_tool(
            name="llm_search",
            description="Intelligent search using LLM query understanding",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "dataset": {"type": "string", "description": "Specific dataset to search in"},
                    "category": {"type": "array", "items": {"type": "string"}, "description": "Restrict to data type: wildlife, domestic_animal, livestock, plants, pests, or animal"},
                    "limit": {"type": "integer", "default": 50, "description": "Max results per page (capped at 50 for performance)"},
                    "offset": {"type": "integer", "default": 0},
                    "dataset_offset": {"type": "integer", "default": 0, "description": "Skip this many datasets to load next batch"}
                }
            },
            handler=self._llm_search_handler,
            tags=["search", "llm", "intelligent", "semantic"]
        )
        
        # Dataset info tool
        self.tool_registry.register_tool(
            name="get_dataset_info",
            description="Get information about available datasets",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Specific dataset name (optional)"}
                }
            },
            handler=self._dataset_info_tool_handler,
            tags=["datasets", "info"]
        )
        
        # Model info tool
        self.tool_registry.register_tool(
            name="get_model_info",
            description="Get information about available ML models",
            input_schema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Specific model name (optional)"}
                }
            },
            handler=self._model_info_tool_handler,
            tags=["models", "info"]
        )
        
        # Croissant dataset crawler tool
        print(f"🔧 Checking Croissant crawler availability: {CROISSANT_CRAWLER_AVAILABLE}")
        print(f"   CROISSANT_CRAWLER_AVAILABLE value: {CROISSANT_CRAWLER_AVAILABLE}")
        print(f"   CroissantCrawler class: {CroissantCrawler}")
        
        if CROISSANT_CRAWLER_AVAILABLE:
            try:
                print(f"🔧 Attempting to register crawl_croissant_datasets tool...")
                self.tool_registry.register_tool(
                    name="crawl_croissant_datasets",
                    description="Crawl AI Institute portals for Croissant-formatted datasets",
                    input_schema={
                        "type": "object",
                        "properties": {}
                    },
                    handler=self._crawl_croissant_datasets_handler,
                    tags=["crawler", "datasets", "croissant"]
                )
                print("✅ Successfully registered crawl_croissant_datasets tool")
                
                # Verify it was actually registered
                all_tools = list(self.tool_registry.get_all_tools().keys())
                if "crawl_croissant_datasets" in all_tools:
                    print(f"✅ Verified: crawl_croissant_datasets is in registered tools list")
                else:
                    print(f"❌ WARNING: crawl_croissant_datasets NOT found in tools list!")
                    print(f"   Registered tools: {all_tools}")
            except Exception as e:
                print(f"❌ Failed to register crawl_croissant_datasets tool: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  Croissant crawler tool NOT registered (CROISSANT_CRAWLER_AVAILABLE is False)")
            print("   This means the import failed. Check the import error messages above.")
    
    # Tool handlers
    def _search_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for search tool"""
        query = input_data.get("query", "")
        dataset = input_data.get("dataset")
        filters = input_data.get("filters", {})
        limit = input_data.get("limit", 50)
        offset = input_data.get("offset", 0)
        
        print(f"🔍 Search request: query='{query}', dataset='{dataset}', filters={filters}")
        
        if dataset:
            # Search in specific dataset using adapter
            print(f"🔍 Searching in specific dataset: {dataset}")
            filtered_results = self.dataset_registry.search_dataset(dataset, query, filters)
            
            if filtered_results is None:
                return {
                    "dataset": dataset,
                    "query": query,
                    "results": [],
                    "total_count": 0,
                    "error": f"Dataset {dataset} not found or has no images"
                }
            
            return {
                "dataset": dataset,
                "query": query,
                "results": filtered_results[offset:offset + limit],
                "total_count": len(filtered_results)
            }
        else:
            # Search across all datasets using adapters
            # Apply category pre-filtering for performance
            category_filter = filters.get("category", [])
            datasets_to_search = []
            
            if category_filter:
                # Filter datasets by category first (same logic as _llm_search_handler)
                print(f"🔍 Category pre-filtering: {category_filter}")
                for dataset_name, dataset_obj in self.dataset_registry.datasets.items():
                    dataset_category = dataset_obj.type.value.lower()
                    # Map category filter to dataset type (animal = wildlife + domestic + livestock)
                    category_mapping = {
                        "pest": ["pests"],
                        "animal": ["wildlife", "domestic_animal", "livestock"],
                        "wildlife": ["wildlife"],
                        "domestic_animal": ["domestic_animal"],
                        "domestic": ["domestic_animal"],
                        "livestock": ["livestock"],
                        "plant": ["plants"]
                    }
                    should_include = False
                    for cat in category_filter:
                        cat_lower = cat.lower()
                        if cat_lower in category_mapping:
                            if dataset_category in category_mapping[cat_lower]:
                                should_include = True
                                break
                        elif cat_lower == dataset_category:
                            should_include = True
                            break
                    
                    if should_include:
                        datasets_to_search.append(dataset_name)
            else:
                # No category filter - search all datasets
                datasets_to_search = list(self.dataset_registry.datasets.keys())
            
            print(f"🔍 Searching {len(datasets_to_search)} datasets (out of {len(self.dataset_registry.datasets)} total)")
            all_results = []
            
            for dataset_name in datasets_to_search:
                print(f"🔍 Searching dataset: {dataset_name}")
                filtered_results = self.dataset_registry.search_dataset(dataset_name, query, filters)
                if filtered_results:
                    # Add dataset info to each result
                    for result in filtered_results:
                        result['dataset'] = dataset_name
                    all_results.extend(filtered_results)
            
            print(f"🔍 Total results found: {len(all_results)}")
            
            return {
                "query": query,
                "results": all_results[offset:offset + limit],
                "total_count": len(all_results),
                "searched_datasets": datasets_to_search
            }
    
    def _apply_search_filters(self, images: List[Dict[str, Any]], query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply search filters to images (legacy method - now uses adapters via search_dataset)"""
        # This method is kept for backward compatibility but search_dataset now uses adapters
        # The adapter-based search is handled in dataset_registry.search_dataset()
        filtered_images = images.copy()
        
        # Apply text search if query provided
        if query.strip():
            query_lower = query.lower()
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_query(img, query_lower)
            ]
        
        # Apply category filter
        if filters.get("category"):
            category_filter = [c.lower() for c in filters["category"]]
            filtered_images = [
                img for img in filtered_images
                if img.get("category", "").lower() in category_filter
            ]
        
        # Apply species filter
        if filters.get("species"):
            species_filter = [s.lower() for s in filters["species"]]
            filtered_images = [
                img for img in filtered_images
                if img.get("collection", "").lower() in species_filter
            ]
        
        # Apply time filter
        if filters.get("time"):
            time_filter = [t.lower() for t in filters["time"]]
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_time(img, time_filter)
            ]
        
        # Apply season filter
        if filters.get("season"):
            season_filter = [s.lower() for s in filters["season"]]
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_season(img, season_filter)
            ]
        
        return filtered_images
    
    async def _llm_search_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM-powered search with intelligent query understanding"""
        query = input_data.get("query", "")
        dataset = input_data.get("dataset")
        category_from_client = input_data.get("category") or []
        # Cap results per request so we never serve thousands of images at once (slow load). User can paginate for more.
        MAX_RESULTS_PER_PAGE = 50
        limit = min(int(input_data.get("limit", 50)), MAX_RESULTS_PER_PAGE)
        offset = int(input_data.get("offset", 0))
        dataset_offset = max(0, int(input_data.get("dataset_offset", 0)))
        
        print(f"🧠 LLM search request: query='{query}', dataset='{dataset}', category={category_from_client}, dataset_offset={dataset_offset}")
        
        try:
            # When user selects a dataset but leaves query empty → show all images from that dataset (no LLM)
            query_understanding = None
            category_only = [
                c.strip() for c in (category_from_client if isinstance(category_from_client, list) else [category_from_client])
                if c and str(c).strip()
            ]
            if (not query or not query.strip()) and dataset and dataset in self.dataset_registry.datasets:
                query_understanding = QueryUnderstanding(
                    intent=f"all images from {dataset}",
                    entities=[],
                    filters={"species": [dataset]},
                    confidence=1.0,
                    reasoning="Dataset-only search: show all images from the selected dataset.",
                    description_query=None,
                )
                print(f"🧠 Dataset-only search: returning all images from '{dataset}' (no query)")
            elif (not query or not query.strip()) and category_only:
                # Empty query + category selected → browse all datasets in that category (no LLM, no species)
                query_understanding = QueryUnderstanding(
                    intent=f"all images in category {category_only}",
                    entities=[],
                    filters={"category": category_only},
                    confidence=1.0,
                    reasoning="Category-only browse: show all images from datasets in the selected category.",
                    description_query=None,
                )
                print(f"🧠 Category-only browse: returning all images in category {category_only} (no query)")
            
            # Get available filters for LLM context
            available_filters = {}
            if dataset:
                if dataset in self.dataset_registry.datasets:
                    available_filters = self._extract_dataset_filters(self.dataset_registry.datasets[dataset])
                    # Also add collections to available_filters for species matching
                    dataset_obj = self.dataset_registry.datasets[dataset]
                    if dataset_obj.collections:
                        if "collections" not in available_filters:
                            available_filters["collections"] = []
                        available_filters["collections"].extend(list(dataset_obj.collections))
            else:
                # Combine filters from all datasets
                all_collections = set()
                for dataset_name, dataset_obj in self.dataset_registry.datasets.items():
                    dataset_filters = self._extract_dataset_filters(dataset_obj)
                    for key, values in dataset_filters.items():
                        if key not in available_filters:
                            available_filters[key] = []
                        available_filters[key].extend(values)
                    # Collect collections from all datasets
                    if dataset_obj.collections:
                        all_collections.update(dataset_obj.collections)
                
                # Add collections to available_filters
                if all_collections:
                    available_filters["collections"] = sorted(list(all_collections))
                
                # Treat dataset names as valid species so e.g. "raspberry" matches dataset "raspberry"
                for dataset_name in self.dataset_registry.datasets:
                    if dataset_name and dataset_name.strip():
                        available_filters.setdefault("species", []).append(dataset_name.strip())
                
                # Remove duplicates
                for key in available_filters:
                    available_filters[key] = sorted(list(set(available_filters[key])))
                # Add opossum/oppossum to species options when we have Virginia opossum dataset so LLM can match and return high confidence
                opossum_datasets = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                if opossum_datasets:
                    for syn in ("opossum", "oppossum", "opossums", "oppossums"):
                        if syn not in available_filters.get("species", []):
                            available_filters.setdefault("species", []).append(syn)
                    available_filters["species"] = sorted(list(set(available_filters["species"])))
            
            # Use LLM to understand the query when we have a query (skip LLM for dataset-only search)
            if query_understanding is None:
                if not self.llm_service or not self.llm_service.is_available():
                    return {
                        "query": query,
                        "error": "LLM service is not available. Please set OPENAI_API_KEY environment variable.",
                        "results": [],
                        "total_count": 0,
                        "llm_understanding": None
                    }
                print(f"🧠 Understanding query with LLM...")
                print(f"   LLM Service available: {self.llm_service.is_available() if self.llm_service else False}")
                if self.llm_service:
                    print(f"   OpenAI available: {self.llm_service.openai_available}")
                    print(f"   Gemini available: {self.llm_service.gemini_available}")
                    print(f"   Provider: {self.llm_service.provider}")
                llm_filters = self._compact_filters_for_llm(available_filters, query, dataset)
                query_understanding = await self.llm_service.understand_query(query, llm_filters)
            
            # Normalize filters so every value is a list of strings (LLM sometimes returns nested lists or non-strings)
            def _flatten_filter_val(v):
                if v is None:
                    return []
                if isinstance(v, str):
                    return [v.strip()] if v.strip() else []
                if isinstance(v, list):
                    out = []
                    for x in v:
                        if isinstance(x, list):
                            out.extend(_flatten_filter_val(x))
                        elif x is not None and str(x).strip():
                            out.append(str(x).strip())
                    return out
                return [str(v).strip()] if str(v).strip() else []
            if getattr(query_understanding, "filters", None):
                for key in list(query_understanding.filters.keys()):
                    query_understanding.filters[key] = _flatten_filter_val(query_understanding.filters[key])
                # Normalize species: LLMs sometimes return "(moth)" or "[moth]" — strip parens/brackets so we match "moth"
                if query_understanding.filters.get("species"):
                    normalized = []
                    for s in query_understanding.filters["species"]:
                        t = str(s).strip()
                        while t and t[0] in "([{\"" and t[-1] in ")]}\"":
                            t = t[1:-1].strip()
                        if t:
                            normalized.append(t)
                    if normalized:
                        query_understanding.filters["species"] = normalized
                # Split comma-separated action values so "foraging, eating" -> ["foraging", "eating"]
                if query_understanding.filters.get("action"):
                    action_vals = query_understanding.filters["action"]
                    split_actions = []
                    for a in action_vals:
                        for part in str(a).split(","):
                            p = part.strip()
                            if p:
                                split_actions.append(p)
                    if split_actions:
                        query_understanding.filters["action"] = split_actions
                # Correct LLM misparse: "cat sleeping" / "dog sleeping" sometimes returned as species "cat, pin" or "dog, pin" (sleeping heard as pin)
                query_lower_pre = query.lower()
                if query_understanding.filters.get("species") and "sleeping" in query_lower_pre:
                    species_list = query_understanding.filters["species"]
                    if len(species_list) == 1:
                        one = species_list[0].lower().strip()
                        if "pin" in one and not query_understanding.filters.get("action"):
                            if "cat" in one:
                                query_understanding.filters["species"] = ["cat"]
                                query_understanding.filters["action"] = ["sleeping"]
                                print(f"   ✅ Corrected LLM misparse: species 'cat, pin' → species ['cat'], action ['sleeping'] (query was 'cat sleeping')")
                            elif "dog" in one:
                                query_understanding.filters["species"] = ["dog"]
                                query_understanding.filters["action"] = ["sleeping"]
                                print(f"   ✅ Corrected LLM misparse: species 'dog, pin' → species ['dog'], action ['sleeping'] (query was 'dog sleeping')")
                # Correct LLM misparse: "dog eating" / "cat feeding" etc. sometimes returned as species "eating" or "feeding" (action word in species)
                action_only_words = {"eating", "feeding", "foraging"}
                if query_understanding.filters.get("species") and (action_only_words & set(query_lower_pre.split())):
                    species_list = list(query_understanding.filters["species"])
                    if len(species_list) == 1:
                        one = species_list[0].lower().strip().replace(" ", "_")
                        if one in action_only_words:
                            # Query has form "X eating" or "X feeding" etc.; use X as species, one as action (fix even if action already set)
                            if "dog" in query_lower_pre:
                                query_understanding.filters["species"] = ["dog"]
                                query_understanding.filters["action"] = [one]
                                print(f"   ✅ Corrected LLM misparse: species [{one!r}] → species ['dog'], action [{one!r}] (query was '{query}')")
                            elif "cat" in query_lower_pre:
                                query_understanding.filters["species"] = ["cat"]
                                query_understanding.filters["action"] = [one]
                                print(f"   ✅ Corrected LLM misparse: species [{one!r}] → species ['cat'], action [{one!r}] (query was '{query}')")
                # Correct metadata misparse: "cat walking" / "dog walking" sometimes returned as species ["and", "king", "pea", "the", "thin", "walking"] (words from action descriptions in filters)
                _nonsense_species = {"and", "king", "pea", "tan", "the", "thin", "walking", "or", "at", "in", "on", "for", "of", "with", "within", "from", "an", "a"}
                if query_understanding.filters.get("species"):
                    species_set = {s.lower().strip() for s in query_understanding.filters["species"]}
                    if species_set and species_set <= _nonsense_species:
                        if re.search(r"\bcat\b", query_lower_pre):
                            query_understanding.filters["species"] = ["cat"]
                            print(f"   ✅ Corrected metadata misparse: species {list(species_set)} → ['cat'] (query was '{query}')")
                        elif re.search(r"\bdog\b", query_lower_pre):
                            query_understanding.filters["species"] = ["dog"]
                            print(f"   ✅ Corrected metadata misparse: species {list(species_set)} → ['dog'] (query was '{query}')")
                # Split species values that contain commas (e.g. "bobcat, dog" → ["bobcat", "dog"]) so we don't look for literal "cat, pin"
                if query_understanding.filters.get("species"):
                    expanded = []
                    for s in query_understanding.filters["species"]:
                        if "," in str(s):
                            expanded.extend(x.strip() for x in str(s).split(",") if x.strip())
                        elif str(s).strip():
                            expanded.append(str(s).strip())
                    if expanded:
                        query_understanding.filters["species"] = expanded
                # Remove time words from species when they belong in time filter (e.g. LLM put "day" in species for "cat at daytime")
                # Otherwise "day" matches pest datasets like "Dog-Day Cicada" and we get cicada images in cat search.
                if query_understanding.filters.get("species") and query_understanding.filters.get("time"):
                    time_vals = {str(t).lower().strip() for t in query_understanding.filters["time"]}
                    time_words_in_species = {"day", "night", "dawn", "dusk", "morning", "afternoon", "evening", "daytime", "nighttime"}
                    if time_vals & time_words_in_species:
                        species_cleaned = [
                            s for s in query_understanding.filters["species"]
                            if str(s).lower().strip() not in time_words_in_species
                        ]
                        if species_cleaned != query_understanding.filters["species"]:
                            query_understanding.filters["species"] = species_cleaned
                            print(f"   ✅ Removed time words from species (kept time in time filter): {species_cleaned}")
                # When user asked for mouse (rodent) but LLM returned a pest like "mouse moth", treat as rodent search
                if query and re.search(r"\bmouse\b|\bmice\b", (query or "").lower()) and query_understanding.filters.get("species"):
                    species_list = query_understanding.filters["species"]
                    # If every species value looks like the insect (e.g. "mouse moth") not the rodent, override to "mouse"
                    all_mouse_moth_like = all(
                        "moth" in str(s).lower() and "mouse" in str(s).lower()
                        for s in species_list if s
                    ) and len(species_list) >= 1
                    if all_mouse_moth_like:
                        query_understanding.filters["species"] = ["mouse"]
                        print(f"   ✅ Query is mouse/mice (rodent) but species was pest (e.g. mouse moth) → species ['mouse']")
                # Map opossum/oppossum to canonical dataset name (e.g. virginia_opossum) so pre-filtering finds the dataset
                if query_understanding.filters.get("species") and self.dataset_registry.datasets:
                    species_list = query_understanding.filters["species"]
                    opossum_vals = {"opossum", "oppossum", "opossums", "oppossums"}
                    if any(str(s).lower() in opossum_vals for s in species_list):
                        canonical = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                        if canonical:
                            rest = [s for s in species_list if str(s).lower() not in opossum_vals]
                            query_understanding.filters["species"] = sorted(list(set(rest + canonical)))
                # When user asked for a generic term (e.g. "bumble bee") but LLM returned one specific species (e.g. "black and gold bumble bee"),
                # expand to all species/datasets containing the query so page 2 and "load more" return results instead of "no images found".
                if query and available_filters and query_understanding.filters.get("species"):
                    species_list = query_understanding.filters["species"]
                    query_lower = query.lower().strip()
                    if len(species_list) == 1 and (" " in query_lower or len(query_lower) >= 8):
                        one = str(species_list[0]).lower().strip()
                        if query_lower in one and len(one) > len(query_lower):
                            # Query is substring of species (e.g. "bumble bee" in "black and gold bumble bee") — expand to all matching
                            from_available = [
                                s for s in (available_filters.get("species") or [])
                                if s and query_lower in str(s).lower()
                            ]
                            if from_available:
                                expanded = sorted(list(set(species_list + from_available)))
                                query_understanding.filters["species"] = expanded
                                print(f"   ✅ Expanded species to all matching query '{query_lower}': {len(expanded)} species (was 1)")
            
            # Reject rule-based fallback results - require real LLM understanding
            if query_understanding.confidence < 0.7:
                print(f"⚠️  Low confidence ({query_understanding.confidence}) - this might be rule-based fallback")
                # Still proceed but log warning
            
            print(f"🧠 LLM Understanding:")
            print(f"   Intent: {query_understanding.intent}")
            print(f"   Entities: {query_understanding.entities}")
            print(f"   Filters: {query_understanding.filters}")
            print(f"   Confidence: {query_understanding.confidence}")
            print(f"   Reasoning: {query_understanding.reasoning}")
            
            # Ensure plant_state is set when query clearly asks for ripeness (e.g. "raspberry ripe")
            # LLM sometimes omits it; without it we return all images and strict filter never runs
            query_lower = query.lower()

            def _query_has_whole_word(term: str) -> bool:
                return bool(re.search(rf"\b{re.escape(str(term or '').lower().strip())}\b", query_lower))

            # The metadata fallback can match filter values as substrings. For terms that are also common
            # prefixes inside species names (e.g. "green" in "greenhouse whitefly"), discard them unless the
            # user actually typed them as standalone words.
            _bad_substring_terms = _GENERIC_ADJECTIVE_WORDS | {"ripe", "unripe", "mature", "immature"}
            _species_before = query_understanding.filters.get("species") or []
            if _species_before:
                _species_after = [
                    s for s in _species_before
                    if str(s).lower().strip() not in _bad_substring_terms or _query_has_whole_word(str(s))
                ]
                if _species_after != _species_before:
                    print(f"🧠 Removed substring-only species filter(s): {sorted(set(_species_before) - set(_species_after))}")
                    query_understanding.filters["species"] = _species_after
            _plant_state_before = query_understanding.filters.get("plant_state") or []
            if _plant_state_before:
                _plant_state_after = [
                    ps for ps in _plant_state_before
                    if str(ps).lower().strip() not in _bad_substring_terms or _query_has_whole_word(str(ps))
                ]
                if _plant_state_after != _plant_state_before:
                    print(f"🧠 Removed substring-only plant_state filter(s): {sorted(set(_plant_state_before) - set(_plant_state_after))}")
                    query_understanding.filters["plant_state"] = _plant_state_after
            if query_understanding.entities:
                _entities_before = list(query_understanding.entities)
                query_understanding.entities = [
                    e for e in _entities_before
                    if str(e).lower().strip() not in _bad_substring_terms or _query_has_whole_word(str(e))
                ]
                if query_understanding.entities != _entities_before:
                    print(f"🧠 Removed substring-only entity/entities: {sorted(set(_entities_before) - set(query_understanding.entities))}")

            # Collapse an over-expanded species filter. The rule-based matcher can expand a bare group word
            # like "grasshopper" into every matching common name (e.g. 60 "* grasshopper" species). Their
            # shared descriptor segments ("eastern", "bird") then wrongly match wildlife datasets (skunk,
            # squirrel, bird). If the query is essentially just ONE generic insect group word, keep only that
            # word so we cleanly match the group and the UI shows a single, sensible filter.
            _expanded_species = query_understanding.filters.get("species") or []
            if len(_expanded_species) >= 3:
                def _sing_tok(t: str) -> str:
                    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
                _q_terms = {_sing_tok(t) for t in re.findall(r"[a-z]+", query_lower)}
                _q_pest = {t for t in _q_terms if t in _PEST_TYPE_WORDS_SET}
                _q_specific = {
                    t for t in _q_terms
                    if len(t) >= 3 and t not in _PEST_TYPE_WORDS_SET and t not in _NON_SUBJECT_WORDS
                }
                if len(_q_pest) == 1 and not _q_specific:
                    _group = next(iter(_q_pest))
                    if all(re.search(rf"\b{re.escape(_group)}s?\b", str(s).lower()) for s in _expanded_species):
                        query_understanding.filters["species"] = [_group]
                        if query_understanding.confidence < 0.9:
                            query_understanding.confidence = 0.9
                        print(f"🧠 Collapsed {len(_expanded_species)} expanded species sharing '{_group}' → ['{_group}'] (query is the group word)")
            plant_fruit_species = {"raspberry", "raspberries", "strawberry", "strawberries", "blueberry", "blueberries", "blackberry", "blackberries"}
            species_in_query = [s for s in (query_understanding.filters.get("species") or []) if s]
            species_lower = {s.lower().strip().replace("_", "") for s in species_in_query}
            is_plant_query = bool(species_lower & {t.replace("_", "") for t in plant_fruit_species})
            if is_plant_query and "ripe" in query_lower and not query_understanding.filters.get("plant_state"):
                query_understanding.filters["plant_state"] = ["ripe"]
                print(f"   ✅ Injected plant_state: ['ripe'] from query (query implied ripe)")
            if is_plant_query and ("unripe" in query_lower or " green " in query_lower or "green fruit" in query_lower) and not query_understanding.filters.get("plant_state"):
                query_understanding.filters["plant_state"] = ["unripe"]
                print(f"   ✅ Injected plant_state: ['unripe'] from query")
            
            # Goat tail position: "tail up" / "tail down" → tail_positions (used by goat_2)
            species_for_tail = {s.lower().strip().replace(" ", "_") for s in (query_understanding.filters.get("species") or [])}
            if species_for_tail & {"goat", "goat_2"} and "tail" in query_lower and not query_understanding.filters.get("tail_positions"):
                if "tail up" in query_lower or "tail  up" in query_lower or ("tail" in query_lower and " up" in query_lower):
                    query_understanding.filters["tail_positions"] = ["up"]
                    print(f"   ✅ Injected tail_positions: ['up'] from query")
                elif "tail down" in query_lower or "tail  down" in query_lower or ("tail" in query_lower and " down" in query_lower):
                    query_understanding.filters["tail_positions"] = ["down"]
                    print(f"   ✅ Injected tail_positions: ['down'] from query")
            
            # If query clearly mentions a fruit/species but LLM left species empty, inject so we only search that dataset.
            query_species = query_understanding.filters.get("species") or []
            # Track whether the LLM itself supplied the species. If it didn't but we resolve it
            # server-side below, the query WAS understood (and matches the catalog exactly), so we
            # later raise the AI-confidence score to reflect that instead of the LLM's low fallback.
            llm_provided_species = bool(query_species)
            # True once we resolve a SPECIFIC dataset from a common name (e.g. "golden beetle" →
            # golden tortoise beetle). When set, we skip the generic pest-type-word injection so we
            # don't broaden the search back out to every beetle/bug dataset.
            cn_injected = False
            dataset_names_lower = {dn.lower(): dn for dn in self.dataset_registry.datasets}
            # Opossum / oppossum (common misspelling): inject first so we never search all datasets
            if not query_species and self.dataset_registry.datasets:
                if "opossum" in query_lower or "oppossum" in query_lower or "opossums" in query_lower or "oppossums" in query_lower:
                    opossum_datasets = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                    if opossum_datasets:
                        query_understanding.filters["species"] = sorted(opossum_datasets)
                        query_species = query_understanding.filters["species"]
                        print(f"   ✅ Injected species (opossum/oppossum → Virginia opossum): {query_understanding.filters['species']}")
                    else:
                        # Dataset not in collection — return clear message instead of searching all (which would show pest/irrelevant images)
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": "This species (opossum) is not available in the collection. Virginia opossum has not been added to this server.",
                            "searched_datasets": [],
                        }
            if not query_species and self.dataset_registry.datasets:
                fruit_to_try = []
                if "blueberry" in query_lower or "blueberries" in query_lower:
                    fruit_to_try.append("blueberry")
                if "raspberry" in query_lower or "raspberries" in query_lower:
                    fruit_to_try.append("raspberry")
                if "strawberry" in query_lower or "strawberries" in query_lower:
                    fruit_to_try.append("strawberry")
                if "blackberry" in query_lower or "blackberries" in query_lower:
                    fruit_to_try.append("blackberry")
                if "mango" in query_lower or "mangoes" in query_lower or "mangos" in query_lower:
                    fruit_to_try.append("mango")
                if "grape" in query_lower or "grapes" in query_lower:
                    fruit_to_try.append("grapes")
                if "apple" in query_lower or "apples" in query_lower:
                    fruit_to_try.append("apple")
                if "citrus" in query_lower or "orange" in query_lower or "oranges" in query_lower:
                    fruit_to_try.append("orange")
                for name in fruit_to_try:
                    norm = name.lower().strip().replace("_", "").replace("-", "")
                    matches = []
                    for d_lower, dataset_name in dataset_names_lower.items():
                        dn_norm = d_lower.replace("_", "").replace("-", "")
                        # Match exact (mango) or prefix with _/- (mango_1, mango_2); use d_lower for prefix so mango_1 matches
                        if (norm == dn_norm or
                                d_lower == norm or
                                d_lower.startswith(norm + "_") or
                                d_lower.startswith(norm + "-")):
                            matches.append(dataset_name)
                    if matches:
                        query_understanding.filters["species"] = sorted(matches)
                        print(f"   ✅ Injected species: {query_understanding.filters['species']} from query (matched '{name}')")
                        break
                    else:
                        # Query clearly asked for this fruit/species but no dataset exists — set species so we return "no dataset" instead of searching all
                        query_understanding.filters["species"] = [name]
                        print(f"   ⚠️  No dataset named '{name}' (or {name}_*) found; will return error instead of searching all datasets")
                        break
                # Fallback: if still no species, check if any query word exactly matches a dataset name (avoids searching all datasets for "mango")
                if not query_understanding.filters.get("species"):
                    words = [w.strip().lower() for w in query.split() if len(w.strip()) >= 3 and w.strip().isalpha()]
                    skip_words = {"the", "and", "for", "with", "images", "pictures", "photos", "species", "show", "find", "search", "ripe", "unripe", "green", "red"}
                    for word in words:
                        if word in skip_words:
                            continue
                        if word in dataset_names_lower:
                            query_understanding.filters["species"] = [dataset_names_lower[word]]
                            print(f"   ✅ Injected species: ['{dataset_names_lower[word]}'] from query word '{word}' (exact dataset name)")
                            break
                        # singular/plural: "mangoes" -> "mango"
                        word_singular = word[:-1] if word.endswith("s") and len(word) > 2 and not word.endswith("ss") else word
                        if word_singular in dataset_names_lower:
                            query_understanding.filters["species"] = [dataset_names_lower[word_singular]]
                            print(f"   ✅ Injected species: ['{dataset_names_lower[word_singular]}'] from query word '{word}'")
                            break
                # Generic subject-word → dataset match: e.g. "squirrel" → eastern_fox_squirrel, eaestern_gray_squirrel;
                # "fox" → red_fox, grey_fox. Match query words as whole segments of dataset names (split on _ or -),
                # preferring non-pest datasets so we don't pull in thousands of pests for a common animal word.
                if not query_understanding.filters.get("species"):
                    subject_words = [
                        w for w in re.findall(r"[a-z]+", query_lower)
                        if len(w) >= 3 and w not in _NON_SUBJECT_WORDS
                    ]
                    non_pest_types = {"wildlife", "domestic_animal", "livestock", "plants"}
                    seg_nonpest, seg_pest = [], []
                    for word in subject_words:
                        word_sing = word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
                        # Plain words match dataset name SEGMENTS (e.g. "squirrel" → eastern_fox_squirrel).
                        seg_terms = {word, word_sing}
                        # Common-name synonyms (e.g. "groundhog" → "woodchuck") match as a substring so spelling
                        # variants in dataset names still resolve.
                        syn_terms = {_SPECIES_SYNONYMS[w] for w in (word, word_sing) if w in _SPECIES_SYNONYMS}
                        for dn in self.dataset_registry.datasets:
                            dn_l = dn.lower()
                            segs = re.split(r"[_\-]", dn_l)
                            if any(t in segs for t in seg_terms) or any(t in dn_l for t in syn_terms):
                                dtype = self.dataset_registry.infer_type_from_name(dn)
                                (seg_nonpest if dtype in non_pest_types else seg_pest).append(dn)
                    chosen = sorted(set(seg_nonpest)) or sorted(set(seg_pest))
                    if chosen:
                        query_understanding.filters["species"] = chosen
                        print(f"   ✅ Injected species from subject word(s) → datasets: {chosen}")
                # Common-name phrase → scientific-name dataset (e.g. "painted lady" → Vanessa_cardui),
                # for datasets named by scientific name where the common name lives in metadata.
                if not query_understanding.filters.get("species"):
                    cn_datasets = self._resolve_datasets_by_common_name(query)
                    if cn_datasets:
                        query_understanding.filters["species"] = sorted(cn_datasets)
                        cn_injected = True
                        print(f"   ✅ Injected species from common name → datasets: {sorted(cn_datasets)}")
                # Rabbit = cottontail = white cottontail: if query mentions any and no species yet, use eastern_cottontail (not white_cottontail)
                if not query_understanding.filters.get("species") and self.dataset_registry.datasets:
                    if "rabbit" in query_lower or "cottontail" in query_lower or "rabbits" in query_lower or "cottontails" in query_lower or "white cottontail" in query_lower or "white cottontails" in query_lower:
                        eastern = [d for d in self.dataset_registry.datasets if d and d.lower() == "eastern_cottontail"]
                        cottontail_datasets = eastern if eastern else [d for d in self.dataset_registry.datasets if "cottontail" in d.lower()]
                        if cottontail_datasets:
                            query_understanding.filters["species"] = sorted(cottontail_datasets)
                            print(f"   ✅ Injected species (rabbit/cottontail/white cottontail → eastern_cottontail): {query_understanding.filters['species']}")
                        else:
                            return {
                                "query": query,
                                "llm_understanding": query_understanding,
                                "results": [],
                                "total_count": 0,
                                "error": "This species (rabbit / cottontail) is not available in the collection.",
                                "searched_datasets": [],
                            }
                # Pest type words: ensure "beetle", "butterfly", "wasp", "moth", "aphid", "stink bug", etc.
                # are in the species filter when the query mentions them so search matches pest images via
                # common_names (e.g. ["French Paper Wasp", "wasp"], ["soybean aphid", "aphid"]).
                # Derived from the module-level _PEST_TYPE_WORDS_SET (so aphid/weevil/midge/caterpillar are
                # included and the two lists can't drift) plus multi-word phrases handled specially.
                _PEST_TYPE_WORDS = sorted(_PEST_TYPE_WORDS_SET) + [
                    "stink bug", "stink bugs", "true bug", "true bugs",
                ]
                query_species_list = query_understanding.filters.get("species") or []
                species_set = {s.lower().strip() for s in query_species_list}
                # Skip generic pest-type injection when a specific common-name dataset was already resolved
                # (e.g. "golden beetle" → golden tortoise beetle; don't also add generic "beetle").
                if not cn_injected:
                    for type_word in _PEST_TYPE_WORDS:
                        type_lower = type_word.lower()
                        if len(type_lower) > 4 and type_lower.endswith("ies"):
                            type_singular = type_lower[:-3] + "y"
                        elif type_lower.endswith("s") and len(type_lower) > 1 and not type_lower.endswith("ss"):
                            type_singular = type_lower[:-1]
                        else:
                            type_singular = type_lower
                        # Match WHOLE words/phrases only, so "bug" does not match inside "ladybug".
                        if re.search(rf"\b{re.escape(type_lower)}\b", query_lower) or re.search(rf"\b{re.escape(type_singular)}\b", query_lower):
                            if type_singular not in species_set and type_lower not in species_set:
                                if not any(type_singular in s or type_lower in s for s in species_set):
                                    query_species_list.append(type_singular)
                                    species_set.add(type_singular)
                                    print(f"   ✅ Injected species (pest type from query): '{type_singular}'")
                if query_species_list != (query_understanding.filters.get("species") or []):
                    query_understanding.filters["species"] = sorted(list(set(query_species_list)))
                # If the query clearly names a pest group ("aphids on a leaf"), context words like
                # "leaf" should not leave plant datasets (e.g. red_leaf) in the species filter.
                _species_after_pest = query_understanding.filters.get("species") or []
                _generic_pest_terms = {
                    s.lower().strip()
                    for s in _species_after_pest
                    if s and (
                        s.lower().strip() in _PEST_TYPE_WORDS_SET
                        or s.lower().strip().rstrip("s") in _PEST_TYPE_WORDS_SET
                    )
                }
                if _generic_pest_terms and len(_species_after_pest) > 1:
                    _filtered_species = []
                    _dropped_species = []
                    _non_pest_types_for_cleanup = {"wildlife", "domestic_animal", "livestock", "plants"}
                    for s in _species_after_pest:
                        s_key = str(s).lower().strip()
                        dataset_name = dataset_names_lower.get(s_key)
                        if dataset_name and self.dataset_registry.infer_type_from_name(dataset_name) in _non_pest_types_for_cleanup:
                            _dropped_species.append(s)
                            continue
                        _filtered_species.append(s)
                    if _dropped_species:
                        query_understanding.filters["species"] = sorted(set(_filtered_species))
                        print(f"🧠 Dropped context plant/non-pest species from pest query: {_dropped_species}")

            # Goat tail position (re-evaluate AFTER species resolution). The earlier pass runs before species
            # injection, so for queries like "goats with tail up" — where the LLM leaves species empty and we
            # resolve "goat" server-side — the tail filter would otherwise never be set.
            if query and not query_understanding.filters.get("tail_positions"):
                _species_for_tail = {s.lower().strip().replace(" ", "_") for s in (query_understanding.filters.get("species") or [])}
                _is_goat = any(s == "goat" or s.startswith("goat_") for s in _species_for_tail)
                if _is_goat and "tail" in query_lower:
                    if "tail up" in query_lower or re.search(r"\btail\b.*\bup\b", query_lower):
                        query_understanding.filters["tail_positions"] = ["up"]
                        print(f"   ✅ Injected tail_positions: ['up'] from query (post-species)")
                    elif "tail down" in query_lower or re.search(r"\btail\b.*\bdown\b", query_lower):
                        query_understanding.filters["tail_positions"] = ["down"]
                        print(f"   ✅ Injected tail_positions: ['down'] from query (post-species)")
                    elif re.search(r"\b(undetermined|undetermend|indeterminate|unknown|not\s+visible|unclear|obscured)\b", query_lower):
                        query_understanding.filters["tail_positions"] = ["undetermined"]
                        print(f"   ✅ Injected tail_positions: ['undetermined'] from query (post-species)")

            # Time: inject canonical buckets when the query mentions time but the LLM omitted it
            if query and not query_understanding.filters.get("time"):
                ql = query.lower()
                injected_times = []
                for bucket, phrases in _TIME_QUERY_PHRASES.items():
                    if re.search(rf"\b{re.escape(bucket)}\b", ql) or any(p in ql for p in phrases):
                        injected_times.append(bucket)
                if injected_times:
                    query_understanding.filters["time"] = sorted(set(injected_times))
                    print(f"   ✅ Injected time: {query_understanding.filters['time']} from query")

            # Action: inject a canonical action when the query names a behavior but the LLM omitted it.
            # Without this, a leftover behavior word (e.g. "eating") would be treated as a required literal
            # description phrase and wrongly exclude images described as "foraging"/"feeding".
            if query and not query_understanding.filters.get("action"):
                qwords = set(re.findall(r"[a-z]+", query.lower()))
                injected_actions = [
                    canonical for canonical, variations in _ACTION_QUERY_MAP.items()
                    if qwords & set(variations)
                ]
                if injected_actions:
                    query_understanding.filters["action"] = sorted(set(injected_actions))
                    print(f"   ✅ Injected action: {query_understanding.filters['action']} from query")

            # Category from the UI dropdown is a HARD restriction: only datasets of that data type are searched.
            ui_categories = [
                c.strip() for c in (category_from_client if isinstance(category_from_client, list) else [category_from_client])
                if c and str(c).strip()
            ]
            if ui_categories:
                query_understanding.filters["category"] = ui_categories
                print(f"   ✅ Category filter from UI (hard restriction): {ui_categories}")
                # If the detected species clearly belongs to a different category, explain instead of returning nonsense.
                detected_species = query_understanding.filters.get("species") or []
                inferred_categories = self._species_implied_categories(detected_species) if detected_species else set()
                if inferred_categories and not self._category_compatible(ui_categories, inferred_categories):
                    species_str = ", ".join(str(s) for s in detected_species)
                    ui_str = ", ".join(ui_categories)
                    inferred_str = ", ".join(sorted(inferred_categories))
                    print(f"   ⚠️  UI category {ui_categories} conflicts with species {detected_species} ({inferred_str})")
                    return {
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": (
                            f"'{species_str}' is in the {inferred_str} category, not {ui_str}. "
                            f"Set Category to 'All categories'"
                            + (f" or '{inferred_str}'" if inferred_str else "")
                            + " to see these results."
                        ),
                        "searched_datasets": [],
                    }

            # Validate filters were extracted correctly
            # Also check if query contains species words that aren't in available filters
            available_species = available_filters.get("species", []) + available_filters.get("collections", [])
            
            # Check if query contains common species words that aren't in available filters
            # This catches cases like "whale" where LLM might not extract it as a filter
            common_species_words = ["whale", "whales", "elephant", "elephants", "tiger", "tigers", "lion", "lions", 
                                   "bear", "bears", "eagle", "eagles", "shark", "sharks", "dolphin", "dolphins"]
            query_species_words = []
            for word in common_species_words:
                if word in query_lower:
                    # Check if this word matches any available species
                    word_matches_available = False
                    for avail in available_species:
                        avail_lower = avail.lower().strip()
                        word_normalized = word.replace("_", "").replace("-", "")
                        avail_normalized = avail_lower.replace("_", "").replace("-", "")
                        if (word == avail_lower or 
                            word_normalized == avail_normalized or
                            word in avail_lower or
                            avail_lower in word):
                            word_matches_available = True
                            break
                    if not word_matches_available:
                        query_species_words.append(word)
            
            # If query contains species words not in available filters, return error (no long species list)
            if query_species_words:
                return {
                    "query": query,
                    "error": f"Species '{', '.join(query_species_words)}' not found in our catalog. Try a different species or check the spelling.",
                    "results": [],
                    "total_count": 0,
                    "llm_understanding": query_understanding,
                }
            
            # If LLM found entities (e.g. "celery") but set species filter empty (no direct match), either
            # inject matching species (e.g. "celery" -> celery dataset, or celery leaftier/looper if no celery dataset) or return "not found".
            # Prefer exact dataset match when present so "celery" returns the celery (plant) dataset, not only pests.
            if not query_understanding.filters.get("species") and getattr(query_understanding, "entities", None):
                entities = [e.strip() for e in query_understanding.entities if e and str(e).strip()]
                if entities:
                    dataset_names_norm = {name.lower().strip().replace("_", "").replace("-", "").replace(" ", ""): name for name in self.dataset_registry.datasets}
                    available_species_norm = {s.lower().strip().replace("_", "").replace("-", "").replace(" ", ""): s for s in available_species}
                    injected = []
                    for ent in entities:
                        ent_lower = ent.lower().strip()
                        ent_norm = ent_lower.replace("_", "").replace("-", "").replace(" ", "")
                        exact_matches = []
                        partial_matches = []
                        for norm, orig in list(dataset_names_norm.items()) + list(available_species_norm.items()):
                            if ent_norm == norm:
                                exact_matches.append(orig)
                            elif (len(ent_norm) >= 3 and ent_norm in norm) or (len(norm) >= 3 and norm in ent_norm):
                                partial_matches.append(orig)
                        if exact_matches:
                            for o in exact_matches:
                                if o not in injected:
                                    injected.append(o)
                        else:
                            for o in partial_matches:
                                if o not in injected:
                                    injected.append(o)
                    if injected:
                        query_understanding.filters["species"] = sorted(list(set(injected)))
                        print(f"   ✅ Injected species from entities (no direct match): {query_understanding.filters['species']}")
                    else:
                        # Entity has no match in catalog — return clear error instead of wrong images
                        return {
                            "query": query,
                            "error": f"Species '{entities[0]}' not found in our catalog. Try a different species or check the spelling.",
                            "results": [],
                            "total_count": 0,
                            "llm_understanding": query_understanding,
                        }
            
            # If the LLM pinned the species only to a GENERIC pest-type word (e.g. "beetles" for
            # "golden beetles"), the earlier common-name resolver was skipped because a species was
            # already set. Try to refine it to the specific dataset via the common-name index so the
            # search isn't left with an unmatched generic word.
            current_species = query_understanding.filters.get("species") or []
            if current_species and not cn_injected and self.dataset_registry.datasets:
                def _is_generic_pest_word(s: str) -> bool:
                    s2 = s.lower().strip().replace("_", " ").replace("-", " ")
                    return s2 in _PEST_TYPE_WORDS_SET or s2.rstrip("s") in _PEST_TYPE_WORDS_SET
                if all(_is_generic_pest_word(s) for s in current_species):
                    specific_datasets = self._resolve_datasets_by_common_name(query)
                    if specific_datasets:
                        query_understanding.filters["species"] = sorted(specific_datasets)
                        cn_injected = True
                        print(f"   ✅ Refined generic pest type {current_species} → specific datasets: {sorted(specific_datasets)}")
            
            if query_understanding.filters.get("species"):
                print(f"   ✅ Species filter: {query_understanding.filters['species']}")
                # Check if species filter matches available species
                species_filter = query_understanding.filters["species"]
                unmatched_species = []
                for species in species_filter:
                    species_lower = species.lower().strip()
                    # Check if species matches any available species (with normalization)
                    matched = False
                    for avail in available_species:
                        avail_lower = avail.lower().strip()
                        # Normalize both for comparison
                        species_norm = species_lower.replace("_", "").replace("-", "")
                        avail_norm = avail_lower.replace("_", "").replace("-", "")
                        if (species_lower == avail_lower or 
                            species_norm == avail_norm or
                            species_lower in avail_lower or
                            avail_lower in species_lower):
                            matched = True
                            break
                    if not matched:
                        unmatched_species.append(species)
                
                # Treat as matched if a dataset name matches (e.g. dataset "raspberry" for species "raspberry")
                dataset_names_norm = {name.lower().strip().replace("_", "").replace("-", "") for name in self.dataset_registry.datasets}
                unmatched_species = [
                    s for s in unmatched_species
                    if (s.lower().strip().replace("_", "").replace("-", "")) not in dataset_names_norm
                ]
                # Also treat as matched if species exactly equals a dataset name or is prefix (e.g. "goat" matches "goat", "goat_2")
                if self.dataset_registry.datasets:
                    def _species_matches_any_dataset(sp: str) -> bool:
                        sp_lo = sp.lower().strip()
                        for dn in self.dataset_registry.datasets:
                            dn_lo = dn.lower()
                            if dn_lo == sp_lo or dn_lo.startswith(sp_lo + "_") or dn_lo.startswith(sp_lo + "-"):
                                return True
                        return False
                    unmatched_species = [s for s in unmatched_species if not _species_matches_any_dataset(s)]
                
                # If species filter doesn't match any available species, return error (no long species list)
                if unmatched_species and len(unmatched_species) == len(species_filter):
                    return {
                        "query": query,
                        "error": f"Species '{', '.join(unmatched_species)}' not found in our catalog. Try a different species or check the spelling.",
                        "results": [],
                        "total_count": 0,
                        "llm_understanding": query_understanding,
                    }
                # The species was resolved server-side (the LLM didn't supply it) and maps to a real
                # catalog dataset, so the query WAS understood. Raise the AI-confidence score from the
                # LLM's low fallback (e.g. 0.5) so it doesn't look uncertain next to a 100% filter match.
                if not llm_provided_species and query_understanding.confidence < 0.9:
                    query_understanding.confidence = 0.9
                    note = "Confidence raised: query resolved to a catalog dataset server-side."
                    query_understanding.reasoning = (
                        f"{query_understanding.reasoning} {note}".strip()
                        if query_understanding.reasoning else note
                    )
                    print("   ✅ AI confidence raised to 0.90 (species resolved server-side)")
            
            if query_understanding.filters.get("time"):
                print(f"   ✅ Time filter: {query_understanding.filters['time']}")
            if query_understanding.filters.get("plant_state"):
                print(f"   ✅ Plant state filter: {query_understanding.filters['plant_state']}")
            if query_understanding.filters.get("action"):
                print(f"   ✅ Action filter: {query_understanding.filters['action']}")
            
            # If no filters were extracted at all and query contains species words, return error
            # This prevents returning all images when query contains unknown species
            has_any_filters = any(query_understanding.filters.values())
            if not has_any_filters and query_species_words:
                return {
                    "query": query,
                    "error": f"Species '{', '.join(query_species_words)}' not found in our catalog. Try a different species or check the spelling.",
                    "results": [],
                    "total_count": 0,
                    "llm_understanding": query_understanding,
                }
            
            # OPTIMIZATION: Pre-filter datasets to reduce search space
            # 1) If species filter: only search datasets that contain that species (or matching dataset name)
            # 2) If category filter: only search datasets in that category
            # 3) Otherwise: search all datasets
            category_filter = query_understanding.filters.get("category", [])
            species_filter = query_understanding.filters.get("species", [])
            datasets_to_search = []

            def _species_match(species_filter_list: List[str], dataset_name: str, dataset_obj: Any) -> bool:
                """True if any requested species is in this dataset's species/collections or in dataset name."""
                if not species_filter_list:
                    return True
                requested = {s.lower().strip().replace(" ", "_") for s in species_filter_list if s}
                # Normalize singular/plural so "moths" matches dataset species "moth"
                def _normalize_plural(s: str) -> str:
                    s = s.lower().strip()
                    if len(s) > 1 and s.endswith("s") and not s.endswith("ss"):
                        return s[:-1]  # "moths" -> "moth"
                    return s
                requested_singular = {_normalize_plural(r) for r in requested}
                # Dataset name match (e.g. "raspberry" matches dataset "raspberry" or "raspberry_1")
                name_normalized = dataset_name.lower().replace(" ", "_")
                for r in requested:
                    if r == name_normalized or name_normalized.startswith(r + "_") or name_normalized.startswith(r + "-"):
                        return True
                    r_singular = _normalize_plural(r)
                    r_is_generic_pest = r in _PEST_TYPE_WORDS_SET or r_singular in _PEST_TYPE_WORDS_SET
                    if r_singular == name_normalized or name_normalized.startswith(r_singular + "_") or name_normalized.startswith(r_singular + "-"):
                        return True
                    # Species as segment or suffix: "cat" matches "domestic_cat" or "bobcat"
                    segments = name_normalized.replace("-", "_").split("_")
                    if r in segments or r_singular in segments:
                        return True
                    if name_normalized.endswith("_" + r) or name_normalized.endswith("_" + r_singular):
                        return True
                    # Do not let short generic pest words match inside unrelated names:
                    # "ant"/"ants" must not match "plant"/"plants".
                    if not r_is_generic_pest and (name_normalized.endswith(r) or name_normalized.endswith(r_singular)):
                        return True
                # Species/collections from this dataset
                opts = dataset_obj.available_filters
                if opts:
                    for lst in (opts.species or [], opts.collections or []):
                        for val in lst or []:
                            v = str(val).lower().strip().replace(" ", "_")
                            v_singular = _normalize_plural(v)
                            if v in requested or v in requested_singular:
                                return True
                            if v_singular in requested or v_singular in requested_singular:
                                return True
                            # Match on whole _-delimited segment sequences only, NOT arbitrary substrings, so
                            # e.g. "striped" (from "striped skunk") does not match inside "greenstriped grasshopper"
                            # and "ant" does not match inside "plant".
                            def _seg_contains(haystack: str, needle: str) -> bool:
                                if not needle:
                                    return False
                                return bool(re.search(rf"(^|_){re.escape(needle)}(_|$)", haystack))
                            for r in requested:
                                r_norm = _normalize_plural(r)
                                r_is_generic_pest = r in _PEST_TYPE_WORDS_SET or r_norm in _PEST_TYPE_WORDS_SET
                                # requested species appears as full segment(s) of the dataset value
                                if _seg_contains(v, r) or _seg_contains(v, r_norm):
                                    return True
                                if r_is_generic_pest:
                                    continue
                                # dataset value appears as full segment(s) of the requested species
                                # (e.g. v="bumble_bee" within r="american_bumble_bee"); require length > 3 to
                                # avoid matching tiny tokens.
                                if (len(v) > 3 and _seg_contains(r, v)) or (len(v_singular) > 3 and _seg_contains(r, v_singular)):
                                    return True
                                if v_singular in r and len(v_singular) > 6:
                                    return True
                return False

            candidate_datasets = list(self.dataset_registry.datasets.items())
            dataset_names_set = set(self.dataset_registry.datasets.keys())

            if species_filter:
                # If exactly one species and it equals a dataset name (e.g. "carrot", "raspberry"), search that dataset.
                # For "goat", search both "goat" and "goat_2" (goat_2 has tail_positions for "goat with tail up").
                dataset_names_lower = {name.lower(): name for name in dataset_names_set}
                if len(species_filter) == 1:
                    s = species_filter[0].lower().strip().replace(" ", "_")
                    if s in dataset_names_lower:
                        # Include all datasets that match this species name (e.g. goat + goat_2)
                        datasets_to_search = [name for name in dataset_names_set if name.lower() == s or name.lower().startswith(s + "_")]
                        datasets_to_search.sort(key=lambda x: (0 if x.lower() == s else 1, x.lower()))
                        print(f"🧠 Species '{s}' matches dataset name(s) → searching: {datasets_to_search}")
                    else:
                        s_singular = s[:-1] if len(s) > 1 and s.endswith("s") and not s.endswith("ss") else s
                        if s_singular in dataset_names_lower:
                            datasets_to_search = [name for name in dataset_names_set if name.lower() == s_singular or name.lower().startswith(s_singular + "_")]
                            datasets_to_search.sort(key=lambda x: (0 if x.lower() == s_singular else 1, x.lower()))
                            print(f"🧠 Species '{s}' (singular '{s_singular}') matches dataset name(s) → searching: {datasets_to_search}")
                # Even in the dataset-name shortcut, honor a UI category restriction (drop datasets of the wrong type).
                if datasets_to_search and category_filter:
                    before = len(datasets_to_search)
                    datasets_to_search = [
                        d for d in datasets_to_search if self._dataset_matches_category(d, category_filter)
                    ]
                    if len(datasets_to_search) != before:
                        print(f"🧠 Category filter {category_filter} restricted dataset-name match: {before} → {len(datasets_to_search)} datasets")
                if not datasets_to_search:
                    # Only datasets that have this species (or name match)
                    for dataset_name, dataset_obj in candidate_datasets:
                        if not _species_match(species_filter, dataset_name, dataset_obj):
                            continue
                        if category_filter:
                            dataset_category = dataset_obj.type.value.lower()
                            category_mapping = {
                                "pest": ["pests"],
                                "animal": ["wildlife", "domestic_animal", "livestock"],
                                "wildlife": ["wildlife"],
                                "domestic_animal": ["domestic_animal"],
                                "domestic": ["domestic_animal"],
                                "livestock": ["livestock"],
                                "plant": ["plants"]
                            }
                            if not any(
                                dataset_category in category_mapping.get(c.lower(), [c.lower()]) or c.lower() == dataset_category
                                for c in category_filter
                            ):
                                continue
                        datasets_to_search.append(dataset_name)
                # When we expanded to many species (e.g. "bumble bee" → 36 species) but _species_match missed pest datasets
                # (their filter may use scientific names), include datasets whose name contains the scientific/genus (e.g. Bombus).
                if query and len(species_filter) > 5:
                    q = query.lower().strip()
                    if "bumble" in q and "bee" in q:
                        bombus_datasets = [d for d in dataset_names_set if d and "bombus" in d.lower()]
                        for d in bombus_datasets:
                            if d not in datasets_to_search:
                                datasets_to_search.append(d)
                        if bombus_datasets:
                            print(f"🧠 Added {len(bombus_datasets)} Bombus datasets by query phrase 'bumble bee'")
                # When user asked for cat or dog, search only DOMESTIC_ANIMAL datasets (exclude pests like "Catocala", "Dog-Day Cicada")
                species_lower = {s.lower().strip().replace(" ", "_") for s in species_filter if s}
                # Normalize plurals ("cats" → "cat", "dogs" → "dog") so the domestic restriction still applies
                # and we don't return wildlife like bobcat (which only matches because it ends in "cat").
                species_lower_singular = {
                    (s[:-1] if len(s) > 1 and s.endswith("s") and not s.endswith("ss") else s)
                    for s in species_lower
                }
                if (species_lower | species_lower_singular) & {"cat", "dog"}:
                    before = len(datasets_to_search)
                    datasets_to_search = [
                        d for d in datasets_to_search
                        if self.dataset_registry.datasets.get(d) and self.dataset_registry.datasets[d].type == DatasetType.DOMESTIC_ANIMAL
                    ]
                    if before != len(datasets_to_search):
                        print(f"🧠 Species includes cat/dog → restricting to DOMESTIC_ANIMAL only ({len(datasets_to_search)} datasets, excluded {before - len(datasets_to_search)} pest/other)")
                # When user asked for mouse (rodent), exclude PEST datasets so we don't return "mouse moth" etc.
                query_lower = (query or "").lower()
                if not ((species_lower | species_lower_singular) & {"cat", "dog"}) and (
                    (species_lower & {"mouse", "mice"}) or re.search(r"\bmouse\b|\bmice\b", query_lower)
                ):
                    before = len(datasets_to_search)
                    datasets_to_search = [
                        d for d in datasets_to_search
                        if self.dataset_registry.datasets.get(d) and self.dataset_registry.datasets[d].type != DatasetType.PESTS
                    ]
                    if before != len(datasets_to_search):
                        print(f"🧠 Query/species mentions mouse/mice (rodent) → excluding PEST datasets ({len(datasets_to_search)} datasets, excluded {before - len(datasets_to_search)} pest e.g. mouse moth)")
                # Species was requested but no dataset matches — return immediately instead of searching all
                if species_filter and not datasets_to_search:
                    # Suggest datasets whose name contains the species term; for "cat"/"dog" prefer domestic_cat/dog
                    term = species_filter[0].lower().strip()
                    related = [name for name in dataset_names_set if term in name.lower()]
                    if related:
                        if term == "cat":
                            related = sorted(related, key=lambda x: (
                                0 if x == "domestic_cat" else 1,
                                0 if (x.endswith("_cat") or x == "bobcat") else 1,
                                len(x),
                                x
                            ))
                        elif term == "dog":
                            related = sorted(related, key=lambda x: (0 if x == "dog" else 1, len(x), x))
                        else:
                            related = sorted(related)
                        related = related[:5]
                        suggestion = f" Did you mean: {', '.join(related)}?"
                    else:
                        suggestion = ""
                    # For "cat"/"dog" with no match, explicitly suggest adding domestic_cat / dog dataset
                    if not related and term in ("cat", "dog"):
                        suggestion = f" Add the {'domestic_cat' if term == 'cat' else 'dog'} dataset to search {'cat' if term == 'cat' else 'dog'} images."
                    # Keep message short: avoid listing many scientific names; suggest browsing or related term
                    available_list = sorted(list(dataset_names_set))
                    # Prefer single-word (common-name) dataset names for the hint
                    single_word = [n for n in available_list if " " not in n and len(n) <= 30][:5]
                    friendly = single_word if single_word else [n for n in available_list if len(n) <= 25][:5]
                    list_part = f" Examples: {', '.join(friendly)}{'...' if len(available_list) > len(friendly) else ''}." if friendly else ""
                    error_msg = f"No dataset found for species '{', '.join(species_filter)}'.{suggestion}{list_part}"
                    return {
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": error_msg,
                        "searched_datasets": []
                    }
            elif category_filter:
                for dataset_name, dataset_obj in candidate_datasets:
                    dataset_category = dataset_obj.type.value.lower()
                    category_mapping = {
                        "pest": ["pests"],
                        "animal": ["wildlife", "domestic_animal", "livestock"],
                        "wildlife": ["wildlife"],
                        "domestic_animal": ["domestic_animal"],
                        "domestic": ["domestic_animal"],
                        "livestock": ["livestock"],
                        "plant": ["plants"]
                    }
                    should_include = any(
                        dataset_category in category_mapping.get(c.lower(), [c.lower()]) or c.lower() == dataset_category
                        for c in category_filter
                    )
                    if should_include:
                        datasets_to_search.append(dataset_name)
            else:
                # No species and no category resolved. If the query clearly names a subject (likely a species)
                # that isn't in our catalog, say so plainly instead of searching every dataset and returning a
                # vague "no images found" message.
                unmatched_subjects = self._unmatched_subject_words(query)
                if unmatched_subjects:
                    pretty = ", ".join(sorted(set(unmatched_subjects)))
                    print(f"🧠 Query subject(s) not in catalog: {unmatched_subjects} → returning not-available message")
                    return {
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": (
                            f"'{pretty}' is not in our catalog yet. "
                            f"Try a different species, or pick a Category to browse what's available."
                        ),
                        "searched_datasets": [],
                    }
                datasets_to_search = list(self.dataset_registry.datasets.keys())

            # When species filter matched both wildlife and pest datasets (e.g. "fox" → red fox + pests with "fox" in name), prefer wildlife and show disambiguation message
            disambiguation_message = None
            if species_filter and len(datasets_to_search) > 1:
                wildlife_only = []
                pest_only = []
                for d in datasets_to_search:
                    obj = self.dataset_registry.datasets.get(d)
                    if not obj:
                        continue
                    if obj.type == DatasetType.WILDLIFE:
                        wildlife_only.append(d)
                    elif obj.type == DatasetType.PESTS:
                        pest_only.append(d)
                if wildlife_only and pest_only:
                    # Restrict to animal/wildlife results and add follow-up message
                    datasets_to_search = wildlife_only
                    term = (species_filter[0] or "this").replace("_", " ").strip()
                    disambiguation_message = f"Showing {term} (animal) images. Would you like to see pests that include '{term}' in their name? Try searching \"{term} pest\" to include them."

            print(f"🧠 Pre-filtering: searching {len(datasets_to_search)} datasets (out of {len(self.dataset_registry.datasets)} total)")
            
            # Cap datasets to search and support "load next 100" via dataset_offset
            MAX_DATASETS_TO_SEARCH = 100
            total_datasets_matching = len(datasets_to_search)
            search_capped = False
            if total_datasets_matching > MAX_DATASETS_TO_SEARCH or dataset_offset > 0:
                search_capped = True
            # Slice to current batch: skip dataset_offset datasets, take up to MAX_DATASETS_TO_SEARCH
            datasets_to_search = datasets_to_search[dataset_offset:dataset_offset + MAX_DATASETS_TO_SEARCH]
            if dataset_offset > 0:
                print(f"🧠 Loading next batch: datasets {dataset_offset + 1}–{dataset_offset + len(datasets_to_search)} of {total_datasets_matching}")
            elif len(datasets_to_search) < total_datasets_matching:
                print(f"🧠 Capped search to first {MAX_DATASETS_TO_SEARCH} datasets (of {total_datasets_matching}) for faster response")
            
            # When only one dataset matches (e.g. "raspberries" → raspberry), use single-dataset path so we
            # return full total_count and pagination (Next/Previous) works. Otherwise we'd use multi-dataset
            # path with early termination and total_count would cap at one page (e.g. 50).
            if not dataset and len(datasets_to_search) == 1:
                dataset = datasets_to_search[0]
                print(f"🧠 Single matching dataset → using single-dataset path for full count and pagination: {dataset}")
            
            # datasets_to_search has ALREADY been restricted to datasets that match the species filter
            # (by dataset name, group word, or common name). Re-applying the species filter per image is
            # redundant and, for the common case where a dataset is keyed by scientific name while the
            # filter is a common/group name (e.g. "grasshopper", "ant", "painted lady"), it wrongly drops
            # every image because metadata.species holds the scientific name. So strip species from the
            # per-image filters here; time/action/plant_state/scene still apply.
            if species_filter:
                print(f"🧠 Species {species_filter} already enforced at dataset level → not re-filtering per image (keeping other filters)")

            def _per_image_filters(base_filters: Dict[str, Any]) -> Dict[str, Any]:
                if species_filter and base_filters.get("species"):
                    return {k: v for k, v in base_filters.items() if k != "species"}
                return base_filters

            # Perform search using the structured understanding and adapters
            # IMPORTANT: Only use filters, not the query string, to avoid incorrect substring matches
            if dataset:
                if dataset in datasets_to_search and dataset in self.dataset_registry.datasets:
                    # Dataset-only search (empty query): pass no filters so we get all images from the dataset
                    search_filters = {} if "Dataset-only search" in getattr(query_understanding, "reasoning", "") else _per_image_filters(query_understanding.filters)
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset, "", search_filters
                    )
                    cache_size = len(self.dataset_registry.images_cache.get(dataset, []))
                    print(f"🧠 Single-dataset search: {dataset} returned {len(filtered_results)} items (cache size: {cache_size})")
                    # Apply strict plant_state filter when user asked for specific state (e.g. ripe only)
                    plant_state_filter = query_understanding.filters.get("plant_state") or []
                    if plant_state_filter:
                        before = len(filtered_results)
                        filtered_results = [r for r in filtered_results if self._passes_plant_state_strict(r, plant_state_filter)]
                        if before != len(filtered_results):
                            print(f"🧠 Plant-state strict filter (single dataset): kept {len(filtered_results)} of {before} results (requested: {plant_state_filter})")
                    action_filter_list = query_understanding.filters.get("action") or []
                    if action_filter_list:
                        before = len(filtered_results)
                        filtered_results = [r for r in filtered_results if self._passes_action_strict(r, action_filter_list)]
                        if before != len(filtered_results):
                            print(f"🧠 Action strict filter (single dataset): kept {len(filtered_results)} of {before} results (requested: {action_filter_list})")
                    tail_filter_list = query_understanding.filters.get("tail_positions") or []
                    if tail_filter_list:
                        before = len(filtered_results)
                        filtered_results = [r for r in filtered_results if self._passes_tail_strict(r, tail_filter_list)]
                        if before != len(filtered_results):
                            print(f"🧠 Tail-position strict filter (single dataset): kept {len(filtered_results)} of {before} results (requested: {tail_filter_list})")
                    # When query specifies a cultivar/variety (e.g. "Cabernet Sauvignon grapes"), require description to contain it so we exclude other varieties (e.g. Syrah).
                    # For broad species-only queries (e.g. "raspberries") _get_required_description_phrase returns None so we don't shrink the set.
                    # Skip when we already filtered by tail_positions/scene — those used metadata; requiring a
                    # literal phrase (e.g. "tail undetermined") would wrongly exclude valid items.
                    req_phrase = None
                    if not query_understanding.filters.get("tail_positions") and not query_understanding.filters.get("scene"):
                        req_phrase = self._get_required_description_phrase(
                            query,
                            query_understanding.filters.get("species") or [],
                            getattr(query_understanding, "description_query", None),
                        )
                    # When species was resolved from a common name (e.g. "golden beetle" → datasets), the query
                    # words ARE the common name and were already used to pick datasets. Don't also require them
                    # literally in the description (image descriptions rarely repeat the common name verbatim).
                    if req_phrase and cn_injected:
                        print(f"🧠 Skipping description phrase '{req_phrase}' (species came from common-name match)")
                        req_phrase = None
                    # Generic pest-group searches (e.g. "aphids on soybean leaves" → species ["aphid"]): the
                    # leftover host/descriptor words ("soybean") shouldn't be forced into the description, since
                    # pest images rarely repeat the host plant verbatim — that would drop all matches.
                    _sp_for_phrase = [s.lower().strip() for s in (query_understanding.filters.get("species") or [])]
                    if req_phrase and _sp_for_phrase and all(
                        (s in _PEST_TYPE_WORDS_SET or (s + "s") in _PEST_TYPE_WORDS_SET or s.rstrip("s") in _PEST_TYPE_WORDS_SET)
                        for s in _sp_for_phrase
                    ):
                        print(f"🧠 Skipping description phrase '{req_phrase}' (generic pest-group search)")
                        req_phrase = None
                    # Explicit: for raspberry dataset, never require "raspberries" in description — return all images.
                    if req_phrase and str(req_phrase).lower().strip() == "raspberries" and str(dataset).lower() == "raspberry":
                        req_phrase = None
                        print(f"🧠 Skipping description phrase 'raspberries' for raspberry dataset (return all images)")
                    # When we already filtered by plant_state (e.g. unripe), don't require that phrase in description (e.g. "unripe raspberries") — descriptions say "unripe berries" etc.
                    plant_state_filter_for_phrase = query_understanding.filters.get("plant_state") or []
                    if req_phrase and plant_state_filter_for_phrase and dataset:
                        pl = req_phrase.lower().strip()
                        for ps in plant_state_filter_for_phrase:
                            ps = (ps or "").lower().strip()
                            if ps and (pl == ps or pl.startswith(ps + " ") or ps in pl.split()):
                                print(f"🧠 Skipping description phrase '{req_phrase}' (already filtered by plant_state '{ps}')")
                                req_phrase = None
                                break
                    # Unconditionally skip description filter when phrase is just this dataset's name (or plural) — we're already on that dataset.
                    if req_phrase and dataset:
                        d = str(dataset).lower().strip().replace("_", " ")
                        phrase_lower = req_phrase.lower().strip()
                        if phrase_lower == d or phrase_lower == d + "s" or (len(d) > 1 and d.endswith("s") and phrase_lower == d[:-1]):
                            req_phrase = None
                            print(f"🧠 Skipping description phrase '{phrase_lower}' (same as dataset '{dataset}' / plural)")
                    if req_phrase:
                        before = len(filtered_results)
                        filtered_results = [r for r in filtered_results if self._passes_description_required(r, req_phrase)]
                        if before != len(filtered_results):
                            print(f"🧠 Description required phrase '{req_phrase}': kept {len(filtered_results)} of {before} results (excluded items without that in description)")
                    
                    if not filtered_results:
                        # No results — return a clear, user-friendly message (do not list all species/catalogs)
                        species_filter = query_understanding.filters.get("species", [])
                        action_filter = query_understanding.filters.get("action", [])
                        plant_state_filter = query_understanding.filters.get("plant_state", [])

                        # If plant_state filter eliminated all results but other plant_states exist, surface them
                        alt_plant_states = set()
                        if plant_state_filter and species_filter:
                            try:
                                alt_filters = dict(query_understanding.filters)
                                alt_filters.pop("plant_state", None)
                                alt_items = self.dataset_registry.search_dataset(dataset, "", alt_filters)
                                for it in alt_items:
                                    meta = it.get("metadata") or {}
                                    raw = meta.get("plant_state") or meta.get("plant_states")
                                    if isinstance(raw, list):
                                        for p in raw:
                                            if str(p).strip():
                                                alt_plant_states.add(str(p).strip())
                                    elif raw and str(raw).strip():
                                        alt_plant_states.add(str(raw).strip())
                            except Exception:
                                alt_plant_states = set()
                        # Normalize action for display: split comma-separated so "foraging, eating" -> ["foraging", "eating"]
                        if action_filter:
                            _split = []
                            for a in (action_filter if isinstance(action_filter, list) else [action_filter]):
                                for part in str(a).split(","):
                                    p = part.strip()
                                    if p:
                                        _split.append(p)
                            if _split:
                                action_filter = _split
                        if species_filter or action_filter or plant_state_filter:
                            parts = []
                            if species_filter:
                                parts.append(f"species '{', '.join(species_filter)}'")
                            if action_filter:
                                parts.append(f"action '{', '.join(action_filter)}'")
                            if plant_state_filter:
                                parts.append(f"plant_state '{', '.join(plant_state_filter)}'")
                            err = f"No images found matching {', '.join(parts)} in our catalog."
                            if plant_state_filter and alt_plant_states:
                                alt_list = ", ".join(sorted(alt_plant_states))
                                err += f" This dataset does have images with other ripeness states (e.g. {alt_list}). Try searching without a ripeness filter, or using one of those states."
                                err += " If you just deployed code changes, restart the server (or call POST /api/reload-datasets to reload data from disk)."
                            elif plant_state_filter:
                                err += " Try searching without a ripeness filter, or use a broader term."
                            elif action_filter:
                                err += " This dataset may have images with other actions (e.g. walking). Try a different action or browse all images without an action filter."
                            else:
                                err += " Try a different species or action."
                        else:
                            err = f"Dataset '{dataset}' has no matching results for your query."
                        return {
                            "dataset": dataset,
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": err,
                        }
                    
                    # Add confidence scores, image URLs, and top-level display fields to each result
                    # When result set is large, only enrich the page we return to keep response fast.
                    total_count = len(filtered_results)
                    if total_count > 500:
                        # Large set: sort by id for stable order, enrich only the requested page
                        filtered_results.sort(key=lambda x: (x.get('id') or ''))
                        page_results = filtered_results[offset:offset + limit]
                        for result in page_results:
                            result['dataset'] = dataset  # set before image URL so canonical dataset_id naming is used
                            result['llm_confidence'] = self._calculate_result_confidence(result, query_understanding, query)
                            result['llm_reasoning'] = query_understanding.reasoning
                            result['llm_intent'] = query_understanding.intent
                            result['image_url'] = self._construct_image_url(result)
                            meta = result.get('metadata') or {}
                            result['background'] = meta.get('background') or meta.get('scene')
                            result['scientific_name'] = meta.get('scientific_name')
                            cn = meta.get('common_names')
                            result['common_names'] = cn if isinstance(cn, list) else ([cn] if cn else None)
                    else:
                        # Smaller set: enrich all, sort by confidence, then slice
                        for result in filtered_results:
                            result['dataset'] = dataset  # set before image URL so canonical dataset_id naming is used
                            result['llm_confidence'] = self._calculate_result_confidence(result, query_understanding, query)
                            result['llm_reasoning'] = query_understanding.reasoning
                            result['llm_intent'] = query_understanding.intent
                            result['image_url'] = self._construct_image_url(result)
                            meta = result.get('metadata') or {}
                            result['background'] = meta.get('background') or meta.get('scene')
                            result['scientific_name'] = meta.get('scientific_name')
                            cn = meta.get('common_names')
                            result['common_names'] = cn if isinstance(cn, list) else ([cn] if cn else None)
                        filtered_results.sort(key=lambda x: (x.get('llm_confidence', 0), x.get('id', '')), reverse=True)
                        page_results = filtered_results[offset:offset + limit]
                    
                    return {
                        "dataset": dataset,
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": page_results,
                        "total_count": total_count
                    }
                else:
                    return {
                        "dataset": dataset,
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": f"Dataset {dataset} not found"
                    }
            else:
                # Search across filtered datasets using adapters (optimized with category pre-filtering)
                all_results = []
                # When few datasets match (e.g. "cat" → 1–2 datasets), collect ALL results so total_count and pagination are correct.
                # When many datasets match, cap collection so first response stays fast; cap high enough for many pages.
                MAX_TOTAL_RESULTS_CAPPED = 5000
                few_datasets = len(datasets_to_search) <= 25
                if few_datasets:
                    enough_results = MAX_TOTAL_RESULTS_CAPPED + 1  # collect all from this batch (no early termination)
                elif dataset_offset == 0:
                    enough_results = max(offset + limit, 500)  # at least 10 pages of 50
                else:
                    enough_results = min(offset + limit + 500, MAX_TOTAL_RESULTS_CAPPED)
                
                for dataset_name in datasets_to_search:
                    if len(all_results) >= enough_results:
                        print(f"🧠 Early termination: have {len(all_results)} results (need up to {enough_results}), stopping dataset search")
                        break
                    print(f"🧠 Searching dataset: {dataset_name}")
                    # Use adapter-based search with ONLY filters (no query string)
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset_name, "", _per_image_filters(query_understanding.filters)
                    )
                    print(f"   Found {len(filtered_results)} matching results in {dataset_name}")
                    if filtered_results:
                        # Log first result for debugging
                        first_result = filtered_results[0]
                        print(f"   Sample result: collection={first_result.get('collection')}, species={first_result.get('metadata', {}).get('species')}")
                        remaining = enough_results - len(all_results)
                        to_add = filtered_results if remaining >= len(filtered_results) else filtered_results[:remaining]
                        for result in to_add:
                            result['dataset'] = dataset_name
                            result['llm_confidence'] = self._calculate_result_confidence(result, query_understanding, query)
                            result['llm_reasoning'] = query_understanding.reasoning
                            result['llm_intent'] = query_understanding.intent
                            result['image_url'] = self._construct_image_url(result)
                            # Top-level fields for UI (background, scientific_name, common_names)
                            meta = result.get('metadata') or {}
                            result['background'] = meta.get('background') or meta.get('scene')
                            result['scientific_name'] = meta.get('scientific_name')
                            cn = meta.get('common_names')
                            result['common_names'] = cn if isinstance(cn, list) else ([cn] if cn else None)
                        all_results.extend(to_add)
                        if len(all_results) >= enough_results:
                            break
                
                # When user asked for a specific plant_state (e.g. ripe), keep only items that pass strict check
                # (uses both metadata and description; excludes mixed/unripe so we return only ripe images)
                plant_state_filter = query_understanding.filters.get("plant_state") or []
                if plant_state_filter:
                    before = len(all_results)
                    all_results = [r for r in all_results if self._passes_plant_state_strict(r, plant_state_filter)]
                    if before != len(all_results):
                        print(f"🧠 Plant-state strict filter: kept {len(all_results)} of {before} results (requested: {plant_state_filter})")
                # When user asked for a specific action (e.g. sleeping), keep only items that pass strict action check (action + description)
                action_filter_list = query_understanding.filters.get("action") or []
                if action_filter_list:
                    before = len(all_results)
                    all_results = [r for r in all_results if self._passes_action_strict(r, action_filter_list)]
                    if before != len(all_results):
                        print(f"🧠 Action strict filter: kept {len(all_results)} of {before} results (requested: {action_filter_list})")
                tail_filter_list = query_understanding.filters.get("tail_positions") or []
                if tail_filter_list:
                    before = len(all_results)
                    all_results = [r for r in all_results if self._passes_tail_strict(r, tail_filter_list)]
                    if before != len(all_results):
                        print(f"🧠 Tail-position strict filter: kept {len(all_results)} of {before} results (requested: {tail_filter_list})")
                # When query specifies a cultivar/variety (e.g. "Cabernet Sauvignon grapes"), require description to contain it
                # Skip when we already filtered by tail_positions or scene — adapter used metadata; requiring phrase would exclude valid items (e.g. "goats field" vs "goats in a field").
                req_phrase = None
                if not query_understanding.filters.get("tail_positions") and not query_understanding.filters.get("scene"):
                    req_phrase = self._get_required_description_phrase(
                        query,
                        query_understanding.filters.get("species") or [],
                        getattr(query_understanding, "description_query", None),
                    )
                # Species resolved from a common name: don't also require those words in the description.
                if req_phrase and cn_injected:
                    print(f"🧠 Skipping description phrase '{req_phrase}' (species came from common-name match)")
                    req_phrase = None
                # Generic pest-group searches (e.g. "aphids on soybean leaves" → species ["aphid"]): don't force
                # leftover host/descriptor words ("soybean") into the description or every match gets dropped.
                _sp_for_phrase = [s.lower().strip() for s in (query_understanding.filters.get("species") or [])]
                if req_phrase and _sp_for_phrase and all(
                    (s in _PEST_TYPE_WORDS_SET or (s + "s") in _PEST_TYPE_WORDS_SET or s.rstrip("s") in _PEST_TYPE_WORDS_SET)
                    for s in _sp_for_phrase
                ):
                    print(f"🧠 Skipping description phrase '{req_phrase}' (generic pest-group search)")
                    req_phrase = None
                if req_phrase:
                    before = len(all_results)
                    all_results = [r for r in all_results if self._passes_description_required(r, req_phrase)]
                    if before != len(all_results):
                        print(f"🧠 Description required phrase '{req_phrase}': kept {len(all_results)} of {before} results (excluded items without that in description)")
                
                print(f"🧠 Total LLM-filtered results found: {len(all_results)}")
                
                # If no results, provide helpful error message
                if not all_results:
                    error_parts = []
                    if query_understanding.filters.get("species"):
                        species_filter = query_understanding.filters["species"]
                        error_parts.append(f"species '{', '.join(species_filter)}'")
                    if query_understanding.filters.get("time"):
                        time_filter = query_understanding.filters["time"]
                        error_parts.append(f"time '{', '.join(time_filter)}'")
                    if query_understanding.filters.get("action"):
                        action_filter = query_understanding.filters["action"]
                        error_parts.append(f"action '{', '.join(action_filter)}'")
                    if query_understanding.filters.get("scene"):
                        scene_filter = query_understanding.filters["scene"]
                        error_parts.append(f"scene '{', '.join(scene_filter)}'")
                    
                    if error_parts:
                        error_msg = f"No images found matching {', '.join(error_parts)} in our catalog."
                        # Short, user-friendly hint (do not list thousands of species/pest names)
                        if query_understanding.filters.get("plant_state"):
                            error_msg += " Try searching without a ripeness filter, or use a broader term."
                            error_msg += " If you just deployed code changes, restart the server (or POST /api/reload-datasets to reload data)."
                        elif query_understanding.filters.get("action"):
                            error_msg += " This dataset may have images with other actions (e.g. walking). Try a different action or browse all images without an action filter."
                        elif query_understanding.filters.get("scene"):
                            error_msg += " Try a different scene (e.g. indoor) or remove the scene filter."
                        elif query_understanding.filters.get("species"):
                            error_msg += " Try a different species or action."
                        
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": error_msg,
                            "searched_datasets": datasets_to_search
                        }
                    else:
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": "No images found matching your query",
                            "searched_datasets": datasets_to_search
                        }
                
                # Sort by confidence (highest first) and then by dataset for consistency
                all_results.sort(key=lambda x: (x.get('llm_confidence', 0), x.get('dataset', ''), x.get('id', '')), reverse=True)
                
                next_dataset_offset = dataset_offset + len(datasets_to_search)
                has_more_datasets = next_dataset_offset < total_datasets_matching
                out = {
                    "query": query,
                    "llm_understanding": query_understanding,
                    "results": all_results[offset:offset + limit],
                    "total_count": len(all_results),
                    "searched_datasets": datasets_to_search,
                    "dataset_offset": dataset_offset,
                    "next_dataset_offset": next_dataset_offset,
                    "total_datasets_matching": total_datasets_matching,
                    "has_more_datasets": has_more_datasets,
                }
                if search_capped:
                    out["search_capped"] = True
                if disambiguation_message is not None:
                    out["disambiguation_message"] = disambiguation_message
                return out
                
        except ValueError as e:
            # LLM service not available or failed
            print(f"❌ LLM search error: {e}")
            return {
                "query": query,
                "error": f"LLM service error: {str(e)}. Please ensure OPENAI_API_KEY is set and valid.",
                "results": [],
                "total_count": 0,
                "llm_understanding": None
            }
        except Exception as e:
            print(f"❌ LLM search error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "error": f"LLM search failed: {str(e)}",
                "results": [],
                "total_count": 0,
                "llm_understanding": None
            }
    
    def _apply_llm_filters(self, images: List[Dict[str, Any]], llm_filters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Apply LLM-understood filters to images"""
        filtered_images = []
        
        for image in images:
            if self._image_matches_llm_filters(image, llm_filters):
                filtered_images.append(image)
        
        return filtered_images
    
    def _image_matches_llm_filters(self, image: Dict[str, Any], llm_filters: Dict[str, List[str]]) -> bool:
        """Check if image matches LLM-understood filters"""
        # Check each filter category
        for filter_type, filter_values in llm_filters.items():
            if not filter_values:  # Skip empty filters
                continue
                
            if filter_type == "species":
                if not self._image_matches_species(image, filter_values):
                    return False
            elif filter_type == "time":
                if not self._image_matches_time(image, filter_values):
                    return False
            elif filter_type == "season":
                if not self._image_matches_season(image, filter_values):
                    return False
            elif filter_type == "action":
                if not self._image_matches_action(image, filter_values):
                    return False
            elif filter_type == "scene":
                if not self._image_matches_scene(image, filter_values):
                    return False
            elif filter_type == "weather":
                if not self._image_matches_weather(image, filter_values):
                    return False
            elif filter_type == "category":
                if not self._image_matches_category(image, filter_values):
                    return False
        
        return True
    
    def _image_matches_species(self, image: Dict[str, Any], species_filters: List[str]) -> bool:
        """Check if image matches species filters"""
        if not species_filters:
            return True
        
        # Look for species in metadata, not top-level
        image_species = image.get("metadata", {}).get("species", "").lower()
        if not image_species:
            return True
        
        return any(species.lower() in image_species for species in species_filters)
    
    def _image_matches_action(self, image: Dict[str, Any], action_filters: List[str]) -> bool:
        """Check if image matches action filters"""
        if not action_filters:
            return True
        
        image_action = image.get("metadata", {}).get("action", "").lower()
        if not image_action:
            return True
        
        return any(action.lower() in image_action for action in action_filters)
    
    def _image_matches_scene(self, image: Dict[str, Any], scene_filters: List[str]) -> bool:
        """
        Check if image matches scene filters.
        
        IMPORTANT: Scene metadata is read ONLY from MCP metadata (metadata.scene).
        The system does NOT infer or set scene values - it only uses what's explicitly
        provided in the MCP data files. If scene values are incorrect, they need to be
        fixed in the source MCP data files.
        
        STRICT MODE: If scene filter is specified, image MUST have matching scene.
        Also validates against description to catch contradictions.
        """
        if not scene_filters:
            return True
        
        # Read scene directly from MCP metadata - no inference or defaults
        image_scene = image.get("metadata", {}).get("scene", "").lower().strip()
        image_description = image.get("metadata", {}).get("description", "").lower()
        
        # If no scene metadata, check description for scene keywords
        if not image_scene:
            scene_keyword_map = {
                "field": ["field", "meadow", "pasture", "open field", "grassland"],
                "forest": ["forest", "woodland", "woods", "trees"],
                "garden": ["garden", "garden area"],
                "farm": ["farm", "farmland", "farm area"],
                "indoor": ["indoor", "inside", "interior", "barn", "shed", "building"],
                "outdoor": ["outdoor", "outside", "exterior", "open air"]
            }
            
            for scene_filter in scene_filters:
                scene_lower = scene_filter.lower().strip()
                if scene_lower in scene_keyword_map:
                    keywords = scene_keyword_map[scene_lower]
                    if any(keyword in image_description for keyword in keywords):
                        return True
            # If no scene metadata and no description match, reject if scene filter is specified
            return False
        
        # Check if scene matches any filter
        for scene_filter in scene_filters:
            scene_lower = scene_filter.lower().strip()
            if scene_lower == image_scene or scene_lower in image_scene or image_scene in scene_lower:
                # VALIDATION: Check description for contradictions
                indoor_keywords = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                outdoor_keywords = ["outdoor", "outside", "field", "meadow", "pasture", "open", "exterior"]
                
                # If scene says "field" but description says "indoor", reject
                if scene_lower == "field" and any(keyword in image_description for keyword in indoor_keywords):
                    print(f"      ⚠️  Scene mismatch: scene='{image_scene}' but description indicates indoor - REJECTING")
                    continue  # Try next scene filter
                
                # If scene says "indoor" but description says outdoor keywords, reject
                if scene_lower in ["indoor", "inside"] and any(keyword in image_description for keyword in outdoor_keywords):
                    print(f"      ⚠️  Scene mismatch: scene='{image_scene}' but description indicates outdoor - REJECTING")
                    continue  # Try next scene filter
                
                return True
        
        # No match found
        return False
    
    def _image_matches_weather(self, image: Dict[str, Any], weather_filters: List[str]) -> bool:
        """Check if image matches weather filters"""
        if not weather_filters:
            return True
        
        image_weather = image.get("metadata", {}).get("weather", "").lower()
        if not image_weather:
            return True
        
        return any(weather.lower() in image_weather for weather in weather_filters)
    
    def _image_matches_category(self, image: Dict[str, Any], category_filters: List[str]) -> bool:
        """Check if image matches category filters"""
        if not category_filters:
            return True
        
        image_category = image.get("category", "").lower()
        if not image_category:
            return True
        
        return any(category.lower() in image_category for category in category_filters)
    
    def _image_matches_query(self, image: Dict[str, Any], query: str) -> bool:
        """Check if image matches search query"""
        # Check collection name
        if query in image.get("collection", "").lower():
            return True
        
        # Check metadata description
        metadata = image.get("metadata", {})
        if query in _metadata_str(metadata.get("description", "")).lower():
            return True
        
        # Check metadata action
        if query in _metadata_str(metadata.get("action", "")).lower():
            return True
        
        # Check metadata scene
        if query in _metadata_str(metadata.get("scene", "")).lower():
            return True
        
        return False
    
    def _image_matches_time(self, image: Dict[str, Any], time_filters: List[str]) -> bool:
        """Check if image matches time filters"""
        metadata = image.get("metadata", {})
        time_info = _metadata_str(metadata.get("time", "")).lower()
        
        for time_filter in time_filters:
            if time_filter == "night" and ("night" in time_info or "dark" in time_info):
                return True
            elif time_filter == "day" and ("day" in time_info or "morning" in time_info or "afternoon" in time_info):
                return True
            elif time_filter == "dawn" and ("dawn" in time_info or "sunrise" in time_info):
                return True
            elif time_filter == "dusk" and ("dusk" in time_info or "sunset" in time_info):
                return True
            elif time_filter == "evening" and (
                "evening" in time_info or "twilight" in time_info or "late afternoon" in time_info
            ):
                return True
        
        return False
    
    def _image_matches_season(self, image: Dict[str, Any], season_filters: List[str]) -> bool:
        """Check if image matches season filters"""
        if not season_filters:
            return True
        
        image_season = image.get("metadata", {}).get("season", "").lower()
        if not image_season:
            return True
        
        return any(season.lower() in image_season for season in season_filters)
    
    def _construct_image_url(self, result: Dict[str, Any]) -> str:
        """Construct image URL from result data. For datasets like goat/goat_2, images on disk use prefix (goat_..., goat_2_...)."""
        try:
            from pathlib import Path
            metadata = result.get("metadata", {})
            result_id = result.get("id", "")
            dataset = result.get("dataset", "")

            # Extension hint from original_filename: disk files are often renamed to the canonical
            # dataset id but keep their real extension (e.g. id "carrot_001" → "carrot_001.png",
            # original_filename "007_image.png"). Used in fallbacks so we don't hardcode .jpg.
            _known_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            _orig_ext = ""
            _of = metadata.get("original_filename") or ""
            if _of:
                _cand = Path(str(_of)).suffix.lower()
                if _cand in _known_exts:
                    _orig_ext = _cand
            _id_ext = _orig_ext or ".jpg"

            # Priority 1: Check if image_url is explicitly set
            if "image_url" in result:
                return result["image_url"]
            
            # Priority 2: When result has a dataset, try dataset-prefixed filename (e.g. goat_001.jpg or goat_Luzignan-....jpg)
            if result_id and dataset:
                from pathlib import Path
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    # If id already has dataset prefix (e.g. goat_001), use it as the filename stem so we request goat_001.jpg
                    if result_id.startswith(dataset + "_"):
                        stem = result_id
                    else:
                        stem = f"{dataset}_{result_id}"
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG", ".GIF"]:
                        potential_filename = f"{stem}{ext}"
                        potential_path = images_dir / potential_filename
                        if potential_path.exists() and potential_path.is_file():
                            return f"/images/{potential_filename}"
                    # Try in dataset subdir without prefix (e.g. goat/Luzignan-20160310_140237.jpg)
                    subdir = images_dir / dataset
                    if subdir.is_dir():
                        for ext in [".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG", ".GIF"]:
                            potential_path = subdir / f"{result_id}{ext}"
                            if potential_path.exists() and potential_path.is_file():
                                return f"/images/{dataset}/{result_id}{ext}"
            
            # Priority 3: Try id with common extensions
            if result_id:
                from pathlib import Path
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG", ".GIF"]:
                        potential_filename = f"{result_id}{ext}"
                        potential_path = images_dir / potential_filename
                        if potential_path.exists() and potential_path.is_file():
                            return f"/images/{potential_filename}"
            
            # Priority 4: Try original_filename from metadata (skip when we have canonical dataset id — serve goat_001.jpg not original name)
            has_canonical_stem = result_id and dataset and result_id.startswith(dataset + "_")
            if not has_canonical_stem and "original_filename" in metadata:
                original_filename = metadata["original_filename"]
                filename = Path(original_filename).name
                from pathlib import Path
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    potential_path = images_dir / filename
                    if potential_path.exists() and potential_path.is_file():
                        return f"/images/{filename}"
                    else:
                        # Try case-insensitive match
                        filename_lower = filename.lower()
                        for item in images_dir.iterdir():
                            if item.is_file() and item.name.lower() == filename_lower:
                                return f"/images/{item.name}"
                # If not found locally, return the filename anyway (might be on server)
                return f"/images/{filename}"
            
            # Priority 5: Fallback — use correct filename (id already prefixed e.g. goat_001, or dataset_id)
            # with the original file's extension (so e.g. carrot_001.png is requested, not .jpg).
            if result_id and dataset:
                stem = result_id if result_id.startswith(dataset + "_") else f"{dataset}_{result_id}"
                return f"/images/{stem}{_id_ext}"
            if result_id:
                return f"/images/{result_id}{_id_ext}"
            
            # If no image info, return placeholder
            return "/images/placeholder.jpg"
            
        except Exception as e:
            print(f"❌ Error constructing image URL for result {result.get('id', 'unknown')}: {e}")
            return "/images/placeholder.jpg"
    
    def _filter_strings(self, value: Any) -> List[str]:
        """Normalize a metadata value to a list of non-empty strings (handles str or list)."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if v is not None and str(v).strip()]
        s = str(value).strip()
        return [s] if s else []

    def _build_common_name_index(self) -> Dict[str, List[str]]:
        """Build (and cache) a map from common-name / scientific-name phrase → dataset name(s).

        Many datasets are named by scientific name (e.g. "Vanessa_cardui") but users search by common
        name (e.g. "painted lady"). Common names live in each item's metadata (common_name, common_names,
        scientific_name). We index the whole phrase (not sub-words) so a multi-word common name only matches
        when it appears in full, avoiding noise from generic words like "moth" or "white".
        """
        if self._common_name_index is not None:
            return self._common_name_index

        index: Dict[str, set] = {}

        def _add(phrase: str, dataset_name: str):
            norm = re.sub(r"[_\-]+", " ", str(phrase or "")).lower().strip()
            norm = re.sub(r"\s+", " ", norm)
            if not norm:
                return
            n_words = norm.split()
            # Skip description-like text and absurdly long phrases.
            if len(norm) > 40 or len(n_words) > 4:
                return
            # Skip single generic words (pest-type words, non-subject words, very short words);
            # those are handled by the existing pest-type and available-filters paths.
            if len(n_words) == 1:
                w = n_words[0]
                if len(w) < 4 or w in _NON_SUBJECT_WORDS or w in _PEST_TYPE_WORDS_SET:
                    return
            index.setdefault(norm, set()).add(dataset_name)

        for dn, items in self.dataset_registry.images_cache.items():
            if not items:
                continue
            # Identity fields are constant within a (pest) dataset, so sampling a few items is enough.
            for item in items[:3]:
                meta = item.get("metadata") or {}
                for val in (meta.get("common_name"), meta.get("scientific_name")):
                    if val:
                        _add(val, dn)
                cn = meta.get("common_names")
                if isinstance(cn, list):
                    for c in cn:
                        _add(c, dn)
                elif cn:
                    _add(cn, dn)

        self._common_name_index = {k: sorted(v) for k, v in index.items()}
        print(f"🧠 Built common-name index: {len(self._common_name_index)} phrases")
        return self._common_name_index

    def _resolve_datasets_by_common_name(self, query: str) -> List[str]:
        """Resolve a query to dataset name(s) via the common-name index, preferring the longest phrase
        match (e.g. "painted lady at night" → ["Vanessa_cardui"]). Returns [] when nothing matches."""
        if not query or not query.strip():
            return []
        index = self._build_common_name_index()
        if not index:
            return []

        def _normalized_query_variants(q: str) -> List[str]:
            norm = re.sub(r"[_\-]+", " ", q.lower()).strip()
            norm = re.sub(r"\s+", " ", norm)
            variants = [norm]
            for alias, replacements in _COMMON_NAME_QUERY_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", norm):
                    for replacement in replacements:
                        variants.append(re.sub(rf"\b{re.escape(alias)}\b", replacement, norm))
            # Also add a singularized form of each variant so plurals resolve against the (singular)
            # common-name index, e.g. "whiteflies" → "whitefly", "mealybugs" → "mealybug". Strips a
            # trailing -s and maps -ies → -y (so "whiteflies" doesn't become "whiteflie").
            def _sing_word(w: str) -> str:
                if len(w) > 4 and w.endswith("ies"):
                    return w[:-3] + "y"
                if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                    return w[:-1]
                return w
            for value in list(variants):
                singular = " ".join(_sing_word(t) for t in value.split())
                if singular != value:
                    variants.append(singular)
            seen = set()
            unique = []
            for value in variants:
                if value and value not in seen:
                    seen.add(value)
                    unique.append(value)
            return unique

        def _singular_pest_type(token: str) -> str:
            if token in _PEST_TYPE_WORDS_SET:
                if len(token) > 3 and token.endswith("ies"):
                    return token[:-3] + "y"
                if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                    return token[:-1]
                return token
            return token

        query_variants = _normalized_query_variants(query)

        # Try longest n-grams first so a specific multi-word common name wins over a single word.
        for query_variant in query_variants:
            tokens = re.findall(r"[a-z]+", query_variant)
            if not tokens:
                continue
            for size in (4, 3, 2, 1):
                if size > len(tokens):
                    continue
                for i in range(len(tokens) - size + 1):
                    phrase = " ".join(tokens[i:i + size])
                    if phrase in index:
                        return list(index[phrase])

        # "golden beetle(s)" is too broad if treated as adjective + pest type: it can match many
        # different golden-colored beetle common names. Keep it limited to the exact/common intended
        # names unless the user provides a more specific common name.
        for query_variant in query_variants:
            if re.search(r"\bgolden\s+beetles?\b", query_variant):
                matches = set()
                for phrase in ("golden beetle", "golden tortoise beetle"):
                    matches.update(index.get(phrase, []))
                if matches:
                    return sorted(matches)
                return []

        # Controlled partial phrase match: an adjective + pest type can match a longer common name,
        # while generic "beetle" alone is still handled by the pest-type path. Very broad phrases with
        # known ambiguity (e.g. "golden beetle") are handled above before this fallback.
        for query_variant in query_variants:
            tokens = re.findall(r"[a-z]+", query_variant)
            if not tokens:
                continue
            query_terms = {_singular_pest_type(t) for t in tokens}
            query_pest_terms = {t for t in query_terms if t in _PEST_TYPE_WORDS_SET}
            query_specific_terms = {
                t for t in query_terms
                if len(t) >= 3 and t not in _PEST_TYPE_WORDS_SET and t not in _NON_SUBJECT_WORDS
            }
            if not query_specific_terms or (len(query_specific_terms) < 2 and not query_pest_terms):
                continue
            # A lone color/quality adjective + a generic pest type ("white fly", "brown beetle") is too
            # broad — it would match any fly/beetle of that color. Require a real (exact/n-gram) phrase
            # match for these instead, so e.g. "white fly" cleanly reports "not in catalog" when there is
            # no whitefly dataset rather than returning unrelated march flies. ("golden beetle" handled above.)
            if query_specific_terms and query_specific_terms.issubset(_GENERIC_ADJECTIVE_WORDS):
                continue
            matches = set()
            for common_phrase, dataset_names in index.items():
                phrase_terms = {_singular_pest_type(t) for t in re.findall(r"[a-z]+", common_phrase)}
                phrase_pest_terms = {t for t in phrase_terms if t in _PEST_TYPE_WORDS_SET}
                if not query_specific_terms.issubset(phrase_terms):
                    continue
                if query_pest_terms and not query_pest_terms.intersection(phrase_pest_terms):
                    continue
                matches.update(dataset_names)
            if matches:
                return sorted(matches)

        # Single group head-noun match: a lone subject noun that only appears as the LAST word of
        # multi-word common names (e.g. "cutworm" → "black cutworm", "caterpillar" → "saddleback
        # caterpillar", "maggot" → "apple maggot", "armyworm" → "beet armyworm"). Requiring the query
        # word to be the head (last) word avoids adjectives like "golden"/"common"/"eastern" matching
        # everything. Larval group nouns in _HEAD_NOUN_SEARCH_WORDS are included even when they are also
        # in _PEST_TYPE_WORDS_SET (e.g. caterpillar).
        def _sing_head(w: str) -> str:
            if len(w) > 4 and w.endswith("ies"):
                return w[:-3] + "y"
            return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w

        for query_variant in query_variants:
            tokens = re.findall(r"[a-z]+", query_variant)
            head_candidates = []
            specific = [
                t for t in tokens
                if len(t) >= 4 and t not in _PEST_TYPE_WORDS_SET and t not in _NON_SUBJECT_WORDS
            ]
            if len(specific) == 1:
                head_candidates.append(_sing_head(specific[0]))
            elif len(tokens) == 1:
                t0 = _sing_head(tokens[0])
                if t0 in _HEAD_NOUN_SEARCH_WORDS or tokens[0] in _HEAD_NOUN_SEARCH_WORDS:
                    head_candidates.append(t0)
            if not head_candidates:
                continue
            head = head_candidates[0]
            matches = set()
            for common_phrase, dataset_names in index.items():
                ph_tokens = re.findall(r"[a-z]+", common_phrase)
                if ph_tokens and _sing_head(ph_tokens[-1]) == head:
                    matches.update(dataset_names)
            if matches:
                return sorted(matches)
        return []

    def _unmatched_subject_words(self, query: str) -> List[str]:
        """Return query words that look like a species subject but match no dataset/species in the catalog.

        Returns [] when the query is only attributes (action/time/scene/etc.) or when any subject word
        does match the catalog — so we only flag the clear "unknown species" case.
        """
        if not query or not query.strip():
            return []
        # A known common-name phrase (e.g. "painted lady" → Vanessa_cardui) means the subject IS in the
        # catalog even though it doesn't appear in any dataset name; don't flag it.
        if self._resolve_datasets_by_common_name(query):
            return []
        words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) >= 3]
        candidates = [w for w in words if w not in _NON_SUBJECT_WORDS]
        if not candidates:
            return []
        dataset_names = [dn.lower() for dn in self.dataset_registry.datasets]
        unmatched = []
        for w in candidates:
            w_sing = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            # Include common-name synonyms (e.g. "groundhog" → "woodchuck") so a known synonym is not flagged.
            terms = {w, w_sing}
            for t in (w, w_sing):
                if t in _SPECIES_SYNONYMS:
                    terms.add(_SPECIES_SYNONYMS[t])
            if any(t in dn for dn in dataset_names for t in terms):
                continue
            unmatched.append(w)
        # Only report when EVERY subject candidate is unmatched (avoid false alarms on partial matches).
        if unmatched and len(unmatched) == len(candidates):
            return unmatched
        return []

    def _compact_filters_for_llm(
        self,
        available_filters: Dict[str, List[str]],
        query: str,
        dataset: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Build a compact filter vocabulary for the LLM (avoids sending thousands of pest tokens)."""
        canonical_times = set()
        for raw in (available_filters.get("times") or []) + (available_filters.get("time") or []):
            canonical_times.update(_canonical_time_from_text(str(raw)))
        if not canonical_times:
            canonical_times = {"day", "night", "dawn", "dusk", "evening"}

        compact: Dict[str, List[str]] = {
            "category": ["wildlife", "domestic_animal", "livestock", "plants", "pests"],
            "time": sorted(canonical_times),
        }

        species = set()
        non_pest_types = {"wildlife", "domestic_animal", "livestock", "plants"}
        for dn in self.dataset_registry.datasets:
            if self.dataset_registry.infer_type_from_name(dn) in non_pest_types:
                species.add(dn)

        if query:
            q_lower = query.lower()
            words = [w for w in re.findall(r"[a-z_]+", q_lower) if len(w) >= 3]
            for dn in self.dataset_registry.datasets:
                dn_l = dn.lower()
                if dn_l in q_lower or any(w in dn_l for w in words):
                    species.add(dn)

        if dataset:
            for s in (available_filters.get("species") or [])[:80]:
                if s:
                    species.add(str(s))

        compact["species"] = sorted(species)

        for out_key, src_key, cap in [
            ("action", "actions", 30),
            ("season", "seasons", 10),
            ("plant_state", "plant_states", 12),
        ]:
            vals = available_filters.get(out_key) or available_filters.get(src_key) or []
            compact[out_key] = sorted({str(v).lower() for v in vals if v})[:cap]

        return compact

    def _species_implied_categories(self, species_filter: List[str]) -> set:
        """Infer data-type categories from species/dataset names."""
        cats = set()
        dataset_names = self.dataset_registry.datasets
        for s in species_filter:
            if not s:
                continue
            sl = str(s).lower().strip().replace(" ", "_")
            matched = False
            for dn in dataset_names:
                dn_l = dn.lower()
                if dn_l == sl or dn_l.startswith(sl + "_") or dn_l.startswith(sl + "-"):
                    cats.add(self.dataset_registry.infer_type_from_name(dn))
                    matched = True
                    break
            if not matched:
                for dn in dataset_names:
                    if sl in dn.lower() or dn.lower() in sl:
                        cats.add(self.dataset_registry.infer_type_from_name(dn))
        return {c for c in cats if c}

    def _dataset_matches_category(self, dataset_name: str, category_filter: List[str]) -> bool:
        """True if the dataset's data type matches any requested category (handles 'animal' grouping)."""
        if not category_filter:
            return True
        dataset_obj = self.dataset_registry.datasets.get(dataset_name)
        dataset_category = (
            dataset_obj.type.value.lower()
            if dataset_obj is not None
            else self.dataset_registry.infer_type_from_name(dataset_name)
        )
        category_mapping = {
            "pest": ["pests"],
            "pests": ["pests"],
            "animal": ["wildlife", "domestic_animal", "livestock"],
            "wildlife": ["wildlife"],
            "domestic_animal": ["domestic_animal"],
            "domestic": ["domestic_animal"],
            "livestock": ["livestock"],
            "plant": ["plants"],
            "plants": ["plants"],
        }
        for c in category_filter:
            cl = c.lower()
            if dataset_category in category_mapping.get(cl, [cl]) or cl == dataset_category:
                return True
        return False

    def _category_compatible(self, ui_categories: List[str], inferred_categories: set) -> bool:
        """True when UI category selection does not conflict with species-implied categories."""
        ui_norm = {c.lower().strip() for c in ui_categories if c}
        inferred_norm = {c.lower().strip() for c in inferred_categories if c}
        if not ui_norm or not inferred_norm:
            return True
        for ui in ui_norm:
            if ui == "animal":
                if inferred_norm & {"wildlife", "domestic_animal", "livestock"}:
                    return True
            elif ui in inferred_norm:
                return True
            elif ui == "plants" and "plants" in inferred_norm:
                return True
            elif ui == "pests" and "pests" in inferred_norm:
                return True
        return False

    def _extract_dataset_filters(self, dataset: Dataset) -> Dict[str, List[str]]:
        """Extract available filters from a dataset"""
        # Get images from the registry instead of accessing dataset.images
        dataset_name = dataset.name
        images = self.dataset_registry.get_images(dataset_name)
        
        filters = {
            "categories": [],
            "species": [],
            "times": [],
            "time": [],
            "seasons": [],
            "actions": [],
            "plant_states": []
        }
        
        for image in images:
            # Extract category
            if image.get("category"):
                filters["categories"].append(image["category"])
            
            # Extract species - prioritize metadata.species (where MCP stores it)
            # Also use common_name, common_names (pest list), and scientific_name so "moth"/"raspberry" etc. are available
            metadata = image.get("metadata", {})
            species_value = (
                metadata.get("species") or metadata.get("common_name") or metadata.get("scientific_name")
                or image.get("species") or image.get("collection")
            )
            for species_str in self._filter_strings(species_value):
                if not species_str:
                    continue
                species_base = species_str.split("_")[0].split("-")[0].strip().lower()
                if species_base:
                    filters["species"].append(species_base)
                if "_" in species_str:
                    parts = [p.strip().lower() for p in species_str.split("_") if p.strip()]
                    if len(parts) >= 2 and not parts[-1].isdigit():
                        compound = "_".join(parts)
                        if compound not in filters["species"]:
                            filters["species"].append(compound)
                        if len(parts) >= 3:
                            two_word_compound = "_".join(parts[:2])
                            if two_word_compound not in filters["species"]:
                                filters["species"].append(two_word_compound)
            # Add pest common_names so queries like "moth" match (common_names can be e.g. ["Raspberry crown moth", "moth"])
            # Words we must NOT add as species (stopwords + action/description words that leak from long action text)
            _species_stopwords = {
                "the", "and", "or", "at", "in", "on", "to", "for", "of", "with", "within", "from", "an", "a",
                "possibly", "appears", "stationary", "standing", "walking", "looking", "staring", "facing",
                "camera", "enclosure", "animal", "appears", "toward", "directly", "resting", "moving", "eating",
                "feeding", "sleeping", "hunting", "alert", "perching", "flying", "running", "sitting", "lowering",
                "king", "pea", "tan", "thin",  # common false positives from substrings in "walking"/description
            }
            for species_str in self._filter_strings(metadata.get("common_names")):
                if not species_str:
                    continue
                species_base = species_str.split("_")[0].split("-")[0].strip().lower()
                if species_base and species_base not in filters["species"] and species_base not in _species_stopwords:
                    filters["species"].append(species_base)
                # Multi-word common name: add each word as a filter (e.g. "moth" from "raspberry crown moth")
                # Skip long description-like strings (e.g. "Animal appears stationary, possibly standing or walking...")
                if len(species_str) > 60:
                    continue
                for word in species_str.replace("_", " ").replace("-", " ").split():
                    w = word.strip().lower()
                    if w and len(w) >= 2 and w not in filters["species"] and w not in _species_stopwords:
                        filters["species"].append(w)
            # Pest type words: add "moth"/"beetle" etc. when they appear as a word in collection/species/scientific_name (e.g. cabbage_moth)
            _pest_type_words = {"beetle", "butterfly", "moth", "wasp", "bee", "ant", "fly", "grasshopper", "dragonfly", "spider", "bug", "insect"}
            for src in (image.get("collection") or "", metadata.get("species") or "", metadata.get("scientific_name") or ""):
                for w in (str(src).replace("_", " ").replace("-", " ").lower().split() or []):
                    w = w.strip()
                    if w in _pest_type_words and w not in filters["species"]:
                        filters["species"].append(w)
                    elif len(w) > 1 and w.endswith("s") and w[:-1] in _pest_type_words:
                        singular = w[:-1]
                        if singular not in filters["species"]:
                            filters["species"].append(singular)
            
            # Extract time, season, action, plant_state (each can be str or list)
            for t in self._filter_strings(metadata.get("time")):
                filters["times"].append(t)
                filters["time"].extend(sorted(_canonical_time_from_text(t)))
            for s in self._filter_strings(metadata.get("season")):
                filters["seasons"].append(s)
            for a in self._filter_strings(metadata.get("action")):
                filters["actions"].append(a)
            for ps in self._filter_strings(metadata.get("plant_state")):
                filters["plant_states"].append(ps)
            
            # Also extract actions from description field
            description = image.get("metadata", {}).get("description", "").lower()
            if description:
                # Action keyword map to identify actions in descriptions
                # IMPORTANT: Map query keywords to canonical action names
                # This ensures "feeding" in description maps to "foraging" if that's the canonical name
                action_keyword_map = {
                    "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],  # Canonical: foraging
                    "sleeping": ["sleep", "sleeping", "rest", "resting"],  # Canonical: sleeping
                    "resting": ["rest", "resting"],  # Canonical: resting
                    "walking": ["walk", "walking", "moving"],  # Canonical: walking
                    "hunting": ["hunt", "hunting"],  # Canonical: hunting
                    "alert": ["alert", "alerts", "watch", "watching", "looking at camera", "looking at the camera", 
                             "staring at camera", "staring at the camera", "facing camera", "facing the camera",
                             "looking toward camera", "looking toward the camera", "staring toward camera", 
                             "staring toward the camera", "facing toward camera", "facing toward the camera",
                             "looking directly at camera", "looking directly at the camera", "staring directly at camera",
                             "staring directly at the camera", "facing directly at camera", "facing directly at the camera"],  # Canonical: alert
                    "moving": ["moving", "move"],  # Canonical: moving
                    "running": ["running", "run"],  # Canonical: running
                    "perching": ["perching", "perch", "sitting", "sit"],  # Canonical: perching
                    "flying": ["flying", "fly"],  # Canonical: flying
                    "blooming": ["blooming", "bloom", "flowering", "flower"],  # Canonical: blooming
                    "fruiting": ["fruiting", "fruit"],  # Canonical: fruiting
                    "growing": ["growing", "grow"],  # Canonical: growing
                    "mature": ["mature", "matured", "ripe"]  # Canonical: mature
                }
                
                # Check description for action keywords and map to canonical names
                for canonical_action, keywords in action_keyword_map.items():
                    if any(keyword in description for keyword in keywords):
                        # Only add canonical action name, not the keyword found
                        if canonical_action not in filters["actions"]:
                            filters["actions"].append(canonical_action)
        
        # Remove duplicates and sort (all values are now strings, so set() is safe)
        for key in filters:
            filters[key] = sorted(list(set(filters[key])))
        
        return filters
    
    def _get_required_description_phrase(self, query: str, species_filter: List[str], description_query: Optional[str] = None) -> Optional[str]:
        """When the query specifies a cultivar/variety (e.g. 'Cabernet Sauvignon grapes'), return the phrase that must appear in the result description to exclude other varieties (e.g. Syrah).
        When the query is only a species name or its plural (e.g. 'raspberries' or 'raspberry'), return None so we return all items for that species."""
        if not query or not query.strip():
            return None
        q = query.strip()
        species_set = set()
        species_words = set()  # all words that appear in any normalized species name (e.g. american, black, bear)
        for s in (species_filter or []):
            s = str(s).lower().strip().replace("_", " ")
            if s:
                species_set.add(s)
                if s.endswith("s") and len(s) > 1:
                    species_set.add(s[:-1])  # singular: raspberries -> raspberry
                else:
                    species_set.add(s + "s")  # common plural: raspberry -> raspberries
                species_words.update(s.split())
        words = q.split()
        stop = {"the", "a", "an", "and", "or", "in", "on", "at", "for", "of", "with", "images", "pictures", "photos", "show", "find"}
        # Behavior/time/scene words (e.g. "eating", "standing", "night") are handled by their own filters and must
        # NOT become a required literal description phrase — otherwise "squirrel eating" would exclude images
        # described as "foraging"/"feeding". Strip them (and punctuation) before forming the cultivar/variety phrase.
        def _norm_word(w: str) -> str:
            return re.sub(r"[^a-z]", "", w.lower())
        remaining = [
            w for w in words
            if w.lower() not in species_set
            and w.lower() not in stop
            and _norm_word(w) not in _NON_SUBJECT_WORDS
        ]
        if not remaining:
            return None
        phrase = " ".join(remaining).strip()
        if len(phrase) < 2:
            return None
        # Don't require phrase in description when the query is only a species synonym (e.g. "american bear" -> american_black_bear)
        remaining_lower = {w.lower() for w in remaining}
        if species_words and remaining_lower <= species_words:
            return None
        return phrase
    
    def _passes_description_required(self, result: Dict[str, Any], required_phrase: str) -> bool:
        """True if the result's description or metadata text contains the required phrase (case-insensitive)."""
        if not required_phrase or not required_phrase.strip():
            return True
        want = required_phrase.lower().strip()
        meta = result.get("metadata") or {}
        desc = (meta.get("description") or "")
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc if x)
        desc = str(desc).lower()
        if want in desc:
            return True
        for key in ("species", "common_name", "scientific_name", "scene", "background"):
            val = meta.get(key)
            if val and want in str(val).lower():
                return True
        return False
    
    def _passes_action_strict(self, result: Dict[str, Any], action_filter: List[str]) -> bool:
        """When user asked for a specific action (e.g. sleeping), exclude items that clearly have a different action (e.g. walking) or that describe awake/observing (e.g. cat at night observing). Do not infer sleeping from 'night'."""
        if not action_filter:
            return True
        meta = result.get("metadata", {})
        item_action = (meta.get("action") or "")
        if isinstance(item_action, list):
            item_action = " ".join(str(x).strip() for x in item_action if x).strip()
        else:
            item_action = str(item_action).strip()
        item_description = (meta.get("description") or "")
        item_canonical = _item_canonical_action(item_action, item_description)
        for requested in action_filter:
            if _action_filter_conflicts(requested, item_canonical):
                return False
        # When user asked for sleeping/resting, exclude items where description says awake/observing (e.g. cat at night observing)
        if any(a.lower().strip() in ("sleeping", "resting") for a in action_filter):
            if _description_indicates_awake_or_observing(item_description):
                return False
        return True
    
    def _passes_tail_strict(self, result: Dict[str, Any], tail_filter: List[str]) -> bool:
        """When the user asked for goats with tail up/down, keep ONLY items that actually say so.

        Checks metadata tail_position/tail_positions, the category label (e.g. "tail-up"), and the
        description ("tail up" / "tail is up"). Items with no explicit tail direction are excluded, so
        the result set is restricted to images whose description/metadata confirms the tail position.
        """
        if not tail_filter:
            return True
        wanted = {str(t).lower().strip() for t in tail_filter if t}
        if not wanted:
            return True
        meta = result.get("metadata") or {}

        _UNDET_INNER = r"(undetermined|undetermend|indeterminate|unknown|not\s+visible|unclear|obscured|n/?a)"
        _UNDET_RE = r"\b" + _UNDET_INNER + r"\b"

        def _explicit_field_direction(text: str) -> str:
            """For dedicated tail fields the value IS the tail position, so a bare up/down is meaningful.
            Use word boundaries so 'group'/'upper' etc. never count."""
            t = str(text or "").lower()
            if not t:
                return ""
            if re.search(_UNDET_RE, t) or "tail-undetermined" in t:
                return "undetermined"
            if re.search(r"\bup\b", t) or "tail-up" in t or "tailup" in t:
                return "up"
            if re.search(r"\bdown\b", t) or "tail-down" in t or "taildown" in t:
                return "down"
            return ""

        def _tail_proximate_direction(text: str) -> str:
            """For free text (category/description) only count up/down when it is clearly about the TAIL,
            e.g. 'tail up', 'tail is up', 'tail pointing up', 'tail held up', 'tail-up'. This avoids false
            positives like 'group of goats' (substring 'up') or 'walking down a hill'."""
            t = str(text or "").lower()
            if not t or "tail" not in t:
                return ""
            if "tail-undetermined" in t or re.search(r"tail\b[^.]{0,20}" + _UNDET_INNER, t):
                return "undetermined"
            if re.search(r"tail[\s\-]*(is\s+|held\s+|pointing\s+|raised\s+|curl(?:ed|ing)?\s+|up\b|standing\s+)*up\b", t) or "tail-up" in t:
                return "up"
            if re.search(r"tail[\s\-]*(is\s+|held\s+|pointing\s+|hanging\s+|tucked\s+|lowered\s+|down\b)*down\b", t) or "tail-down" in t:
                return "down"
            # Generic "tail" + nearby up/down within a few words
            m = re.search(r"tail\b[^.]{0,20}\bup\b", t)
            if m:
                return "up"
            m = re.search(r"tail\b[^.]{0,20}\bdown\b", t)
            if m:
                return "down"
            return ""

        raw_tail = meta.get("tail_positions") or meta.get("tail_position")
        if isinstance(raw_tail, list):
            raw_tail = " ".join(str(x) for x in raw_tail if x)
        item_dir = _explicit_field_direction(raw_tail)
        if not item_dir:
            item_dir = _tail_proximate_direction(meta.get("category"))
        if not item_dir:
            desc = meta.get("description") or result.get("description") or ""
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc if x)
            item_dir = _tail_proximate_direction(desc)
        if not item_dir:
            return False
        return item_dir in wanted

    def _passes_plant_state_strict(self, result: Dict[str, Any], plant_state_filter: List[str]) -> bool:
        """When user asked for a specific plant_state (e.g. ripe/unripe), keep items that match.
        For 'ripe': exclude mixed unless description says ripe. For 'unripe': allow mixed if description mentions unripe.
        """
        if not plant_state_filter:
            return True
        # Normalize to list (in case passed as string)
        if isinstance(plant_state_filter, str):
            plant_state_filter = [plant_state_filter.strip()] if plant_state_filter.strip() else []
        if not plant_state_filter:
            return True
        meta = result.get("metadata", {})
        ps_val = meta.get("plant_state") or meta.get("plant_states")
        if isinstance(ps_val, list):
            item_ps = " ".join(str(x).strip() for x in ps_val if x).strip().lower()
        else:
            item_ps = (str(ps_val).strip().lower() if ps_val else "")
        desc = (meta.get("description") or result.get("description") or "").lower()
        for requested in plant_state_filter:
            r = requested.lower().strip()
            if r not in ["ripe", "ripening", "unripe", "mature", "green", "red", "blooming", "fruiting", "buds", "bud"]:
                continue
            # ---- Unripe: accept immediately if metadata says unripe, or blooming/buds, or description mentions unripe ----
            if r == "unripe":
                if item_ps == "unripe":
                    return True
                # Unripe should translate to early growth stages: blooming, buds
                if item_ps in ("blooming", "buds", "bud"):
                    return True
                # If "unripe" appears anywhere in the description caption, include the image
                if "unripe" in desc or "unripened" in desc:
                    return True
                # When metadata has no plant_state, accept if description clearly describes unripe fruit/berries
                if not item_ps or not item_ps.strip():
                    unripe_desc_phrases = [
                        "unripe berr", "unripe fruit", "unripe raspberr", "green berr", "green fruit",
                        "developing fruit", "developing berr", "developing raspberr", "immature",
                    ]
                    if any(p in desc for p in unripe_desc_phrases):
                        return True
                # Allow "mixed" for unripe when description mentions unripe (image has unripe berries even if some ripe too)
                if item_ps == "mixed":
                    unripe_phrases = [
                        "unripe berr", "unripe fruit", "unripe raspberr", "unripe strawberr", "developing fruit",
                        "developing berr", "developing raspberr", "green fruit", "green berr",
                        "young raspberr", "young fruit", "young berr", "unripe berries",
                    ]
                    if any(p in desc for p in unripe_phrases):
                        return True
                    continue
                if item_ps == "fruiting":
                    if any(p in desc for p in ["developing fruit", "developing berr", "green fruit"]):
                        return True
                continue
            # ---- Ripe: allow "mixed" if description mentions ripe (image has ripe berries even if some unripe too) ----
            if r == "ripe" and item_ps == "mixed":
                if _description_has_ripe_phrase(desc):
                    return True
            # Ripe: exclude when description says unripe but has no ripe (e.g. "flowers and buds and unripe berries")
            if r == "ripe" and ("unripe" in desc or "unripened" in desc):
                if not re.search(r"\bripe\b", desc) and not _description_has_ripe_phrase(desc):
                    return False
            # Exclude mixed for ripe only when description doesn't suggest ripe content
            if item_ps == "mixed":
                return False
            # Ripening ≠ ripe: when user asks for "ripe", exclude items that are only "ripening"
            if r == "ripe" and item_ps == "ripening":
                return False
            if r == "ripe" and item_ps == "ripe":
                return True
            if r == "ripe":
                mixed_phrases = [
                    "varying stages", "various stages", "various ripening", "stages of ripeness", "varying ripeness",
                    "at various ripening", "at various stages", "displaying raspberries at various",
                    "mix of unripe and ripe", "unripe and ripe", "unripe raspberr", "unripe berry",
                    "unripe berries", "unripe fruit", "small, unripe", "developing raspberr",
                    "developing fruit", "developing berries", "different stages", "multiple stages",
                    "various stages of", "ripening stages",
                ]
                if any(p in desc for p in mixed_phrases):
                    return False
                if item_ps == "fruiting" and not _description_has_ripe_phrase(desc):
                    return False
            if item_ps == r:
                return True
            item_ps_words = (item_ps.split() if item_ps else [])
            if r in item_ps_words:
                return True
            if r == "ripe" and _description_has_ripe_phrase(desc):
                return True
        return False
    
    def _calculate_result_confidence(self, result: Dict[str, Any], query_understanding, query: Optional[str] = None) -> float:
        """Calculate per-image filter match score based on how well it matches the query filters.
        
        This is NOT an AI confidence score - it's a relevance/match score that measures
        how well the image's metadata matches the structured filters extracted from the query.
        The actual AI confidence (how well the LLM understood the query) is in query_understanding.confidence.
        When query is provided, results whose description contains the query or key phrases (e.g. "standing upright")
        get a boost so they rank above results that only match filters.
        """
        filters = query_understanding.filters
        metadata = result.get('metadata', {})
        
        # Count how many filters are specified in the query
        total_filters = sum(1 for key, values in filters.items() if values)
        if total_filters == 0:
            return query_understanding.confidence
        
        # Track match quality and count matches vs misses
        match_scores = []
        matched_filters = 0
        total_filter_checks = 0
        
        # Check species match quality (CRITICAL - highest weight)
        if filters.get("species"):
            total_filter_checks += 1
            item_species = _metadata_str(metadata.get("species", "")).lower().strip()
            item_collection = _metadata_str(result.get("collection", "")).lower().strip()
            # The species filter is often the DATASET name (a common name like "coyote"), while metadata.species
            # may hold the scientific name (e.g. "Canis latrans"). Match against all of the image's identity
            # fields so a result that genuinely belongs to the queried dataset scores as a strong match.
            item_dataset = _metadata_str(result.get("dataset", "")).lower().strip()
            item_scientific = _metadata_str(metadata.get("scientific_name", "")).lower().strip()
            item_common = _metadata_str(metadata.get("common_name", "")).lower().strip()
            item_common_names = [
                str(c).lower().strip()
                for c in (metadata.get("common_names") or [])
                if c and str(c).strip()
            ]
            # Exact-identity values: an exact (normalized) match against any of these is a perfect species match.
            identity_values = [v for v in [item_dataset, item_collection, item_scientific, item_common] if v] + item_common_names
            identity_norm = {v.replace("_", "").replace("-", "").replace(" ", "") for v in identity_values}
            best_match = 0.0
            for species_filter in filters["species"]:
                species_lower = species_filter.lower().strip()
                species_norm = species_lower.replace("_", "").replace("-", "").replace(" ", "")
                # Exact match in metadata.species is strongest
                if item_species and item_species == species_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    # Exact match: 0.98 to 1.0 (perfect match!)
                    best_match = max(best_match, 0.98 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif species_norm and (species_norm in identity_norm or species_norm == item_species.replace("_", "").replace("-", "").replace(" ", "")):
                    # Filter matches the dataset/collection/common/scientific name (e.g. dataset-name search) — perfect match.
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.97 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif (item_species.startswith(species_lower + "_") or item_collection.startswith(species_lower + "_")
                      or item_dataset.startswith(species_lower + "_")):
                    best_match = max(best_match, 0.90)
                    matched_filters += 1
                else:
                    # Normalized substring match (handles variations like compound common names)
                    if species_norm and any(species_norm in v or v in species_norm for v in identity_norm if v):
                        best_match = max(best_match, 0.87)
                        matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check time match quality (daytime requested => exclude items that indicate night)
        if filters.get("time"):
            total_filter_checks += 1
            item_time = _metadata_str(metadata.get("time", "")).lower().strip()
            item_desc_lower = _metadata_str(metadata.get("description", "")).lower()
            best_match = 0.0
            for time_filter in filters["time"]:
                time_lower = time_filter.lower().strip()
                is_day_request = any(x in time_lower for x in ("day", "morning", "afternoon", "daytime"))
                item_is_night = ("night" in item_time or "nighttime" in item_time) and "daytime" not in item_time
                item_desc_says_night = "night" in item_desc_lower and "daytime" not in item_desc_lower
                if is_day_request and (item_is_night or item_desc_says_night):
                    continue  # user asked for day but item is night — do not count time match
                if item_time == time_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif time_lower in item_time or item_time in time_lower:
                    best_match = max(best_match, 0.91)
                    matched_filters += 1
                elif is_day_request and any(x in item_time or x in item_desc_lower for x in ("day", "daytime", "morning", "afternoon")):
                    # e.g. filter "during daytime" vs item "Daytime (11:40 AM)"
                    if not item_is_night and not item_desc_says_night:
                        best_match = max(best_match, 0.91)
                        matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check action match quality: check against action field AND against description (either can match)
        if filters.get("action"):
            total_filter_checks += 1
            item_action = _metadata_str(metadata.get("action", "")).lower().strip()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            _action_keyword_map = {
                "sleeping": ["sleep", "sleeping", "rest", "resting"],
                "feeding": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
                "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
                "resting": ["rest", "resting", "sleep", "sleeping"],
                "walking": ["walk", "walking", "moving"],
                "hunting": ["hunt", "hunting"],
                "alert": ["alert", "alerts", "watch", "watching", "looking at camera", "looking at the camera"],
                "moving": ["move", "moving", "walk", "walking"],
                "running": ["run", "running", "moving"],
                "perching": ["perch", "perching", "sitting", "sit"],
                "flying": ["fly", "flying"],
            }
            best_match = 0.0
            for action_filter in filters["action"]:
                action_lower = action_filter.lower().strip()
                matched_this = False
                # Check against action field
                if item_action == action_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_this = True
                elif action_lower in item_action:
                    best_match = max(best_match, 0.92)
                    matched_this = True
                # Check against description field (in addition to action field)
                desc_matched = False
                if action_lower in item_description:
                    best_match = max(best_match, 0.88)
                    desc_matched = True
                if not desc_matched:
                    for _keyword, _variations in _action_keyword_map.items():
                        if action_lower == _keyword or action_lower in _variations:
                            if any(v in item_description for v in _variations):
                                best_match = max(best_match, 0.88)
                                desc_matched = True
                                break
                if not desc_matched:
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if action_base in item_description and len(action_base) >= 3:
                        best_match = max(best_match, 0.88)
                        desc_matched = True
                if desc_matched:
                    matched_this = True
                if matched_this:
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check season match quality
        if filters.get("season"):
            total_filter_checks += 1
            item_season = _metadata_str(metadata.get("season", "")).lower().strip()
            best_match = 0.0
            for season_filter in filters["season"]:
                season_lower = season_filter.lower().strip()
                if item_season == season_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif season_lower in item_season:
                    best_match = max(best_match, 0.91)
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check scene match quality (IMPORTANT: Scene mismatches should be heavily penalized)
        if filters.get("scene"):
            total_filter_checks += 1
            item_scene = _metadata_str(metadata.get("scene", "")).lower().strip()
            best_match = 0.0
            scene_matched = False
            for scene_filter in filters["scene"]:
                scene_lower = scene_filter.lower().strip()
                if item_scene == scene_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.97 + (id_hash % 3) / 100.0)  # Higher score for exact match
                    matched_filters += 1
                    scene_matched = True
                elif scene_lower in item_scene or item_scene in scene_lower:
                    best_match = max(best_match, 0.93)  # Good partial match
                    matched_filters += 1
                    scene_matched = True
            
            if scene_matched:
                match_scores.append(best_match)
            else:
                # HEAVY PENALTY: Scene filter specified but doesn't match
                # This ensures "field" queries prioritize field images over indoor images
                match_scores.append(0.3)  # Low score for scene mismatch
                # Don't increment matched_filters - this counts as a miss
        
        # Check weather match quality
        if filters.get("weather"):
            total_filter_checks += 1
            item_weather = _metadata_str(metadata.get("weather", "")).lower().strip()
            best_match = 0.0
            for weather_filter in filters["weather"]:
                weather_lower = weather_filter.lower().strip()
                if item_weather == weather_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.94 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif weather_lower in item_weather:
                    best_match = max(best_match, 0.89)
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check plant_state match quality (important for ripeness/color queries)
        # Prioritize exact plant_state match (e.g. ripe) over mixed/fruiting when user asked for ripe
        if filters.get("plant_state"):
            total_filter_checks += 1
            item_plant_state = _metadata_str(metadata.get("plant_state", "")).lower().strip()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            best_match = 0.0
            for plant_state_filter in filters["plant_state"]:
                plant_state_lower = plant_state_filter.lower().strip()
                if item_plant_state == plant_state_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.97 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif plant_state_lower in ("ripe", "unripe", "mature", "green", "red") and item_plant_state in ("mixed", "fruiting"):
                    # User asked for specific state but item is mixed/fruiting — deprioritize so exact matches sort first
                    best_match = max(best_match, 0.62)
                    matched_filters += 1
                elif plant_state_lower == "ripe" and item_plant_state == "ripening":
                    # Ripening ≠ ripe: do not treat ripening as a match for "ripe"
                    best_match = max(best_match, 0.55)
                    matched_filters += 1
                elif plant_state_lower in (item_plant_state.split() if item_plant_state else []):
                    # Whole-word match only (so "ripe" does not match "ripening")
                    best_match = max(best_match, 0.93)
                    matched_filters += 1
                elif plant_state_lower in item_description:
                    # If item has mixed/fruiting but user asked for specific state, don't over-reward description match
                    desc_score = 0.91
                    if item_plant_state in ("mixed", "fruiting") and plant_state_lower in ("ripe", "unripe", "mature", "green", "red"):
                        desc_score = 0.68  # so exact plant_state=ripe still sorts first
                    # For "ripe": require whole-word or explicit phrases so "unripe berries" / "ripening" do not match
                    if plant_state_lower == "ripe":
                        ripe_ok = (re.search(r"\bripe\b", item_description) and "ripening" not in item_description) or _description_has_ripe_phrase(item_description)
                        if ripe_ok:
                            best_match = max(best_match, desc_score)
                            matched_filters += 1
                    else:
                        plant_state_keywords = {
                            "mature": ["mature", "ripe", "ready", "fully developed"],
                            "unripe": ["unripe", "green", "immature", "young", "developing"],
                            "green": ["green", "unripe", "immature"],
                        }
                        for key, variations in plant_state_keywords.items():
                            if plant_state_lower == key:
                                if any(v in item_description for v in variations):
                                    best_match = max(best_match, desc_score)
                                    matched_filters += 1
                                    break
                        if best_match < desc_score:
                            best_match = max(best_match, min(0.89, desc_score))
                            matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Calculate base confidence from match quality
        if match_scores:
            # Use weighted average (species gets more weight if present)
            if filters.get("species") and len(match_scores) > 0:
                # Give species match 40% weight, others share remaining 60%
                species_score = match_scores[0] if filters.get("species") else 0
                other_scores = match_scores[1:] if filters.get("species") else match_scores
                if other_scores:
                    avg_other = sum(other_scores) / len(other_scores)
                    avg_match_quality = 0.4 * species_score + 0.6 * avg_other
                else:
                    avg_match_quality = species_score
            else:
                avg_match_quality = sum(match_scores) / len(match_scores)
            
            # Calculate filter match ratio (how many filters matched); cap at 1.0 so full match => high confidence
            match_ratio = min(1.0, matched_filters / total_filter_checks if total_filter_checks > 0 else 1.0)
            
            # Base confidence: start high for good matches, penalize for missing filters
            if match_ratio == 1.0:
                # All filters matched - high confidence
                base_confidence = 0.90 + (avg_match_quality - 0.90) * 0.3  # 90-96%
            elif match_ratio >= 0.75:
                # Most filters matched - good confidence
                base_confidence = 0.80 + (avg_match_quality - 0.85) * 0.4  # 75-88%
            elif match_ratio >= 0.5:
                # Half filters matched - moderate confidence
                base_confidence = 0.65 + (avg_match_quality - 0.80) * 0.3  # 60-75%
            else:
                # Few filters matched - lower confidence
                base_confidence = 0.50 + (avg_match_quality - 0.75) * 0.3  # 50-65%
        else:
            # No matches - low confidence
            base_confidence = 0.45
        
        # Calculate metadata completeness boost
        metadata_fields = ['species', 'time', 'season', 'action', 'scene', 'weather', 'description']
        populated_fields = sum(1 for field in metadata_fields if metadata.get(field))
        metadata_completeness = populated_fields / len(metadata_fields) if metadata_fields else 0.7
        completeness_boost = 1.0 + (metadata_completeness - 0.7) * 0.05  # Up to +1.5% boost
        
        # Description quality boost
        description = metadata.get("description", "")
        description_length = len(description) if description else 0
        description_quality = min(1.0, description_length / 100.0) if description_length > 0 else 0.6
        description_boost = 1.0 + (description_quality - 0.6) * 0.03  # Up to +1.2% boost
        
        adjusted_confidence = base_confidence * completeness_boost * description_boost
        
        # Add variation based on ID hash for differentiation (±1.5%)
        import hashlib
        item_id = result.get('id', '')
        hash_value = int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)
        variation = (hash_value % 31 - 15) / 1000.0  # Range: -0.015 to +0.015
        adjusted_confidence = adjusted_confidence + variation
        
        # Cap at 50% to 98% (wider range, higher max for perfect matches)
        adjusted_confidence = min(0.98, max(0.50, adjusted_confidence))
        
        # Prioritize results whose description explicitly mentions the query or key phrase (e.g. "standing upright")
        if query and isinstance(query, str) and query.strip():
            item_desc = _metadata_str(metadata.get("description", "")).lower()
            if item_desc:
                q = query.strip().lower()
                if q in item_desc:
                    adjusted_confidence = max(adjusted_confidence, 0.96)  # Full query in description → top rank
                else:
                    words = q.split()
                    if len(words) >= 2:
                        key_phrase = " ".join(words[1:])  # e.g. "standing upright" from "woodchuck standing upright"
                        if key_phrase in item_desc:
                            adjusted_confidence = max(adjusted_confidence, 0.94)  # Key phrase in description
                    if len(words) >= 3:
                        key_phrase_3 = " ".join(words[1:4])
                        if key_phrase_3 in item_desc:
                            adjusted_confidence = max(adjusted_confidence, 0.95)
        
        # Round to 1 decimal place
        return round(adjusted_confidence, 1)
    
    def _inference_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for inference tool"""
        dataset_name = input_data["dataset_name"]
        model_name = input_data["model_name"]
        image_ids = input_data["image_ids"]
        parameters = input_data.get("parameters", {})
        
        # Create inference request
        inference_request = InferenceRequest(
            dataset_name=dataset_name,
            model_name=model_name,
            image_ids=image_ids,
            parameters=parameters
        )
        
        # Run inference (this would be async in practice)
        # For now, return a placeholder
        return {
            "message": "Inference tool handler - will be implemented with async support",
            "request": input_data
        }
    
    def _dataset_info_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for dataset info tool"""
        dataset_name = input_data.get("dataset_name")
        
        if dataset_name:
            dataset = self.dataset_registry.get_dataset(dataset_name)
            if not dataset:
                return {"error": f"Dataset {dataset_name} not found"}
            return {"dataset": dataset}
        else:
            return {
                "datasets": self.dataset_registry.get_all_datasets(),
                "total": len(self.dataset_registry.get_all_datasets())
            }
    
    def _model_info_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for model info tool"""
        from models import ModelInfo
        
        model_name = input_data.get("model_name")
        
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                return {"error": f"Model {model_name} not found"}
            
            # Convert Model to ModelInfo (removes non-serializable handler)
            model_info = ModelInfo(
                name=model.name,
                type=model.type,
                description=model.description,
                version=model.version,
                supported_datasets=model.supported_datasets,
                parameters=model.parameters,
                metadata=model.metadata
            )
            # Convert to dict for JSON serialization (handles Enum types)
            return {
                "model": {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            }
        else:
            # Get all models and convert to serializable format
            all_models = self.model_registry.get_all_models()
            models_dict = {}
            for name, model_info in all_models.items():
                models_dict[name] = {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            return {
                "models": models_dict,
                "total": len(models_dict)
            }
    
    async def _crawl_croissant_datasets_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Croissant dataset crawling"""
        if not CROISSANT_CRAWLER_AVAILABLE:
            return {
                "error": "Croissant crawler not available. Ensure croissant_crawler.py is in the same directory.",
                "datasets": [],
                "total_count": 0
            }
        
        try:
            print(f"🔍 Starting Croissant dataset crawling...")
            
            crawler = CroissantCrawler()
            datasets = await crawler.crawl_all_portals()
            
            print(f"📊 Crawler returned {len(datasets)} datasets")
            for i, dataset in enumerate(datasets):
                print(f"  {i+1}. {dataset.name} (source: {dataset.source_portal})")
            
            # Convert to MCP format
            results = []
            for i, dataset in enumerate(datasets):
                try:
                    result = {
                        'name': dataset.name,
                        'description': dataset.description,
                        'url': dataset.url,
                        'source': dataset.source_portal,
                        'fields': dataset.fields,
                        'keywords': dataset.keywords or [],
                        'license': dataset.license,
                        'download_urls': dataset.download_urls or [],
                        'created_date': dataset.created_date,
                        'updated_date': dataset.updated_date
                    }
                    results.append(result)
                    print(f"  ✅ Converted dataset {i+1}: {dataset.name}")
                except Exception as e:
                    print(f"  ❌ Error converting dataset {i+1} ({dataset.name}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next dataset instead of failing completely
            
            print(f"✅ Converted {len(results)} datasets to MCP format")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['name']} (source: {result['source']})")
            
            from datetime import datetime
            return {
                "datasets": results,
                "total_count": len(results),
                "sources": list(crawler.portals.keys()),
                "crawl_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Croissant crawling error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Croissant crawling failed: {str(e)}",
                "datasets": [],
                "total_count": 0
            }
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], 
                      handler: Callable, tags: List[str] = None):
        """Register a new tool with the server"""
        self.tool_registry.register_tool(name, description, input_schema, handler, tags)
    
    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for external access"""
        return self.tool_registry
    
    def get_dataset_registry(self) -> DatasetRegistry:
        """Get the dataset registry for external access"""
        return self.dataset_registry
    
    def get_model_registry(self) -> ModelRegistry:
        """Get the model registry for external access"""
        return self.model_registry
    
    def run(self, host: str = None, port: int = None):
        """Run the MCP server"""
        host = host or MCP_CONFIG.get("mcp_host", "0.0.0.0")
        port = port or MCP_CONFIG.get("mcp_port", 8188)
        
        print(f"🚀 Starting {self.name} on {host}:{port}")
        print(f"✅ LLM search: broad species queries (e.g. 'raspberries') return full dataset — no description filter")
        all_tools = self.tool_registry.get_all_tools()
        tool_names = list(all_tools.keys())
        print(f"🔧 Available tools ({len(tool_names)}): {', '.join(tool_names)}")
        print(f"📁 Available datasets: {len(self.dataset_registry.get_all_datasets())}")
        print(f"🤖 Available models: {len(self.model_registry.get_all_models())}")
        print(f"🔗 MCP Discovery: http://{host}:{port}/.well-known/mcp")
        print(f"🔧 Tools endpoint: http://{host}:{port}/mcp/tools")
        print(f"📁 Datasets endpoint: http://{host}:{port}/api/datasets")
        print(f"🤖 Models endpoint: http://{host}:{port}/api/models")
        print(f"💚 Health check: http://{host}:{port}/health")
        
        # Check if croissant tool was registered
        if "crawl_croissant_datasets" not in tool_names:
            print(f"⚠️  WARNING: crawl_croissant_datasets tool is NOT registered!")
            print(f"   CROISSANT_CRAWLER_AVAILABLE was: {CROISSANT_CRAWLER_AVAILABLE}")
        else:
            print(f"✅ crawl_croissant_datasets tool is registered and available")
        
        uvicorn.run(self.app, host=host, port=port)

# Global MCP server instance
mcp_server = MCPServer()

if __name__ == "__main__":
    # Run the core MCP server
    mcp_server.run()
