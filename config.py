#!/usr/bin/env python3
"""
Configuration for the MCP server and web interface
"""

import os
from pathlib import Path

# Base configuration (defined first so we can use it for .env file location)
BASE_DIR = Path("/opt/mcp-data-server")

# Load environment variables from .env file if it exists.
# By default, only variables that are NOT already set in the environment are loaded (env takes precedence).
# Set DOTENV_OVERWRITE=1 to load all variables from .env and overwrite existing env vars.
# Place a .env file in /opt/mcp-data-server/ (or project root) with lines like:
# GOOGLE_API_KEY=your-api-key-here
# OPENAI_API_KEY=your-api-key-here
# MCP_IMAGES_DIR=/taiga/ncsa/radiant/bbgp/rgpu02/owodd/mcp-images
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    try:
        _overwrite = os.getenv("DOTENV_OVERWRITE", "").strip().lower() in ("1", "true", "yes")
        with open(_env_file, 'r') as f:
            loaded_count = 0
            skipped_count = 0
            total_in_file = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if not key:
                        continue
                    total_in_file += 1
                    if not value:
                        skipped_count += 1
                        continue
                    if _overwrite or key not in os.environ:
                        os.environ[key] = value
                        loaded_count += 1
                    else:
                        skipped_count += 1
            if total_in_file > 0:
                if skipped_count and not _overwrite:
                    print(f"📝 Loaded {loaded_count} of {total_in_file} variable(s) from .env ({skipped_count} skipped: already set in environment or empty; set DOTENV_OVERWRITE=1 to use .env for all)")
                else:
                    print(f"📝 Loaded {loaded_count} environment variable(s) from .env file")
    except Exception as e:
        print(f"⚠️  Warning: Could not load .env file: {e}")

# Directory configuration
# IMAGES_DIR: default is BASE_DIR/images. Set MCP_IMAGES_DIR to Taiga mcp-images path. Two layouts supported:
#   Flat: all images in MCP_IMAGES_DIR/ (e.g. almonds_001.jpg, grapes_281.jpg) — common on Taiga.
#   Species subdirs: MCP_IMAGES_DIR/grapes/, MCP_IMAGES_DIR/carrot/, etc.
# Example: MCP_IMAGES_DIR=/taiga/ncsa/radiant/bbgp/rgpu02/owodd/mcp-images
IMAGES_DIR = Path(os.getenv("MCP_IMAGES_DIR", str(BASE_DIR / "images")))
if "MCP_IMAGES_DIR" not in os.environ and IMAGES_DIR == (BASE_DIR / "images"):
    print("📁 IMAGES_DIR: {} (set MCP_IMAGES_DIR to use Taiga, e.g. /taiga/ncsa/radiant/bbgp/rgpu02/owodd/mcp-images)".format(IMAGES_DIR))
else:
    print("📁 IMAGES_DIR: {}".format(IMAGES_DIR))
# Startup check: detect layout (species subdirs vs flat). Both are supported.
# IMAGES_LAYOUT: "auto" (detect), "species_subdir" (try subdir first), "flat" (try flat first)
IMAGES_LAYOUT = os.getenv("IMAGES_LAYOUT", "auto").strip().lower()
if IMAGES_LAYOUT not in ("auto", "species_subdir", "flat"):
    IMAGES_LAYOUT = "auto"
_IMAGES_TRY_SPECIES_FIRST = False  # set below from detection or env
try:
    _resolved = IMAGES_DIR.resolve()
    _grapes = _resolved / "grapes"
    _species_exists = _grapes.exists() and _grapes.is_dir()
    _detected_layout = "unknown"
    print("📁 IMAGES_DIR (resolved): {}".format(_resolved))
    if _species_exists:
        try:
            _sample = list(_grapes.iterdir())[:3]
            _names = [p.name for p in _sample if p.is_file()]
            print("📁 Layout: species subdirs (e.g. grapes/). Sample in grapes/: {}".format(_names if _names else "(none)"))
        except OSError:
            _names = []
            print("📁 Layout: species subdirs (e.g. grapes/). Sample: (listdir skipped)")
        _detected_layout = "species_subdir"
    else:
        # No grapes/ subdir: treat as flat. Do NOT call _resolved.iterdir() here — on large/NFS
        # dirs (e.g. /taiga/.../mcp-images) it can hang or take minutes.
        _detected_layout = "flat"
        if IMAGES_LAYOUT == "flat":
            print("📁 Layout: flat (from IMAGES_LAYOUT).")
        else:
            print("📁 Layout: flat (no grapes/ subdir). Set IMAGES_LAYOUT=flat or species_subdir to override.")
    # When layout is species_subdir (or auto and we detected it), try species subdir first when serving images
    if IMAGES_LAYOUT == "species_subdir":
        _IMAGES_TRY_SPECIES_FIRST = True
    elif IMAGES_LAYOUT == "flat":
        _IMAGES_TRY_SPECIES_FIRST = False
    else:
        _IMAGES_TRY_SPECIES_FIRST = _detected_layout == "species_subdir"
    if _IMAGES_TRY_SPECIES_FIRST:
        print("📁 Image lookup: species subdir first (e.g. grapes/grapes_285.jpg)")
    else:
        print("📁 Image lookup: flat path first (e.g. grapes_285.jpg)")
except Exception as e:
    print("📁 IMAGES_DIR resolve/list check failed: {}".format(e))
    _IMAGES_TRY_SPECIES_FIRST = False
IMAGES_TRY_SPECIES_FIRST = _IMAGES_TRY_SPECIES_FIRST
TEMPLATES_DIR = BASE_DIR / "templates"
PLUGINS_DIR = BASE_DIR / "plugins"
DATASETS_DIR = BASE_DIR / "datasets"
# Directory for MCP JSON data files (discovery uses BASE_DIR by default; can override with MCP_JSON_DIR env)
MCP_JSON_DIR = Path(os.getenv("MCP_JSON_DIR", str(BASE_DIR)))
# Optional: comma-separated extra dirs to search for *_mcp_data.json
# Example (pest subfolder): MCP_JSON_EXTRA_DIRS=/opt/mcp-data-server/mcp_json/-1423391161
# Example (wildlife/plants/domestic/livestock): add the dir where those JSONs live so they load and appear under Wildlife/Plants etc. in the UI:
#   MCP_JSON_EXTRA_DIRS=/path/to/animals_plants
# Or use a single env with multiple paths: MCP_JSON_EXTRA_DIRS=/path/to/mcp_json,/path/to/animals_plants
_env_extra = os.getenv("MCP_JSON_EXTRA_DIRS", "").strip()
MCP_JSON_EXTRA_DIRS = [Path(p.strip()) for p in _env_extra.split(",") if p.strip()] if _env_extra else []

# LLM Configuration
# Supports: Azure OpenAI, OpenAI, and Gemini
# Priority: Azure (if AZURE_OPENAI_* set) -> OpenAI (OPENAI_API_KEY) -> Gemini (GOOGLE_API_KEY) -> Metadata fallback
# Azure: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT (e.g. gpt-5-mini-2), optionally AZURE_OPENAI_API_VERSION
LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),  # OpenAI API key (also used as fallback key for Azure)
    "model": os.getenv("LLM_MODEL", "gpt-5-mini-2"),  # Model/deployment name (OpenAI model or Azure deployment)
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "provider": os.getenv("LLM_PROVIDER", "auto").lower(),  # "openai", "gemini", or "auto" (auto prefers OpenAI first)
    "enabled": os.getenv("LLM_ENABLED", "true").lower() == "true",
    "fallback_to_rules": os.getenv("LLM_FALLBACK_TO_RULES", "true").lower() == "true",
    # Azure OpenAI (optional; if set, used for OpenAI-compatible calls)
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
    "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("LLM_MODEL", "gpt-5-mini-2")),
    "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
}

# MCP Configuration
MCP_CONFIG = {
    "mcp_host": os.getenv("MCP_HOST", "0.0.0.0"),
    "mcp_port": int(os.getenv("MCP_PORT", 8188)),
    "mcp_base_url": os.getenv("MCP_BASE_URL", "http://127.0.0.1:8188")
}

WEB_CONFIG = {
    "web_host": os.getenv("WEB_HOST", "0.0.0.0"),
    "web_port": int(os.getenv("WEB_PORT", 8187)),
    "mcp_server_url": os.getenv("MCP_SERVER_URL", "https://mcp.aifarms.org/")
}

# Dataset configuration
DATASET_CONFIG = {
    "auto_discover": True,
    "supported_formats": [".json", ".csv"],
    "default_limit": 50,
    "max_limit": 1000
}

# Animals and plants: dataset key (common name) -> scientific_name.
# Data for wildlife/crops is usually keyed by common names; they need scientific names here.
# Pests are the opposite: they already have scientific names and need common names (see SPECIES_AND_PEST_NAMES.md).
SPECIES_SCIENTIFIC_NAMES = {
    # Plants / crops
    "raspberry": "Rubus idaeus",
    "strawberry": "Fragaria × ananassa",
    "strawberry_1": "Fragaria × ananassa",
    "strawberry_2": "Fragaria × ananassa",
    "strawberry_3": "Fragaria × ananassa",
    "blueberry": "Vaccinium sect. Cyanococcus",
    "carrot": "Daucus carota",
    "celery": "Apium graveolens",
    "red_leaf": "Acer rubrum",
    "romaine": "Lactuca sativa var. longifolia",
    "tomatoes": "Solanum lycopersicum",
    # Wildlife
    "striped_skunk": "Mephitis mephitis",
    "white_tailed_deer": "Odocoileus virginianus",
    "wild_turkey": "Meleagris gallopavo",
    "woodchuck": "Marmota monax",
    "bobcat": "Lynx rufus",
    "coyote": "Canis latrans",
    "red_fox": "Vulpes vulpes",
    "gray_fox": "Urocyon cinereoargenteus",
    "american_crow": "Corvus brachyrhynchos",
    "eastern_chipmunk": "Tamias striatus",
    "eastern_cottontail": "Sylvilagus floridanus",
    "eastern_fox_squirrel": "Sciurus niger",
    "eastern_gray_squirrel": "Sciurus carolinensis",
    "northern_raccoon": "Procyon lotor",
    "virginia_opossum": "Didelphis virginiana",
}

# Model configuration
MODEL_CONFIG = {
    "default_model": "baseline_classifier",
    "supported_types": ["classification", "detection", "segmentation"],
    "cache_results": True,
    "max_batch_size": 100
}

# MCP Protocol configuration
MCP_PROTOCOL_CONFIG = {
    "server_name": "AIFARMS Extensible MCP Server",
    "server_version": "2.0.0",
    "server_description": "Extensible MCP server for AIFARMS datasets and models",
    "capabilities": {
        "resources": True,
        "tools": True,
        "prompts": False,
        "agents": False
    }
}

# Croissant Crawler configuration
_croissant_hf_datasets_env = os.getenv("CROISSANT_HF_DATASETS")
CROISSANT_CRAWLER_CONFIG = {
    # Hugging Face datasets to crawl (can be overridden via environment variable CROISSANT_HF_DATASETS)
    # Format: comma-separated list, e.g., "dataset1,dataset2,dataset3"
    # Note: If auto-discovery is enabled, these will be combined with discovered datasets
    "huggingface_datasets": (
        _croissant_hf_datasets_env.split(",") if _croissant_hf_datasets_env
        else [
            "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled",
            "AgMMU/AgMMU_v1"
        ]
    ),
    # Whether to automatically discover Croissant datasets from Hugging Face API
    "auto_discover": os.getenv("CROISSANT_AUTO_DISCOVER", "true").lower() == "true",
    # Maximum number of datasets to discover from Hugging Face API
    "discovery_limit": int(os.getenv("CROISSANT_DISCOVERY_LIMIT", "100")),
    # Whether to filter discovered datasets to agriculture-related only
    "filter_agriculture": os.getenv("CROISSANT_FILTER_AGRICULTURE", "true").lower() == "true",
    # Agriculture-related keywords to search for (comma-separated)
    "agriculture_keywords": os.getenv(
        "CROISSANT_AGRICULTURE_KEYWORDS",
        "agriculture,agricultural,farming,farm,crop,plant,livestock,soil,harvest,agronomy,agtech,agrifood"
    ).split(",") if os.getenv("CROISSANT_AGRICULTURE_KEYWORDS") else [
        "agriculture", "agricultural", "farming", "farm", "crop", "plant", 
        "livestock", "soil", "harvest", "agronomy", "agtech", "agrifood"
    ],
    # Whether to create synthetic datasets for datasets without explicit Croissant metadata
    "create_synthetic": os.getenv("CROISSANT_CREATE_SYNTHETIC", "true").lower() == "true",
    # SSL verification (can be disabled for problematic portals)
    "verify_ssl": os.getenv("CROISSANT_VERIFY_SSL", "true").lower() == "true"
}

# Ensure directories exist (create parent directories if needed)
for directory in [TEMPLATES_DIR, PLUGINS_DIR, DATASETS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        # Silently fail if we can't create directories (might not have permissions)
        # This allows the module to be imported even if directories can't be created
        pass
