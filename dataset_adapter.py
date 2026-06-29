#!/usr/bin/env python3
"""
Dataset Adapter System for Extensible Dataset Type Support

This module provides an adapter pattern for different dataset types,
allowing the MCP server to handle datasets with different schemas and structures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from models import DatasetType, FilterOptions

# Keys that FilterOptions accepts. tail_positions is only valid for goat_2 and is NOT in FilterOptions.
# Animals have tail_positions; plants/pests do not. plant_states only for plants.
FILTER_OPTIONS_ALLOWED_KEYS = frozenset({
    "categories", "species", "times", "seasons", "actions", "plant_states", "collections"
})
# Schema/item field names that map to FilterOptions keys (only these are extracted for FilterOptions)
FILTER_OPTIONS_SCHEMA_FIELDS = frozenset({
    "category", "species", "time", "season", "action", "plant_state", "collection"
})


@dataclass
class DatasetSchema:
    """Schema definition for a dataset type"""
    dataset_type: DatasetType
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    filter_fields: List[str] = field(default_factory=list)
    metadata_fields: List[str] = field(default_factory=list)
    description: str = ""


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters"""
    
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
    
    @abstractmethod
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract available filter options from dataset items"""
        pass
    
    @abstractmethod
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate that an item conforms to this dataset type's schema"""
        pass
    
    @abstractmethod
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an item to a standard format for searching"""
        pass
    
    @abstractmethod
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if an item matches a search query"""
        pass
    
    @abstractmethod
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Check if an item matches the given filters"""
        pass
    
    def get_collections(self, items: List[Dict[str, Any]]) -> List[str]:
        """Extract collection names from items (default implementation)"""
        collections = set()
        for item in items:
            collection = item.get('collection') or item.get('category') or item.get('type')
            if collection:
                collections.add(collection)
        return sorted(list(collections))


def _metadata_strings(value: Any) -> List[str]:
    """Normalize a metadata value to a list of non-empty strings (handles str or list)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _metadata_str(value: Any, default: str = "") -> str:
    """Single string for comparison; if list, join with space. Used for filter matching."""
    if value is None:
        return default
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return " ".join(parts).lower() if parts else default
    s = str(value).strip()
    return s.lower() if s else default


def _get_description_str(metadata: Dict[str, Any]) -> str:
    """Get description from metadata; try common keys so scene/description matching works regardless of schema."""
    for key in ("description", "desc", "Description", "summary"):
        val = metadata.get(key)
        if val is not None and (isinstance(val, str) and val.strip() or isinstance(val, list)):
            return _metadata_str(val)
    return ""


def _get_item_description(item: Dict[str, Any]) -> str:
    """Get description from item metadata or top-level; ensures we never miss a description."""
    meta = item.get("metadata") or {}
    desc = _get_description_str(meta)
    if desc:
        return desc
    for key in ("description", "desc", "Description", "summary"):
        val = item.get(key)
        if val is not None and (isinstance(val, str) and val.strip() or isinstance(val, list)):
            return _metadata_str(val)
    return ""


def _get_item_plant_state(metadata: Dict[str, Any]) -> str:
    """Get plant_state from metadata; support both 'plant_state' and 'plant_states' keys."""
    val = metadata.get("plant_state") or metadata.get("plant_states")
    return _metadata_str(val).strip()


def _description_has_ripe_phrase(desc: str) -> bool:
    """True if description contains a ripe phrase with word boundaries (so 'unripe berries' does not match)."""
    if not desc:
        return False
    d = desc.lower()
    patterns = [
        r"\bripe\s+raspberr", r"\bripe\s+strawberr", r"\bripe\s+berr", r"\bripe\s+fruit",
        r"\bred\s+ripe", r"\bfully\s+ripe", r"\bmature\s+berry",
    ]
    return any(re.search(p, d) for p in patterns)


def _words_from_field(s: str) -> List[str]:
    """Split a field (collection, species, scientific_name) into words for type matching (e.g. 'cabbage_moth' -> ['cabbage', 'moth'])."""
    if not s or not str(s).strip():
        return []
    return [w.strip().lower() for w in str(s).replace("_", " ").replace("-", " ").split() if w.strip()]


# Canonical action keywords for strict matching: when user asks for "sleeping", exclude items that clearly say "walking".
# Do not infer sleeping from "night" — only match when action or description explicitly says sleep/rest.
ACTION_KEYWORD_MAP_FOR_STRICT = {
    "sleeping": ["sleep", "sleeping", "rest", "resting"],
    "walking": ["walk", "walking", "moving"],
    "feeding": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
    "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
    "resting": ["rest", "resting", "sleep", "sleeping"],
    "alert": ["alert", "alerts", "watch", "watching", "observing", "observe", "looking at camera", "staring at camera", "facing camera", "awake"],
    "moving": ["move", "moving", "walk", "walking"],
    "running": ["run", "running", "moving"],
    "hunting": ["hunt", "hunting"],
}


def _item_canonical_action(item_action: str, item_description: str) -> Optional[str]:
    """Return the canonical action (e.g. 'walking', 'sleeping') if the item clearly indicates one; else None.
    Uses whole-word match in description for sleep/rest and feed/eat so 'forest' does not imply 'resting', 'featuring' does not imply 'feeding'."""
    action_lower = (item_action or "").lower().strip()
    desc_lower = (item_description or "").lower()
    for canonical, variations in ACTION_KEYWORD_MAP_FOR_STRICT.items():
        if any(v in action_lower for v in variations):
            return canonical
    for canonical, variations in ACTION_KEYWORD_MAP_FOR_STRICT.items():
        if canonical in ("sleeping", "resting", "feeding", "foraging"):
            if any(_description_contains_action_word(desc_lower, v) for v in variations):
                return canonical
        else:
            if any(v in desc_lower for v in variations):
                return canonical
    return None


def _action_filter_conflicts(requested_action: str, item_canonical: Optional[str]) -> bool:
    """True if requested action (e.g. sleeping) conflicts with item's canonical action (e.g. walking)."""
    if not item_canonical:
        return False
    requested = requested_action.lower().strip()
    if requested == item_canonical:
        return False
    if requested in ("resting", "sleeping") and item_canonical in ("resting", "sleeping"):
        return False
    if requested in ("feeding", "foraging", "eating") and item_canonical in ("feeding", "foraging", "eating"):
        return False
    if requested in ("walking", "moving", "running") and item_canonical in ("walking", "moving", "running"):
        return False
    return True


def _description_indicates_awake_or_observing(description: str) -> bool:
    """True if description clearly indicates the subject is awake, observing, or alert (not sleeping).
    Used to exclude night images that are 'cat observing' when user asked for 'cat sleeping'. Do not infer sleeping from 'night'."""
    if not description or not isinstance(description, str):
        return False
    d = description.lower()
    awake_indicators = [
        "observing", "observe", "awake", "watching", "watch", "alert", "looking at", "staring at",
        "facing camera", "facing the camera", "looking toward", "staring toward", "eyes open",
        "looking at camera", "staring at camera", "vigilant", "aware of", "attention",
    ]
    return any(ind in d for ind in awake_indicators)


def _description_contains_action_word(description: str, word: str) -> bool:
    """True only if word appears as a whole word in description (not as substring of another word, e.g. 'rest' in 'forest')."""
    if not description or not word:
        return False
    d = description.lower()
    w = word.lower().strip()
    if not w:
        return False
    return bool(re.search(r"\b" + re.escape(w) + r"\b", d))


def _normalize_species_filter_val(s: str) -> str:
    """Strip leading/trailing parentheses/brackets so '(moth)' or '[moth]' from LLM becomes 'moth' for matching."""
    t = str(s).strip()
    while len(t) >= 2 and t[0] in "([{\"" and t[-1] in ")]}\"":
        t = t[1:-1].strip()
    return t.lower()


# Pest type words: when these appear as a word in collection/species/scientific_name, items match that type filter (e.g. "moth" matches "Raspberry crown moth", "cabbage_moth").
PEST_TYPE_WORDS = frozenset([
    "beetle", "beetles", "butterfly", "butterflies", "moth", "moths",
    "wasp", "wasps", "bee", "bees", "ant", "ants", "fly", "flies",
    "grasshopper", "grasshoppers", "dragonfly", "dragonflies",
    "spider", "spiders", "bug", "bugs", "insect", "insects",
])

# Short species terms that must match whole-word only in common_name/scientific_name to avoid "cat" matching "Catalpa"/"catalpae", "dog" matching "catalog", etc.
SPECIES_WHOLE_WORD_ONLY = frozenset(["cat", "cats", "dog", "dogs", "fox", "red", "fly", "bee"])


def _species_in_common_name(s: str, text: str) -> bool:
    """True if species s matches in common-name text as exact or whole word; avoids 'cat' matching 'Catalpa'."""
    if not text:
        return False
    text_lower = text.lower()
    if s == text_lower:
        return True
    words = _words_from_field(text)
    if s in words:
        return True
    if s in SPECIES_WHOLE_WORD_ONLY or len(s) <= 4:
        return False  # no substring match for short/animal terms
    return s in text_lower


class SpeciesObservationAdapter(DatasetAdapter):
    """Adapter for species observation datasets (default/legacy)"""

    def __init__(self, schema: Optional[DatasetSchema] = None):
        if schema is None:
            schema = DatasetSchema(
                dataset_type=DatasetType.WILDLIFE,
                required_fields=['id', 'collection'],
                optional_fields=['category', 'metadata'],
                filter_fields=['category', 'species', 'time', 'season', 'action', 'plant_state', 'collection'],
                metadata_fields=['species', 'action', 'time', 'season', 'scene', 'weather', 'date', 'description'],
                description="Species observation dataset with wildlife, plants, or pest observations"
            )
        super().__init__(schema)
    
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract filters for species observation datasets"""
        categories = set()
        species = set()
        times = set()
        seasons = set()
        actions = set()
        plant_states = set()
        collections = set()
        
        for item in items:
            metadata = item.get('metadata', {})
            collections.add(item.get('collection', 'unknown'))
            
            # Extract category
            if 'category' in item:
                categories.add(item['category'])
            
            # Extract time information (time can be str or list)
            for time_val in _metadata_strings(metadata.get('time', '')):
                time_info = time_val.lower()
                if 'night' in time_info or 'dark' in time_info:
                    times.add('night')
                elif 'day' in time_info or 'morning' in time_info or 'afternoon' in time_info:
                    times.add('day')
                elif 'dawn' in time_info or 'sunrise' in time_info:
                    times.add('dawn')
                elif 'dusk' in time_info or 'sunset' in time_info:
                    times.add('dusk')
                elif 'evening' in time_info or 'twilight' in time_info or 'late afternoon' in time_info:
                    times.add('evening')
            
            # Extract other filters (action, season, plant_state can be str or list)
            for action_val in _metadata_strings(metadata.get('action')):
                actions.add(action_val.lower())
            for season_val in _metadata_strings(metadata.get('season')):
                seasons.add(season_val.lower())
            for ps_val in _metadata_strings(metadata.get('plant_state')):
                plant_states.add(ps_val.lower())
            
            # Extract species - prioritize metadata.species (where MCP stores it)
            # Also check collection (may be set from species during normalization)
            species_name = metadata.get('species', '') or item.get('collection') or item.get('species', '')
            for name in _metadata_strings(species_name):
                if name:
                    species_base = name.split("_")[0].split("-")[0].strip().lower()
                    species.add(species_base)
            # Pest datasets: add common_name and common_names so pre-filtering and LLM see "moth" etc.
            for name in _metadata_strings(metadata.get('common_name')):
                if name:
                    species_base = name.split("_")[0].split("-")[0].strip().lower()
                    if len(species_base) >= 2:
                        species.add(species_base)
            for name in _metadata_strings(metadata.get('common_names')):
                if name:
                    species_base = name.split("_")[0].split("-")[0].strip().lower()
                    if len(species_base) >= 2:
                        species.add(species_base)
                    for word in name.replace("_", " ").replace("-", " ").split():
                        w = word.strip().lower()
                        if len(w) >= 2:
                            species.add(w)
            # Pest type words: add "moth"/"beetle" etc. when they appear as a word in collection/species/scientific_name (e.g. "cabbage_moth" -> moth)
            for src in (item.get('collection') or '', metadata.get('species') or '', metadata.get('scientific_name') or ''):
                for w in _words_from_field(src):
                    if w in PEST_TYPE_WORDS:
                        species.add(w)
                    elif len(w) > 1 and w.endswith('s') and w[:-1] in PEST_TYPE_WORDS:
                        species.add(w[:-1])
        
        return FilterOptions(
            categories=sorted(list(categories)),
            species=sorted(list(species)),
            times=sorted(list(times)),
            seasons=sorted(list(seasons)),
            actions=sorted(list(actions)),
            plant_states=sorted(list(plant_states)),
            collections=sorted(list(collections))
        )
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate species observation item"""
        return 'id' in item and ('collection' in item or 'category' in item)
    
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize species observation item.
        
        IMPORTANT: This method only normalizes the 'collection' field based on species metadata.
        All other metadata fields (including scene, time, season, action, etc.) are passed through
        unchanged from the original MCP data. Scene and other metadata fields are NEVER inferred
        or set as defaults - they only come from what's explicitly in the MCP data files.
        """
        normalized = item.copy()
        # Ensure collection field exists - use species from metadata if available
        if 'collection' not in normalized:
            # Try to get species from metadata first (most reliable)
            metadata = normalized.get('metadata', {})
            species = metadata.get('species', '')
            if not species and metadata.get('common_name'):
                # Livestock/goat_2 etc.: common_name is the species for filtering (e.g. "goat")
                species = _metadata_str(metadata.get('common_name')).strip()
            if species:
                # Use species as collection (e.g., "bobcat" from metadata.species or "goat" from common_name)
                normalized['collection'] = species
            else:
                # Fallback to category or unknown
                normalized['collection'] = normalized.get('category', 'unknown')
        # Note: All metadata fields (scene, time, season, action, etc.) are preserved as-is
        # from the original MCP data - we do NOT modify or infer them
        return normalized
    
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if item matches search query"""
        query_lower = query.lower()
        
        # Check collection name
        if query_lower in item.get("collection", "").lower():
            return True
        
        # Check metadata description
        metadata = item.get("metadata", {})
        if query_lower in metadata.get("description", "").lower():
            return True
        
        # Check metadata action (can be str or list)
        if query_lower in _metadata_str(metadata.get("action")):
            return True
        
        # Check metadata scene
        if query_lower in metadata.get("scene", "").lower():
            return True
        
        return False
    
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Check if item matches filters"""
        metadata = item.get('metadata', {})
        
        # Check category filter
        if filters.get("category"):
            category_filter = [c.lower() for c in filters["category"]]
            if item.get("category", "").lower() not in category_filter:
                return False
        
        # Check species filter - use exact matching based on MCP metadata
        # IMPORTANT: In MCP data, species is stored in metadata.species, not collection
        # For pests, also match metadata.common_name and metadata.common_names (e.g. "moth")
        if filters.get("species"):
            species_filter = [x for s in filters["species"] if (x := _normalize_species_filter_val(s))]
            
            # Primary: Check metadata.species (this is where species is actually stored in MCP data)
            item_species_meta = _metadata_str(metadata.get("species")).strip()
            
            # Secondary: Check collection (may be set from species during normalization)
            item_collection = item.get("collection", "").lower().strip()
            
            # Tertiary: Check top-level species field (if it exists)
            item_species_top = item.get("species", "").lower().strip()
            
            # Pest/common name: single common_name and list common_names (e.g. ["Raspberry crown moth", "moth"])
            item_common_name = _metadata_str(metadata.get("common_name")).strip()
            item_common_names = _metadata_strings(metadata.get("common_names"))
            item_common_names_lower = [s.lower() for s in item_common_names]
            
            # Match if any of these match the filter
            # This ensures "bobcat" matches items with metadata.species="bobcat" but NOT "red_fox" or "strawberry"
            species_matches = False
            for species in species_filter:
                species_lower = species.lower().strip()
                # Normalize for comparison: no spaces/underscores/hyphens so "Polistes gallicus" matches "Polistes_gallicus"
                species_normalized = species_lower.replace("_", "").replace("-", "").replace(" ", "")
                
                # Normalize item values too
                item_collection_normalized = item_collection.replace("_", "").replace("-", "").replace(" ", "")
                item_species_meta_normalized = item_species_meta.replace("_", "").replace("-", "").replace(" ", "")
                item_species_top_normalized = item_species_top.replace("_", "").replace("-", "").replace(" ", "")
                
                # Priority: metadata.species is most reliable (this is where MCP stores it)
                if (item_species_meta == species_lower or
                    item_species_meta_normalized == species_normalized):
                    species_matches = True
                    break
                
                # Fallback: collection (may be derived from species)
                if (item_collection == species_lower or 
                    item_collection.startswith(species_lower + "_") or
                    item_collection.startswith(species_lower + "-") or
                    item_collection_normalized == species_normalized):
                    species_matches = True
                    break
                
                # Fallback: top-level species field
                if (item_species_top == species_lower or
                    item_species_top_normalized == species_normalized):
                    species_matches = True
                    break
                
                # Pest: common_name or common_names (e.g. "moth" in common_names or in "Raspberry crown moth")
                # Use whole-word match for short species terms (cat, dog) so "cat" does not match "Catalpa" or "catalpae"
                if _species_in_common_name(species_lower, item_common_name):
                    species_matches = True
                    break
                for cn in item_common_names_lower:
                    if _species_in_common_name(species_lower, cn):
                        species_matches = True
                        break
                # Pest type word: match when "moth"/"beetle" etc. appears as a word in collection, metadata.species, or scientific_name (e.g. cabbage_moth, Raspberry crown moth)
                if species_lower in PEST_TYPE_WORDS or (species_lower.endswith("s") and species_lower[:-1] in PEST_TYPE_WORDS):
                    item_scientific = _metadata_str(metadata.get("scientific_name")).strip()
                    for part in _words_from_field(item_collection) + _words_from_field(item_species_meta) + _words_from_field(item_scientific):
                        if part == species_lower or (species_lower.endswith("s") and part == species_lower[:-1]) or (part.endswith("s") and part[:-1] == species_lower):
                            species_matches = True
                            break
                if species_matches:
                    break
            
            if not species_matches:
                return False
        
        # Check time filter (normalize filter values: "Nighttime", "night time" -> night; check time field and description)
        if filters.get("time"):
            time_filter = [t.lower().strip() for t in filters["time"]]
            time_info = _metadata_str(metadata.get("time")).lower()
            description = _metadata_str(metadata.get("description", "")).lower()
            matched = False
            for time_val in time_filter:
                # Normalize: "nighttime", "night time", "night" all mean night
                is_night = "night" in time_val or "dark" in time_val
                is_day = "day" in time_val or "morning" in time_val or "afternoon" in time_val or "daytime" in time_val
                is_dawn = "dawn" in time_val or "sunrise" in time_val
                is_dusk = "dusk" in time_val or "sunset" in time_val
                is_evening = "evening" in time_val or "twilight" in time_val
                # Check time field (exclude "daytime" when matching night - "daytime" contains substring "night")
                if is_night and "daytime" not in time_info and ("night" in time_info or "dark" in time_info):
                    matched = True
                    break
                # In description, match only "night" (not "dark") to avoid matching "dark-colored bird" etc.
                if is_night and "daytime" not in description and "night" in description:
                    matched = True
                    break
                # Daytime: require day/morning/afternoon/daytime in time or description, AND exclude items that indicate night
                if is_day and ("day" in time_info or "morning" in time_info or "afternoon" in time_info or "daytime" in time_info):
                    if ("night" in time_info or "nighttime" in time_info) and "daytime" not in time_info:
                        continue  # item is night, not day — skip
                    matched = True
                    break
                if is_dawn and ("dawn" in time_info or "sunrise" in time_info):
                    matched = True
                    break
                if is_dusk and ("dusk" in time_info or "sunset" in time_info):
                    matched = True
                    break
                if is_evening and ("evening" in time_info or "dusk" in time_info or "sunset" in time_info or "twilight" in time_info or "late afternoon" in time_info):
                    matched = True
                    break
                # Check description if not matched in time field
                if is_day and ("day" in description or "morning" in description or "afternoon" in description or "daytime" in description):
                    if "night" in description:
                        continue  # description says night — do not count as daytime match
                    matched = True
                    break
                if is_dawn and ("dawn" in description or "sunrise" in description):
                    matched = True
                    break
                if is_dusk and ("dusk" in description or "sunset" in description):
                    matched = True
                    break
                if is_evening and ("evening" in description or "dusk" in description or "sunset" in description or "twilight" in description or "late afternoon" in description):
                    matched = True
                    break
            if not matched:
                return False
        
        # Check season filter
        if filters.get("season"):
            season_filter = [s.lower() for s in filters["season"]]
            item_season = _metadata_str(metadata.get("season"))
            if not any(season.lower() in item_season for season in season_filter):
                return False
        
        # Check plant_state filter (uses both plant_state field and description)
        if filters.get("plant_state"):
            raw_ps = filters["plant_state"]
            plant_state_filter = [str(ps).lower().strip() for ps in (raw_ps if isinstance(raw_ps, list) else [raw_ps]) if ps]
            item_plant_state = _get_item_plant_state(metadata)
            item_description = _get_item_description(item)
            if isinstance(item_description, str):
                item_description = item_description.lower()
            else:
                item_description = ""
            
            print(f"      🔍 Checking plant_state filter: {plant_state_filter} against item {item.get('id', 'unknown')}")
            print(f"         Item plant_state: '{item_plant_state}', description: {item_description[:100]}...")
            
            # Define opposite states to exclude
            opposite_states = {
                "ripe": ["unripe", "immature", "green"],
                "unripe": ["ripe", "mature", "red"],
                "mature": ["unripe", "immature", "green"],
                "green": ["ripe", "mature", "red"],
                "red": ["unripe", "green", "immature"]
            }
            
            plant_state_matches = False
            for plant_state in plant_state_filter:
                plant_state_lower = plant_state.lower().strip()
                
                # Early exact match: if metadata says unripe and user asked for unripe, accept immediately (no exclusion logic)
                if plant_state_lower == "unripe" and item_plant_state == "unripe":
                    plant_state_matches = True
                    print(f"         ✅ Matched: exact plant_state 'unripe' (metadata)")
                    break
                
                # Unripe: allow "mixed" if description mentions unripe (image has unripe berries even if some ripe too)
                if plant_state_lower == "unripe" and item_plant_state == "mixed":
                    unripe_mixed_phrases = [
                        "unripe", "unripened",
                        "green berr", "green fruit",
                        "developing fruit", "developing berr", "developing raspberr",
                        "immature",
                    ]
                    if any(p in item_description for p in unripe_mixed_phrases):
                        plant_state_matches = True
                        print(f"         ✅ Matched: description contains unripe-related phrase, filter is 'unripe' (mixed image OK)")
                        break
                # Unripe should also match early growth stages: blooming, buds (semantically pre-ripe)
                if plant_state_lower == "unripe" and item_plant_state in ("blooming", "buds", "bud"):
                    plant_state_matches = True
                    print(f"         ✅ Matched: filter 'unripe' includes early growth '{item_plant_state}'")
                    break
                # Ripe: allow "mixed" if description mentions ripe (image has ripe berries even if some unripe too)
                if plant_state_lower == "ripe" and item_plant_state == "mixed":
                    if _description_has_ripe_phrase(item_description):
                        plant_state_matches = True
                        print(f"         ✅ Matched: description contains ripe, filter is 'ripe' (mixed image OK)")
                        break
                # Ripe: exclude when description clearly says unripe (or unripened) but has no ripe — e.g. "flowers and buds and unripe berries"
                if plant_state_lower == "ripe" and (("unripe" in item_description or "unripened" in item_description)):
                    has_ripe_word = bool(re.search(r"\bripe\b", item_description))
                    has_ripe_phrase = _description_has_ripe_phrase(item_description)
                    if not has_ripe_word and not has_ripe_phrase:
                        print(f"         ❌ Excluded: description says unripe but has no ripe, filter is 'ripe'")
                        plant_state_matches = False
                        continue
                # Exclude "mixed" when searching for specific states (except unripe/ripe + mixed when description supports it, handled above)
                if plant_state_lower in ["ripe", "unripe", "mature", "blooming", "fruiting"]:
                    if item_plant_state == "mixed":
                        print(f"         ❌ Excluded: item has 'mixed' plant_state, but filter is '{plant_state_lower}'")
                        plant_state_matches = False
                        continue
                    # When filter is "ripe", exclude "fruiting" unless description clearly says ripe
                    if plant_state_lower == "ripe" and item_plant_state == "fruiting":
                        if not _description_has_ripe_phrase(item_description):
                            print(f"         ❌ Excluded: item has 'fruiting' plant_state and description doesn't say ripe, filter is 'ripe'")
                            plant_state_matches = False
                            continue
                
                # When filter is "ripe", exclude items whose description suggests mixed/unripe (unless metadata says "ripe")
                if plant_state_lower == "ripe" and item_plant_state != "ripe":
                    mixed_desc_phrases = [
                        "varying stages", "various stages", "various ripening", "at various ripening", "at various stages",
                        "mix of unripe and ripe", "unripe and ripe", "stages of ripeness", "varying ripeness",
                        "unripe raspberr", "unripe berry", "unripe berries", "developing raspberr", "developing fruit",
                        "developing berries", "different stages", "multiple stages", "various stages of", "ripening stages",
                    ]
                    if any(p in item_description for p in mixed_desc_phrases):
                        print(f"         ❌ Excluded: description suggests mixed/unripe, filter is 'ripe'")
                        plant_state_matches = False
                        continue
                
                # Check for opposite states - exclude items that explicitly have opposite state
                if plant_state_lower in opposite_states:
                    opposites = opposite_states[plant_state_lower]
                    # Check plant_state field for opposite
                    if item_plant_state in opposites:
                        print(f"         ❌ Excluded: item has opposite plant_state '{item_plant_state}', filter is '{plant_state_lower}'")
                        plant_state_matches = False
                        continue
                    # When filter is 'ripe' or 'unripe' and item explicitly matches, trust metadata — skip description opposite check
                    if not ((plant_state_lower == "ripe" and item_plant_state == "ripe") or (plant_state_lower == "unripe" and item_plant_state == "unripe")):
                        # Check description for opposite keywords (but only if they're clearly about the fruit/berry, not leaves)
                        for opposite in opposites:
                            if opposite == "green" and "green leaves" in item_description:
                                continue
                            # When filter is "unripe", "immature" means unripe — don't treat "mature" inside "immature" as opposite
                            if plant_state_lower == "unripe" and opposite == "mature" and "immature" in item_description:
                                continue
                            opposite_patterns = [
                                f"{opposite} fruit", f"{opposite} berry", f"{opposite} berries",
                                f"{opposite} raspberr", f"{opposite} strawberr", f"{opposite} blueberr"
                            ]
                            if any(pattern in item_description for pattern in opposite_patterns):
                                print(f"         ❌ Excluded: description contains opposite state '{opposite}' in fruit context, filter is '{plant_state_lower}'")
                                plant_state_matches = False
                                break
                        if not plant_state_matches:
                            continue
                
                # Direct match in plant_state field (exact match preferred)
                if plant_state_lower == item_plant_state:
                    plant_state_matches = True
                    print(f"         ✅ Matched: exact plant_state match '{item_plant_state}'")
                    break
                elif item_plant_state and plant_state_lower in item_plant_state:
                    plant_state_matches = True
                    print(f"         ✅ Matched: plant_state contains '{plant_state_lower}'")
                    break
                
                # SPECIAL HANDLING FOR "green" - must be specific to fruits/berries, not leaves
                # This check must come BEFORE the general keyword check to avoid matching "green leaves"
                if plant_state_lower == "green":
                    # Look for green in context of fruits/berries, not just "green leaves"
                    green_contexts = ["green fruit", "green berry", "green berries", "green raspberr", "unripe", "unripe fruit", "unripe berries", "unripe raspberr"]
                    if any(ctx in item_description for ctx in green_contexts):
                        plant_state_matches = True
                        print(f"         ✅ Matched: found green context in description")
                        break
                    # Also check if "green" appears near fruit/berry keywords (but not "green leaves")
                    if re.search(r'green\s+(fruit|berry|berries|raspberr)', item_description):
                        plant_state_matches = True
                        print(f"         ✅ Matched: 'green' near fruit/berry keywords")
                        break
                    # Explicitly exclude if only "green leaves" is present
                    if "green leaves" in item_description and not any(ctx in item_description for ctx in ["green fruit", "green berry", "green berries", "unripe"]):
                        # Skip this item - only has "green leaves", not green fruits
                        print(f"         ❌ Excluded: only has 'green leaves', no green fruits")
                        plant_state_matches = False
                        break  # Break out of plant_state loop - this item doesn't match
                
                # Check description for plant state keywords (for non-green states)
                # Common mappings: "green" -> "unripe", "green", "immature"
                plant_state_keywords = {
                    "ripe": ["ripe", "mature", "ready"],
                    "unripe": ["unripe", "immature"],
                    "mature": ["mature", "ripe", "ready"],
                    "blooming": ["blooming", "flowering", "bloom", "flower"],
                    "fruiting": ["fruiting", "fruits", "berries"],
                    "growing": ["growing", "developing"]
                }
                
                # Check if plant_state matches any keyword variations (skip "green" - already handled above)
                # For "ripe" use whole-word match so "unripe" does not match
                if plant_state_lower != "green":
                    for keyword, variations in plant_state_keywords.items():
                        if plant_state_lower == keyword or plant_state_lower in variations:
                            if keyword == "ripe" or (plant_state_lower == "mature" and "ripe" in variations):
                                # Whole-word "ripe" so "unripe" doesn't match; other terms can be substring
                                if re.search(r"\bripe\b", item_description) or any(
                                    v in item_description for v in variations if v != "ripe"
                                ):
                                    plant_state_matches = True
                                    break
                            elif any(v in item_description for v in variations):
                                plant_state_matches = True
                                break
                        if plant_state_matches:
                            break
                
                # Direct check: if plant_state word is in description (for non-green states)
                # Use whole-word match for "ripe" so "unripe" / "unripened" don't match the ripe filter
                if plant_state_lower != "green":
                    if plant_state_lower == "ripe":
                        if re.search(r"\bripe\b", item_description):
                            plant_state_matches = True
                            break
                    elif plant_state_lower in item_description:
                        plant_state_matches = True
                        break
                
                if plant_state_matches:
                    break
            
            if not plant_state_matches:
                return False
        
        # Check action filter - check both action field and description (action can be str or list)
        if filters.get("action") or filters.get("Action"):
            raw_action = filters.get("action") or filters.get("Action")
            # Normalize to list and split comma-separated values (e.g. "foraging, eating" -> ["foraging", "eating"])
            _raw_list = raw_action if isinstance(raw_action, list) else [raw_action]
            action_filter = []
            for a in _raw_list:
                for part in str(a).split(","):
                    p = part.strip().lower()
                    if p:
                        action_filter.append(p)
            item_action = _metadata_str(metadata.get("action"))
            item_description = metadata.get("description", "").lower()
            # Strict: exclude items that clearly have a different primary action (e.g. walking when user asked for sleeping)
            item_canonical = _item_canonical_action(item_action, item_description)
            for action in action_filter:
                if _action_filter_conflicts(action, item_canonical):
                    return False
            # When user asked for sleeping/resting, do not include items where description says awake/observing (e.g. cat at night observing)
            if action_filter and any(a in ("sleeping", "resting") for a in action_filter):
                if _description_indicates_awake_or_observing(item_description):
                    return False
            # Action keyword variations map (same as in llm_service.py)
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
            
            action_matches = False
            for action in action_filter:
                action_lower = action.lower().strip()
                
                # Direct match in action field
                if action_lower in item_action:
                    action_matches = True
                    break
                # Eating/feeding/foraging are synonyms: match if item's action field is any of them
                feed_forage_synonyms = {"eating", "feeding", "foraging", "eat", "feed", "forage"}
                if action_lower in feed_forage_synonyms:
                    item_action_lower = item_action.lower().strip()
                    if item_action_lower in feed_forage_synonyms or any(s in item_action_lower for s in feed_forage_synonyms):
                        action_matches = True
                        break
                
                # Check description for action or variations
                for keyword, variations in action_keyword_map.items():
                    if keyword == action_lower or action_lower in variations:
                        if keyword in ("sleeping", "resting", "feeding", "foraging"):
                            if any(_description_contains_action_word(item_description, v) for v in variations):
                                action_matches = True
                                break
                        else:
                            if any(v in item_description for v in variations):
                                action_matches = True
                                break
                        if not action_matches:
                            action_base = action_lower.rstrip('ing').rstrip('ed')
                            if len(action_base) >= 3 and _description_contains_action_word(item_description, action_base):
                                action_matches = True
                                break
                    if action_matches:
                        break
                
                # Direct check: whole-word for sleeping/resting/feeding/foraging/eating (so "eat" not in "featuring")
                if action_lower in ("sleeping", "resting", "feeding", "foraging", "eating"):
                    if _description_contains_action_word(item_description, action_lower):
                        action_matches = True
                        break
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if len(action_base) >= 3 and _description_contains_action_word(item_description, action_base):
                        action_matches = True
                        break
                else:
                    if action_lower in item_description:
                        action_matches = True
                        break
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if action_base in item_description and len(action_base) >= 3:
                        action_matches = True
                        break
                
                if action_matches:
                    break
            
            if not action_matches:
                return False
        
        # tail_positions: only used for goat_2. Match metadata.tail_position (e.g. "tail pointing up"),
        # metadata.category ("tail-up" / "tail-down"), or description ("tail up" / "tail down").
        if filters.get("tail_positions"):
            tail_filter = [t.lower().strip() for t in filters["tail_positions"] if t]
            if tail_filter:
                raw_tail = _metadata_str(metadata.get("tail_positions") or metadata.get("tail_position")).lower().strip()
                item_tail = ""
                _undet_keys = ("undetermined", "undetermend", "indeterminate", "unknown", "not visible", "unclear", "obscured")
                if raw_tail:
                    # Normalize phrases like "tail pointing up" / "tail pointing down" to "up" / "down";
                    # explicit "undetermined"/"unknown" annotations map to "undetermined".
                    if any(k in raw_tail for k in _undet_keys):
                        item_tail = "undetermined"
                    elif "up" in raw_tail:
                        item_tail = "up"
                    elif "down" in raw_tail:
                        item_tail = "down"
                if not item_tail:
                    # Fallback: category can be "tail-up" / "tail-down" / "tail-undetermined"
                    cat = _metadata_str(metadata.get("category")).lower().strip()
                    if cat == "tail-up" or cat.endswith("tail-up"):
                        item_tail = "up"
                    elif cat == "tail-down" or cat.endswith("tail-down"):
                        item_tail = "down"
                    elif "tail-undetermined" in cat or any(k in cat for k in _undet_keys):
                        item_tail = "undetermined"
                if not item_tail:
                    # Fallback: description only (e.g. "tail up", "tail down")
                    desc = (metadata.get("description") or "").lower()
                    if "tail up" in desc or "tail is up" in desc or "with tail up" in desc:
                        item_tail = "up"
                    elif "tail down" in desc or "tail is down" in desc or "with tail down" in desc:
                        item_tail = "down"
                if not item_tail:
                    return False
                if not any(t in item_tail or item_tail in t for t in tail_filter):
                    return False
        
        # Check scene filter - STRICT matching required
        if filters.get("scene"):
            scene_filter = [s.lower().strip() for s in filters["scene"]]
            item_scene = _metadata_str(metadata.get("scene")).strip()
            item_description = _get_item_description(item)
            
            scene_matches = False
            
            # Scene synonyms: "field" filter matches metadata.scene meadow/pasture/grassland
            scene_synonyms = {
                "field": ["meadow", "pasture", "grassland", "outdoor", "field"],
                "outdoor": ["field", "meadow", "pasture", "grassland", "outdoor"],
                "indoor": ["indoor", "inside"],
            }
            
            # First, check if scene field matches (exact, substring, or synonym)
            if item_scene:
                for scene in scene_filter:
                    scene_lower = scene.lower().strip()
                    # Exact match or substring match
                    if scene_lower == item_scene or scene_lower in item_scene or item_scene in scene_lower:
                        # VALIDATION: Check description for contradictions
                        indoor_keywords = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                        outdoor_keywords = ["outdoor", "outside", "field", "meadow", "pasture", "open", "exterior"]
                        
                        # If searching for "field" but description says "indoor", reject
                        if scene_lower == "field":
                            if any(keyword in item_description for keyword in indoor_keywords):
                                # Description contradicts scene metadata - reject this item
                                print(f"      ⚠️  Scene mismatch: scene='{item_scene}' but description indicates indoor")
                                continue  # Skip this scene match, try next
                        
                        # If searching for "indoor" but description says outdoor keywords, reject
                        if scene_lower in ["indoor", "inside"]:
                            if any(keyword in item_description for keyword in outdoor_keywords):
                                # Description contradicts scene metadata - reject this item
                                print(f"      ⚠️  Scene mismatch: scene='{item_scene}' but description indicates outdoor")
                                continue  # Skip this scene match, try next
                        
                        scene_matches = True
                        break
                    # Semantic equivalence: "field" <-> "outdoor" (field implies outdoor; outdoor may be field)
                    if (scene_lower == "field" and item_scene == "outdoor") or (scene_lower == "outdoor" and item_scene == "field"):
                        scene_matches = True
                        break
                    # Field synonyms: filter "field" matches metadata.scene "meadow", "pasture", "grassland"
                    if scene_lower in scene_synonyms and item_scene in scene_synonyms.get(scene_lower, []):
                        # Reject if description clearly contradicts (e.g. scene=meadow but description says indoor)
                        if scene_lower == "field":
                            indoor_kw = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                            if any(k in item_description for k in indoor_kw):
                                continue
                        scene_matches = True
                        break
            
            # If scene field doesn't match, also check description for scene keywords
            if not scene_matches and item_description:
                scene_keyword_map = {
                    "field": ["field", "fields", "meadow", "pasture", "open field", "grassland", "grazing", "grassy field", "grassy pasture", "agricultural field", "open fields"],
                    "forest": ["forest", "woodland", "woods", "trees"],
                    "garden": ["garden", "garden area"],
                    "farm": ["farm", "farmland", "farm area"],
                    "indoor": ["indoor", "inside", "interior", "barn", "shed", "building"],
                    "outdoor": ["outdoor", "outside", "exterior", "open air"]
                }
                
                for scene in scene_filter:
                    scene_lower = scene.lower().strip()
                    if scene_lower in scene_keyword_map:
                        keywords = scene_keyword_map[scene_lower]
                        desc_lower = item_description.lower() if isinstance(item_description, str) else item_description
                        if any(keyword in desc_lower for keyword in keywords):
                            scene_matches = True
                            break
            
            if not scene_matches:
                return False
        
        # Check weather filter (substring match on metadata.weather and description)
        if filters.get("weather"):
            weather_filter = [w.lower().strip() for w in filters["weather"]]
            item_weather = _metadata_str(metadata.get("weather", "")).lower()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            weather_matches = any(
                w in item_weather or w in item_description
                for w in weather_filter
            )
            if not weather_matches:
                return False
        
        return True


class GenericDatasetAdapter(DatasetAdapter):
    """Generic adapter for custom dataset types with flexible schema"""

    def __init__(self, schema: DatasetSchema):
        super().__init__(schema)
        # Build dynamic filter extraction based on schema
        self.filter_field_map = {field: field for field in schema.filter_fields}
    
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract filters dynamically based on schema. Only FilterOptions-supported fields are included (tail_positions is only valid for goat_2 and is not in FilterOptions)."""
        # Only collect fields that FilterOptions supports; ignore tail_positions etc.
        allowed_fields = [f for f in self.schema.filter_fields if f in FILTER_OPTIONS_SCHEMA_FIELDS]
        filter_values = {field: set() for field in allowed_fields}
        collections = set()
        
        for item in items:
            # Extract collection
            collection = item.get('collection') or item.get('category') or item.get('type', 'unknown')
            collections.add(collection)
            
            # Extract filter values based on allowed schema fields only
            for field in allowed_fields:
                value = item.get(field) or item.get('metadata', {}).get(field)
                if value:
                    if isinstance(value, list):
                        filter_values[field].update(str(v).lower() for v in value)
                    else:
                        filter_values[field].add(str(value).lower())
            # Pest datasets: add common_name and common_names to species for pre-filtering
            if 'species' in filter_values:
                meta = item.get('metadata', {})
                for name in _metadata_strings(meta.get('common_name')):
                    if name and len(name) >= 2:
                        filter_values['species'].add(name.strip().lower())
                for name in _metadata_strings(meta.get('common_names')):
                    if name and len(name) >= 2:
                        filter_values['species'].add(name.strip().lower())
                    for word in name.replace("_", " ").replace("-", " ").split():
                        w = word.strip().lower()
                        if len(w) >= 2:
                            filter_values['species'].add(w)
                # Pest type words from collection/species/scientific_name (e.g. "cabbage_moth" -> moth)
                for src in (item.get('collection') or '', meta.get('species') or '', meta.get('scientific_name') or ''):
                    for w in _words_from_field(src):
                        if w in PEST_TYPE_WORDS:
                            filter_values['species'].add(w)
                        elif len(w) > 1 and w.endswith('s') and w[:-1] in PEST_TYPE_WORDS:
                            filter_values['species'].add(w[:-1])
        
        # Convert to FilterOptions format: only pass keys that FilterOptions accepts (ignore tail_positions etc.)
        def _to_list(s: Set) -> List[str]:
            return sorted(list(s)) if s else []
        result = FilterOptions(
            categories=_to_list(filter_values.get('category', set())),
            species=_to_list(filter_values.get('species', set()) or collections),
            times=_to_list(filter_values.get('time', set())),
            seasons=_to_list(filter_values.get('season', set())),
            actions=_to_list(filter_values.get('action', set())),
            plant_states=_to_list(filter_values.get('plant_state', set())),
            collections=_to_list(collections),
        )
        
        return result
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate item against schema"""
        # Check required fields
        for field in self.schema.required_fields:
            if field not in item and field not in item.get('metadata', {}):
                return False
        return True
    
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize item to standard format.
        
        IMPORTANT: This method only normalizes the 'collection' field based on species metadata.
        All other metadata fields (including scene, time, season, action, etc.) are passed through
        unchanged from the original MCP data. Scene and other metadata fields are NEVER inferred
        or set as defaults - they only come from what's explicitly in the MCP data files.
        """
        normalized = item.copy()
        # Ensure basic structure
        if 'collection' not in normalized:
            # Try to get species from metadata first (most reliable for MCP data)
            metadata = normalized.get('metadata', {})
            species = metadata.get('species', '')
            if species:
                # Use species as collection (e.g., "raspberry" from metadata.species)
                normalized['collection'] = species
            else:
                # Fallback to category or type
                normalized['collection'] = normalized.get('category') or normalized.get('type', 'unknown')
        # Note: All metadata fields (scene, time, season, action, etc.) are preserved as-is
        # from the original MCP data - we do NOT modify or infer them
        return normalized
    
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Generic query matching - search in all text fields"""
        query_lower = query.lower()
        
        # Search in all fields
        for key, value in item.items():
            if isinstance(value, str) and query_lower in value.lower():
                return True
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and query_lower in sub_value.lower():
                        return True
        
        return False
    
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Generic filter matching"""
        metadata = item.get('metadata', {})
        
        # Special handling for species filter - check collection, metadata.species, common_name, common_names
        if filters.get("species"):
            species_filter = [x for s in filters["species"] if (x := _normalize_species_filter_val(s))]
            # Check collection (primary identifier)
            item_collection = item.get("collection", "").lower().strip()
            # Check top-level species
            item_species = item.get("species", "").lower().strip()
            # Check metadata species (can be str or list)
            item_species_meta = _metadata_str(metadata.get("species")).strip()
            # Pest: common_name and common_names (e.g. "moth")
            item_common_name = _metadata_str(metadata.get("common_name")).strip()
            item_common_names = _metadata_strings(metadata.get("common_names"))
            item_common_names_lower = [s.lower() for s in item_common_names]
            
            species_matches = False
            for species in species_filter:
                species_lower = species.lower().strip()
                # Normalize for comparison: no spaces/underscores/hyphens so "Polistes gallicus" matches "Polistes_gallicus"
                species_normalized = species_lower.replace("_", "").replace("-", "").replace(" ", "")
                
                # Normalize item values too
                item_collection_normalized = item_collection.replace("_", "").replace("-", "").replace(" ", "")
                item_species_normalized = item_species.replace("_", "").replace("-", "").replace(" ", "")
                item_species_meta_normalized = item_species_meta.replace("_", "").replace("-", "").replace(" ", "")
                
                if (item_collection == species_lower or
                    item_collection.startswith(species_lower + "_") or
                    item_collection.startswith(species_lower + "-") or
                    item_species == species_lower or
                    item_species_meta == species_lower or
                    # Normalized matching (handles case, space, underscore variations)
                    item_collection_normalized == species_normalized or
                    item_species_normalized == species_normalized or
                    item_species_meta_normalized == species_normalized):
                    species_matches = True
                    break
                
                # Pest: match common_name or common_names (e.g. "moth" in common_names or in "Raspberry crown moth")
                # Use whole-word match so "cat" does not match "Catalpa"/"catalpae"
                if _species_in_common_name(species_lower, item_common_name):
                    species_matches = True
                    break
                for cn in item_common_names_lower:
                    if _species_in_common_name(species_lower, cn):
                        species_matches = True
                        break
                # Pest type word: match when "moth"/"beetle" etc. appears as word in collection/species/scientific_name
                if species_lower in PEST_TYPE_WORDS or (species_lower.endswith("s") and species_lower[:-1] in PEST_TYPE_WORDS):
                    item_scientific = _metadata_str(metadata.get("scientific_name")).strip()
                    for part in _words_from_field(item_collection) + _words_from_field(item_species_meta) + _words_from_field(item_scientific):
                        if part == species_lower or (species_lower.endswith("s") and part == species_lower[:-1]) or (part.endswith("s") and part[:-1] == species_lower):
                            species_matches = True
                            break
                if species_matches:
                    break
            
            if not species_matches:
                return False
        
        # Special handling for time filter (normalize "Nighttime"/"night time" -> night; check time field and description)
        if filters.get("time"):
            time_filter = [t.lower().strip() for t in filters["time"]]
            time_info = _metadata_str(metadata.get("time")).lower()
            description = _metadata_str(metadata.get("description", "")).lower()
            time_matched = False
            for time_val in time_filter:
                is_night = "night" in time_val or "dark" in time_val
                is_day = "day" in time_val or "morning" in time_val or "afternoon" in time_val or "daytime" in time_val
                is_dawn = "dawn" in time_val or "sunrise" in time_val
                is_dusk = "dusk" in time_val or "sunset" in time_val
                is_evening = "evening" in time_val or "twilight" in time_val
                # Exclude "daytime" when matching night (substring "night" appears in "daytime")
                if is_night and "daytime" not in time_info and ("night" in time_info or "dark" in time_info):
                    time_matched = True
                    break
                # In description, match only "night" (not "dark") to avoid matching "dark-colored bird" etc.
                if is_night and "daytime" not in description and "night" in description:
                    time_matched = True
                    break
                # Daytime: require day in time/description and exclude items that indicate night
                if is_day and ("day" in time_info or "morning" in time_info or "afternoon" in time_info or "daytime" in time_info):
                    if ("night" in time_info or "nighttime" in time_info) and "daytime" not in time_info:
                        continue
                    time_matched = True
                    break
                if is_dawn and ("dawn" in time_info or "sunrise" in time_info):
                    time_matched = True
                    break
                if is_dusk and ("dusk" in time_info or "sunset" in time_info):
                    time_matched = True
                    break
                if is_evening and ("evening" in time_info or "dusk" in time_info or "sunset" in time_info or "twilight" in time_info or "late afternoon" in time_info):
                    time_matched = True
                    break
                if is_day and ("day" in description or "morning" in description or "afternoon" in description or "daytime" in description):
                    if "night" in description:
                        continue
                    time_matched = True
                    break
                if is_dawn and ("dawn" in description or "sunrise" in description):
                    time_matched = True
                    break
                if is_dusk and ("dusk" in description or "sunset" in description):
                    time_matched = True
                    break
                if is_evening and ("evening" in description or "dusk" in description or "sunset" in description or "twilight" in description or "late afternoon" in description):
                    time_matched = True
                    break
            if not time_matched:
                return False
        
        # Special handling for action filter - check both action field and description (action can be str or list)
        if filters.get("action") or filters.get("Action"):
            raw_action = filters.get("action") or filters.get("Action")
            # Normalize to list and split comma-separated values (e.g. "foraging, eating" -> ["foraging", "eating"])
            _raw_list = raw_action if isinstance(raw_action, list) else [raw_action]
            action_filter = []
            for a in _raw_list:
                for part in str(a).split(","):
                    p = part.strip().lower()
                    if p:
                        action_filter.append(p)
            item_action = _metadata_str(metadata.get("action"))
            item_description = metadata.get("description", "").lower()
            # Strict: exclude items that clearly have a different primary action (e.g. walking when user asked for sleeping)
            item_canonical = _item_canonical_action(item_action, item_description)
            for action in action_filter:
                if _action_filter_conflicts(action, item_canonical):
                    return False
            # When user asked for sleeping/resting, do not include items where description says awake/observing (e.g. cat at night observing)
            if action_filter and any(a in ("sleeping", "resting") for a in action_filter):
                if _description_indicates_awake_or_observing(item_description):
                    return False
            # Action keyword variations map (same as in llm_service.py)
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
            
            action_matches = False
            for action in action_filter:
                action_lower = action.lower().strip()
                
                # Direct match in action field
                if action_lower in item_action:
                    action_matches = True
                    break
                # Eating/feeding/foraging are synonyms: match if item's action field is any of them
                feed_forage_synonyms = {"eating", "feeding", "foraging", "eat", "feed", "forage"}
                if action_lower in feed_forage_synonyms:
                    item_action_lower = item_action.lower().strip()
                    if item_action_lower in feed_forage_synonyms or any(s in item_action_lower for s in feed_forage_synonyms):
                        action_matches = True
                        break
                
                # Check description for action or variations
                for keyword, variations in action_keyword_map.items():
                    if keyword == action_lower or action_lower in variations:
                        if keyword in ("sleeping", "resting", "feeding", "foraging"):
                            if any(_description_contains_action_word(item_description, v) for v in variations):
                                action_matches = True
                                break
                        else:
                            if any(v in item_description for v in variations):
                                action_matches = True
                                break
                        if not action_matches:
                            action_base = action_lower.rstrip('ing').rstrip('ed')
                            if len(action_base) >= 3 and _description_contains_action_word(item_description, action_base):
                                action_matches = True
                                break
                    if action_matches:
                        break
                
                # Direct check: whole-word for sleeping/resting/feeding/foraging/eating (so "eat" not in "featuring")
                if action_lower in ("sleeping", "resting", "feeding", "foraging", "eating"):
                    if _description_contains_action_word(item_description, action_lower):
                        action_matches = True
                        break
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if len(action_base) >= 3 and _description_contains_action_word(item_description, action_base):
                        action_matches = True
                        break
                else:
                    if action_lower in item_description:
                        action_matches = True
                        break
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if action_base in item_description and len(action_base) >= 3:
                        action_matches = True
                        break
                
                if action_matches:
                    break
            
            if not action_matches:
                return False
        
        # Special handling for plant_state filter - check both plant_state field and description (plant_state can be str or list)
        if filters.get("plant_state"):
            raw_ps = filters["plant_state"]
            plant_state_filter = [str(ps).lower().strip() for ps in (raw_ps if isinstance(raw_ps, list) else [raw_ps]) if ps]
            item_plant_state = _get_item_plant_state(metadata)
            item_description = _get_item_description(item)
            if isinstance(item_description, str):
                item_description = item_description.lower()
            else:
                item_description = ""
            
            # Define opposite states to exclude
            opposite_states = {
                "ripe": ["unripe", "immature", "green"],
                "unripe": ["ripe", "mature", "red"],
                "mature": ["unripe", "immature", "green"],
                "green": ["ripe", "mature", "red"],
                "red": ["unripe", "green", "immature"]
            }
            
            plant_state_matches = False
            for plant_state in plant_state_filter:
                plant_state_lower = plant_state.lower().strip()
                
                # Early exact match: if metadata says unripe and user asked for unripe, accept immediately (no exclusion logic)
                if plant_state_lower == "unripe" and item_plant_state == "unripe":
                    plant_state_matches = True
                    print(f"         ✅ Matched: exact plant_state 'unripe' (metadata)")
                    break
                
                # Unripe: allow "mixed" if description mentions unripe (image has unripe berries even if some ripe too)
                if plant_state_lower == "unripe" and item_plant_state == "mixed":
                    unripe_mixed_phrases = [
                        "unripe", "unripened",
                        "green berr", "green fruit",
                        "developing fruit", "developing berr", "developing raspberr",
                        "immature",
                    ]
                    if any(p in item_description for p in unripe_mixed_phrases):
                        plant_state_matches = True
                        print(f"         ✅ Matched: description contains unripe-related phrase, filter is 'unripe' (mixed image OK)")
                        break
                # Unripe should also match early growth stages: blooming, buds (semantically pre-ripe)
                if plant_state_lower == "unripe" and item_plant_state in ("blooming", "buds", "bud"):
                    plant_state_matches = True
                    print(f"         ✅ Matched: filter 'unripe' includes early growth '{item_plant_state}'")
                    break
                # Ripe: allow "mixed" if description mentions ripe (image has ripe berries even if some unripe too)
                if plant_state_lower == "ripe" and item_plant_state == "mixed":
                    if _description_has_ripe_phrase(item_description):
                        plant_state_matches = True
                        print(f"         ✅ Matched: description contains ripe, filter is 'ripe' (mixed image OK)")
                        break
                # Ripe: exclude when description clearly says unripe (or unripened) but has no ripe — e.g. "flowers and buds and unripe berries"
                if plant_state_lower == "ripe" and (("unripe" in item_description or "unripened" in item_description)):
                    has_ripe_word = bool(re.search(r"\bripe\b", item_description))
                    has_ripe_phrase = _description_has_ripe_phrase(item_description)
                    if not has_ripe_word and not has_ripe_phrase:
                        print(f"         ❌ Excluded: description says unripe but has no ripe, filter is 'ripe'")
                        plant_state_matches = False
                        continue
                # Exclude "mixed" when searching for specific states (except unripe/ripe + mixed when description supports it, handled above)
                if plant_state_lower in ["ripe", "unripe", "mature", "blooming", "fruiting"]:
                    if item_plant_state == "mixed":
                        print(f"         ❌ Excluded: item has 'mixed' plant_state, but filter is '{plant_state_lower}'")
                        plant_state_matches = False
                        continue
                    # When filter is "ripe", exclude "fruiting" unless description clearly says ripe
                    if plant_state_lower == "ripe" and item_plant_state == "fruiting":
                        if not _description_has_ripe_phrase(item_description):
                            print(f"         ❌ Excluded: item has 'fruiting' plant_state and description doesn't say ripe, filter is 'ripe'")
                            plant_state_matches = False
                            continue
                
                # When filter is "ripe", exclude items whose description suggests mixed/unripe (unless metadata says "ripe")
                if plant_state_lower == "ripe" and item_plant_state != "ripe":
                    mixed_desc_phrases = [
                        "varying stages", "various stages", "various ripening", "at various ripening", "at various stages",
                        "mix of unripe and ripe", "unripe and ripe", "stages of ripeness", "varying ripeness",
                        "unripe raspberr", "unripe berry", "unripe berries", "developing raspberr", "developing fruit",
                        "developing berries", "different stages", "multiple stages", "various stages of", "ripening stages",
                    ]
                    if any(p in item_description for p in mixed_desc_phrases):
                        print(f"         ❌ Excluded: description suggests mixed/unripe, filter is 'ripe'")
                        plant_state_matches = False
                        continue
                
                # Check for opposite states - exclude items that explicitly have opposite state
                if plant_state_lower in opposite_states:
                    opposites = opposite_states[plant_state_lower]
                    # Check plant_state field for opposite
                    if item_plant_state in opposites:
                        print(f"         ❌ Excluded: item has opposite plant_state '{item_plant_state}', filter is '{plant_state_lower}'")
                        plant_state_matches = False
                        continue
                    # When filter is 'ripe' or 'unripe' and item explicitly matches, trust metadata — skip description opposite check
                    if not ((plant_state_lower == "ripe" and item_plant_state == "ripe") or (plant_state_lower == "unripe" and item_plant_state == "unripe")):
                        # Check description for opposite keywords (fruit/berry context)
                        for opposite in opposites:
                            if opposite == "green" and "green leaves" in item_description:
                                continue
                            opposite_patterns = [
                                f"{opposite} fruit", f"{opposite} berry", f"{opposite} berries",
                                f"{opposite} raspberr", f"{opposite} strawberr", f"{opposite} blueberr"
                            ]
                            if any(pattern in item_description for pattern in opposite_patterns):
                                print(f"         ❌ Excluded: description contains opposite state '{opposite}' in fruit context, filter is '{plant_state_lower}'")
                                plant_state_matches = False
                                break
                        if not plant_state_matches:
                            continue
                
                # Direct match in plant_state field (exact match preferred)
                if plant_state_lower == item_plant_state:
                    plant_state_matches = True
                    print(f"         ✅ Matched: exact plant_state match '{item_plant_state}'")
                    break
                elif item_plant_state and plant_state_lower in item_plant_state:
                    plant_state_matches = True
                    print(f"         ✅ Matched: plant_state contains '{plant_state_lower}'")
                    break
                
                # SPECIAL HANDLING FOR "green" - must be specific to fruits/berries, not leaves
                # This check must come BEFORE the general keyword check to avoid matching "green leaves"
                if plant_state_lower == "green":
                    # Look for green in context of fruits/berries, not just "green leaves"
                    green_contexts = ["green fruit", "green berry", "green berries", "green raspberr", "unripe", "unripe fruit", "unripe berries", "unripe raspberr"]
                    if any(ctx in item_description for ctx in green_contexts):
                        plant_state_matches = True
                        print(f"         ✅ Matched: found green context in description")
                        break
                    # Also check if "green" appears near fruit/berry keywords (but not "green leaves")
                    if re.search(r'green\s+(fruit|berry|berries|raspberr)', item_description):
                        plant_state_matches = True
                        print(f"         ✅ Matched: 'green' near fruit/berry keywords")
                        break
                    # Explicitly exclude if only "green leaves" is present
                    if "green leaves" in item_description and not any(ctx in item_description for ctx in ["green fruit", "green berry", "green berries", "unripe"]):
                        # Skip this item - only has "green leaves", not green fruits
                        print(f"         ❌ Excluded: only has 'green leaves', no green fruits")
                        plant_state_matches = False
                        break  # Break out of plant_state loop - this item doesn't match
                
                # Check description for plant state keywords (for non-green states)
                plant_state_keywords = {
                    "ripe": ["ripe", "mature", "ready"],
                    "unripe": ["unripe", "immature"],
                    "mature": ["mature", "ripe", "ready"],
                    "blooming": ["blooming", "flowering", "bloom", "flower"],
                    "fruiting": ["fruiting", "fruits", "berries"],
                    "growing": ["growing", "developing"]
                }
                
                # Check if plant_state matches any keyword variations (skip "green" - already handled above)
                # For "ripe" use whole-word match so "unripe" does not match
                if plant_state_lower != "green":
                    for keyword, variations in plant_state_keywords.items():
                        if plant_state_lower == keyword or plant_state_lower in variations:
                            if keyword == "ripe" or (plant_state_lower == "mature" and "ripe" in variations):
                                # Whole-word "ripe" so "unripe" doesn't match; other terms can be substring
                                if re.search(r"\bripe\b", item_description) or any(
                                    v in item_description for v in variations if v != "ripe"
                                ):
                                    plant_state_matches = True
                                    break
                            elif any(v in item_description for v in variations):
                                plant_state_matches = True
                                break
                        if plant_state_matches:
                            break
                
                # Direct check: if plant_state word is in description (for non-green states)
                # Use whole-word match for "ripe" so "unripe" / "unripened" don't match the ripe filter
                if plant_state_lower != "green":
                    if plant_state_lower == "ripe":
                        if re.search(r"\bripe\b", item_description):
                            plant_state_matches = True
                            break
                    elif plant_state_lower in item_description:
                        plant_state_matches = True
                        break
                
                if plant_state_matches:
                    break
            
            if not plant_state_matches:
                return False
        
        # Check scene filter - STRICT matching with description validation
        if filters.get("scene"):
            scene_filter = [s.lower().strip() for s in filters["scene"]]
            item_scene = _metadata_str(metadata.get("scene")).strip()
            item_description = _get_item_description(item)
            
            scene_matches = False
            
            # Scene synonyms: "field" filter matches metadata.scene meadow/pasture/grassland
            scene_synonyms = {
                "field": ["meadow", "pasture", "grassland", "outdoor", "field"],
                "outdoor": ["field", "meadow", "pasture", "grassland", "outdoor"],
                "indoor": ["indoor", "inside"],
            }
            
            # First, check if scene field matches (exact, substring, or synonym)
            if item_scene:
                for scene in scene_filter:
                    scene_lower = scene.lower().strip()
                    # Exact match or substring match
                    if scene_lower == item_scene or scene_lower in item_scene or item_scene in scene_lower:
                        # VALIDATION: Check description for contradictions
                        indoor_keywords = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                        outdoor_keywords = ["outdoor", "outside", "field", "meadow", "pasture", "open", "exterior"]
                        
                        # If searching for "field" but description says "indoor", reject
                        if scene_lower == "field":
                            if any(keyword in item_description for keyword in indoor_keywords):
                                # Description contradicts scene metadata - reject this item
                                print(f"      ⚠️  Scene mismatch: scene='{item_scene}' but description indicates indoor")
                                continue  # Skip this scene match, try next
                        
                        # If searching for "indoor" but description says outdoor keywords, reject
                        if scene_lower in ["indoor", "inside"]:
                            if any(keyword in item_description for keyword in outdoor_keywords):
                                # Description contradicts scene metadata - reject this item
                                print(f"      ⚠️  Scene mismatch: scene='{item_scene}' but description indicates outdoor")
                                continue  # Skip this scene match, try next
                        
                        scene_matches = True
                        break
                    # Semantic equivalence: "field" <-> "outdoor"
                    if (scene_lower == "field" and item_scene == "outdoor") or (scene_lower == "outdoor" and item_scene == "field"):
                        scene_matches = True
                        break
                    # Field synonyms: filter "field" matches metadata.scene "meadow", "pasture", "grassland"
                    if scene_lower in scene_synonyms and item_scene in scene_synonyms.get(scene_lower, []):
                        # Reject if description clearly contradicts (e.g. scene=meadow but description says indoor)
                        if scene_lower == "field":
                            indoor_kw = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                            if any(k in item_description for k in indoor_kw):
                                continue
                        scene_matches = True
                        break
            
            # If scene field doesn't match, also check description for scene keywords
            if not scene_matches and item_description:
                scene_keyword_map = {
                    "field": ["field", "fields", "meadow", "pasture", "open field", "grassland", "grazing", "grassy field", "grassy pasture", "agricultural field", "open fields"],
                    "forest": ["forest", "woodland", "woods", "trees"],
                    "garden": ["garden", "garden area"],
                    "farm": ["farm", "farmland", "farm area"],
                    "indoor": ["indoor", "inside", "interior", "barn", "shed", "building"],
                    "outdoor": ["outdoor", "outside", "exterior", "open air"]
                }
                
                for scene in scene_filter:
                    scene_lower = scene.lower().strip()
                    if scene_lower in scene_keyword_map:
                        keywords = scene_keyword_map[scene_lower]
                        desc_lower = item_description.lower() if isinstance(item_description, str) else item_description
                        if any(keyword in desc_lower for keyword in keywords):
                            scene_matches = True
                            break
            
            if not scene_matches:
                return False
        
        # Check weather filter (substring match on metadata.weather and description)
        if filters.get("weather"):
            weather_filter = [w.lower().strip() for w in filters["weather"]]
            item_weather = _metadata_str(metadata.get("weather", "")).lower()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            weather_matches = any(
                w in item_weather or w in item_description
                for w in weather_filter
            )
            if not weather_matches:
                return False
        
        # Generic handling for other filters
        for filter_type, filter_values in filters.items():
            if filter_type in ["species", "time", "action", "plant_state", "scene", "weather"]:  # Already handled above
                continue
            if not filter_values:
                continue
            
            # Check in top-level fields
            item_value = item.get(filter_type, "")
            if item_value:
                if not any(fv.lower() in str(item_value).lower() for fv in filter_values):
                    return False
            
            # Check in metadata
            item_value = metadata.get(filter_type, "")
            if item_value:
                if not any(fv.lower() in str(item_value).lower() for fv in filter_values):
                    return False
        
        return True


class DatasetAdapterRegistry:
    """Registry for dataset adapters"""
    
    def __init__(self):
        self.adapters: Dict[DatasetType, DatasetAdapter] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self):
        """Register default adapters"""
        # Build schema and pass explicitly so adapter works even when __init__ requires schema
        species_schema = DatasetSchema(
            dataset_type=DatasetType.WILDLIFE,
            required_fields=['id', 'collection'],
            optional_fields=['category', 'metadata'],
            filter_fields=['category', 'species', 'time', 'season', 'action', 'plant_state', 'collection'],
            metadata_fields=['species', 'action', 'time', 'season', 'scene', 'weather', 'date', 'description'],
            description="Species observation dataset with wildlife, plants, or pest observations",
        )
        species_adapter = SpeciesObservationAdapter(species_schema)
        self.adapters[DatasetType.WILDLIFE] = species_adapter
        self.adapters[DatasetType.DOMESTIC_ANIMAL] = species_adapter
        self.adapters[DatasetType.LIVESTOCK] = species_adapter
        self.adapters[DatasetType.PLANTS] = species_adapter
        self.adapters[DatasetType.PESTS] = species_adapter
    
    def register_adapter(self, dataset_type: DatasetType, adapter: DatasetAdapter):
        """Register a custom adapter for a dataset type"""
        self.adapters[dataset_type] = adapter
        print(f"📦 Registered adapter for dataset type: {dataset_type.value}")
    
    def get_adapter(self, dataset_type: DatasetType) -> DatasetAdapter:
        """Get adapter for a dataset type"""
        adapter = self.adapters.get(dataset_type)
        if not adapter:
            # Fallback to generic adapter
            schema = DatasetSchema(
                dataset_type=dataset_type,
                description=f"Generic adapter for {dataset_type.value} datasets"
            )
            adapter = GenericDatasetAdapter(schema)
            self.adapters[dataset_type] = adapter
        return adapter
    
    def create_custom_adapter(self, dataset_type: DatasetType, schema: DatasetSchema) -> DatasetAdapter:
        """Create and register a custom adapter with a specific schema"""
        adapter = GenericDatasetAdapter(schema)
        self.register_adapter(dataset_type, adapter)
        return adapter

