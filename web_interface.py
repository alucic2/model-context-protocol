#!/usr/bin/env python3
"""
FastAPI Web Interface
- Web UI that consumes the MCP server
- Handles web-specific concerns and user interface
- Communicates with MCP server via HTTP
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import httpx
import os

# Import configuration
from config import WEB_CONFIG, BASE_DIR, IMAGES_DIR, IMAGES_TRY_SPECIES_FIRST

class WebInterface:
    """Web interface that consumes the MCP server"""
    
    def __init__(self):
        self.app = FastAPI(title="AIFARMS Web Interface", version="2.0.0")
        self.mcp_server_url = WEB_CONFIG["mcp_server_url"]
        
        # Serve static files (e.g. logo)
        _static_dir = Path(__file__).resolve().parent / "static"
        if _static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_templates()
        
        print(f"🌐 Initialized Web Interface")
        print(f"🔗 MCP Server URL: {self.mcp_server_url}")
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        # Return 404 for common scanner/probe paths (reduces log noise from bots)
        _PROBE_PATHS = frozenset({
            "/s3.yml", "/s3.yaml", "/s3.properties", "/s3.key", "/s3.secret",
            "/aws_s3_config.json", "/.env.aws", "/.env.bak", "/s3/.env.bak",
        })
        class BlockProbePathsMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                path = request.url.path.rstrip("/") or "/"
                if path in _PROBE_PATHS or path.startswith("/s3/"):
                    return JSONResponse(status_code=404, content={"detail": "Not Found"})
                return await call_next(request)
        self.app.add_middleware(BlockProbePathsMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup web interface routes"""
        
        @self.app.get("/health")
        async def health():
            """Quick check that the web server is up (no MCP dependency)."""
            return {"status": "ok", "service": "web_interface"}
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with search interface"""
            try:
                # Get available filters from MCP server (timeout so page still loads if MCP is down)
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    raw_list = datasets_data.get("datasets") if isinstance(datasets_data, dict) else []
                    datasets_list = self._normalize_datasets_for_template(raw_list)
                    all_filters = self._extract_filters_from_datasets(raw_list if isinstance(raw_list, list) else [])
                    # Count by type for UI (so user can see if wildlife/plants etc. loaded)
                    by_type = {}
                    for d in datasets_list:
                        t = (d.get("type") or "unknown").strip().lower() or "unknown"
                        by_type[t] = by_type.get(t, 0) + 1
                    datasets_by_type = dict(sorted(by_type.items()))
                    
                return self.templates.TemplateResponse(request, "search_interface.html", {
                    "request": request,
                    "filters": all_filters,
                    "datasets": datasets_list,
                    "datasets_by_type": datasets_by_type,
                    "mcp_server_url": self.mcp_server_url
                })
            except Exception as e:
                print(f"Error loading home page: {e}")
                # Return basic interface if MCP server is unavailable
                return self.templates.TemplateResponse(request, "search_interface.html", {
                    "request": request,
                    "filters": {},
                    "datasets": [],
                    "datasets_by_type": {},
                    "mcp_server_url": self.mcp_server_url,
                    "error": "MCP server unavailable"
                })
        
        @self.app.get("/search", response_class=HTMLResponse)
        async def search_page(request: Request):
            """Search results page"""
            return self.templates.TemplateResponse(request, "search_results.html", {
                "request": request,
                "results": {},
                "query": "",
                "filters": {}
            })
        
        @self.app.post("/search", response_class=HTMLResponse)
        async def search_images(
            request: Request,
            query: str = Form(""),
            dataset: str = Form(""),
            category: List[str] = Form([]),
            species: List[str] = Form([]),
            time: List[str] = Form([]),
            season: List[str] = Form([]),
            action: List[str] = Form([]),
            plant_state: List[str] = Form([]),
            limit: int = Form(50),
            page: int = Form(1)
        ):
            """Handle search form submission"""
            try:
                print(f"🔍 Web interface search request: query='{query}', dataset='{dataset}'")
                
                # Prepare search request
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "filters": {
                        "category": category if category else [],
                        "species": species if species else [],
                        "time": time if time else [],
                        "season": season if season else [],
                        "action": action if action else [],
                        "plant_state": plant_state if plant_state else []
                    },
                    "limit": limit,
                    "offset": (page - 1) * limit
                }
                
                print(f"🔍 Calling MCP server with: {search_data}")
                
                # Call MCP server search tool
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🔍 MCP server response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                    
                    print(f"🔍 Extracted search results: {search_results}")
                
                # Get filters for the results page
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    all_filters = self._extract_filters_from_datasets(datasets_data["datasets"])
                
                return self.templates.TemplateResponse(request, "search_results.html", {
                    "request": request,
                    "results": search_results,
                    "query": query,
                    "filters": all_filters,
                    "datasets": datasets_data["datasets"],
                    "current_page": page,
                    "limit": limit
                })
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                import traceback
                traceback.print_exc()
                return self.templates.TemplateResponse(request, "search_results.html", {
                    "request": request,
                    "results": {"error": str(e)},
                    "query": query,
                    "filters": {},
                    "datasets": {},
                    "current_page": page,
                    "limit": limit
                })
        
        @self.app.post("/llm_search", response_class=HTMLResponse)
        async def llm_search_images(
            request: Request,
            query: str = Form(""),
            dataset: str = Form(""),
            category: str = Form(""),
            limit: int = Form(50),
            page: int = Form(1),
            dataset_offset: int = Form(0)
        ):
            """Handle LLM-powered search with intelligent query understanding"""
            try:
                print(f"🧠 LLM search request: query='{query}', dataset='{dataset}', category='{category}', dataset_offset={dataset_offset}")
                
                # Prepare LLM search request (category restricts by data type: wildlife, domestic_animal, livestock, plants, pests)
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "limit": limit,
                    "offset": (page - 1) * limit,
                    "dataset_offset": dataset_offset
                }
                if category and category.strip():
                    search_data["category"] = [category.strip()]
                
                print(f"🧠 Calling MCP server LLM search with: {search_data}")
                
                # Call MCP server LLM search tool (long timeout: LLM + dataset search; server caps datasets for broad queries)
                llm_search_timeout = httpx.Timeout(30.0, read=300.0)  # 5 min read for slow LLM + many datasets
                async with httpx.AsyncClient(timeout=llm_search_timeout) as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/llm_search",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🧠 MCP server LLM response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP LLM response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results and sort by confidence
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                        
                        # Sort results by confidence (highest first) if available
                        if any('llm_confidence' in result for result in search_results["results"]):
                            search_results["results"].sort(key=lambda x: x.get('llm_confidence', 0), reverse=True)
                            print(f"🧠 Results sorted by confidence (highest first)")
                    
                    print(f"🧠 Extracted LLM search results: {search_results}")
                
                # Get filters for the results page
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    raw_list = datasets_data.get("datasets") if isinstance(datasets_data, dict) else []
                    all_filters = self._extract_filters_from_datasets(raw_list if isinstance(raw_list, list) else [])
                    datasets_list = self._normalize_datasets_for_template(raw_list)
                
                return self.templates.TemplateResponse(request, "llm_search_results.html", {
                    "request": request,
                    "results": search_results,
                    "query": query,
                    "dataset": dataset,
                    "category": category,
                    "filters": all_filters,
                    "datasets": datasets_list,
                    "current_page": page,
                    "limit": limit,
                    "dataset_offset": search_results.get("dataset_offset", 0),
                    "next_dataset_offset": search_results.get("next_dataset_offset", 0),
                    "total_datasets_matching": search_results.get("total_datasets_matching", 0),
                    "has_more_datasets": search_results.get("has_more_datasets", False),
                })
                
            except Exception as e:
                print(f"❌ LLM search error: {e}")
                import traceback
                traceback.print_exc()
                return self.templates.TemplateResponse(request, "llm_search_results.html", {
                    "request": request,
                    "results": {"error": str(e)},
                    "query": query,
                    "dataset": dataset,
                    "category": category,
                    "filters": {},
                    "datasets": [],
                    "current_page": page,
                    "limit": limit,
                    "dataset_offset": 0,
                    "next_dataset_offset": 0,
                    "total_datasets_matching": 0,
                    "has_more_datasets": False,
                })
        
        @self.app.get("/api/search")
        async def api_search(
            query: str = "",
            dataset: str = "",
            category: List[str] = Query([]),
            species: List[str] = Query([]),
            time: List[str] = Query([]),
            season: List[str] = Query([]),
            action: List[str] = Query([]),
            plant_state: List[str] = Query([]),
            limit: int = Query(50),
            offset: int = Query(0)
        ):
            """API endpoint for search"""
            try:
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "filters": {
                        "category": category if category else [],
                        "species": species if species else [],
                        "time": time if time else [],
                        "season": season if season else [],
                        "action": action if action else [],
                        "plant_state": plant_state if plant_state else []
                    },
                    "limit": limit,
                    "offset": offset
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    return response.json()
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/datasets")
        async def api_datasets():
            """Get available datasets"""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models")
        async def api_models():
            """Get available models"""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/models")
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/inference")
        async def api_inference(request: Request):
            """Run model inference"""
            try:
                body = await request.json()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/api/inference",
                        json=body
                    )
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/image/{filename:path}")
        async def serve_image(filename: str):
            """Serve images from the MCP server"""
            try:
                # Redirect to MCP server for images
                image_url = f"{self.mcp_server_url}/image/{filename}"
                return JSONResponse({"image_url": image_url})
            except Exception as e:
                raise HTTPException(status_code=404, detail="Image not found")
        
        @self.app.get("/croissant_datasets", response_class=HTMLResponse)
        async def croissant_datasets(request: Request):
            """Browse discovered Croissant datasets"""
            try:
                # Call MCP server to get Croissant datasets
                async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for crawling
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/crawl_croissant_datasets",
                        json={}
                    )
                    
                    # Check response status
                    if response.status_code != 200:
                        error_detail = response.text
                        try:
                            error_json = response.json()
                            error_detail = error_json.get("detail", error_detail)
                        except:
                            pass
                        print(f"❌ MCP server returned error {response.status_code}: {error_detail}")
                        return self.templates.TemplateResponse(request, "croissant_datasets.html", {
                            "request": request,
                            "datasets": [],
                            "total_count": 0,
                            "error": f"MCP server error ({response.status_code}): {error_detail}"
                        })
                    
                    mcp_response = response.json()
                    
                    # Extract datasets from MCP response
                    datasets = []
                    
                    # Handle MCP format response (wrapped by tool registry)
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                data = content_item["data"]
                                # Check for datasets in the data
                                if "datasets" in data:
                                    datasets = data["datasets"]
                                    break
                                # Also check if data itself is a dict with datasets at root
                                elif isinstance(data, dict) and "datasets" in data:
                                    datasets = data["datasets"]
                                    break
                    
                    # Also check if datasets are in the response root (direct format)
                    if not datasets and "datasets" in mcp_response:
                        datasets = mcp_response["datasets"]
                    
                    # Debug: log what we got
                    if not datasets:
                        print(f"⚠️  No datasets found in response. Response keys: {list(mcp_response.keys())}")
                        if "detail" in mcp_response:
                            print(f"⚠️  Error detail: {mcp_response['detail']}")
                        if "content" in mcp_response:
                            for i, item in enumerate(mcp_response["content"]):
                                print(f"  Content item {i}: type={item.get('type')}, keys={list(item.keys())}")
                                if "data" in item:
                                    print(f"    Data keys: {list(item['data'].keys())}")
                                    print(f"    Data: {item['data']}")
                    
                    error_msg = None
                    if "detail" in mcp_response:
                        error_msg = mcp_response["detail"]
                    
                    return self.templates.TemplateResponse(request, "croissant_datasets.html", {
                        "request": request,
                        "datasets": datasets,
                        "total_count": len(datasets),
                        "error": error_msg
                    })
                    
            except httpx.TimeoutException:
                error_msg = "Request timed out - the crawler may be taking too long. Try again later."
                print(f"❌ Timeout loading Croissant datasets: {error_msg}")
                return self.templates.TemplateResponse(request, "croissant_datasets.html", {
                    "request": request,
                    "datasets": [],
                    "total_count": 0,
                    "error": error_msg
                })
            except Exception as e:
                print(f"❌ Error loading Croissant datasets: {e}")
                import traceback
                traceback.print_exc()
                return self.templates.TemplateResponse(request, "croissant_datasets.html", {
                    "request": request,
                    "datasets": [],
                    "total_count": 0,
                    "error": str(e)
                })
        
        @self.app.get("/debug/search")
        async def debug_search():
            """Debug endpoint to test search functionality"""
            try:
                # Test search with a simple query
                search_data = {
                    "query": "bobcat",
                    "limit": 5
                }
                
                print(f"🔍 Debug search request: {search_data}")
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🔍 Debug search response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                    
                    print(f"🔍 Debug extracted results: {search_results}")
                
                return {
                    "debug_info": "Search test completed",
                    "request": search_data,
                    "response": mcp_response,
                    "extracted_results": search_results,
                    "mcp_server_url": self.mcp_server_url
                }
                
            except Exception as e:
                print(f"❌ Debug search error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/debug/files")
        async def debug_files():
            """Debug endpoint to show file structure"""
            try:
                import os
                from pathlib import Path
                
                base_dir = Path("/opt/mcp-data-server")
                file_info = {
                    "base_directory": str(base_dir),
                    "base_exists": base_dir.exists(),
                    "base_contents": [],
                    "image_files": [],
                    "mcp_files": []
                }
                
                if base_dir.exists():
                    # List base directory contents
                    for item in base_dir.iterdir():
                        if item.is_dir():
                            file_info["base_contents"].append(f"📁 {item.name}/")
                            # Look for images in subdirectories
                            for subitem in item.iterdir():
                                if subitem.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                                    file_info["image_files"].append(f"{item.name}/{subitem.name}")
                        elif item.suffix == '.json':
                            file_info["mcp_files"].append(item.name)
                        elif item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                            file_info["image_files"].append(item.name)
                
                return file_info
                
            except Exception as e:
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/test/images")
        async def test_images():
            """Test endpoint to check if image files exist"""
            try:
                import os
                from pathlib import Path
                
                images_dir = Path(IMAGES_DIR)
                test_files = []
                
                if images_dir.exists():
                    # List first 10 image files
                    for item in images_dir.iterdir():
                        if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                            test_files.append(item.name)
                            if len(test_files) >= 10:
                                break
                
                return {
                    "images_directory": str(images_dir),
                    "directory_exists": images_dir.exists(),
                    "sample_files": test_files,
                    "total_files": len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/debug/coyote-images")
        async def debug_coyote_images():
            """Debug endpoint to check coyote image file matching"""
            try:
                import json
                from pathlib import Path
                
                # Load coyote dataset
                coyote_file = BASE_DIR / "coyote_mcp_data.json"
                if not coyote_file.exists():
                    return {"error": f"Coyote dataset file not found: {coyote_file}"}
                
                with open(coyote_file, 'r') as f:
                    coyote_data = json.load(f)
                
                images_dir = Path(IMAGES_DIR)
                if not images_dir.exists():
                    return {"error": f"Images directory not found: {images_dir}"}
                
                # Get all image files in directory (case-insensitive)
                existing_files = {}
                for item in images_dir.iterdir():
                    if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                        # Store both original and lowercase versions for matching
                        existing_files[item.name.lower()] = item.name
                        existing_files[item.name] = item.name
                
                # Check first 20 coyote images
                coyote_images = coyote_data.get("images", [])[:20]
                results = []
                missing = []
                found = []
                
                for img_entry in coyote_images:
                    metadata = img_entry.get("metadata", {})
                    original_filename = metadata.get("original_filename", "")
                    img_id = img_entry.get("id", "")
                    
                    # Check if file exists (case-insensitive)
                    found_file = None
                    if original_filename:
                        # Try exact match
                        if original_filename in existing_files:
                            found_file = existing_files[original_filename]
                        # Try case-insensitive match
                        elif original_filename.lower() in existing_files:
                            found_file = existing_files[original_filename.lower()]
                        # Try with different case
                        else:
                            # Try all variations
                            for existing_lower, existing_orig in existing_files.items():
                                if existing_lower == original_filename.lower():
                                    found_file = existing_orig
                                    break
                    
                    result = {
                        "id": img_id,
                        "original_filename": original_filename,
                        "expected_path": f"{IMAGES_DIR}/{original_filename}",
                        "file_exists": found_file is not None,
                        "actual_filename": found_file
                    }
                    
                    if found_file:
                        found.append(result)
                    else:
                        missing.append(result)
                    
                    results.append(result)
                
                # Get sample of actual files that might be coyote images
                sample_files = []
                for item in list(images_dir.iterdir())[:50]:
                    if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                        sample_files.append(item.name)
                
                return {
                    "coyote_dataset_file": str(coyote_file),
                    "images_directory": str(images_dir),
                    "total_coyote_entries_checked": len(coyote_images),
                    "files_found": len(found),
                    "files_missing": len(missing),
                    "results": results,
                    "missing_files": missing,
                    "found_files": found,
                    "sample_existing_files": sample_files[:20],
                    "total_images_in_directory": len([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']])
                }
                
            except Exception as e:
                import traceback
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/images/{filename:path}")
        async def serve_image(filename: str):
            """Serve images: order depends on IMAGES_TRY_SPECIES_FIRST (Taiga = subdirs only → try species subdir first)."""
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
                # Use resolved base for species subdirs so we look in the same place as the flat path
                images_base = images_dir_resolved
                requested_suffix = Path(filename).suffix

                def unique_values(values):
                    seen = set()
                    unique = []
                    for value in values:
                        if value and value not in seen:
                            seen.add(value)
                            unique.append(value)
                    return unique

                filename_no_ext_variants = unique_values([
                    filename_no_ext,
                    filename_no_ext.replace("-", "_"),
                    filename_no_ext.replace("ref_leaf", "red_leaf"),
                    filename_no_ext.replace("ref_leaf", "red_leaf").replace("-", "_"),
                ])
                filename_variants = unique_values(
                    [filename] +
                    [f"{stem}{requested_suffix}" for stem in filename_no_ext_variants if requested_suffix]
                )
                
                def try_species_subdir():
                    if "_" not in filename_no_ext:
                        return None
                    # Try dataset-name subdir first (e.g. domestic_cat for domestic_cat_001), then first segment
                    import re
                    subdir_candidates = []
                    for stem in filename_no_ext_variants:
                        if re.search(r"_\d+$", stem):
                            subdir_candidates.append(re.sub(r"_\d+$", "", stem))  # wild_turkey_001 -> wild_turkey
                        elif "_" in stem:
                            subdir_candidates.append(stem)  # wild_turkey (no number) -> wild_turkey
                        subdir_candidates.append(stem.split("_")[0])
                    subdir_candidates = unique_values(subdir_candidates)
                    # Case-insensitive subdir match first (e.g. dir is Daktulosphaira_vitifoliae, we have daktulosphaira_vitifoliae)
                    try:
                        for candidate in subdir_candidates:
                            if not candidate:
                                continue
                            target_lower = candidate.lower().replace(" ", "_").replace("-", "_")
                            for child in images_base.iterdir():
                                if not child.is_dir():
                                    continue
                                if child.name.lower().replace(" ", "_").replace("-", "_") == target_lower:
                                    for name in filename_variants:
                                        sub_path = child / name
                                        if sub_path.exists() and sub_path.is_file():
                                            return FileResponse(str(sub_path))
                                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                        for stem in filename_no_ext_variants:
                                            sub_path = child / f"{stem}{ext}"
                                            if sub_path.exists() and sub_path.is_file():
                                                return FileResponse(str(sub_path))
                                    # Fallback: subdir exists but files may be named by number only (e.g. 252.jpg)
                                    num_suffix = re.search(r"(\d+)$", filename_no_ext)
                                    if num_suffix:
                                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                            sub_path = child / f"{num_suffix.group(1)}{ext}"
                                            if sub_path.exists() and sub_path.is_file():
                                                return FileResponse(str(sub_path))
                            break  # only need first candidate for case-insensitive; then try literal
                    except OSError:
                        pass
                    for subdir in subdir_candidates:
                        if not subdir:
                            continue
                        species_dir = images_base / subdir
                        for name in filename_variants:
                            sub_path = species_dir / name
                            if sub_path.exists() and sub_path.is_file():
                                return FileResponse(str(sub_path))
                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                            for stem in filename_no_ext_variants:
                                sub_path = species_dir / f"{stem}{ext}"
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
                
                def try_flat():
                    if flat_to_try.exists() and flat_to_try.is_file():
                        return FileResponse(str(flat_to_try))
                    for name in filename_variants:
                        flat_variant = images_base / name
                        if flat_variant.exists() and flat_variant.is_file():
                            return FileResponse(str(flat_variant))
                    return None
                
                # Order: species subdir first when Taiga layout (subdirs only), else flat first
                if IMAGES_TRY_SPECIES_FIRST:
                    r = try_species_subdir()
                    if r is not None:
                        return r
                    r = try_flat()
                    if r is not None:
                        return r
                else:
                    r = try_flat()
                    if r is not None:
                        return r
                    r = try_species_subdir()
                    if r is not None:
                        return r
                
                # Case-insensitive in flat dir
                filename_lower = filename.lower()
                try:
                    for item in images_base.iterdir():
                        if item.is_file() and item.name.lower() in {name.lower() for name in filename_variants}:
                            return FileResponse(str(item))
                except OSError:
                    pass
                
                # 4) Different extensions in flat dir
                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                    for stem in filename_no_ext_variants:
                        potential_file = images_base / f"{stem}{ext}"
                        if potential_file.exists() and potential_file.is_file():
                            return FileResponse(str(potential_file))
                
                # Not found – log why (directory missing vs wrong filenames)
                # Use same subdir list as try_species_subdir (e.g. wild_turkey then wild, same as domestic_cat then domestic)
                if "_" in filename_no_ext:
                    import re
                    subdir_candidates_log = []
                    if re.search(r"_\d+$", filename_no_ext):
                        subdir_candidates_log.append(re.sub(r"_\d+$", "", filename_no_ext))
                    elif "_" in filename_no_ext:
                        subdir_candidates_log.append(filename_no_ext)
                    subdir_candidates_log.append(filename_no_ext.split("_")[0])
                    species_dirs_tried = [images_base / d for d in subdir_candidates_log]
                else:
                    species_dirs_tried = [images_base / filename_no_ext] if filename_no_ext else []
                print(f"❌ Image not found: {filename}")
                print(f"   Flat path resolved: {image_path_flat_resolved} (exists={image_path_flat_resolved.exists()})")
                print(f"   Tried species subdirs: {species_dirs_tried}")
                # Report first candidate we tried (e.g. wild_turkey, domestic_cat) not just first segment
                if species_dirs_tried:
                    first_tried = species_dirs_tried[0]
                    exists = first_tried.exists() and first_tried.is_dir()
                    print(f"   Species dir exists ({first_tried.name}): {exists}")
                    if exists:
                        try:
                            sample = list(first_tried.iterdir())[:5]
                            names = [p.name for p in sample if p.is_file()]
                            print(f"   Sample files in {first_tried.name}/: {names}")
                        except OSError as e:
                            print(f"   (cannot list {first_tried.name}/: {e})")
                    else:
                        try:
                            print(f"   Resolved path: {first_tried.resolve()}")
                        except OSError as e:
                            print(f"   Resolve failed (permissions?): {e}")
                print(f"   Tried flat: {image_path_flat}")
                print(f"   Tried case-insensitive and other extensions for: {filename}")
                # List files with same prefix (e.g. grapes_*) to see if any exist
                prefix = filename_no_ext.split("_")[0] if "_" in filename_no_ext else filename_no_ext
                try:
                    same_prefix = [p.name for p in images_base.iterdir() if p.is_file() and p.name.lower().startswith(prefix.lower() + "_")]
                    if same_prefix:
                        print(f"   Files with prefix '{prefix}_' in images dir: {same_prefix[:10]}{'...' if len(same_prefix) > 10 else ''}")
                    else:
                        print(f"   No files with prefix '{prefix}_' in images dir.")
                except OSError as e:
                    print(f"   (could not list images dir: {e})")
                
                # List some similar files for debugging
                similar_files = []
                filename_lower_short = filename_lower[:10]
                try:
                    for item in images_base.iterdir():
                        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                            if filename_lower_short in item.name.lower() or item.name.lower().startswith(filename_lower_short):
                                similar_files.append(item.name)
                                if len(similar_files) >= 5:
                                    break
                except OSError:
                    pass
                
                error_detail = f"Image not found: {filename}"
                if similar_files:
                    error_detail += f". Similar files found: {', '.join(similar_files[:3])}"
                
                # Fallback: redirect to MCP server /images/ so images on the server (e.g. wild_turkey) still load
                if getattr(self, "mcp_server_url", None):
                    mcp_images_url = f"{self.mcp_server_url.rstrip('/')}/images/{filename}"
                    print(f"🖼️  Redirecting to MCP server for image: {mcp_images_url}")
                    return RedirectResponse(url=mcp_images_url, status_code=302)
                
                raise HTTPException(status_code=404, detail=error_detail)
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ Error serving image {filename}: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/download/{filename:path}")
        async def download_image(filename: str):
            """Download images from the web interface with proper download headers"""
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
                disp = lambda n: f'attachment; filename="{n}"'
                
                print(f"📥 Download request for image: {filename}")
                flat_to_try = image_path_flat_resolved if image_path_flat_resolved != image_path_flat else image_path_flat
                images_base = images_dir_resolved
                def _dl_species():
                    if "_" not in filename_no_ext:
                        return None
                    import re
                    subdir_candidates = []
                    if re.search(r"_\d+$", filename_no_ext):
                        subdir_candidates.append(re.sub(r"_\d+$", "", filename_no_ext))
                    elif "_" in filename_no_ext:
                        subdir_candidates.append(filename_no_ext)
                    subdir_candidates.append(filename_no_ext.split("_")[0])
                    for subdir in subdir_candidates:
                        if not subdir:
                            continue
                        species_dir = images_base / subdir
                        for p in [species_dir / filename, *(species_dir / f"{filename_no_ext}{e}" for e in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF'])]:
                            if p.exists() and p.is_file():
                                return FileResponse(str(p), media_type="application/octet-stream", filename=p.name, headers={"Content-Disposition": disp(p.name)})
                        # Fallback: numeric suffix only (e.g. wild_turkey/252.jpg)
                        num_suffix = re.search(r"(\d+)$", filename_no_ext)
                        if num_suffix:
                            for e in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                                p = species_dir / f"{num_suffix.group(1)}{e}"
                                if p.exists() and p.is_file():
                                    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name, headers={"Content-Disposition": disp(p.name)})
                    return None
                if IMAGES_TRY_SPECIES_FIRST:
                    r = _dl_species()
                    if r is not None:
                        return r
                if flat_to_try.exists() and flat_to_try.is_file():
                    return FileResponse(str(flat_to_try), media_type="application/octet-stream", filename=filename, headers={"Content-Disposition": disp(filename)})
                if not IMAGES_TRY_SPECIES_FIRST:
                    r = _dl_species()
                    if r is not None:
                        return r
                # Case-insensitive, then other extensions in flat dir
                filename_lower = filename.lower()
                try:
                    for item in images_base.iterdir():
                        if item.is_file() and item.name.lower() == filename_lower:
                            return FileResponse(str(item), media_type="application/octet-stream", filename=item.name, headers={"Content-Disposition": disp(item.name)})
                except OSError:
                    pass
                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                    potential_file = images_base / f"{filename_no_ext}{ext}"
                    if potential_file.exists() and potential_file.is_file():
                        return FileResponse(str(potential_file), media_type="application/octet-stream", filename=potential_file.name, headers={"Content-Disposition": disp(potential_file.name)})
                
                print(f"❌ Image not found for download: {filename}")
                raise HTTPException(status_code=404, detail=f"Image {filename} not found")
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ Error downloading image {filename}: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_templates(self):
        """Setup Jinja2 templates"""
        templates_dir = BASE_DIR / "templates"
        templates_dir.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # Create templates if they don't exist
        self._create_templates()
    
    def _create_templates(self):
        """Create HTML templates"""
        templates_dir = BASE_DIR / "templates"
        
        # Search interface template (always regenerate to ensure latest version)
        search_template = templates_dir / "search_interface.html"
        with open(search_template, "w") as f:
            f.write(self._get_search_template())
        
        # Search results template (always regenerate for disclaimer footer)
        results_template = templates_dir / "search_results.html"
        with open(results_template, "w") as f:
            f.write(self._get_results_template())
        
        # LLM search results template (always regenerate to ensure latest version)
        llm_results_template = templates_dir / "llm_search_results.html"
        with open(llm_results_template, "w") as f:
            f.write(self._get_llm_results_template())
        
        # Croissant datasets template (always regenerate to ensure latest version)
        croissant_template = templates_dir / "croissant_datasets.html"
        with open(croissant_template, "w") as f:
            f.write(self._get_croissant_datasets_template())
    
    def _normalize_datasets_for_template(self, raw: Any) -> List[Dict[str, Any]]:
        """Ensure datasets is a list of dicts with 'name' and 'type' for the dropdown (type used for category filter)."""
        if isinstance(raw, list):
            out = []
            for d in raw:
                name = (d.get("name") if isinstance(d, dict) else str(d)) or "unknown"
                dtype = (d.get("type") if isinstance(d, dict) else None) or ""
                if hasattr(dtype, "value"):
                    dtype = dtype.value
                out.append({"name": name, "type": str(dtype).strip().lower() if dtype else ""})
            return out
        if isinstance(raw, dict):
            # API sometimes returns {"dataset_name": DatasetInfo or dict, ...}; extract type from value
            out = []
            for k, v in raw.items():
                t = ""
                if isinstance(v, dict):
                    t = v.get("type") or ""
                elif hasattr(v, "type"):
                    t = getattr(v, "type", "")
                if hasattr(t, "value"):
                    t = t.value
                out.append({"name": k, "type": str(t).strip().lower() if t else ""})
            return out
        return []

    def _extract_filters_from_datasets(self, datasets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract available filters from datasets"""
        all_filters = {
            "categories": [],
            "species": [],
            "times": [],
            "seasons": [],
            "actions": [],
            "plant_states": []
        }
        
        for dataset in datasets:
            if "filters" in dataset:
                for filter_type, values in dataset["filters"].items():
                    if filter_type in all_filters and isinstance(values, list):
                        all_filters[filter_type].extend(values)
        
        # Remove duplicates and sort
        for filter_type in all_filters:
            all_filters[filter_type] = sorted(list(set(all_filters[filter_type])))
        
        return all_filters
    
    def _construct_image_url(self, result: Dict[str, Any]) -> str:
        """Construct image URL from result data"""
        try:
            metadata = result.get("metadata", {})
            result_id = result.get("id", "")

            # Extension hint from original_filename so id-based fallbacks don't hardcode .jpg.
            # Disk files are often renamed to match the id but keep their real extension
            # (e.g. id "carrot_001" → file "carrot_001.png", original_filename "007_image.png").
            _known_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            _orig_ext = ""
            _of = metadata.get("original_filename") or ""
            if _of:
                _candidate_ext = Path(str(_of)).suffix.lower()
                if _candidate_ext in _known_exts:
                    _orig_ext = _candidate_ext
            _id_fallback_ext = _orig_ext or ".jpg"
            
            # Strategy: Try multiple filename patterns in order of likelihood
            # For coyote: id is "coyote_001" and file is "coyote_001.jpg" (id matches filename)
            # For red_leaf: id is "red_leaf_001" but actual file is "585652.jpg" (from original_filename)
            # For raspberry: id is "raspberry_001" but actual file might be "raspberry_458.JPEG"
            
            # Priority 1: Check if image_url is explicitly set (most reliable)
            if "image_url" in result:
                return result["image_url"]
            
            # Priority 2: Try id with common extensions FIRST (works for datasets like coyote where id matches filename)
            # This should be checked before original_filename for datasets where files are named like "coyote_001.jpg"
            if result_id:
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    # Try uppercase extensions first (JPEG is common for some datasets)
                    for ext in [".JPEG", ".JPG", ".PNG", ".jpg", ".jpeg", ".png"]:
                        potential_filename = f"{result_id}{ext}"
                        potential_path = images_dir / potential_filename
                        if potential_path.exists() and potential_path.is_file():
                            print(f"✅ Found image using id: {potential_filename} (for {result_id})")
                            return f"/images/{potential_filename}"
                
                # If file doesn't exist locally, still return the most likely extension
                # (server might have it even if local check fails)
                # But only if id looks like a filename pattern (contains underscore or matches common patterns)
                if "_" in result_id or result_id.replace("_", "").replace("-", "").isalnum():
                    print(f"⚠️  Image not found locally for id {result_id}, using {_id_fallback_ext} (likely on server)")
                    return f"/images/{result_id}{_id_fallback_ext}"
            
            # Priority 3: Try original_filename from metadata (for datasets like red_leaf where id doesn't match filename)
            # original_filename contains the actual filename from the source dataset
            if "original_filename" in metadata:
                original_filename = metadata["original_filename"]
                filename = Path(original_filename).name
                # Try to verify file exists locally if possible, but still use it even if check fails
                # (file might exist on server even if not locally accessible)
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    potential_path = images_dir / filename
                    if potential_path.exists() and potential_path.is_file():
                        print(f"✅ Found image using original_filename: {filename} (for {result_id})")
                        return f"/images/{filename}"
                    else:
                        # Try case-insensitive match
                        filename_lower = filename.lower()
                        for item in images_dir.iterdir():
                            if item.is_file() and item.name.lower() == filename_lower:
                                print(f"✅ Found image using original_filename (case-insensitive): {item.name} (requested: {filename}, for {result_id})")
                                return f"/images/{item.name}"
                
                # Still use original_filename even if local check fails - it's the source filename
                # The file likely exists on the server even if not locally accessible
                print(f"⚠️  original_filename '{filename}' not found locally for {result_id}, but using it anyway (likely on server)")
                return f"/images/{filename}"
            
            # Priority 4: For raspberry and similar datasets, try to find files by searching
            # Extract species name and try to find matching files
            species = metadata.get("species", "").lower()
            if species and result_id:
                # Try to extract number from id (e.g., "raspberry_001" -> 1)
                import re
                id_match = re.search(r'_(\d+)$', result_id)
                if id_match:
                    id_num = int(id_match.group(1))
                    # Try different offsets (some datasets have offset numbering)
                    # For raspberry: id_001 might map to file_458, so try id_num + 457
                    # But this is fragile, so we'll try a range
                    for offset in [0, 457, 458, 459]:  # Common offsets
                        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                            potential_filename = f"{species}_{id_num + offset}{ext}"
                            potential_path = f"{IMAGES_DIR}/{potential_filename}"
                            if os.path.exists(potential_path):
                                print(f"✅ Found image using offset pattern: {potential_filename}")
                                return f"/images/{potential_filename}"
            
            # Priority 5: Try mcp_id from metadata
            if "mcp_id" in metadata:
                mcp_id = metadata["mcp_id"]
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    potential_filename = f"{mcp_id}{ext}"
                    potential_path = f"{IMAGES_DIR}/{potential_filename}"
                    if os.path.exists(potential_path):
                        print(f"✅ Found image using mcp_id: {potential_filename}")
                        return f"/images/{potential_filename}"
            
            # Priority 6: Fallback to id with the original file's extension (or .jpg if unknown)
            if result_id:
                print(f"⚠️  Using id as fallback: {result_id}{_id_fallback_ext}")
                return f"/images/{result_id}{_id_fallback_ext}"
            
            # If no image info, return placeholder
            print(f"❌ No image identifier found for result: {result.get('id', 'unknown')}")
            return "/images/placeholder.jpg"
                
        except Exception as e:
            print(f"❌ Error constructing image URL for result {result.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return "/images/placeholder.jpg"
    
    def _get_site_footer_css(self) -> str:
        """Shared footer styles for disclaimer and acknowledgments."""
        return """
        .site-footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 13px; color: #444; line-height: 1.5; }
        .site-footer summary { cursor: pointer; font-weight: bold; color: #1a1a1a; }
        .disclaimer-box { background: #fff8e6; border: 1px solid #f0d58c; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; }
        .disclaimer-box ul { margin: 8px 0 0 1.2em; padding: 0; }
        .disclaimer-box li { margin-bottom: 4px; }
        .resource-ack { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; background: #f8f9fa; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }
        .resource-ack img, .resource-ack svg { height: 44px; width: auto; display: block; flex-shrink: 0; }
        .resource-ack-body { flex: 1; min-width: 260px; font-size: 12px; color: #333; }
        .resource-ack-body p { margin: 0 0 8px 0; }
        .resource-ack-body ul { margin: 0; padding-left: 1.2em; }
        .resource-ack-body li { margin-bottom: 4px; }
        """

    def _get_site_footer_html(self) -> str:
        """Disclaimer + resource acknowledgments (species.aifarms.org)."""
        return """
        <footer class="site-footer">
            <details class="disclaimer-box" open>
                <summary>Data quality &amp; AI-generated metadata</summary>
                <p>
                    Image metadata in this catalog were produced in part by vision-language models
                    (<a href="https://azure.microsoft.com/en-us/products/ai-services/openai-service" target="_blank" rel="noopener noreferrer">Azure OpenAI</a>
                    <strong>GPT-4.1</strong> at time of ingest). They are intended for
                    <strong>discovery and research support</strong>, not as verified ground truth.
                </p>
                <ul>
                    <li><strong>Species / taxonomy:</strong> common and scientific names were post-processed and checked against curated lists.</li>
                    <li><strong>Scene fields</strong> (description, time, weather, lighting, setting, background):
                        model-generated and <strong>not manually verified</strong> for every image.</li>
                    <li><strong>Agreement research:</strong> under strict multi-model rules, most free-text captions
                        would require human review before automated publishing — that does not mean all other metadata
                        is unusable, but thresholds for auto-trust are an active research question.</li>
                    <li>Do not rely on this resource alone for species identification, pest management, or regulatory decisions.</li>
                </ul>
            </details>
            <div class="resource-ack">
                <a href="https://nairrpilot.org/" target="_blank" rel="noopener noreferrer" title="NAIRR Pilot" aria-label="NAIRR Pilot">
                    <svg xmlns="http://www.w3.org/2000/svg" width="220" height="48" viewBox="0 0 220 48" role="img" aria-hidden="true">
                      <rect width="220" height="48" rx="6" fill="#1a4480"/>
                      <text x="110" y="21" fill="#ffffff" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" text-anchor="middle">NAIRR Pilot</text>
                      <text x="110" y="36" fill="#c7dafc" font-family="Arial, Helvetica, sans-serif" font-size="9" text-anchor="middle">National AI Research Resource</text>
                    </svg>
                </a>
                <div class="resource-ack-body">
                    <p><strong>Computational resources &amp; partners.</strong> AIFARMS acknowledges:</p>
                    <ul>
                        <li>The <a href="https://nairrpilot.org/" target="_blank" rel="noopener noreferrer">National Artificial Intelligence Research Resource (NAIRR) Pilot</a>
                            and the Delta advanced computing and data resource (NSF award
                            <a href="https://nairrpilot.org/about" target="_blank" rel="noopener noreferrer">OAC-2005572</a>).</li>
                        <li><a href="https://www.cloudbank.org/" target="_blank" rel="noopener noreferrer">CloudBank</a>,
                            which provides NSF-sponsored access to commercial cloud resources through ACCESS and the NAIRR Pilot.</li>
                        <li><a href="https://azure.microsoft.com/en-us" target="_blank" rel="noopener noreferrer">Microsoft Azure</a>
                            and Azure OpenAI for vision-language metadata generation and query understanding.</li>
                        <li><a href="https://cloud.sambanova.ai/" target="_blank" rel="noopener noreferrer">SambaNova Cloud</a>
                            for multi-model inference and comparison resources used in metadata evaluation.</li>
                    </ul>
                </div>
            </div>
        </footer>
        """

    def _get_search_template(self) -> str:
        """Get the search interface HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AIFARMS Species Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .search-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .form-row { display: flex; gap: 20px; margin-bottom: 15px; }
        .form-group { flex: 1; min-width: 0; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #1a1a1a; font-size: 14px; }
        .form-group select, .form-group input { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; background: #fff; color: #222; }
        .form-group.form-group-dataset label { color: #1a1a1a; font-weight: bold; }
        .form-group.form-group-dataset select { border-color: #ccc; background: #fff; color: #222; }
        .search-button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .search-button:hover { background: #0056b3; }
        .mcp-info { background: #d4edda; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .dataset-selector { margin-bottom: 20px; }
        .llm-search-section { margin-top: 30px; padding: 20px; background: #f0f7ff; border-radius: 8px; }
        .llm-search-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .llm-search-button { background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .llm-search-button:hover { background: #218838; }
        .llm-examples { margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; }
        .llm-examples h4 { margin-bottom: 10px; }
        .llm-examples ul { list-style: none; padding: 0; margin: 0; }
        .llm-examples li { margin-bottom: 5px; }
        """ + self._get_site_footer_css() + """
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="/static/aifarms_logo.png" alt="AIFARMS" style="max-width: 320px; height: auto; margin-bottom: 15px;">
            <h1>🌿 AIFARMS Species Observation Search</h1>
            <p>Search across multiple datasets using AI-powered natural language queries</p>
        </div>
        
        {% if error %}
        <div class="error">
            <strong>Warning:</strong> {{ error }}. Some features may be limited.
        </div>
        {% endif %}
        
        <div class="mcp-info">
            <h3>🤖 MCP Server Integration</h3>
            <p>This interface connects to the AIFARMS MCP Server at <code>{{ mcp_server_url }}</code></p>
            <p><strong>How to search:</strong> You can type a natural language query (e.g. “coyote at night”) to search for a species, or choose a <strong>Category</strong> and <strong>Dataset</strong> from the dropdowns below and run the search to browse that dataset. Leave the query blank to see all images from the selected dataset.</p>
            <p><strong>Available Datasets:</strong> {{ datasets|length }} total</p>
            <p><strong>Features:</strong> AI-powered semantic search, ML model inference, extensible tools</p>
        </div>
            
        <!-- AI-Powered Search Section -->
        <div class="llm-search-section">
            <h3>🧠 AI-Powered Search <span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px;">AI + FILTER MATCHING</span></h3>
            <p>Use natural language to find exactly what you're looking for. Results are ranked by filter match scores for optimal relevance:</p>
            
            <form action="/llm_search" method="post" class="llm-search-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="llm_query">Natural Language Query:</label>
                        <input type="text" id="llm_query" name="query" placeholder="e.g., 'coyote at night' or leave blank to show all images from the selected dataset">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="llm_category">Category (optional):</label>
                        <select id="llm_category" name="category">
                            <option value="">All categories</option>
                            <option value="wildlife">Wildlife</option>
                            <option value="domestic_animal">Domestic animal</option>
                            <option value="livestock">Livestock</option>
                            <option value="plants">Plants</option>
                            <option value="pests">Pests</option>
                            <option value="animal">Animal (all types)</option>
                        </select>
                    </div>
                    <div class="form-group form-group-dataset">
                        <label for="llm_dataset">Dataset (optional)</label>
                        <select id="llm_dataset" name="dataset" title="Restrict search to a specific dataset (filtered by category when set)">
                            <option value="">All datasets</option>
                            <option value="" disabled class="no-match-msg" style="display: none;">(no datasets in this category)</option>
                            {% for dataset in datasets %}
                            <option value="{{ dataset.name }}" data-type="{{ dataset.type|default('') }}">{{ dataset.name|title }}</option>
                            {% else %}
                            <option value="" disabled>(no datasets loaded — check MCP server)</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="llm_limit">Results per page (max 50):</label>
                        <select id="llm_limit" name="limit">
                            <option value="10">10</option>
                            <option value="20">20</option>
                            <option value="50" selected>50</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>&nbsp;</label>
                        <button type="submit" class="llm-search-button">🧠 AI Search</button>
                    </div>
                </div>
            </form>
            
            <div class="llm-examples">
                <h4>Example Queries:</h4>
                <ul>
                    <li><strong>"coyote at night"</strong> - Find coyote images taken at night</li>
                    <li><strong>"goats in the field"</strong> - Goats in field or pasture settings</li>
                    <li><strong>"unripe raspberries"</strong> - Raspberry plants with unripe fruit</li>
                    <li><strong>"Painted Lady"</strong> - Painted Lady butterfly (or moth) images</li>
                </ul>
                
                <div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 6px; border-left: 4px solid #28a745;">
                    <h5 style="margin: 0 0 8px 0; color: #155724;">🎯 Score-Based Ranking</h5>
                    <p style="margin: 0; font-size: 14px; color: #155724;">
                        Results are automatically ranked by filter match scores. The AI confidence score shows how well the query was understood, while filter match scores show how well each image matches the search criteria.
                    </p>
                </div>
            </div>
        </div>
        """ + self._get_site_footer_html() + """
    </div>
    <script>
    (function() {
        var catSelect = document.getElementById('llm_category');
        var dsSelect = document.getElementById('llm_dataset');
        if (!catSelect || !dsSelect) return;
        // Infer type from dataset name (mirrors server); livestock/domestic use exact names only
        function inferTypeFromName(name) {
            if (!name) return 'pests';
            var n = name.toLowerCase().replace(/-/g, '_').replace(/[ \t\n]+/g, '_').trim();
            // Blocklist: never treat as wildlife (pest species)
            if (n === 'stictocephala_bisonia' || n === 'stictocephala_diceros') return 'pests';
            // Domestic animal: only these exact names
            if (n === 'dog' || n === 'domestic_cat') return 'domestic_animal';
            // Livestock: only these exact names
            if (n === 'goat' || n === 'goat_2' || n === 'chicken') return 'livestock';
            // Wildlife: exact allowlist only
            var wildlife = ['american_black_bear','american_crow','bird','bobcat','coyote','eastern_chipmunk','eastern_cottontail','eastern_fox_squirrel','eaestern_gray_squirrel','grey_fox','horse','northern_raccoon','red_fox','striped_skunk','virginia_oppossum','white_tailed_deer','wild_turkey','woodchuck'];
            if (wildlife.indexOf(n) !== -1) return 'wildlife';
            // Plants: exact allowlist only
            var plants = ['almonds','apple','avocado','beets','blueberry','broccoli','capsicum','carrots','celery','citrus','grapes','green_cabbage','iceberg','mango_1','mango_2','orange','raspberry','red_leaf_5_6','red_leaf_8_9','rockmelon','romaine','strawberry_1','strawberry_2','strawberry_3','tomatoes'];
            if (plants.indexOf(n) !== -1) return 'plants';
            // Everything else is pests (3000+ pest datasets)
            return 'pests';
        }
        function filterDatasetsByCategory() {
            var category = (catSelect.value || '').toLowerCase();
            var animalTypes = ['wildlife', 'domestic_animal', 'livestock'];
            var matchCount = 0;
            for (var i = 0; i < dsSelect.options.length; i++) {
                var opt = dsSelect.options[i];
                if (opt.value === '' && !opt.classList.contains('no-match-msg')) { opt.style.display = ''; opt.disabled = false; continue; }
                if (opt.classList.contains('no-match-msg')) { opt.style.display = 'none'; continue; }
                var dtype = (opt.getAttribute('data-type') || '').toLowerCase();
                if (dtype === '' || dtype === 'unknown') dtype = inferTypeFromName(opt.value);
                var show = true;
                if (category === '') show = true;
                else if (category === 'animal') show = animalTypes.indexOf(dtype) !== -1;
                else show = dtype === category;
                if (show) matchCount++;
                opt.style.display = show ? '' : 'none';
                opt.disabled = !show;
            }
            var noMatchOpt = dsSelect.querySelector('option.no-match-msg');
            if (noMatchOpt) {
                noMatchOpt.style.display = (category !== '' && matchCount === 0) ? '' : 'none';
                noMatchOpt.disabled = true;
            }
            var firstShown = null;
            for (var j = 0; j < dsSelect.options.length; j++) {
                if (dsSelect.options[j].classList.contains('no-match-msg')) continue;
                if (dsSelect.options[j].value === '' || dsSelect.options[j].style.display !== 'none') {
                    firstShown = dsSelect.options[j];
                    break;
                }
            }
            if (dsSelect.value && dsSelect.selectedOptions.length && dsSelect.selectedOptions[0].style.display === 'none') {
                dsSelect.value = firstShown ? firstShown.value : '';
            }
        }
        catSelect.addEventListener('change', filterDatasetsByCategory);
        filterDatasetsByCategory();
    })();
    </script>
</body>
</html>
"""
    
    def _get_results_template(self) -> str:
        """Get the search results HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Search Results - AIFARMS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #007bff; text-decoration: none; font-weight: bold; }
        .results-info { background: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }
        .image-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }
        .metadata { margin-top: 10px; font-size: 14px; }
        .download-button { display: inline-block; margin-top: 10px; padding: 8px 16px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; font-size: 14px; }
        .download-button:hover { background: #0056b3; }
        .no-results { text-align: center; padding: 40px; color: #666; }
        """ + self._get_site_footer_css() + """
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">← Back to Search</a>
        </div>
        
        <h1>Search Results</h1>
        
        {% if results.error %}
        <div class="results-info">
            <p><strong>Error:</strong> {{ results.error }}</p>
        </div>
        {% elif results.results %}
        <div class="results-info">
            <p><strong>Query:</strong> {{ query or "All images" }}</p>
            <p><strong>Showing {{ results.results|length }} of {{ results.total_count or results.results|length }} results</strong> — use the buttons below to load more.</p>
            <p><strong>Page:</strong> {{ current_page }} of {{ (results.total_count / limit)|round(0, 'ceil')|int if results.total_count else 1 }}</p>
        </div>
        
        <div class="image-grid">
            {% for result in results.results %}
            <div class="image-card">
                <img src="{{ result.image_url }}" alt="{{ result.metadata.species|title if result.metadata and result.metadata.species else 'Image' }}">
                <div class="metadata">
                    {% if result.image_url %}
                        {% set image_filename = result.image_url.split('/')[-1] %}
                        {% if image_filename %}
                            <a href="/download/{{ image_filename }}" class="download-button" download>📥 Download Image</a>
                        {% endif %}
                    {% endif %}
                    {% if result.metadata %}
                        {% if result.metadata.species %}
                        <div><strong>Species:</strong> {{ result.metadata.species|title }}</div>
                        {% endif %}
                        {% if result.scientific_name %}
                        <div><strong>Scientific name:</strong> <em>{{ result.scientific_name }}</em></div>
                        {% endif %}
                        {% if result.common_names %}
                        <div><strong>Common names:</strong> {{ result.common_names|join(', ')|title }}</div>
                        {% endif %}
                        {% if result.background %}
                        <div><strong>Background:</strong> {{ result.background }}</div>
                        {% elif result.metadata.scene %}
                        <div><strong>Scene:</strong> {{ result.metadata.scene }}</div>
                        {% endif %}
                        {% if result.metadata.action %}
                        <div><strong>Action:</strong> {{ result.metadata.action }}</div>
                        {% endif %}
                        {% if result.metadata.plant_state %}
                        <div><strong>Plant State:</strong> {{ result.metadata.plant_state }}</div>
                        {% endif %}
                        {% if result.metadata.time %}
                        <div><strong>Time:</strong> {{ result.metadata.time }}</div>
                        {% endif %}
                        {% if result.metadata.season %}
                        <div><strong>Season:</strong> {{ result.metadata.season }}</div>
                        {% endif %}
                        {% if result.metadata.weather %}
                        <div><strong>Weather:</strong> {{ result.metadata.weather|title }}</div>
                        {% endif %}
                        {% if result.metadata.lighting %}
                        <div><strong>Lighting:</strong> {{ result.metadata.lighting }}</div>
                        {% endif %}
                        {% if result.metadata.date %}
                        <div><strong>Date:</strong> {{ result.metadata.date }}</div>
                        {% endif %}
                        {% if result.metadata.description %}
                        <div><strong>Description:</strong> {{ result.metadata.description[:100] }}{% if result.metadata.description|length > 100 %}...{% endif %}</div>
                        {% endif %}
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h2>No results found</h2>
            <p>Try adjusting your search criteria or filters.</p>
            <p><strong>Debug Info:</strong></p>
            <p>Query: {{ query }}</p>
            <p>Results structure: {{ results }}</p>
        </div>
        {% endif %}
        """ + self._get_site_footer_html() + """
    </div>
</body>
</html>
"""
    
    def _get_llm_results_template(self) -> str:
        """Get the LLM search results HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM Search Results - AIFARMS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #007bff; text-decoration: none; font-weight: bold; }
        .results-info { background: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .confidence-panel { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .confidence-score { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
        .confidence-bar { background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin: 10px 0; }
        .confidence-fill { background: #4CAF50; height: 100%; border-radius: 4px; transition: width 0.3s ease; }
        .confidence-details { font-size: 14px; opacity: 0.9; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; position: relative; }
        .image-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }
        .confidence-badge { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .metadata { margin-top: 10px; font-size: 14px; }
        .no-results { text-align: center; padding: 40px; color: #666; }
        """ + self._get_site_footer_css() + """
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">← Back to Search</a>
        </div>
        
        <h1>🧠 LLM Search Results</h1>
        
        {% if results.error %}
        <div class="results-info">
            <p><strong>Error:</strong> {{ results.error }}</p>
        </div>
        {% elif results.results %}
        <div class="results-info">
            <p><strong>Query:</strong> {{ query or "All images" }}</p>
            <p><strong>Total results in this batch:</strong> {{ results.total_count or results.results|length }}</p>
            <p><strong>Page:</strong> {{ current_page }} of {{ (results.total_count / limit)|round(0, 'ceil')|int if results.total_count else 1 }}</p>
            {% if total_datasets_matching and total_datasets_matching > 100 %}
            <p><strong>Datasets:</strong> Batch (datasets {{ dataset_offset + 1 }}–{{ dataset_offset + (results.searched_datasets|length) }} of {{ total_datasets_matching }} matching)</p>
            {% endif %}
        </div>
        
        {% if results.llm_understanding %}
        <div class="confidence-panel">
            <div class="confidence-score">
                🧠 AI Confidence: {{ "%.0f"|format(results.llm_understanding.confidence * 100) }}%
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {{ "%.0f"|format(results.llm_understanding.confidence * 100) }}%"></div>
            </div>
            <div class="confidence-details">
                <p><strong>Query Understanding:</strong> This score reflects how confident the AI is that it correctly transformed your natural language query into structured search filters.</p>
                <p><strong>Intent:</strong> {{ results.llm_understanding.intent }}</p>
                <p><strong>Reasoning:</strong> {{ results.llm_understanding.reasoning }}</p>
                <p><strong>Applied Filters:</strong> 
                    {% for filter_type, values in results.llm_understanding.filters.items() %}
                        {% if values %}
                            <strong>{{ filter_type|title }}:</strong> {{ values|join(", ") }}{% if not loop.last %} | {% endif %}
                        {% endif %}
                    {% endfor %}
                </p>
            </div>
        </div>
        {% endif %}
        
        <div class="image-grid">
            {% for result in results.results %}
            <div class="image-card">
                {% if result.llm_confidence %}
                <div class="confidence-badge" title="Filter Match Score: How well this image matches the search criteria">
                    {{ "%.0f"|format(result.llm_confidence * 100) }}%
                </div>
                {% endif %}
                <img src="{{ result.image_url }}" alt="{{ result.metadata.species|title if result.metadata and result.metadata.species else 'Image' }}">
                <div class="metadata">
                    {% if result.metadata %}
                        {% if result.metadata.species %}
                        <div><strong>Species:</strong> {{ result.metadata.species|title }}</div>
                        {% endif %}
                        {% if result.scientific_name %}
                        <div><strong>Scientific name:</strong> <em>{{ result.scientific_name }}</em></div>
                        {% endif %}
                        {% if result.common_names %}
                        <div><strong>Common names:</strong> {{ result.common_names|join(', ')|title }}</div>
                        {% endif %}
                        {% if result.background %}
                        <div><strong>Background:</strong> {{ result.background }}</div>
                        {% endif %}
                        {% if result.metadata.action %}
                        <div><strong>Action:</strong> {{ result.metadata.action }}</div>
                        {% endif %}
                        {% if result.metadata.plant_state %}
                        <div><strong>Plant State:</strong> {{ result.metadata.plant_state }}</div>
                        {% endif %}
                        {% if result.metadata.time %}
                        <div><strong>Time:</strong> {{ result.metadata.time }}</div>
                        {% endif %}
                        {% if result.metadata.season %}
                        <div><strong>Season:</strong> {{ result.metadata.season }}</div>
                        {% endif %}
                        {% if result.metadata.scene and not result.background %}
                        <div><strong>Scene:</strong> {{ result.metadata.scene }}</div>
                        {% endif %}
                        {% if result.metadata.weather %}
                        <div><strong>Weather:</strong> {{ result.metadata.weather|title }}</div>
                        {% endif %}
                        {% if result.metadata.lighting %}
                        <div><strong>Lighting:</strong> {{ result.metadata.lighting }}</div>
                        {% endif %}
                        {% if result.metadata.date %}
                        <div><strong>Date:</strong> {{ result.metadata.date }}</div>
                        {% endif %}
                        {% if result.metadata.description %}
                        <div><strong>Description:</strong> {{ result.metadata.description[:100] }}{% if result.metadata.description|length > 100 %}...{% endif %}</div>
                        {% endif %}
                    {% endif %}
                    {% if result.llm_confidence %}
                    <div style="margin-top: 10px; padding: 5px; background: #f8f9fa; border-radius: 4px;">
                        <small><strong>Filter Match Score:</strong> {{ "%.0f"|format(result.llm_confidence * 100) }}%</small>
                        <br><small style="color: #666;">How well this image matches the search filters</small>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if has_more_datasets %}
        <div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 8px; border: 1px solid #b3d9ff;">
            <p style="margin: 0 0 10px 0;">There are more matching datasets. Load the next batch of up to 100 results.</p>
            <form method="post" action="/llm_search" style="display: inline;">
                <input type="hidden" name="query" value="{{ query }}">
                <input type="hidden" name="dataset" value="{{ dataset or '' }}">
                <input type="hidden" name="category" value="{{ category or '' }}">
                <input type="hidden" name="limit" value="{{ limit }}">
                <input type="hidden" name="page" value="1">
                <input type="hidden" name="dataset_offset" value="{{ next_dataset_offset }}">
                <button type="submit" style="padding: 12px 24px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Load next 100 results</button>
            </form>
            <span style="margin-left: 10px; color: #666;">(datasets {{ next_dataset_offset + 1 }}–{{ [next_dataset_offset + 100, total_datasets_matching]|min }} of {{ total_datasets_matching }})</span>
        </div>
        {% endif %}
        
        {% if results.total_count and results.total_count > limit %}
        <div class="pagination" style="margin-top: 30px; text-align: center; padding: 20px;">
            {% set total_pages = (results.total_count / limit)|round(0, 'ceil')|int %}
            <p style="margin-bottom: 15px;">Page {{ current_page }} of {{ total_pages }} ({{ results.total_count }} total results in this batch)</p>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                {% if current_page > 1 %}
                <form method="post" action="/llm_search" style="display: inline;">
                    <input type="hidden" name="query" value="{{ query }}">
                    <input type="hidden" name="dataset" value="{{ dataset or '' }}">
                    <input type="hidden" name="category" value="{{ category or '' }}">
                    <input type="hidden" name="limit" value="{{ limit }}">
                    <input type="hidden" name="page" value="{{ current_page - 1 }}">
                    <input type="hidden" name="dataset_offset" value="{{ dataset_offset }}">
                    <button type="submit" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">← Previous</button>
                </form>
                {% endif %}
                
                {% for page_num in range(1, total_pages + 1) %}
                    {% if page_num == current_page %}
                    <span style="padding: 10px 15px; background: #007bff; color: white; border-radius: 4px; font-weight: bold;">{{ page_num }}</span>
                    {% elif page_num <= 3 or page_num > total_pages - 3 or (page_num >= current_page - 1 and page_num <= current_page + 1) %}
                    <form method="post" action="/llm_search" style="display: inline;">
                        <input type="hidden" name="query" value="{{ query }}">
                        <input type="hidden" name="dataset" value="{{ dataset or '' }}">
                        <input type="hidden" name="category" value="{{ category or '' }}">
                        <input type="hidden" name="limit" value="{{ limit }}">
                        <input type="hidden" name="page" value="{{ page_num }}">
                        <input type="hidden" name="dataset_offset" value="{{ dataset_offset }}">
                        <button type="submit" style="padding: 10px 15px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;">{{ page_num }}</button>
                    </form>
                    {% elif page_num == 4 or page_num == total_pages - 3 %}
                    <span style="padding: 10px 5px;">...</span>
                    {% endif %}
                {% endfor %}
                
                {% if current_page < total_pages %}
                <form method="post" action="/llm_search" style="display: inline;">
                    <input type="hidden" name="query" value="{{ query }}">
                    <input type="hidden" name="dataset" value="{{ dataset or '' }}">
                    <input type="hidden" name="category" value="{{ category or '' }}">
                    <input type="hidden" name="limit" value="{{ limit }}">
                    <input type="hidden" name="page" value="{{ current_page + 1 }}">
                    <input type="hidden" name="dataset_offset" value="{{ dataset_offset }}">
                    <button type="submit" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Next →</button>
                </form>
                {% endif %}
            </div>
        </div>
        {% endif %}
        {% else %}
        <div class="no-results">
            <h2>No results found</h2>
            <p>Try adjusting your search criteria or filters.</p>
            <p><strong>Debug Info:</strong></p>
            <p>Query: {{ query }}</p>
            <p>Results structure: {{ results }}</p>
        </div>
        {% endif %}
        """ + self._get_site_footer_html() + """
    </div>
</body>
</html>
"""
    
    def _get_croissant_datasets_template(self) -> str:
        """Get the Croissant datasets HTML template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Croissant Datasets - AIFARMS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #007bff; text-decoration: none; font-weight: bold; }
        .dataset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
        .dataset-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: white; }
        .dataset-card h3 { margin-top: 0; color: #333; }
        .dataset-meta { font-size: 14px; color: #666; margin: 10px 0; }
        .dataset-fields { margin: 15px 0; }
        .field-tag { display: inline-block; background: #e9ecef; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 12px; }
        .source-badge { background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .no-results { text-align: center; padding: 40px; color: #666; }
        """ + self._get_site_footer_css() + """
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">← Back to Search</a>
        </div>
        
        <h1>🔍 Croissant Datasets</h1>
        <p>Discovered datasets from AI Institute portals with Croissant metadata</p>
        <p><strong>Total found: {{ total_count }}</strong></p>
        {% if datasets and datasets|length > 0 %}
        <div class="dataset-grid">
            {% for dataset in datasets %}
            <div class="dataset-card">
                <h3>{{ dataset.name }}</h3>
                <div class="dataset-meta">
                    <span class="source-badge">{{ dataset.source }}</span>
                    {% if dataset.license %}
                    <span style="margin-left: 10px;">License: {{ dataset.license }}</span>
                    {% endif %}
                </div>
                <p>{{ dataset.description }}</p>
                
                {% if dataset.fields %}
                <div class="dataset-fields">
                    <strong>Fields:</strong><br>
                    {% for field in dataset.fields[:5] %}
                    <span class="field-tag">{{ field.name }} ({{ field.data_type }})</span>
                    {% endfor %}
                    {% if dataset.fields|length > 5 %}
                    <span class="field-tag">... and {{ dataset.fields|length - 5 }} more</span>
                    {% endif %}
                </div>
                {% endif %}
                
                {% if dataset.keywords %}
                <div class="dataset-fields">
                    <strong>Keywords:</strong><br>
                    {% for keyword in dataset.keywords[:5] %}
                    <span class="field-tag">{{ keyword }}</span>
                    {% endfor %}
                </div>
                {% endif %}
                
                <div style="margin-top: 15px;">
                    <a href="{{ dataset.url }}" target="_blank" style="color: #007bff;">View Dataset →</a>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h2>No Croissant datasets found</h2>
            <p>Try running the crawler to discover datasets from AI Institute portals.</p>
            {% if error %}
            <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
            {% endif %}
            <p><strong>Debug:</strong> datasets variable is {{ datasets }}, length is {{ datasets|length if datasets else 0 }}</p>
        </div>
        {% endif %}
        """ + self._get_site_footer_html() + """
    </div>
</body>
</html>"""
    
    def run(self, host: str = None, port: int = None):
        """Run the web interface"""
        host = host or WEB_CONFIG.get("web_host", "0.0.0.0")
        port = port or WEB_CONFIG.get("web_port", 8187)
        
        print(f"🌐 Starting Web Interface on {host}:{port}")
        print(f"🔗 MCP Server: {self.mcp_server_url}")
        print(f"📱 Open in browser: http://localhost:{port}  or  http://127.0.0.1:{port}")
        if host == "0.0.0.0":
            print(f"   (From another machine use: http://<this-machine-ip>:{port})")
        print(f"🔍 Search: http://localhost:{port}/")
        
        uvicorn.run(self.app, host=host, port=port)

# Global web interface instance
web_interface = WebInterface()

if __name__ == "__main__":
    # Run the web interface
    web_interface.run()
