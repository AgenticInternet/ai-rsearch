"""
Helper functions for properly formatting Algolia MCP tool parameters.
This module ensures correct parameter formatting to avoid validation errors.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


def format_save_objects_batch(
    objects: List[Dict[str, Any]], 
    index_name: str = "ai_rsearch"
) -> Dict[str, str]:
    """
    Format objects for the save_objects_batch Algolia tool.
    
    Args:
        objects: List of dictionaries to save to Algolia
        index_name: Name of the Algolia index
        
    Returns:
        Properly formatted parameters dict with objects_json as a JSON string
    """
    # Ensure each object has an objectID
    for i, obj in enumerate(objects):
        if 'objectID' not in obj:
            # Generate a unique ID using timestamp and UUID
            obj['objectID'] = f"{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
    
    # Convert the list of objects to a JSON string
    objects_json = json.dumps(objects, ensure_ascii=False)
    
    return {
        "index_name": index_name,
        "objects_json": objects_json  # MUST be a string, not a list!
    }


def format_save_object(
    obj: Dict[str, Any], 
    index_name: str = "ai_rsearch"
) -> Dict[str, str]:
    """
    Format a single object for the save_object Algolia tool.
    
    Args:
        obj: Dictionary to save to Algolia
        index_name: Name of the Algolia index
        
    Returns:
        Properly formatted parameters dict
    """
    # Ensure object has an objectID
    if 'objectID' not in obj:
        obj['objectID'] = f"{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
    
    return {
        "index_name": index_name,
        "object": json.dumps(obj, ensure_ascii=False)
    }


def format_search_index(
    query: str,
    index_name: str = "ai_rsearch",
    hits_per_page: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Format parameters for the search_index Algolia tool.
    
    Args:
        query: Search query string
        index_name: Name of the Algolia index
        hits_per_page: Number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        Properly formatted parameters dict
    """
    params = {
        "index_name": index_name,
        "query": query,
        "hitsPerPage": hits_per_page
    }
    
    # Add any additional parameters
    params.update(kwargs)
    
    return params


def create_search_result_object(
    title: str,
    content: str,
    url: Optional[str] = None,
    source: str = "web_search",
    category: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a properly formatted object for indexing search results.
    
    Args:
        title: Title of the document
        content: Main content/description
        url: Optional URL source
        source: Source of the data (default: "web_search")
        category: Optional category/classification
        metadata: Optional additional metadata
        
    Returns:
        Formatted object ready for Algolia indexing
    """
    obj = {
        "objectID": f"{source}_{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}",
        "title": title,
        "content": content,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }
    
    if url:
        obj["url"] = url
    
    if category:
        obj["category"] = category
    
    if metadata:
        obj["metadata"] = metadata
    
    return obj


# Example usage for agents:
if __name__ == "__main__":
    # Example 1: Format multiple search results for batch saving
    search_results = [
        {
            "title": "AI Multiagent Systems",
            "content": "Recent advances in multiagent AI systems...",
            "url": "https://example.com/article1"
        },
        {
            "title": "MCP Server Architecture",
            "content": "Building scalable MCP servers for AI agents...",
            "url": "https://example.com/article2"
        }
    ]
    
    # Convert to Algolia objects
    algolia_objects = [
        create_search_result_object(
            title=result["title"],
            content=result["content"],
            url=result.get("url"),
            category="research"
        )
        for result in search_results
    ]
    
    # Format for save_objects_batch tool
    batch_params = format_save_objects_batch(algolia_objects)
    print("Batch save parameters:")
    print(json.dumps(batch_params, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Format search parameters
    search_params = format_search_index(
        query="multiagent AI systems",
        hits_per_page=20,
        attributesToHighlight=["title", "content"]
    )
    print("Search parameters:")
    print(json.dumps(search_params, indent=2))