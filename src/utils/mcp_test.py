#!/usr/bin/env python
"""
Test MCP Server Connectivity
Tests if the MCP servers are accessible and responding correctly.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

async def test_mcp_server(name: str, url: str) -> Dict[str, Any]:
    """Test connectivity to a single MCP server."""
    result = {
        "name": name,
        "url": url,
        "status": "unknown",
        "error": None,
        "headers": None,
        "response_time_ms": 0
    }
    
    try:
        import time
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Try to connect with SSE headers
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            }
            
            print(f"🔍 Testing {name} at {url}...")
            
            # Test basic connectivity
            async with session.get(url, headers=headers, timeout=10) as response:
                result["response_time_ms"] = (time.time() - start_time) * 1000
                result["status"] = f"HTTP {response.status}"
                result["headers"] = dict(response.headers)
                
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    if "event-stream" in content_type:
                        result["status"] = "✅ Connected (SSE Ready)"
                        
                        # Try to read initial events
                        try:
                            # Read a small chunk to see if we get SSE data
                            chunk = await response.content.read(512)
                            if chunk:
                                result["initial_data"] = chunk.decode('utf-8', errors='ignore')[:200]
                        except Exception as e:
                            result["read_error"] = str(e)
                    else:
                        result["status"] = f"⚠️ Connected but wrong content-type: {content_type}"
                else:
                    result["status"] = f"❌ HTTP Error {response.status}"
                    
    except asyncio.TimeoutError:
        result["status"] = "❌ Timeout"
        result["error"] = "Connection timed out after 10 seconds"
    except Exception as e:
        result["status"] = "❌ Failed"
        result["error"] = str(e)
    
    return result

async def test_mcp_initialization():
    """Test MCP initialization with streamable_http transport."""
    print("\n🧪 Testing MCP Client Initialization...")
    
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        
        serpapi_url = os.getenv("MCP_SERPAPI_URL", "http://localhost:3001")
        
        print(f"📡 Creating SSE client for {serpapi_url}...")
        
        # Create SSE client with streamable_http transport
        async with sse_client(serpapi_url) as (read, write):
            # Create session
            session = ClientSession(read, write)
            
            print("🔗 Initializing session...")
            await session.initialize()
            
            print("📋 Listing tools...")
            tools_result = await session.list_tools()
            
            if tools_result and tools_result.tools:
                print(f"✅ Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools[:5]:  # Show first 5 tools
                    print(f"  • {tool.name}")
                return True
            else:
                print("⚠️ No tools found")
                return False
                
    except Exception as e:
        print(f"❌ MCP initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 MCP Server Connectivity Test")
    print("=" * 70)
    
    # Load environment variables
    servers = {
        "SerpApi": os.getenv("MCP_SERPAPI_URL", "http://localhost:3001"),
        "Algolia": os.getenv("MCP_ALGOLIA_URL", "http://localhost:3002"),
        "OpenSearch": os.getenv("MCP_OPENSEARCH_URL", "http://localhost:3003")
    }
    
    print("\n📡 Testing server connectivity...\n")
    
    # Test each server
    results = []
    for name, url in servers.items():
        result = await test_mcp_server(name, url)
        results.append(result)
        
        print(f"\n{name}:")
        print(f"  Status: {result['status']}")
        if result['error']:
            print(f"  Error: {result['error']}")
        if result['response_time_ms'] > 0:
            print(f"  Response time: {result['response_time_ms']:.1f}ms")
        if result.get('initial_data'):
            print(f"  Initial data preview: {result['initial_data'][:100]}...")
    
    # Test MCP client initialization
    print("\n" + "=" * 70)
    success = await test_mcp_initialization()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    connected = sum(1 for r in results if "✅" in r["status"])
    failed = sum(1 for r in results if "❌" in r["status"])
    
    print(f"✅ Connected: {connected}")
    print(f"❌ Failed: {failed}")
    
    if connected == len(servers):
        print("\n🎉 All servers are accessible!")
    elif connected > 0:
        print(f"\n⚠️ {connected}/{len(servers)} servers are accessible")
    else:
        print("\n❌ No servers are accessible. Please check:")
        print("  1. Server URLs are correct")
        print("  2. Servers are running and accessible")
        print("  3. Network/firewall settings")
    
    return connected > 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)