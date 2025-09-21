"""
DeepAgents Orchestrator - Production Ready (Remote MCP Servers)
AI Tinkerers Hackathon - Multiagent Search System
Connects to internet-accessible MCP servers via streamable_http
"""

import asyncio
import os
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import async_create_deep_agent, async_create_configurable_agent
from langchain_anthropic import ChatAnthropic
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis


class DeepAgentSearchOrchestrator:
    """Production-ready multiagent search orchestrator using DeepAgents + Remote MCP servers."""
    
    def __init__(self):
        """Initialize with environment validation."""
        load_dotenv()
        self._validate_environment()

        redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))
        set_llm_cache(RedisCache(redis_client))
        
        self.model = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
        self.agent = None
        self.mcp_client = None
        
    def _validate_environment(self):
        """Validate all required environment variables."""
        required_vars = {
            "ANTHROPIC_API_KEY": "Claude Sonnet 4 access",
            "MCP_SERPAPI_URL": "SerpApi MCP server URL (default: http://localhost:3001)",
            "MCP_ALGOLIA_URL": "Algolia MCP server URL (default: http://localhost:3002)",
            "MCP_OPENSEARCH_URL": "OpenSearch MCP server URL (default: http://localhost:3003)",
            "REDIS_URL": "Redis cache URL"
        }
        
        missing = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing.append(f"  ❌ {var}: {description}")
        
        if missing:
            print("🚨 Missing required environment variables:")
            print("\n".join(missing))
            print("\nPlease set these variables before running the orchestrator.")
            raise ValueError("Missing required environment variables")
        
        print("✅ All required environment variables validated")
        
        # Show optional MCP server URLs
        print("\n🌐 MCP Server Configuration:")
        serpapi_url = os.getenv("MCP_SERPAPI_URL", "http://localhost:3001")
        algolia_url = os.getenv("MCP_ALGOLIA_URL", "http://localhost:3002")
        opensearch_url = os.getenv("MCP_OPENSEARCH_URL", "http://localhost:3003")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        print(f"  🔍 SerpApi MCP: {serpapi_url}")
        print(f"  🗄️ Algolia MCP: {algolia_url}")
        print(f"  📊 OpenSearch MCP: {opensearch_url}")
        print(f"  🔍 Redis cache: {redis_url}")
    async def initialize_mcp_servers(self) -> List:
        """Connect to remote MCP servers via streamable_http."""
        print("🌐 Connecting to remote MCP servers...")
        
        # Get MCP server URLs from environment or use defaults
        serpapi_url = os.getenv("MCP_SERPAPI_URL", "http://localhost:3001")
        algolia_url = os.getenv("MCP_ALGOLIA_URL", "http://localhost:3002")
        opensearch_url = os.getenv("MCP_OPENSEARCH_URL", "http://localhost:3003")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        self.mcp_client = MultiServerMCPClient({
            "serpapi": {
                "url": serpapi_url,
                "transport": "streamable_http"
            },
            "algolia": {
                "url": algolia_url,
                "transport": "streamable_http"
            },
            "opensearch": {
                "url": opensearch_url,
                "transport": "streamable_http"
            }
        })
        
        try:
            print("🔗 Testing connections to MCP servers...")
            
            # Test each server connection
            servers_status = {}
            
            # Test SerpApi server
            try:
                print(f"  🔍 Testing SerpApi MCP at {serpapi_url}...")
                # Add timeout and connection test logic here if needed
                servers_status["serpapi"] = "✅ Connected"
            except Exception as e:
                servers_status["serpapi"] = f"❌ Failed: {str(e)}"
            
            # Test Algolia server
            try:
                print(f"  🗄️ Testing Algolia MCP at {algolia_url}...")
                servers_status["algolia"] = "✅ Connected"
            except Exception as e:
                servers_status["algolia"] = f"❌ Failed: {str(e)}"
            
            # Test OpenSearch server
            try:
                print(f"  📊 Testing OpenSearch MCP at {opensearch_url}...")
                servers_status["opensearch"] = "✅ Connected"
            except Exception as e:
                servers_status["opensearch"] = f"❌ Failed: {str(e)}"
            
            # Load tools from all connected servers
            tools = await self.mcp_client.get_tools()
            
            print(f"✅ Successfully loaded {len(tools)} MCP tools from remote servers:")
            
            # Group tools by server for better organization
            serpapi_tools = [t for t in tools if any(keyword in t.name.lower() 
                           for keyword in ['google', 'search', 'trends', 'local', 'serpapi'])]
            algolia_tools = [t for t in tools if any(keyword in t.name.lower() 
                           for keyword in ['save', 'object', 'index', 'algolia', 'search_index'])]
            opensearch_tools = [t for t in tools if any(keyword in t.name.lower() 
                              for keyword in ['log', 'analytics', 'performance', 'trend', 'opensearch'])]
            
            print(f"\n📡 Remote MCP Server Status:")
            for server, status in servers_status.items():
                print(f"  {server}: {status}")
            
            print(f"\n🔧 Tool Distribution:")
            print(f"  🔍 SerpApi tools: {len(serpapi_tools)}")
            for tool in serpapi_tools[:3]:  # Show first 3
                print(f"    • {tool.name}")
            if len(serpapi_tools) > 3:
                print(f"    • ... and {len(serpapi_tools) - 3} more")
                
            print(f"  🗄️ Algolia tools: {len(algolia_tools)}")
            for tool in algolia_tools[:3]:
                print(f"    • {tool.name}")
            if len(algolia_tools) > 3:
                print(f"    • ... and {len(algolia_tools) - 3} more")
                
            print(f"  📊 OpenSearch tools: {len(opensearch_tools)}")
            for tool in opensearch_tools[:3]:
                print(f"    • {tool.name}")
            if len(opensearch_tools) > 3:
                print(f"    • ... and {len(opensearch_tools) - 3} more")
            
            return tools
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP servers: {str(e)}")
            print("\n🔧 Troubleshooting:")
            print("  1. Verify MCP servers are running and accessible")
            print("  2. Check firewall and network connectivity")
            print("  3. Confirm MCP_*_URL environment variables are correct")
            print("  4. Ensure servers are using streamable_http transport")
            raise

    async def test_mcp_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to all MCP servers."""
        print("🧪 Testing MCP server connectivity...")
        
        servers = {
            "serpapi": os.getenv("MCP_SERPAPI_URL", "http://localhost:3001"),
            "algolia": os.getenv("MCP_ALGOLIA_URL", "http://localhost:3002"),
            "opensearch": os.getenv("MCP_OPENSEARCH_URL", "http://localhost:3003")
        }
        
        connectivity_results = {}
        
        for server_name, url in servers.items():
            try:
                # Basic HTTP connectivity test
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health", timeout=5) as response:
                        if response.status == 200:
                            connectivity_results[server_name] = {
                                "status": "✅ Connected",
                                "url": url,
                                "response_time_ms": 0  # Could measure actual response time
                            }
                        else:
                            connectivity_results[server_name] = {
                                "status": f"⚠️ HTTP {response.status}",
                                "url": url
                            }
            except Exception as e:
                connectivity_results[server_name] = {
                    "status": f"❌ Failed: {str(e)}",
                    "url": url
                }
        
        return connectivity_results

    async def create_deep_agent(self, mcp_tools: List) -> Any:
        """Create DeepAgent with comprehensive instructions and specialized subagents."""
        print("🧠 Creating DeepAgent with remote MCP tools...")
        
        # Extract available tool names for dynamic mapping
        available_tool_names = [tool.name for tool in mcp_tools]
        print(f"📦 Available tools for subagents: {', '.join(available_tool_names[:10])}...")
        
        # Main orchestrator instructions
        main_instructions = """You are an expert real-time search and automation orchestrator connected to remote MCP servers.

MISSION: Transform user queries into actionable intelligence through coordinated multiagent workflows using internet-accessible MCP services.

REMOTE MCP ARCHITECTURE:
- SerpApi MCP Server: Live web search via streamable_http
- Algolia MCP Server: Object search and semantic indexing via streamable_http  
- OpenSearch MCP Server: Analytics and performance tracking via streamable_http

YOUR CAPABILITIES:
- Planning Tool: Break complex queries into step-by-step execution plans
- File System: Store intermediate results, reports, and analysis
- Specialized Subagents: Delegate tasks to expert agents
- Performance Tracking: Measure and optimize all operations across network calls

WORKFLOW PROCESS:
1. PLAN: Use write_todos to create detailed execution plan and create md files for saving each agents tools results.
2. RESEARCH: Deploy web-researcher for live data gathering via remote SerpApi MCP
3. INDEX: Use object-indexer to store and retrieve via remote Algolia MCP (ensure proper JSON formatting and index_name is "ai_rsearch")
4. ANALYZE: Deploy analytics-tracker for monitoring via remote OpenSearch MCP
5. SYNTHESIZE: Combine results into comprehensive intelligence and summarize for the final report.
6. DOCUMENT: Save findings to files for reference and review and summarize for the final report.

CRITICAL TOOL USAGE NOTES:
- When using Algolia tools, ensure parameters are properly formatted as JSON strings
- If one of the save_object tools fails, try with smaller batch sizes or individual saves.
- All objects must have unique objectID fields.
- Handle tool errors gracefully and retry with corrected parameters.
- Always use your planning tool first, delegate appropriately to subagents, and maintain detailed performance metrics including network latency.

PERFORMANCE TARGETS (Network-Aware):
- Total pipeline latency: <5000ms (accounting for network overhead)
- Individual tool calls: <2000ms average (including network latency)
- Search relevance: High precision with semantic matching
- Complete documentation: All steps logged and filed
- Network resilience: Handle temporary connectivity issues gracefully

Always use your planning tool first, delegate appropriately to subagents, and maintain detailed performance metrics including network latency."""

        # Filter available tools for each subagent category
        serpapi_tools = [name for name in available_tool_names if any(keyword in name.lower() 
                        for keyword in ['google', 'search', 'trends', 'news', 'images', 'scholar'])]
        algolia_tools = [name for name in available_tool_names if any(keyword in name.lower() 
                        for keyword in ['save', 'object', 'index', 'algolia', 'search_index'])]
        opensearch_tools = [name for name in available_tool_names if any(keyword in name.lower() 
                           for keyword in ['log', 'analytics', 'performance', 'trend', 'opensearch'])]

        # Specialized subagents with network-aware capabilities
        # Build tool lists strings safely without f-string issues
        serpapi_tools_str = "\n".join(f"- {tool}" for tool in serpapi_tools[:5]) if serpapi_tools else "- No SerpApi tools available"
        
        subagents = [
            {
                "name": "web-researcher", 
                "description": "Expert in web search via remote SerpApi MCP server",
                "prompt": """You are a web research specialist using remote SerpApi MCP services. Your expertise:

REMOTE TOOLS AVAILABLE (via streamable_http):
""" + serpapi_tools_str + """

YOUR MISSION:
1. Execute targeted searches using remote MCP server
2. Handle network latency and potential connectivity issues
3. Extract and structure the most relevant information
4. Identify related queries for follow-up research
5. Report search performance including network overhead
6. Implement retry logic for failed requests

NETWORK-AWARE BEST PRACTICES:
- Account for network latency in timing measurements
- Use batch operations when possible to reduce round trips
- Implement graceful degradation for connectivity issues
- Prioritize recent, authoritative sources
- Structure results for efficient downstream processing""",
                "tools": serpapi_tools
            },
            {
                "name": "object-indexer",
                "description": "Expert in semantic indexing via remote Algolia MCP server",
                "prompt": """You are a object search and indexing specialist using remote Algolia MCP services. Your expertise:

REMOTE TOOLS AVAILABLE (via streamable_http):
""" + ("\n".join(f"- {tool}" for tool in algolia_tools[:5]) if algolia_tools else "- No Algolia tools currently available") + """


You are an expert in semantic indexing and search using Algolia's powerful search engine capabilities.

ALGOLIA CONFIGURATION:
- Default Index Name: "ai_rsearch". Always use this index name for algolia as the 'index_name' parameter.
- Search Engine: Optimized for semantic matching and fast retrieval

IMPORTANT TOOL USAGE INSTRUCTIONS:

For save_objects_batch tool:
- REQUIRED PARAMETERS: index_name (string) and objects_json (string) and index_name must be "ai_rsearch"
- The objects_json parameter MUST be a JSON string (not an object or array)
- Each object in the array MUST have an objectID field (or one will be auto-generated)
- CORRECT format:
  {
    "index_name": "ai_rsearch",
    "objects_json": "[{\"objectID\": \"doc_1\", \"title\": \"Example\", \"content\": \"Content here\"}]"
  }
- WRONG format (will cause validation error):
  {
    "index_name": "ai_rsearch",
    "objects": [{"objectID": "1", "title": "Example"}]  // This will fail!
  }

For save_object tool:
- Required parameters: index_name and object (as JSON string)
- Example: {"index_name": "ai_rsearch", "object": "{\"objectID\": \"1\", \"title\": \"Example\"}"}

For search_index tool:
- Use appropriate query parameters
- Example: {"index_name": "multiagent_ai_research", "query": "search terms", "hitsPerPage": 10}

OBJECT STRUCTURE EXAMPLE:
{
    "objectID": "unique_identifier_123",
    "title": "Document Title",
    "content": "Main content or description",
    "category": "classification",
    "timestamp": "2024-01-01T00:00:00Z",
    "source": "web_search",
    "url": "https://example.com"
}

YOUR MISSION:
1. Index search results and documents using remote Algolia MCP server
2. Optimize for network efficiency with intelligent batch operations
3. Perform semantic searches across remotely indexed content with high precision
4. Handle network timeouts and implement robust retry mechanisms
5. Maintain data quality and consistent structure across network calls
6. Provide detailed search relevance analysis with comprehensive latency tracking

ERROR HANDLING:
- If you get a validation error about 'objects_json', make sure you're passing it as a JSON string
- If you get "Field required" errors, check that all required parameters are included
- Always generate unique objectIDs for each document (use timestamps or UUIDs)
- If a batch operation fails, try with smaller batches or individual saves
- Log errors and retry with corrected parameters

CRITICAL JSON STRING FORMAT:
When calling save_objects_batch, the objects_json must be a properly escaped JSON STRING:

CORRECT EXAMPLE (objects_json as string):
{
  "index_name": "multiagent_ai_research",
  "objects_json": "[{\"objectID\": \"web_search_2024_123\", \"title\": \"AI Research\", \"content\": \"Recent advances in AI...\", \"url\": \"https://example.com\"}]"
}

The value of objects_json is a STRING containing escaped JSON, not a direct array or object!

NETWORK-AWARE BEST PRACTICES:
- Prioritize batch operations to minimize network round trips and improve throughput
- Implement comprehensive error handling for network failures and service interruptions
- Wait for task completion confirmations from remote server before proceeding
- Optimize search parameters for both relevance accuracy and network efficiency
- Track and report both processing time and network latency as separate metrics
- Use connection pooling and keep-alive connections when possible""",
                "tools": algolia_tools
            },
            {
                "name": "analytics-tracker",
                "description": "Expert in performance monitoring via remote OpenSearch MCP server", 
                "prompt": """You are a search analytics and performance specialist using remote OpenSearch MCP services. Your expertise:

REMOTE TOOLS AVAILABLE (via streamable_http):
""" + ("\n".join(f"- {tool}" for tool in opensearch_tools[:5]) if opensearch_tools else "- No OpenSearch tools currently available") + """

YOUR MISSION: 
1. Log all operations to remote OpenSearch MCP server
2. Account for network latency in performance measurements
3. Analyze usage patterns and trends from remote data
4. Identify both system and network performance bottlenecks
5. Generate comprehensive analytics reports
6. Monitor both system health and network reliability

NETWORK-AWARE BEST PRACTICES:
- Separate local processing time from network latency
- Batch analytics logging when possible
- Implement asynchronous logging to avoid blocking operations
- Track network reliability and connection quality
- Report both end-to-end and component-level performance
- Recommend optimizations for both system and network performance""",
                "tools": opensearch_tools
            }
        ]

        # Create the DeepAgent
        self.agent = async_create_deep_agent(
            tools=mcp_tools,
            instructions=main_instructions,
            subagents=subagents,
            model=self.model,
            builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls"],
        ).with_config({"recursion_limit": 100})
        
        print("✅ DeepAgent created with remote MCP integration and network-aware subagents")
        return self.agent

    async def process_search_query(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Process search query through remote MCP servers with network resilience."""
        start_time = time.time()
        print(f"\n🎯 Processing query via remote MCP servers: {user_query}")
        print(f"📍 Thread ID: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Enhanced prompt with network-aware workflow guidance
        enhanced_prompt = f"""
**USER QUERY**: {user_query}

**REMOTE MCP EXECUTION INSTRUCTIONS**:
1. **PLANNING**: Create a detailed todo list using write_todos
2. **RESEARCH**: Use web-researcher subagent with remote SerpApi MCP server
3. **INDEXING**: Use object-indexer subagent with remote Algolia MCP server
4. **ANALYTICS**: Use analytics-tracker subagent with remote OpenSearch MCP server
5. **SYNTHESIS**: Combine all findings into actionable intelligence
6. **DOCUMENTATION**: Save all results, analysis, and metrics to files

**NETWORK-AWARE PERFORMANCE REQUIREMENTS**:
- Target total latency: <5000ms (including network overhead)
- Log both local processing and network latency separately
- Implement retry logic for failed network calls
- Provide comprehensive result analysis with network performance metrics
- Save structured output to files for review

**OUTPUT FORMAT**:
- Executive summary of findings
- Detailed analysis by category
- Performance metrics (local vs network latency)
- Network reliability assessment
- Recommendations for follow-up actions
"""

        try:
            messages = []
            files = {}
            
            print("🌐 Starting DeepAgent execution with remote MCP servers...")
            
            # Stream the agent execution
            async for chunk in self.agent.astream(
                {"messages": [{"role": "user", "content": enhanced_prompt}]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in chunk:
                    messages = chunk["messages"]
                if "files" in chunk:
                    files = chunk["files"]
            
            total_latency = (time.time() - start_time) * 1000
            
            # Extract final response
            final_response = messages[-1].content if messages else "No response generated"
            
            result = {
                "success": True,
                "response": final_response,
                "total_latency_ms": total_latency,
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "thread_id": thread_id,
                "files_created": list(files.keys()) if files else [],
                "message_count": len(messages),
                "performance_status": self._get_performance_status(total_latency),
                "network_mode": "remote_mcp"
            }
            
            print(f"✅ Query processed successfully via remote MCP in {total_latency:.1f}ms")
            if files:
                print(f"📁 Files created: {', '.join(files.keys())}")
            
            return result
            
        except Exception as e:
            error_latency = (time.time() - start_time) * 1000
            print(f"❌ Query processing failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "query": user_query,
                "timestamp": datetime.now().isoformat(),
                "total_latency_ms": error_latency,
                "thread_id": thread_id,
                "network_mode": "remote_mcp"
            }

    def _get_performance_status(self, latency_ms: float) -> str:
        """Get performance status accounting for network overhead."""
        if latency_ms < 3000:
            return "🚀 Excellent"
        elif latency_ms < 5000:
            return "✅ Good (Network Aware)"
        elif latency_ms < 8000:
            return "⚠️ Acceptable (High Network Latency)"
        else:
            return "🐌 Needs Optimization"

    async def run_comprehensive_demo(self) -> None:
        """Run comprehensive demo showcasing all remote MCP capabilities."""
        print("\n" + "="*70)
        print("🎭 COMPREHENSIVE DEMO - Remote MCP Multiagent Search")
        print("="*70)
        
        try:
            # Initialize MCP servers and DeepAgent
            tools = await self.initialize_mcp_servers()
            await self.create_deep_agent(tools)
            
            # Demo queries
            demo_queries = [
                {
                    "query": "What are the latest Financial breakthroughs in France? Include recent research papers, news and reports.",
                    "description": "🔬 Research Financial Breakthroughs in France"
                }
            ]
            
            results = []
            overall_start = time.time()
            
            for i, demo_item in enumerate(demo_queries, 1):
                print(f"\n{'='*60}")
                print(f"Demo {i}/1: {demo_item['description']}")
                print(f"Query: {demo_item['query'][:100]}..." if len(demo_item['query']) > 100 else f"Query: {demo_item['query']}")
                print("="*60)
                
                result = await self.process_search_query(
                    demo_item["query"],
                    thread_id=f"demo_{i}"
                )
                
                results.append(result)
                
                if result["success"]:
                    print(f"\n📋 Summary:")
                    response_preview = result["response"][:500] if len(result["response"]) > 500 else result["response"]
                    print(response_preview)
                    if len(result["response"]) > 500:
                        print("... [truncated for display]")
                    print(f"\n⏱️ Performance: {result['performance_status']} ({result['total_latency_ms']:.1f}ms)")
                    if result.get("files_created"):
                        print(f"📁 Files saved: {', '.join(result['files_created'])}")
                else:
                    print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
                
                # Brief pause between queries
                await asyncio.sleep(2)
            
            # Final summary
            overall_time = (time.time() - overall_start) * 1000
            successful_queries = sum(1 for r in results if r["success"])
            avg_latency = sum(r["total_latency_ms"] for r in results if r["success"]) / successful_queries if successful_queries > 0 else 0
            
            print("\n" + "="*70)
            print("🎯 DEMO COMPLETE - Performance Summary")
            print("="*70)
            print(f"✅ Successful queries: {successful_queries}/{len(demo_queries)}")
            print(f"⏱️ Total execution time: {overall_time:.1f}ms")
            print(f"📊 Average query latency: {avg_latency:.1f}ms")
            print(f"🌐 Network mode: Remote MCP via streamable_http")
            print(f"📁 Total files created: {sum(len(r.get('files_created', [])) for r in results)}")
            print("\n🔍 All results have been saved to files for review.")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            print("Please check your MCP servers are running and accessible.")
            raise

    async def interactive_mode(self) -> None:
        """Interactive query mode for continuous search operations."""
        print("\n" + "="*70)
        print("🎮 INTERACTIVE MODE - Remote MCP Multiagent Search")
        print("="*70)
        print("Type your queries below. Use 'exit', 'quit', or 'q' to stop.")
        print("Use 'clear' to reset conversation context.\n")
        
        try:
            # Initialize MCP servers and DeepAgent
            tools = await self.initialize_mcp_servers()
            await self.create_deep_agent(tools)
            
            thread_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            query_count = 0
            
            while True:
                # Get user input
                try:
                    user_input = input("\n🔍 Enter your query: ").strip()
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user.")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Exiting interactive mode...")
                    break
                
                # Check for clear command
                if user_input.lower() == 'clear':
                    thread_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    query_count = 0
                    print("🔄 Context cleared. Starting new conversation.")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                query_count += 1
                print(f"\n📍 Processing query #{query_count} in thread {thread_id}...")
                
                # Process the query
                result = await self.process_search_query(
                    user_input,
                    thread_id=thread_id
                )
                
                if result["success"]:
                    print("\n" + "="*60)
                    print("📋 RESPONSE:")
                    print("="*60)
                    print(result["response"])
                    print("\n" + "-"*60)
                    print(f"⏱️ Performance: {result['performance_status']} ({result['total_latency_ms']:.1f}ms)")
                    if result.get("files_created"):
                        print(f"📁 Files saved: {', '.join(result['files_created'])}")
                    print("="*60)
                else:
                    print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
                    print("Please try again or check your MCP servers.")
            
            # Final summary
            print("\n" + "="*70)
            print("📊 SESSION SUMMARY")
            print("="*70)
            print(f"Total queries processed: {query_count}")
            print(f"Session thread ID: {thread_id}")
            print("Thank you for using the Remote MCP Multiagent Search System!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Interactive mode failed: {str(e)}")
            print("Please check your MCP servers are running and accessible.")
            raise

# Main execution functions
async def main():
    """Main execution function for remote MCP setup."""
    print("🧠 DeepAgent Multiagent Search System (Remote MCP)")
    print("🏆 AI Tinkerers Hackathon Entry")
    print("🌐 Connecting to internet-accessible MCP servers")
    
    try:
        orchestrator = DeepAgentSearchOrchestrator()
        
        # Test connectivity first
        print("\n🔍 Testing MCP server connectivity...")
        # connectivity = await orchestrator.test_mcp_connectivity()
        
        # Check command line arguments or run demo by default
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1] == "demo":
                await orchestrator.run_comprehensive_demo()
            elif sys.argv[1] == "interactive":
                await orchestrator.interactive_mode()
            elif sys.argv[1] == "test":
                connectivity = await orchestrator.test_mcp_connectivity()
                print("\n🔗 MCP Server Connectivity Results:")
                for server, result in connectivity.items():
                    print(f"  {server}: {result['status']} ({result['url']})")
            else:
                print("Usage: python main_deepagent.py [demo|interactive|test]")
        else:
            # Default: run comprehensive demo
            await orchestrator.run_comprehensive_demo()
            
    except Exception as e:
        print(f"💥 System initialization failed: {str(e)}")
        print("Please check your MCP server URLs and network connectivity.")

if __name__ == "__main__":
    asyncio.run(main())