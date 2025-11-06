"""
Chainlit UI for DeepAgents Search Orchestrator
Interactive web interface for the multi-agent search system
"""

import chainlit as cl
from typing import Optional
from datetime import datetime

from src.core import DeepAgentSearchOrchestrator, Config


# Global orchestrator instance (initialized per session)
orchestrator: Optional[DeepAgentSearchOrchestrator] = None


@cl.on_chat_start
async def start():
    """Initialize the orchestrator when a chat session starts."""
    global orchestrator
    
    await cl.Message(
        content="🚀 **Welcome to DeepAgents Search System!**\n\nInitializing the multi-agent orchestrator with remote MCP servers..."
    ).send()
    
    try:
        # Create config and orchestrator
        config = Config()
        orchestrator = DeepAgentSearchOrchestrator(config)
        
        # Show initialization step
        async with cl.Step(name="Initializing MCP Servers", type="tool") as step:
            step.output = "Connecting to remote MCP servers..."
            
            # Initialize MCP servers
            tools = await orchestrator.initialize_mcp_servers()
            
            step.output = f"✅ Successfully connected to MCP servers\n📦 Loaded {len(tools)} tools"
        
        # Create deep agent
        async with cl.Step(name="Creating DeepAgent", type="tool") as step:
            step.output = "Setting up specialized subagents..."
            
            await orchestrator.create_deep_agent(tools)
            
            step.output = "✅ DeepAgent initialized with:\n- Web Researcher\n- Object Indexer\n- Analytics Tracker"
        
        # Send welcome message
        await cl.Message(
            content="""
✅ **System Ready!**

I'm your AI-powered search orchestrator with access to:
- 🔍 **SerpApi** - Live web search (Google, News, Trends)
- 🗄️ **Algolia** - Semantic indexing and search
- 📊 **OpenSearch** - Analytics and performance tracking

**What would you like to search for?**

Examples:
- "Latest AI breakthroughs in 2024"
- "Financial technology innovations in Europe"  
- "Research on quantum computing applications"
"""
        ).send()
        
        # Store orchestrator in user session
        cl.user_session.set("orchestrator", orchestrator)
        cl.user_session.set("thread_id", f"chainlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    except Exception as e:
        error_msg = f"❌ **Initialization Failed**\n\n```\n{str(e)}\n```\n\nPlease check your environment configuration and MCP servers."
        await cl.Message(content=error_msg).send()
        raise


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and process search queries."""
    
    # Get orchestrator from session
    orchestrator = cl.user_session.get("orchestrator")
    thread_id = cl.user_session.get("thread_id")
    
    if not orchestrator:
        await cl.Message(
            content="❌ Orchestrator not initialized. Please restart the chat."
        ).send()
        return
    
    # Show processing message
    processing_msg = await cl.Message(
        content=f"🔍 **Processing your query...**\n\n> {message.content}\n\nThis may take a few moments as I coordinate multiple agents across remote MCP servers..."
    ).send()
    
    try:
        # Create a step for the query processing
        async with cl.Step(name="Query Processing", type="llm") as main_step:
            main_step.input = message.content
            
            # Process the query
            result = await orchestrator.process_search_query(
                user_query=message.content,
                thread_id=thread_id
            )
            
            if result["success"]:
                main_step.output = f"✅ Query processed successfully in {result['total_latency_ms']:.1f}ms"
                
                # Create response message
                response_content = f"""
## 📋 Search Results

{result['response']}

---

### ⏱️ Performance Metrics
- **Status**: {result['performance_status']}
- **Latency**: {result['total_latency_ms']:.1f}ms
- **Messages**: {result['message_count']}
- **Mode**: {result['network_mode']}
"""
                
                # Add files info if any were created
                if result.get('files_created'):
                    response_content += f"\n### 📁 Files Created\n"
                    for file in result['files_created']:
                        response_content += f"- `{file}`\n"
                
                # Remove the processing message
                await processing_msg.remove()
                
                # Send the final response
                await cl.Message(content=response_content).send()
                
            else:
                main_step.output = f"❌ Query failed: {result.get('error', 'Unknown error')}"
                
                # Remove processing message
                await processing_msg.remove()
                
                # Send error message
                error_content = f"""
❌ **Query Processing Failed**

**Error**: {result.get('error', 'Unknown error')}

**Latency**: {result['total_latency_ms']:.1f}ms

Please try:
- Simplifying your query
- Checking MCP server connectivity
- Reviewing the error details above
"""
                await cl.Message(content=error_content).send()
    
    except Exception as e:
        # Remove processing message
        await processing_msg.remove()
        
        # Send error message
        await cl.Message(
            content=f"❌ **An error occurred**\n\n```\n{str(e)}\n```\n\nPlease try again or restart the chat."
        ).send()
        raise


@cl.on_chat_end
async def end():
    """Clean up when chat session ends."""
    await cl.Message(
        content="👋 **Chat session ended.**\n\nThank you for using DeepAgents Search System!"
    ).send()


# Optional: Add settings/profile callback
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Optional authentication callback.
    Remove this or implement proper authentication as needed.
    """
    # For now, allow all (you can implement real auth here)
    return cl.User(
        identifier=username,
        metadata={"role": "user"}
    )
