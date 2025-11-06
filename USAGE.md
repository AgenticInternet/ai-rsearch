# Usage Guide

## Getting Started

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required Environment Variables
ANTHROPIC_API_KEY=sk-ant-api03-...
MCP_SERPAPI_URL=http://your-serpapi-mcp-url
MCP_ALGOLIA_URL=http://your-algolia-mcp-url
MCP_OPENSEARCH_URL=http://your-opensearch-mcp-url
REDIS_URL=redis://your-redis-url

# Optional
MODEL_NAME=claude-sonnet-4-5-20250929
MODEL_TEMPERATURE=0.1
```

## Running the Application

### Option 1: Chainlit Web Interface (Recommended)

The Chainlit interface provides an interactive chat-based UI with real-time streaming and step visualization.

```bash
# Start the web interface
chainlit run app.py -w

# Access at http://localhost:8000
```

**Features:**
- 💬 Interactive chat interface
- 🔄 Real-time response streaming
- 📊 Step-by-step visualization
- 📁 Session management
- 🎨 Dark/Light theme support

**Usage:**
1. Open http://localhost:8000 in your browser
2. Wait for system initialization (connects to MCP servers)
3. Type your search query in the chat
4. Watch the agents work through the steps
5. View results with performance metrics

**Example Queries:**
```
What are the latest AI breakthroughs in 2024?
Financial technology innovations in Europe
Research on quantum computing applications
Trends in renewable energy technology
```

### Option 2: Command Line Interface

#### Demo Mode

Run a comprehensive demo with a predefined query:

```bash
python main.py demo
```

**What it does:**
- Initializes MCP servers
- Creates DeepAgent
- Runs a demo query about financial breakthroughs in France
- Shows performance metrics
- Saves results to files

#### Interactive Mode

Start an interactive CLI session:

```bash
python main.py interactive
```

**Usage:**
- Enter queries at the prompt
- Type 'clear' to reset conversation context
- Type 'exit', 'quit', or 'q' to quit
- Press Ctrl+C to interrupt

**Features:**
- Conversation context preservation
- Performance metrics
- File output tracking

#### Test Mode

Test MCP server connectivity:

```bash
python main.py test
```

**What it checks:**
- SerpApi MCP server connection
- Algolia MCP server connection
- OpenSearch MCP server connection
- Response times
- Server status

## Understanding the Output

### Chainlit Interface

The Chainlit UI shows:

1. **Initialization Steps**
   - MCP server connection status
   - Tool loading progress
   - Agent creation

2. **Query Processing**
   - Main step showing overall progress
   - Sub-steps for each agent action
   - Real-time status updates

3. **Results**
   - Executive summary
   - Detailed findings
   - Performance metrics
   - Created files list

### CLI Output

The CLI shows:

1. **Initialization**
   ```
   ✅ All required environment variables validated
   🌐 MCP Server Configuration
   🔗 Testing connections to MCP servers...
   ✅ Successfully loaded X MCP tools
   ```

2. **Query Processing**
   ```
   🎯 Processing query via remote MCP servers: [your query]
   🌐 Starting DeepAgent execution...
   ```

3. **Results**
   ```
   ✅ Query processed successfully in Xms
   📁 Files created: [list of files]
   ```

## Advanced Usage

### Using the Orchestrator in Code

```python
import asyncio
from src.core import DeepAgentSearchOrchestrator

async def my_search():
    # Create orchestrator
    orchestrator = DeepAgentSearchOrchestrator()
    
    # Initialize MCP servers
    tools = await orchestrator.initialize_mcp_servers()
    
    # Create agent
    await orchestrator.create_deep_agent(tools)
    
    # Process query
    result = await orchestrator.process_search_query(
        user_query="Your search query here",
        thread_id="my_thread_123"
    )
    
    if result["success"]:
        print(result["response"])
        print(f"Latency: {result['total_latency_ms']:.1f}ms")
    else:
        print(f"Error: {result['error']}")

# Run
asyncio.run(my_search())
```

### Using Individual Components

```python
from src.core import Config
from src.helpers import format_save_objects_batch

# Configuration
config = Config()
print(config.serpapi_url)

# Algolia helpers
objects = [
    {"title": "Doc 1", "content": "Content 1"},
    {"title": "Doc 2", "content": "Content 2"}
]
params = format_save_objects_batch(objects)
```

### Testing Connectivity Programmatically

```python
import asyncio
from src.core import DeepAgentSearchOrchestrator

async def test_servers():
    orchestrator = DeepAgentSearchOrchestrator()
    results = await orchestrator.test_mcp_connectivity()
    
    for server, result in results.items():
        print(f"{server}: {result['status']}")

asyncio.run(test_servers())
```

## Troubleshooting

### Common Issues

#### 1. "Missing required environment variables"

**Solution:** Create a `.env` file with all required variables:
```bash
ANTHROPIC_API_KEY=...
MCP_SERPAPI_URL=...
MCP_ALGOLIA_URL=...
MCP_OPENSEARCH_URL=...
REDIS_URL=...
```

#### 2. "Failed to connect to MCP servers"

**Solutions:**
- Check MCP server URLs are correct and accessible
- Verify servers are running
- Test connectivity: `python main.py test`
- Check firewall/network settings
- Ensure servers use `streamable_http` transport

#### 3. Import errors

**Solutions:**
```bash
# Reinstall dependencies
uv sync

# Verify Python version (3.13+ required)
python --version

# Test imports
python -c "from src.core import Config; print('OK')"
```

#### 4. Redis connection errors

**Solutions:**
- Check Redis URL format: `redis://[password@]host:port/db`
- Verify Redis server is running
- Test connection: `redis-cli ping`

#### 5. Chainlit not starting

**Solutions:**
```bash
# Check Chainlit is installed
chainlit --version

# Try with verbose output
chainlit run app.py -w --debug

# Check port 8000 is available
lsof -i :8000
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export CHAINLIT_DEBUG=true
```

## Performance Optimization

### Target Metrics

- Total latency: <5000ms (including network)
- Individual tool calls: <2000ms average
- Network overhead: Tracked separately

### Optimization Tips

1. **Use Redis caching**
   - Caches LLM responses
   - Reduces API calls
   - Improves response time

2. **Batch operations**
   - Use batch tools when possible
   - Reduces network round trips

3. **Monitor performance**
   - Check latency in results
   - Identify slow operations
   - Optimize network calls

## Security Best Practices

1. **Never commit `.env`**
   - Add to `.gitignore`
   - Use environment variables in production

2. **Secure API keys**
   - Use secrets management
   - Rotate keys regularly
   - Limit key permissions

3. **Authentication**
   - Implement proper auth in Chainlit
   - See `chainlit_app.py` for auth callback
   - Use OAuth or JWT in production

4. **Network security**
   - Use HTTPS for MCP servers
   - Implement rate limiting
   - Monitor for abuse

## Next Steps

- Explore `STRUCTURE.md` for project organization
- Check `README.md` for architecture details
- Review code in `src/` for customization
- Add new features following the modular structure

## Getting Help

1. Check environment variables
2. Test MCP connectivity: `python main.py test`
3. Review error messages
4. Check README.md for setup
5. Review STRUCTURE.md for architecture
