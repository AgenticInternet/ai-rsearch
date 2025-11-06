# Quick Start Guide 🚀

Get up and running with the DeepAgents Search System in 3 steps!

## Prerequisites

- Python 3.13+
- `uv` package manager installed
- API keys and MCP server access

## Step 1: Install Dependencies

```bash
uv sync
```

This installs all required packages including Chainlit.

## Step 2: Configure Environment

Create a `.env` file in the project root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
MCP_SERPAPI_URL=http://your-serpapi-server-url
MCP_ALGOLIA_URL=http://your-algolia-server-url
MCP_OPENSEARCH_URL=http://your-opensearch-server-url
REDIS_URL=redis://your-redis-url

# Optional (with defaults)
MODEL_NAME=claude-sonnet-4-5-20250929
MODEL_TEMPERATURE=0.1
```

## Step 3: Run the Application

### Option A: Web Interface (Recommended) 🌐

```bash
chainlit run app.py -w
```

Then open **http://localhost:8000** in your browser!

**Features:**
- 💬 Interactive chat interface
- 🔄 Real-time agent execution
- 📊 Step-by-step visualization
- 📈 Performance metrics

### Option B: Command Line 💻

```bash
# Interactive CLI
python main.py interactive

# Or run a demo
python main.py demo

# Or test connectivity
python main.py test
```

## Example Queries

Try these in the chat interface:

```
What are the latest AI breakthroughs in 2024?
```

```
Financial technology innovations in Europe
```

```
Research on quantum computing applications
```

```
Trends in renewable energy technology
```

## Troubleshooting

### Problem: Environment variables not found

**Solution:** Make sure `.env` file exists in the project root with all required variables.

### Problem: Can't connect to MCP servers

**Solution:** Test connectivity first:
```bash
python main.py test
```

### Problem: Chainlit won't start

**Solution:** Check Chainlit is installed:
```bash
chainlit --version
```

If not found:
```bash
uv sync
```

## Need More Help?

- **Usage Guide**: See `USAGE.md` for detailed instructions
- **Project Structure**: See `STRUCTURE.md` for architecture
- **Migration Info**: See `MIGRATION_SUMMARY.md` for changes
- **Main README**: See `README.md` for comprehensive docs

## What's New? ✨

This project has been restructured with:
- ✅ Modular architecture (`src/` directory)
- ✅ Chainlit web interface
- ✅ Improved CLI
- ✅ Better configuration management
- ✅ Comprehensive documentation

Enjoy your AI-powered multi-agent search system! 🎉
