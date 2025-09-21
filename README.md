# 🚀 DeepAgents Real-time Search (RSearch) System

> **AI-Powered Multi-Agent Search Orchestration with Remote MCP Servers**

A production-ready real-time search system that combines DeepAgents orchestration with Model Context Protocol (MCP) servers to deliver intelligent, coordinated search capabilities across multiple data sources.

## 🌟 Overview

The RSearch (Real-time Search) system is an advanced AI orchestration platform that leverages:
- **DeepAgents**: Multi-agent coordination for complex search workflows
- **MCP Servers**: Standardized protocol for tool interactions
- **Claude 4 Sonnet**: Advanced language model for intelligent processing
- **Distributed Architecture**: Remote MCP servers for scalable search operations

## 🏗️ Architecture

![Technical Architecture](/images/rsearch-architecture.png )

### DeepAgents Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   User Query Interface                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 DeepAgent Orchestrator                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Planning Engine (Todo Management)                 │   │
│  │  • Workflow Coordination                            │   │
│  │  • Result Synthesis                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────┬──────────────────┬──────────────────┬───────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐      ┌──────────┐
    │ SerpApi  │       │ Algolia  │      │OpenSearch│
    │   MCP    │       │   MCP    │      │   MCP    │
    └──────────┘       └──────────┘      └──────────┘
    Web Search         Indexing          Analytics
```

## 📋 Features

### 🔍 Real-time Search Capabilities
- **Web Search**: Live internet search via SerpApi MCP
- **Semantic Indexing**: Document storage and retrieval with Algolia
- **Analytics Tracking**: Performance monitoring via OpenSearch
- **Multi-Source Aggregation**: Combines results from multiple sources

### 🤖 Intelligent Orchestration
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Adaptive Workflow**: Dynamic task planning and execution
- **Error Recovery**: Graceful degradation and retry mechanisms
- **Performance Optimization**: Network-aware latency management

### 🛠️ Technical Features
- **MCP Protocol Support**: SSE/Streamable HTTP transports
- **Redis Caching**: Response caching for improved performance
- **File Management**: Automatic result documentation
- **Thread Management**: Conversation context preservation

## 🚀 Quick Start

### Prerequisites
```bash
# Required
- Python 3.11+
- uv package manager
- Anthropic API key

# Optional
- Redis server (for caching)
- MCP server endpoints
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd agentic-internet-rsearch-deepagents
```

2. **Install dependencies**
```bash
uv sync
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - ANTHROPIC_API_KEY=your-api-key
# - MCP_SERPAPI_URL=https://your-serpapi-mcp-url
# - MCP_ALGOLIA_URL=https://your-algolia-mcp-url
# - MCP_OPENSEARCH_URL=https://your-opensearch-mcp-url
# - REDIS_URL=redis://your-redis-url (optional)
```

### Running the System

#### Demo Mode
```bash
uv run python main.py demo
```

#### Interactive Mode
```bash
uv run python main.py interactive
```

#### Testing Connectivity
```bash
uv run python test_mcp_connectivity.py
```

## 💡 How It Works

### 1. Query Processing
The system receives a user query and creates an execution plan:
```python
# Example query flow
Query → Planning → Research → Indexing → Analysis → Synthesis → Output
```

### 2. Multi-Agent Coordination
Specialized agents handle different aspects:
- **Web Researcher**: Executes targeted searches
- **Object Indexer**: Stores and retrieves structured data
- **Analytics Tracker**: Monitors performance and trends

### 3. MCP Server Integration
Remote MCP servers provide:
- **SerpApi**: Google search, news, images, trends
- **Algolia**: Object storage, indexing, semantic search
- **OpenSearch**: Logging, analytics, performance tracking

### 4. Result Synthesis
The orchestrator combines results:
- Aggregates findings from multiple sources
- Generates comprehensive reports
- Saves results to files for review

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
ANTHROPIC_API_KEY=sk-ant-api03-...     # Required: Claude API access

# MCP Server URLs (Optional - will degrade gracefully)
MCP_SERPAPI_URL=https://...            # Web search server
MCP_ALGOLIA_URL=https://...            # Indexing server
MCP_OPENSEARCH_URL=https://...         # Analytics server

# Optional Services
REDIS_URL=redis://...                  # Cache server
LANGSMITH_API_KEY=...                  # Tracing/monitoring
```

### Performance Tuning
```python
# Adjust in main.py
recursion_limit = 100    # Max agent recursion depth
temperature = 0.1        # LLM creativity (0.0-1.0)
timeout = 10.0          # MCP connection timeout (seconds)
```

## 📊 Example Workflows

### Research Query
```python
# Input
"What are the latest AI breakthroughs in 2024?"

# Workflow
1. Create research plan
2. Search web for recent AI news
3. Index relevant findings
4. Analyze trends and patterns
5. Generate comprehensive report
```

### Financial Analysis
```python
# Input
"Analyze recent financial technology innovations in Europe"

# Workflow
1. Search financial news sources
2. Index company and technology data
3. Track investment trends
4. Generate market analysis report
```

## 🚨 Error Handling

The system implements multiple layers of error recovery:

1. **Connection Failures**: Automatic fallback to available servers
2. **Timeout Management**: Configurable timeouts with graceful degradation
3. **Partial Failures**: Continue with available tools
4. **Retry Logic**: Automatic retry for transient failures

## 📈 Performance Metrics

The system tracks:
- **Query Latency**: End-to-end processing time
- **Network Overhead**: Separate tracking of network vs processing
- **Tool Performance**: Individual tool execution metrics
- **Success Rates**: Query completion statistics

Target Performance:
- Total latency: <5000ms
- Individual tools: <2000ms average
- Network resilience: Handle up to 33% server failures

## 🔒 Security

- **API Key Management**: Secure environment variable storage
- **Redis Authentication**: Optional password protection
- **MCP Server Auth**: HTTPS connections with authentication
- **No Sensitive Data Logging**: Automatic credential masking

## 🐛 Troubleshooting

### Common Issues

1. **MCP Connection Failures**
```bash
# Test connectivity
uv run python test_mcp_connectivity.py

# Check server URLs and network access
```

2. **Redis Connection Issues**
```bash
# System works without Redis (degraded performance)
# Check REDIS_URL format: redis://[password@]host:port/db
```

3. **Import Errors**
```bash
# Ensure dependencies are installed
uv sync

# Check Python version (3.11+ required)
python --version
```

## 📚 Project Structure

```
agentic-internet-rsearch-deepagents/
├── main.py                 # Main orchestrator
├── test_mcp_connectivity.py # MCP server testing
├── pyproject.toml          # Project dependencies
├── .env                    # Environment configuration
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request

## 📄 License

[License Type] - See LICENSE file for details

## 🙏 Acknowledgments

- **DeepAgents**: Advanced multi-agent orchestration framework
- **MCP Protocol**: Standardized tool interaction protocol
- **Anthropic**: Claude 3.5 Sonnet language model
- **AI Tinkerers Hackathon**: Platform for innovation

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section
- Review test scripts for connectivity debugging

---

**Built with ❤️ for the AI Tinkerers Hackathon**