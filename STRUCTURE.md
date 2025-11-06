# Project Structure

## Overview

This project has been restructured to follow best practices with clear separation of concerns and modular organization.

## Directory Structure

```
agentic-internet-rsearch-deepagents/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── core/                     # Core orchestration logic
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration and environment management
│   │   └── orchestrator.py      # DeepAgentSearchOrchestrator class
│   ├── helpers/                  # Helper utilities
│   │   ├── __init__.py
│   │   └── algolia.py           # Algolia formatting helpers
│   ├── ui/                       # User interfaces
│   │   ├── __init__.py
│   │   └── chainlit_app.py      # Chainlit web interface
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       └── mcp_test.py          # MCP connectivity testing
├── app.py                        # Chainlit entry point
├── main.py                       # CLI entry point
├── pyproject.toml                # Project dependencies
├── chainlit.md                   # Chainlit welcome page
├── .chainlit                     # Chainlit configuration
├── README.md                     # Main documentation
├── STRUCTURE.md                  # This file
└── .env                          # Environment variables (not in git)
```

## Module Descriptions

### `src/core/`

Core business logic for the search orchestrator.

- **`config.py`**: Manages environment variables and configuration
  - `Config` class: Centralized configuration management
  - Environment variable validation
  - MCP server configuration

- **`orchestrator.py`**: Main orchestration logic
  - `DeepAgentSearchOrchestrator` class: Multi-agent coordinator
  - MCP server initialization
  - Agent creation and management
  - Query processing
  - Demo and interactive modes

### `src/helpers/`

Helper utilities for specific services.

- **`algolia.py`**: Algolia-specific helpers
  - `format_save_objects_batch()`: Format objects for batch save
  - `format_save_object()`: Format single object save
  - `format_search_index()`: Format search parameters
  - `create_search_result_object()`: Create search result objects

### `src/ui/`

User interface implementations.

- **`chainlit_app.py`**: Chainlit web interface
  - Chat-based UI for search queries
  - Real-time streaming responses
  - Step-by-step visualization
  - Session management

### `src/utils/`

Utility modules for testing and tools.

- **`mcp_test.py`**: MCP server connectivity testing
  - Test individual server connectivity
  - Test MCP client initialization
  - Connectivity diagnostics

## Entry Points

### Web UI (Chainlit)

```bash
# Start the Chainlit interface
chainlit run app.py -w

# Access at http://localhost:8000
```

### CLI

```bash
# Run demo mode
python main.py demo

# Run interactive mode
python main.py interactive

# Test MCP connectivity
python main.py test
```

## Configuration

All configuration is managed through environment variables in `.env`:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
MCP_SERPAPI_URL=http://...
MCP_ALGOLIA_URL=http://...
MCP_OPENSEARCH_URL=http://...
REDIS_URL=redis://...

# Optional
MODEL_NAME=claude-sonnet-4-5-20250929
MODEL_TEMPERATURE=0.1
```

## Development Workflow

### Installing Dependencies

```bash
uv sync
```

### Running Tests

```bash
# Test MCP connectivity
python main.py test

# Or use the utility directly
python -m src.utils.mcp_test
```

### Running the Application

#### Option 1: Web UI (Recommended)

```bash
chainlit run app.py -w
```

Features:
- Interactive chat interface
- Real-time response streaming
- Step-by-step visualization
- Session management

#### Option 2: CLI

```bash
# Demo mode
python main.py demo

# Interactive mode
python main.py interactive
```

## Architecture Benefits

### Separation of Concerns

- **Configuration**: Centralized in `config.py`
- **Business Logic**: Isolated in `orchestrator.py`
- **UI Layer**: Separate interfaces (Chainlit, CLI)
- **Utilities**: Modular helpers and tools

### Modularity

- Easy to add new interfaces
- Simple to extend with new helpers
- Clear import paths
- Testable components

### Maintainability

- Clear file organization
- Logical grouping
- Easy navigation
- Scalable structure

## Migrating from Old Structure

### Old Structure

```
.
├── main.py                    (everything)
├── algolia_helper.py
└── test_mcp_connectivity.py
```

### New Structure

```
.
├── src/
│   ├── core/                  (main.py → orchestrator.py + config.py)
│   ├── helpers/               (algolia_helper.py → algolia.py)
│   ├── ui/                    (NEW: chainlit_app.py)
│   └── utils/                 (test_mcp_connectivity.py → mcp_test.py)
├── app.py                     (NEW: Chainlit entry)
└── main.py                    (NEW: CLI wrapper)
```

## Best Practices

### Import Paths

```python
# Core modules
from src.core import Config, DeepAgentSearchOrchestrator

# Helpers
from src.helpers import format_save_objects_batch

# Utils
from src.utils.mcp_test import test_mcp_server
```

### Adding New Features

1. **New Helper**: Add to `src/helpers/`
2. **New UI**: Add to `src/ui/`
3. **New Utility**: Add to `src/utils/`
4. **Core Logic**: Extend `src/core/orchestrator.py`

### Configuration Changes

All configuration changes should be made in `src/core/config.py` and exposed through the `Config` class.

## Future Enhancements

Potential additions:
- `src/agents/` - Individual agent implementations
- `src/api/` - REST API interface
- `src/database/` - Database models and connections
- `tests/` - Unit and integration tests
- `docs/` - Extended documentation

## Support

For issues or questions:
1. Check environment variables in `.env`
2. Test MCP connectivity: `python main.py test`
3. Review logs for error messages
4. Check README.md for setup instructions
