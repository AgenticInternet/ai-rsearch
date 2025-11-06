# DeepAgents Integration Review

## Issues Found

### 1. ❌ Wrong Parameter Name
**Current Code (Line 455):**
```python
self.agent = async_create_deep_agent(
    tools=mcp_tools,
    instructions=main_instructions,  # ❌ WRONG
    subagents=subagents,
    model=self.model,
    builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls"],
)
```

**Should Be (According to Official Docs):**
```python
self.agent = async_create_deep_agent(
    tools=mcp_tools,
    system_prompt=main_instructions,  # ✅ CORRECT
    subagents=subagents,
    model=self.model,
    builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls"],
)
```

**Reference:** https://docs.langchain.com/oss/python/deepagents/customization

### 2. ⚠️ Missing `task` Tool in builtin_tools

According to the official docs, the default built-in tools include:
- `write_todos` ✅
- `ls` ✅
- `read_file` ✅
- `write_file` ✅
- `edit_file` ✅
- `task` ❌ (MISSING - needed for subagent spawning)

**Current:**
```python
builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls"]
```

**Should Include:**
```python
builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls", "task"]
```

### 3. ⚠️ Package Version

**Current (pyproject.toml):**
```toml
deepagents>=0.0.5
```

**Latest Available:**
```toml
deepagents>=0.0.6
```

Consider updating to the latest version for bug fixes and improvements.

## Impact Assessment

### write_file Validation Error
The validation error you're experiencing is NOT caused by the parameter name issue. The error is correctly about the agent not providing the `content` parameter when calling the `write_file` tool. Our fix to add explicit warnings in the prompts was the right approach.

### Parameter Name Issue
The `instructions` vs `system_prompt` parameter might work (backward compatibility), but it's better to use the official parameter name to ensure future compatibility.

### Missing `task` Tool
Without the `task` tool, the agent cannot properly spawn subagents. This could limit the agent's ability to delegate work effectively.

## Recommended Changes

1. **Change parameter name from `instructions` to `system_prompt`** (Line 455)
2. **Add `task` to builtin_tools list** (Line 457)
3. **Consider updating to deepagents 0.0.6** (pyproject.toml)

## Official Documentation Reference

- **Overview:** https://docs.langchain.com/oss/python/deepagents/overview
- **Quickstart:** https://docs.langchain.com/oss/python/deepagents/quickstart
- **Customization:** https://docs.langchain.com/oss/python/deepagents/customization
- **Subagents:** https://docs.langchain.com/oss/python/deepagents/subagents

## Correct Usage Pattern

```python
from deepagents import async_create_deep_agent
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

agent = async_create_deep_agent(
    tools=[your_tools],
    system_prompt="Your custom instructions here",
    subagents=[your_subagents],
    model=model,
    builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls", "task"],
)
```

## Files to Update

1. `/src/core/orchestrator.py` - Line 455: Change `instructions` to `system_prompt`
2. `/src/core/orchestrator.py` - Line 457: Add `"task"` to builtin_tools list
3. `/pyproject.toml` - Update deepagents version to `>=0.0.6`
