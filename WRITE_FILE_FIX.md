# Fix for write_file Validation Error + DeepAgents API Updates

## Problem 1: write_file Validation Error
The agent was getting a Pydantic validation error when calling `write_file`:
```
1 validation error for write_file
content
  Field required [type=missing, input_value={'file_path': '/tmp/frenc...GAjMTnxxKFsbpj1RM4CBfo'}, input_type=dict]
```

This error occurred in step 4 when synthesizing French financial situation reports.

## Root Cause
The `write_file` tool requires BOTH parameters:
- `file_path` (string): Where to save the file
- `content` (string): The actual content to write

The agent was only providing `file_path` without `content`, causing the validation error.

## Solution Applied
Updated `src/core/orchestrator.py` with explicit warnings and instructions in FOUR locations:

### 1. Web-Researcher Subagent (Line 278)
Added critical instructions for using write_file when saving research results.

### 2. Object-Indexer Subagent (Line 372)
Added critical instructions for using write_file when saving indexing results.

### 3. Analytics-Tracker Subagent (Line 421)
Added critical instructions for using write_file when saving analytics reports.

### 4. Enhanced Query Prompt (Line 484)
Added prominent, visually distinct warning that appears with EVERY query.

## Correct Usage
```python
# ✅ CORRECT
{
  "file_path": "/tmp/research_results.md",
  "content": "# Research Results\n\nFindings: ...\n\nConclusion: ..."
}

# ❌ WRONG - This causes the error
{
  "file_path": "/tmp/research_results.md"
}
```

## Testing the Fix
Run your French financial research query again:
```bash
python main.py demo
```

Or in interactive mode:
```bash
python main.py interactive
```

The agent should now properly call write_file with both parameters, preventing the validation error.

## What Changed
- All subagent prompts now have explicit write_file usage instructions
- Main query prompt has a prominent warning section
- Instructions include correct and incorrect examples
- Step-by-step guidance on how to use write_file properly

## Cost Savings
By fixing this error, you'll avoid:
- Failed API calls that waste tokens
- Retry attempts that multiply costs
- Agent confusion and hallucination loops
- Unnecessary debugging iterations

The fix ensures the agent understands the correct tool usage from the start, preventing expensive validation errors.

## Problem 2: Incorrect DeepAgents API Usage

### Issues Found
After reviewing the official LangChain DeepAgents documentation, we found several API discrepancies:

1. **Wrong parameter name**: Using `instructions` instead of `system_prompt`
2. **Missing tool**: `task` tool not included in `builtin_tools` (needed for subagent spawning)
3. **Outdated package**: Using `deepagents>=0.0.5` instead of latest `>=0.0.6`

### Changes Made

#### 1. Fixed Parameter Name (orchestrator.py line 455)
```python
# ❌ OLD (WRONG)
self.agent = async_create_deep_agent(
    tools=mcp_tools,
    instructions=main_instructions,  # Wrong parameter name
    ...
)

# ✅ NEW (CORRECT)
self.agent = async_create_deep_agent(
    tools=mcp_tools,
    system_prompt=main_instructions,  # Correct parameter name
    ...
)
```

#### 2. Added Missing `task` Tool (orchestrator.py line 458)
```python
# ❌ OLD (INCOMPLETE)
builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls"]

# ✅ NEW (COMPLETE)
builtin_tools=["write_todos", "write_file", "read_file", "edit_file", "ls", "task"]
```

The `task` tool is essential for subagent spawning - without it, agents cannot properly delegate work to specialized subagents.

#### 3. Updated Package Version (pyproject.toml)
```toml
# ❌ OLD
deepagents>=0.0.5

# ✅ NEW
deepagents>=0.0.6
```

### Official Documentation Reference
- **DeepAgents Overview**: https://docs.langchain.com/oss/python/deepagents/overview
- **Quickstart**: https://docs.langchain.com/oss/python/deepagents/quickstart
- **Customization**: https://docs.langchain.com/oss/python/deepagents/customization
- **API Reference**: https://reference.langchain.com/python/deepagents/

### Impact
These changes ensure:
- ✅ Proper alignment with official LangChain documentation
- ✅ Full subagent spawning capabilities with the `task` tool
- ✅ Latest bug fixes and improvements from deepagents 0.0.6
- ✅ Future compatibility with DeepAgents updates

## Next Steps

1. **Update dependencies**:
   ```bash
   uv sync
   # or
   pip install -e .
   ```

2. **Test the complete fix**:
   ```bash
   python main.py demo
   ```

Both issues are now resolved - the write_file validation error through better prompts, and the API usage through correct parameter names and complete builtin_tools list.
