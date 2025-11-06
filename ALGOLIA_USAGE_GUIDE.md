# Algolia MCP Tool Usage Guide

## Common Error: "Field required" for objects_json

If you see this error:
```
Error executing tool save_objects_batch: 1 validation error for save_objects_batchArguments
objects_json
  Field required [type=missing, input_value={'index_name': 'ai_rsearch'}, input_type=dict]
```

**Cause**: The `save_objects_batch` tool was called with only `index_name` but missing the required `objects_json` parameter.

## Correct Usage

### save_objects_batch Tool

**Required Parameters:**
- `index_name`: String - Always use `"ai_rsearch"`
- `objects_json`: String - A JSON string containing an array of objects

**IMPORTANT**: Both parameters are required. Never call with only `index_name`.

### Correct Examples

#### Example 1: Single Document
```json
{
  "index_name": "ai_rsearch",
  "objects_json": "[{\"objectID\": \"doc_123\", \"title\": \"AI Research\", \"content\": \"Content about AI\", \"url\": \"https://example.com\"}]"
}
```

#### Example 2: Multiple Documents
```json
{
  "index_name": "ai_rsearch",
  "objects_json": "[{\"objectID\": \"doc_1\", \"title\": \"First Doc\", \"content\": \"Content 1\"}, {\"objectID\": \"doc_2\", \"title\": \"Second Doc\", \"content\": \"Content 2\"}]"
}
```

#### Example 3: With Full Metadata
```json
{
  "index_name": "ai_rsearch",
  "objects_json": "[{\"objectID\": \"search_2024_001\", \"title\": \"Latest Financial News in France\", \"content\": \"Breaking news about French economy...\", \"timestamp\": \"2024-01-15T10:30:00Z\", \"source\": \"web_search\", \"url\": \"https://example.com/article\", \"category\": \"finance\"}]"
}
```

### Wrong Usage (Will Fail)

❌ **Missing objects_json** (This is what caused your error):
```json
{
  "index_name": "ai_rsearch"
}
```

❌ **objects_json as array instead of string**:
```json
{
  "index_name": "ai_rsearch",
  "objects_json": [{"objectID": "1", "title": "Example"}]
}
```

## Using the Helper Function

For Python code, use the helper function to ensure correct formatting:

```python
from src.helpers import format_save_objects_batch

# Prepare your data
search_results = [
    {
        "title": "Financial Breakthroughs in France",
        "content": "Recent developments...",
        "url": "https://example.com/article1"
    }
]

# Convert to Algolia format
from src.helpers.algolia import create_search_result_object

algolia_objects = [
    create_search_result_object(
        title=result["title"],
        content=result["content"],
        url=result.get("url"),
        category="finance",
        source="web_search"
    )
    for result in search_results
]

# Format for the tool (automatically adds objectIDs and creates JSON string)
params = format_save_objects_batch(algolia_objects, index_name="ai_rsearch")

# Now params is correctly formatted:
# {
#   "index_name": "ai_rsearch",
#   "objects_json": "[{...}]"  # Properly formatted JSON string
# }
```

## Object Structure Requirements

Every object MUST have:
- `objectID`: Unique identifier (string)

Recommended fields:
- `title`: Document title
- `content`: Main content/description
- `url`: Source URL (if applicable)
- `timestamp`: ISO 8601 timestamp
- `source`: Origin of the data (e.g., "web_search")
- `category`: Classification/type

## Troubleshooting

### "Field required" error
- Check that you're providing BOTH `index_name` AND `objects_json`
- Verify both parameters are present in your tool call

### "Invalid JSON" error
- Ensure `objects_json` is a STRING, not an array or object
- Make sure the JSON is properly escaped

### "Duplicate objectID" error
- Each object must have a unique `objectID`
- Use timestamps or UUIDs to generate unique IDs

## Alternative: save_object for Single Items

If you only need to save one object, use `save_object` instead:

```json
{
  "index_name": "ai_rsearch",
  "object": "{\"objectID\": \"doc_001\", \"title\": \"Single Document\", \"content\": \"Content here\"}"
}
```

## Quick Checklist

Before calling `save_objects_batch`, verify:
- ✓ Both `index_name` and `objects_json` are provided
- ✓ `index_name` is `"ai_rsearch"`
- ✓ `objects_json` is a STRING (has quotes around the JSON)
- ✓ Each object has a unique `objectID`
- ✓ JSON is properly escaped (use `json.dumps()` in Python)
