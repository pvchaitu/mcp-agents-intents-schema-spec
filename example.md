## 📊 Example Spec (MCP Docs Server)

```json
{
  "openmcpspec": "0.1",
  "info": {
    "title": "MCP Docs Server",
    "version": "0.9.0",
    "description": "Open-source MCP server exposing intents for documentation search and retrieval."
  },
  "server": {
    "name": "docs-mcp",
    "hostname": "docs.mcp.org",
    "port": 443,
    "protocol": "https",
    "base_path": "/api",
    "endpoints": [
      "https://docs.mcp.org/api/intents",
      "https://docs.mcp.org/api/enumerate"
    ]
  },
  "intents": [
    {
      "name": "searchDocs",
      "description": "Search documentation by keyword or phrase.",
      "category": "documentation",
      "parameters": [
        { "name": "query", "type": "string", "required": true },
        { "name": "limit", "type": "integer", "default": 10 }
      ],
      "responses": {
        "success": {
          "schema": { "results": "array<object>" },
          "examples": [
            { "results": [{ "title": "Quickstart Guide", "url": "https://docs.mcp.org/quickstart" }] }
          ]
        },
        "error": {
          "schema": { "code": "string", "message": "string" },
          "codes": ["SERVER_ERROR", "INVALID_QUERY"]
        }
      },
      "metadata": {
        "roles": ["user"],
        "security": ["none"],
        "nlp_hints": ["search", "docs", "manual"],
        "version": "0.1",
        "deprecated": false
      }
    },
    {
      "name": "getDoc",
      "description": "Retrieve a specific documentation page by ID.",
      "category": "documentation",
      "parameters": [
        { "name": "doc_id", "type": "string", "required": true }
      ],
      "responses": {
        "success": {
          "schema": { "doc": "object" },
          "examples": [
            { "doc": { "id": "doc-123", "title": "Installation", "content": "Step 1: Download..." } }
          ]
        },
        "error": {
          "schema": { "code": "string", "message": "string" },
          "codes": ["NOT_FOUND", "SERVER_ERROR"]
        }
      },
      "metadata": {
        "roles": ["user"],
        "security": ["none"],
        "nlp_hints": ["open", "read", "manual"],
        "version": "0.1",
        "deprecated": false
      }
    }
  ],
  "enumeration": {
    "enabled": true,
    "filters": {
      "categories": ["documentation"],
      "roles": ["user"],
      "deprecated": false
    },
    "pagination": {
      "cursor": "cur-001",
      "limit": 50,
      "has_more": false
    },
    "versioning": {
      "schema_version": "0.1",
      "compatibility": ">=0.1"
    },
    "intents": [
      {
        "name": "searchDocs",
        "description": "Search documentation by keyword or phrase.",
        "category": "documentation",
        "version": "0.1",
        "deprecated": false
      },
      {
        "name": "getDoc",
        "description": "Retrieve a specific documentation page by ID.",
        "category": "documentation",
        "version": "0.1",
        "deprecated": false
      }
    ]
  },
  "lifecycle": {
    "changelog": [
      "Version 0.1: Initial draft of OpenMCPSpec schema and MCP Docs Server example."
    ],
    "compatibility": ">=0.1"
  },
  "testing": {
    "mock_data": { "query": "installation" },
    "validation_cases": ["empty query", "invalid doc_id"]
  },
  "analytics": {
    "logging": true,
    "metrics": ["intent_usage", "error_rate"],
    "monitoring_hooks": ["prometheus"]
  }
}
