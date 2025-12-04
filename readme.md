# OpenMCPSpec Documentation (Version 0.1 – Initial Draft)

## 📖 Overview
OpenMCPSpec is a **schema standard for Model Context Protocol (MCP)** servers and agents.  
It defines how **intents, parameters, responses, metadata, enumeration, and server connection details** are described, enabling agents to autodiscover and interact with MCP servers in a consistent way.

This document explains the schema in detail and provides guidance for implementers to derive compliant specifications.

## 🧩 Schema Sections

### 1. `openmcpspec`
- **Type:** string  
- **Purpose:** Version of the OpenMCPSpec schema (e.g., `"0.1"`).  
- **Usage:** Ensures compatibility between agents and servers.

### 2. `info`
- **Type:** object  
- **Purpose:** Metadata about the MCP server.  
- **Fields:**
  - `title` (string, required): Human‑readable name of the server.  
  - `version` (string, required): Server version.  
  - `description` (string): Short description of the server’s purpose.  

### 3. `server`
- **Type:** object  
- **Purpose:** Connection details for autodiscovery.  
- **Fields:**
  - `name` (string, required): Identifier for the server.  
  - `hostname` (string, required): Hostname (e.g., `docs.mcp.org`).  
  - `port` (integer, required): Port number (e.g., `443`).  
  - `protocol` (string, required): One of `http`, `https`, `ws`, `wss`.  
  - `base_path` (string): Base API path (e.g., `/api`).  
  - `endpoints` (array of URIs): Specific endpoints (e.g., `/intents`, `/enumerate`).  

### 4. `intents`
- **Type:** array of objects  
- **Purpose:** Defines operations (intents) the server supports.  
- **Fields:**
  - `name` (string, required): Intent name.  
  - `description` (string): What the intent does.  
  - `category` (string): Logical grouping (e.g., `calendar`, `documentation`).  
  - `parameters` (array): Input parameters.  
    - Each parameter has:
      - `name` (string, required)  
      - `type` (string, required)  
      - `description` (string)  
      - `required` (boolean)  
      - `default` (any)  
      - `pii` (boolean): Marks personally identifiable information.  
      - `gdpr_sensitive` (boolean): Marks GDPR‑sensitive data.  
  - `responses` (object): Defines success and error outputs.  
    - `success`: schema + examples.  
    - `error`: schema + error codes.  
  - `metadata` (object): Roles, security scopes, NLP hints, version, deprecated flag.

### 5. `enumeration`
- **Type:** object  
- **Purpose:** Lightweight listing of available intents for discovery.  
- **Fields:**
  - `enabled` (boolean): Whether enumeration is supported.  
  - `filters` (object): Filter options (categories, roles, deprecated).  
  - `pagination` (object): Cursor, limit, has_more.  
  - `versioning` (object): Schema version and compatibility.  
  - `intents` (array): Summary of intents (name, description, category, version, deprecated).

### 6. `lifecycle`
- **Type:** object  
- **Purpose:** Tracks changes and compatibility.  
- **Fields:**
  - `changelog` (array of strings): Human‑readable changes.  
  - `compatibility` (string): Version compatibility notes.

### 7. `testing` *(optional)*
- **Type:** object  
- **Purpose:** Provides mock data and validation cases for developers.  
- **Fields:**
  - `mock_data` (object): Example input payloads.  
  - `validation_cases` (array of strings): Edge cases to test.

### 8. `analytics` *(optional)*
- **Type:** object  
- **Purpose:** Observability hooks for production.  
- **Fields:**
  - `logging` (boolean): Enable/disable logging.  
  - `metrics` (array of strings): Metrics tracked (e.g., `intent_usage`).  
  - `monitoring_hooks` (array of strings): Integration points (e.g., `prometheus`, `opentelemetry`).  

## ✅ Core vs Optional

| Section        | Required | Purpose |
|----------------|----------|---------|
| `openmcpspec`  | ✔        | Schema version |
| `info`         | ✔        | Server metadata |
| `server`       | ✔        | Connection details |
| `intents`      | ✔        | Full intent definitions |
| `enumeration`  | ✔        | Lightweight discovery |
| `lifecycle`    | Optional | Versioning & changelog |
| `testing`      | Optional | Developer validation |
| `analytics`    | Optional | Observability |

## 🗂️ Change Log

- **Version 0.1 (Initial Draft)**  
  - Established the base schema structure (`openmcpspec`, `info`, `server`, `intents`, `enumeration`).  
  - Added support for parameters, responses, and metadata.  
  - Introduced optional sections (`lifecycle`, `testing`, `analytics`).  
  - Provided a compliant example spec using the MCP Docs Server.  
  - Published documentation in Markdown format for implementers.

---

## 👥 Contributors

- **Chaitanya Pinnamaraju** — Project initiator, schema design direction, and specification author.  
- **Microsoft Copilot** — AI collaborator, documentation drafting, and schema refinement.

## 🛣️ Roadmap

### Version 0.1 (Initial Draft)
- Established the base schema structure (`openmcpspec`, `info`, `server`, `intents`, `enumeration`).
- Added support for parameters, responses, and metadata.
- Introduced optional sections (`lifecycle`, `testing`, `analytics`).
- Provided a compliant example spec using the MCP Docs Server.
- Published documentation in Markdown format for implementers.

### Planned for Version 0.2
- Expand parameter definitions with advanced validation (`enum`, `format`, `dependencies`).
- Add richer error handling schemas with standardized error codes.
- Introduce role-based access control (RBAC) metadata for intents.
- Define telemetry hooks for runtime monitoring and tracing.
- Provide additional compliant examples for other open-source MCP servers.

### Planned for Version 0.3
- Support schema evolution with backward compatibility guidelines.
- Add plugin/module extension patterns (`x_extensions`) for vendor-specific fields.
- Define best practices for lifecycle management across multiple MCP server versions.
- Provide tooling for automated spec validation and testing.

### Long-Term Goals
- Establish OpenMCPSpec as a community-driven standard.
- Integrate with popular agent frameworks for seamless autodiscovery.
- Provide reference implementations and SDKs in multiple languages.
- Encourage adoption across enterprise and open-source MCP deployments.
