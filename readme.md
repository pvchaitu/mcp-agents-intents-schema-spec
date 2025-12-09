# OpenMCPSpec Documentation (Version 0.2 – Standardized Draft)

## **📖 Overview**

OpenMCPSpec is a schema standard for Model Context Protocol (MCP) servers and agents.  
It defines how intents, parameters, responses, metadata, enumeration, and server connection details are described, enabling agents to autodiscover and interact with MCP servers in a consistent way.  
This document explains the schema in detail and provides guidance for implementers to derive compliant specifications.

## **🧩 Schema Sections**

### **1\. openmcpspec**

* **Type:** string  
* **Purpose:** Version of the OpenMCPSpec schema (e.g., "0.2.0").  
* **Usage:** Ensures compatibility between agents and servers.

### **2\. info**

* **Type:** object  
* **Purpose:** Metadata about the MCP server.  
* **Fields:**  
  * title (string, required): Human-readable name of the server.  
  * version (string, required): Server version.  
  * description (string): Short description of the server’s purpose.

### **3\. server**

* **Type:** object  
* **Purpose:** Connection details for autodiscovery.  
* **Fields:**  
  * name (string, required): Identifier for the server.  
  * hostname (string, required): Hostname (e.g., docs.mcp.org).  
  * port (integer, required): Port number (e.g., 443).  
  * protocol (string, required): One of http, https, ws, wss.  
  * base\_path (string): Base API path (e.g., /api).  
  * endpoints (array of URIs): Specific endpoints (e.g., /intents, /enumerate).

### **4\. intents**

* **Type:** array of objects  
* **Purpose:** Defines operations (intents) the server supports.  
* **Fields (Key Updates in v0.2):**  
  * name (string, required), description (string), category (string).  
  * **parameters** (array): Input parameters.  
    * Includes core fields: name, type, description, required, default.  
    * **Data Governance:** pii (boolean) and gdpr\_sensitive (boolean) fields are retained.  
    * **Advanced Validation (NEW):** Added fields like enum, pattern, format, minimum, maximum, and dependencies to enable stricter, machine-readable validation constraints on input arguments.  
  * **responses** (object): Defines success and error outputs.  
    * error: Includes schema and error codes.  
    * **Richer Error Handling (NEW):** Added recovery\_hints (suggestions for agents on failure) and a standardized flag for error responses.  
  * **metadata** (object): Roles, security scopes, NLP hints, version, deprecated flag.  
    * **Advanced Governance & Lifecycle (NEW):** Expanded with sunset\_date, migration\_hint, **rbac\_scopes** (Role-Based Access Control), **data\_residency** (e.g., EU, US), and **audit\_hooks** for compliance and long-term planning.

### **5\. enumeration**

* **Type:** object  
* **Purpose:** Lightweight listing of available intents for discovery.  
* **Fields (Key Updates in v0.2):**  
  * enabled, filters, pagination.  
  * versioning: Includes schema\_version, compatibility.  
    * **Versioning Enhancement (NEW):** Added migration\_paths to explicitly guide agents between compatible versions.  
  * **Discovery Enhancement (NEW):** Added **capability\_tags** and **intent\_groups** for finer-grained filtering and clustering of intents by the agent.

### **6\. lifecycle**

* **Type:** object  
* **Purpose:** Tracks changes and compatibility.  
* **Fields:**  
  * changelog (array of strings): Human-readable changes.  
  * compatibility (string): Version compatibility notes.

### **7\. testing *(optional)***

* **Type:** object  
* **Purpose:** Provides mock data and validation cases for developers.  
* **Fields (Key Updates in v0.2):**  
  * mock\_data, validation\_cases.  
  * **Richer Test Specifications (NEW):** Added **bdd\_cases** (Behavior Driven Development specifications) and **conformance\_tests** for formal compliance testing.

### **8\. analytics *(optional)***

* **Type:** object  
* **Purpose:** Observability hooks for production.  
* **Fields (Key Updates in v0.2):**  
  * logging, metrics, monitoring\_hooks.  
  * **Deep Observability (NEW):** Added **trace\_id\_support** (boolean), **usage\_quota** (integer), and **latency\_sla\_ms** (Service Level Agreement for latency) to enhance runtime monitoring.

## **✅ Core vs Optional (Updated)**

| Section | Required | Purpose |
| :---- | :---- | :---- |
| openmcpspec | ✔ | Schema version |
| info | ✔ | Server metadata |
| server | ✔ | Connection details |
| intents | ✔ | Full intent definitions |
| enumeration | ✔ | Lightweight discovery |
| lifecycle | Optional | Versioning & changelog |
| testing | Optional | Developer validation |
| analytics | Optional | Observability |

## ---

**🗂️ Change Log**

* **Version 0.2 (Standardized Draft) — *NEW***  
  * **Richer Data Validation:** Expanded parameter definitions with JSON schema properties (enum, pattern, minimum, maximum, dependencies).  
  * **Enhanced Governance:** Introduced rbac\_scopes, data\_residency, sunset\_date, and audit\_hooks in intent metadata.  
  * **Improved Reliability:** Added recovery\_hints and a standardized flag to error responses.  
  * **Advanced Discovery:** Added capability\_tags and intent\_groups to the enumeration section.  
  * **Deeper Observability:** Added trace\_id\_support, usage\_quota, and latency\_sla\_ms to analytics.  
  * **Formal Testing:** Introduced bdd\_cases and conformance\_tests fields.  
* **Version 0.1 (Initial Draft)**  
  * Established the base schema structure (openmcpspec, info, server, intents, enumeration).  
  * Added support for parameters, responses, and metadata.  
  * Introduced optional sections (lifecycle, testing, analytics).  
  * Provided a compliant example spec using the MCP Docs Server.  
  * Published documentation in Markdown format for implementers.

## ---

**👥 Contributors**

* **Chaitanya Pinnamaraju** — Project initiator, schema design direction, and specification author.  
* **Microsoft Copilot** — AI collaborator, documentation drafting, and schema refinement.

## **🛣️ Roadmap (Updated)**

### **Planned for Version 0.3**

* Support schema evolution with backward compatibility guidelines.  
* Add plugin/module extension patterns (x\_extensions) for vendor-specific fields.  
* Define best practices for lifecycle management across multiple MCP server versions.  
* Provide tooling for automated spec validation and testing.

### **Long-Term Goals**

* Establish OpenMCPSpec as a community-driven standard.  
* Integrate with popular agent frameworks for seamless autodiscovery.  
* Provide reference implementations and SDKs in multiple languages.  
* Encourage adoption across enterprise and open-source MCP deployments.