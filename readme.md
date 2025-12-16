# **OpenMCPSpec Documentation (Version 0.4.1 – Production Standard)**

## **📖 Overview**

OpenMCPSpec is a schema standard for Model Context Protocol (MCP) servers and agents.  
It defines how intents, parameters, responses, metadata, enumeration, and server connection details are described, enabling large language model (LLM) agents to autodiscover and interact with MCP servers in a consistent and secure way.  
This specification is designed to support **System Reliability** (via standardized envelopes and tracing) and **LLM Governance** (via explicit data sensitivity flags) required for production implementations.

## **🧩 Schema Sections**

### **1\. openmcpspec**

* **Type:** string  
* **Purpose:** Version of the OpenMCPSpec schema (e.g., "v0.4.1").  
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
* **Purpose:** Connection and Security details for autodiscovery.  
* **Fields:**  
  * name (string, required): Identifier for the server.  
  * hostname (string, required): Hostname (e.g., docs.mcp.org).  
  * port (integer, required): Port number (e.g., 443).  
  * protocol (string, required): Transport protocol (http, https, ws, wss).  
  * base\_path (string): Base path for API endpoints (e.g., /api/v1).  
  * endpoints (array of string/URI): List of entry points for the server.  
  * **NEW auth (object):** Defines supported authentication methods (V0.3/V0.4.1 Feature).  
    * methods (array of string): Allowed methods (mtls, oidc, jwt, hmac).  
    * jwks\_uri (string/URI): URI for the JSON Web Key Set endpoint.  
    * token\_scopes (array of string): Required OAuth/JWT scopes for access.

### **4\. envelope (NEW SECTION)**

* **Type:** object  
* **Purpose:** Standardized wrapper for all request/response payloads to ensure reliability, tracing, and streaming in distributed production environments (V0.3/V0.4.1 Feature).  
* **Fields:**  
  * **request (object):** Fields used by the agent when calling the MCP server.  
    * id (string): Unique request ID for auditing.  
    * deadline\_ms (integer): Timeout for the request.  
    * idempotency\_key (string): Key to ensure safe retries.  
    * trace\_id (string): Distributed tracing identifier.  
  * **response (object):** Fields returned by the MCP server.  
    * status (string): Result status (success, error, partial).  
    * error (ref): References the errorEnvelope definition for standardized error handling.  
  * **stream\_chunk (object):** Defines the structure for real-time data streaming (e.g., LLM token streams or database row batches).  
    * chunk\_type (string): Type of payload (token, row\_batch, log).  
    * sequence (integer): Ordering of the chunk within the stream.

### **5\. intents**

* **Type:** array  
* **Purpose:** List of functional capabilities (actions) the server provides.

#### **5.1. intents/parameters**

This section defines the input requirements for an intent.

* **Type:** array of objects **(V0.2 Structure Restored)**  
* **Purpose:** Explicitly defines the contract and **governance flags** for each argument.  
* **Critical Fields for LLM Governance & ACI (Argument Correctness Improvement):**  
  * name, type, description, required, default.  
  * **pii (boolean):** **(V0.2 Restoration)** Critical flag indicating if the parameter contains Personal Identifiable Information. Used for Compliance Pre-emption Rate (CPR) calculation and refusal by the LLM agent.  
  * **gdpr\_sensitive (boolean):** **(V0.2 Restoration)** Critical flag indicating data subject to GDPR/CCPA regulations. Also used for CPR.  
  * **Validation Fields:** enum, pattern, minimum, maximum, format. These enable ACI checks by the agent before the call is made.

#### **5.2. intents/responses**

* **Type:** object  
* **Purpose:** Defines the expected output structures for successful and failed execution.  
  * success/schema (object): The JSON Schema for the successful data payload.  
  * error (ref): References the standardized errorEnvelope definition.

### **6\. definitions/errorEnvelope (NEW/UPDATED DEFINITION)**

* **Type:** object  
* **Purpose:** Standardized taxonomy for handling system and domain errors across all intents.  
* **Fields:**  
  * code (string, enum): Categorical error code (e.g., MCP-ERR-AUTH-FAILED, SQL-ERR-SYNTAX). **(V0.3 Feature)**  
  * message (string): Human-readable error message.  
  * retryable (boolean): Flag for automated client retry logic.  
  * correlation\_id (string): ID linking the error to internal logs.  
  * **details (object):** The intent-specific error data payload.  
  * **payload\_schema (object):** **(V0.4.1 Fix)** An optional JSON Schema that defines the structure of the **details** object for this specific error type, allowing for intent-specific rich error data within the standardized envelope.  
  * recovery\_hints (array of string): Suggested actions for the client.

### **7\. enumeration**

* **Type:** object  
* **Purpose:** Defines mechanisms for agents to efficiently discover available intents based on criteria.  
* **Fields:** filters, pagination, versioning, capability\_tags, intent\_groups.

### **8\. lifecycle**

* **Type:** object  
* **Purpose:** Defines metadata about the specification's management.  
* **Fields:** changelog, compatibility.

### **9\. testing**

* **Type:** object  
* **Purpose:** Defines hooks and metadata for testing compliance and correctness.  
* **Fields:** mock\_data, validation\_cases, bdd\_cases, conformance\_tests.

### **10\. analytics**

* **Type:** object  
* **Purpose:** Defines hooks for monitoring and observability.  
* **Fields:** logging, metrics, monitoring\_hooks, trace\_id\_support, usage\_quota, latency\_sla\_ms.

## **⚙️ Changelog**

* **Version 0.4.1 (Production Consolidation)**  
  * **Consolidation:** Merged system reliability features from the V0.3 draft with the LLM governance features from V0.2.  
  * **Error Fix:** Restored V0.2's rich error structure by introducing the payload\_schema field within the global errorEnvelope to define intent-specific error data schemas.  
  * **Governance Reinstated:** Explicitly restored the array structure for intents/parameters including the critical pii and gdpr\_sensitive boolean flags, ensuring full Compliance Pre-emption (CPR) capabilities for agents.  
  * **API Reliability:** Incorporated the top-level envelope for standard request/response/streaming protocols, including fields like idempotency\_key and trace\_id.  
  * **Security Standardized:** Added the server/auth object to define explicit security contracts (mtls, oidc, jwt).  
* **Version 0.2 (Standardized Draft)**  
  * Established the base schema structure (openmcpspec, info, server, intents, enumeration).  
  * Introduced mandatory LLM governance flags (pii, gdpr\_sensitive) and ACI fields (pattern, minimum, maximum) in parameters.  
* **Version 0.1 (Initial Draft)**  
  * Established the base schema structure (openmcpspec, info, server, intents, enumeration).  
  * Added support for parameters, responses, and metadata.  
  * Introduced optional sections (lifecycle, testing, analytics).

## **\---**

**👥 Contributors**

* **Chaitanya Pinnamaraju** — Project initiator, schema design direction, and specification author.  
* **Microsoft Copilot** — AI collaborator, documentation drafting, and schema refinement.

## **🛣️ Roadmap (Updated)**

### **Planned for Version 0.5**

* Support schema evolution with backward compatibility guidelines.  
* Add plugin/module extension patterns (x\_extensions) for vendor-specific fields.  
* Define best practices for lifecycle management across multiple MCP server versions.  
* Provide tooling for automated spec validation and testing.

### **Long-Term Goals**

* Establish OpenMCPSpec as a community-driven standard.  
* Integrate with popular agent frameworks for seamless autodiscovery.  
* Provide reference implementations and SDKs in multiple languages.  
* Encourage adoption across enterprise and open-source MCP deployments.