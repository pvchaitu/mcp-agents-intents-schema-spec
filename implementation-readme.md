

# **💡 MCP Server Implementation Guide (Version 0.1 – Initial Draft)**

This guide provides a step-by-step approach to implementing an **MCP Server** using the custom openmcpspec-spring-starter Maven library.

## ---

**🗺️ Flow of Thought for Implementation**

The implementation process is divided into two main parts, demonstrating the creation and subsequent use of a custom Spring Boot Starter library.

1. **Generate the Core Library:** Create the openmcpspec-core module (containing data models and a loader) and the openmcpspec-spring-starter module (providing auto-configuration and web controllers).  
2. **Use the Starter:** Implement a sample MCP server (sample-mcp-server) that consumes the custom starter, loads an **OpenMCPSpec** JSON file, and exposes the defined endpoints automatically.

## ---

**🛠️ Section A: Generate your own openmcpspec-spring-starter Maven library**

This section outlines the structure and content for building the two core library modules: openmcpspec-core and openmcpspec-spring-starter.

### **Project Structure**

.  
├── openmcpspec-core/  
└── openmcpspec-spring-starter/

### **1\. openmcpspec-core Module**

This module contains the core Java models that map to the **OpenMCPSpec** JSON structure, and a utility class for loading the specification.

#### **openmcpspec-core/pom.xml**

\<project xmlns\="http://maven.apache.org/POM/4.0.0" xmlns:xsi\="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation\="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd"\>  
  \<modelVersion\>4.0.0\</modelVersion\>  
  \<groupId\>org.openmcp\</groupId\>  
  \<artifactId\>openmcpspec-core\</artifactId\>  
  \<version\>0.1.0\</version\>  
  \<name\>OpenMCPSpec Core\</name\>  
  \<properties\>  
    \<maven.compiler.source\>17\</maven.compiler.source\>  
    \<maven.compiler.target\>17\</maven.compiler.target\>  
    \<jackson.version\>2.17.1\</jackson.version\>  
  \</properties\>  
  \<dependencies\>  
    \<dependency\>  
      \<groupId\>com.fasterxml.jackson.core\</groupId\>  
      \<artifactId\>jackson-databind\</artifactId\>  
      \<version\>${jackson.version}\</version\>  
    \</dependency\>  
  \</dependencies\>  
\</project\>

#### **Core Data Models (POJOs)**

*(The following files contain the Java classes for mapping the OpenMCPSpec JSON structure, including OpenMCPSpec.java, Info.java, Server.java, Intent.java, Parameter.java, Responses.java, Metadata.java, Enumeration.java, Lifecycle.java, Testing.java, and Analytics.java. Due to their length, only one example is included here for brevity, but all should be present in the final structure.)*

**Example:** openmcpspec-core/src/main/java/org/openmcp/core/model/OpenMCPSpec.java

package org.openmcp.core.model;

import java.util.List;

public class OpenMCPSpec {  
  private String openmcpspec;  
  private Info info;  
  private Server server;  
  private List\<Intent\> intents;  
  private Enumeration enumeration;  
  private Lifecycle lifecycle;  
  private Testing testing;  
  private Analytics analytics;

  // Getters and Setters...  
  public String getOpenmcpspec() { return openmcpspec; }  
  public void setOpenmcpspec(String openmcpspec) { this.openmcpspec \= openmcpspec; }  
  public Info getInfo() { return info; }  
  // ... and so on for all fields.  
}

#### **Specification Loader**

**openmcpspec-core/src/main/java/org/openmcp/core/OpenMCPSpecLoader.java**

package org.openmcp.core;

import com.fasterxml.jackson.databind.ObjectMapper;  
import org.openmcp.core.model.OpenMCPSpec;  
import java.io.InputStream;

public class OpenMCPSpecLoader {  
  private final ObjectMapper mapper \= new ObjectMapper();  
  public OpenMCPSpec load(InputStream json) {  
    try {  
      return mapper.readValue(json, OpenMCPSpec.class);  
    } catch (Exception e) {  
      throw new RuntimeException("Failed to load OpenMCPSpec", e);  
    }  
  }  
}

### **2\. openmcpspec-spring-starter Module**

This module integrates the core models into Spring Boot by providing configuration, loading logic, and REST controllers.

#### **openmcpspec-spring-starter/pom.xml**

\<project xmlns\="http://maven.apache.org/POM/4.0.0" xmlns:xsi\="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation\="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd"\>  
  \<modelVersion\>4.0.0\</modelVersion\>  
  \<groupId\>org.openmcp\</groupId\>  
  \<artifactId\>openmcpspec-spring-starter\</artifactId\>  
  \<version\>0.1.0\</version\>  
  \<name\>OpenMCPSpec Spring Starter\</name\>  
  \<properties\>  
    \<maven.compiler.source\>17\</maven.compiler.source\>  
    \<maven.compiler.target\>17\</maven.compiler.target\>  
    \<spring.boot.version\>3.3.2\</spring.boot.version\>  
  \</properties\>  
  \<dependencies\>  
    \<dependency\>  
      \<groupId\>org.openmcp\</groupId\>  
      \<artifactId\>openmcpspec-core\</artifactId\>  
      \<version\>0.1.0\</version\>  
    \</dependency\>  
    \<dependency\>  
      \<groupId\>org.springframework.boot\</groupId\>  
      \<artifactId\>spring-boot-starter-web\</artifactId\>  
      \<version\>${spring.boot.version}\</version\>  
    \</dependency\>  
    \<dependency\>   
      \<groupId\>com.fasterxml.jackson.core\</groupId\>  
      \<artifactId\>jackson-databind\</artifactId\>  
      \<version\>2.17.1\</version\>  
    \</dependency\>  
  \</dependencies\>  
\</project\>

#### **Configuration and Auto-Configuration**

**openmcpspec-spring-starter/src/main/java/org/openmcp/spring/OpenMCPSpecProperties.java**

package org.openmcp.spring;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix \= "openmcp.spec")  
public class OpenMCPSpecProperties {  
  private String path;  
  private String apiPath \= "/api";  
  public String getPath() { return path; }  
  public void setPath(String path) { this.path \= path; }  
  public String getApiPath() { return apiPath; }  
  public void setApiPath(String apiPath) { this.apiPath \= apiPath; }  
}

**openmcpspec-spring-starter/src/main/java/org/openmcp/spring/OpenMCPSpecAutoConfiguration.java**

package org.openmcp.spring;

import org.openmcp.core.OpenMCPSpecLoader;  
import org.openmcp.core.model.OpenMCPSpec;  
import org.springframework.boot.autoconfigure.AutoConfiguration;  
import org.springframework.boot.context.properties.EnableConfigurationProperties;  
import org.springframework.context.annotation.Bean;  
import org.springframework.core.io.Resource;  
import org.springframework.core.io.ResourceLoader;  
import java.io.InputStream;

@AutoConfiguration  
@EnableConfigurationProperties(OpenMCPSpecProperties.class)  
public class OpenMCPSpecAutoConfiguration {  
  @Bean  
  public OpenMCPSpec openMcpSpec(OpenMCPSpecProperties props, ResourceLoader loader) throws Exception {  
    Resource specRes \= loader.getResource(props.getPath());  
    try (InputStream specIn \= specRes.getInputStream()) {  
      return new OpenMCPSpecLoader().load(specIn);  
    }  
  }  
}

#### **Web Controllers**

These controllers expose the specification data as REST endpoints.

**openmcpspec-spring-starter/src/main/java/org/openmcp/spring/web/IntentsController.java** (Exposes the full spec for intents)

package org.openmcp.spring.web;

import org.openmcp.core.model.OpenMCPSpec;  
import org.springframework.web.bind.annotation.GetMapping;  
import org.springframework.web.bind.annotation.RequestMapping;  
import org.springframework.web.bind.annotation.RestController;

@RestController  
@RequestMapping("${openmcp.spec.api-path:/api}")  
public class IntentsController {  
  private final OpenMCPSpec spec;  
  public IntentsController(OpenMCPSpec spec) { this.spec \= spec; }  
  @GetMapping("/intents")  
  public OpenMCPSpec intents() { return spec; }  
}

**openmcpspec-spring-starter/src/main/java/org/openmcp/spring/web/EnumerationController.java** (Exposes the specific enumeration object)

package org.openmcp.spring.web;

import org.openmcp.core.model.OpenMCPSpec;  
import org.openmcp.core.model.Enumeration;  
import org.springframework.web.bind.annotation.GetMapping;  
import org.springframework.web.bind.annotation.RequestMapping;  
import org.springframework.web.bind.annotation.RestController;

@RestController  
@RequestMapping("${openmcp.spec.api-path:/api}")  
public class EnumerationController {  
  private final OpenMCPSpec spec;  
  public EnumerationController(OpenMCPSpec spec) { this.spec \= spec; }  
  @GetMapping("/enumerate")  
  public Enumeration enumerate() { return spec.getEnumeration(); }  
}

#### **Spring Boot Auto-Configuration Metadata**

This file enables Spring Boot to find and use the auto-configuration class.

**openmcpspec-spring-starter/src/main/resources/META-INF/spring/org.openmcp.spring.autoconfigure.imports**

org.openmcp.spring.OpenMCPSpecAutoConfiguration

### **Build the Library**

Run this command from the root directory of the two modules to build and install the artifacts into your local Maven repository:

mvn \-q \-DskipTests install

## ---

**🚀 Section B: Use openmcpspec-spring-starter as a Maven dependency**

This section describes setting up a sample Spring Boot application (sample-mcp-server) that utilizes the starter library and an OpenMCPSpec JSON file.

### **sample-mcp-server/pom.xml**

The sample server adds the **Spring Starter** as a dependency.

\<project xmlns\="http://maven.apache.org/POM/4.0.0" xmlns:xsi\="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation\="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd"\>  
  \<modelVersion\>4.0.0\</modelVersion\>  
  \<groupId\>org.example\</groupId\>  
  \<artifactId\>sample-mcp-server\</artifactId\>  
  \<version\>0.1.0\</version\>  
  \<properties\>  
    \<java.version\>17\</java.version\>  
    \<spring.boot.version\>3.3.2\</spring.boot.version\>  
  \</properties\>  
  \<dependencies\>  
    \<dependency\>  
      \<groupId\>org.openmcp\</groupId\>  
      \<artifactId\>openmcpspec-spring-starter\</artifactId\>  
      \<version\>0.1.0\</version\>  
    \</dependency\>  
    \<dependency\>  
      \<groupId\>org.springframework.boot\</groupId\>  
      \<artifactId\>spring-boot-starter\</artifactId\>  
      \<version\>${spring.boot.version}\</version\>  
    \</dependency\>  
    \<dependency\>  
      \<groupId\>org.springframework.boot\</ விடுதலை="spring-boot-starter-web"\>  
      \<artifactId\>spring-boot-starter-web\</artifactId\>  
      \<version\>${spring.boot.version}\</version\>  
    \</dependency\>  
  \</dependencies\>  
\</project\>

### **Application Configuration**

**sample-mcp-server/src/main/resources/application.properties**

This configures the location of the OpenMCPSpec file and the API path.

openmcp.spec.path\=classpath:/specs/openmcpspec-docs-server.json  
openmcp.spec.api-path\=/api  
server.port\=8080

### **Main Application Class**

**sample-mcp-server/src/main/java/org/example/DocsMcpServerApplication.java**

package org.example;

import org.springframework.boot.SpringApplication;  
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication  
public class DocsMcpServerApplication {  
  public static void main(String\[\] args) {  
    SpringApplication.run(DocsMcpServerApplication.class, args);  
  }  
}

### **OpenMCPSpec Definition**

**sample-mcp-server/src/main/resources/specs/openmcpspec-docs-server.json**

*(This file defines the actual MCP server specification, including its intents, server details, and enumeration properties.)*

{  
  "openmcpspec": "0.1",  
  "info": {  
    "title": "MCP Docs Server",  
    "version": "0.1.0",  
    "description": "MCP server exposing intents for documentation search and retrieval."  
  },  
  "server": {  
    "name": "docs-mcp",  
    "hostname": "localhost",  
    "port": 8080,  
    "protocol": "http",  
    "base\_path": "/api",  
    "endpoints": \[  
      "http://localhost:8080/api/intents",  
      "http://localhost:8080/api/enumerate"  
    \]  
  },  
  "intents": \[  
    {  
      "name": "searchDocs",  
      "description": "Search documentation by keyword or phrase.",  
      "category": "documentation",  
      "parameters": \[  
        { "name": "query", "type": "string", "required": true },  
        { "name": "limit", "type": "integer", "default": 10 }  
      \],  
      "responses": { /\* ... details omitted for brevity ... \*/ },  
      "metadata": { /\* ... details omitted for brevity ... \*/ }  
    },  
    {  
      "name": "getDoc",  
      "description": "Retrieve a specific documentation page by ID.",  
      "category": "documentation",  
      "parameters": \[  
        { "name": "doc\_id", "type": "string", "required": true }  
      \],  
      "responses": { /\* ... details omitted for brevity ... \*/ },  
      "metadata": { /\* ... details omitted for brevity ... \*/ }  
    }  
  \],  
  "enumeration": { /\* ... details omitted for brevity ... \*/ },  
  "lifecycle": { /\* ... details omitted for brevity ... \*/ },  
  "testing": { /\* ... details omitted for brevity ... \*/ },  
  "analytics": { /\* ... details omitted for brevity ... \*/ }  
}

### **Run Sample Server**

Start the application from the sample-mcp-server directory:

mvn \-q \-DskipTests spring-boot:run

## ---

**💻 Section C: Hands-on example with an agent using curl**

Once the sample server is running on http://localhost:8080, you can test the exposed endpoints.

### **1\. List Intents (Full Spec)**

Retrieve the complete **OpenMCPSpec** JSON containing all intent definitions.

curl \-s http://localhost:8080/api/intents

### **2\. Enumerate Intents (Summary)**

Retrieve the summary **Enumeration** object.

curl \-s http://localhost:8080/api/enumerate

### **3\. Simulate LLM using searchDocs intent parameters**

Simulate a call that an LLM would make to use the searchDocs intent.

curl \-s \-X POST http://localhost:8080/api/intents/searchDocs \\  
  \-H "Content-Type: application/json" \\  
  \-d '{"query":"installation","limit":5}'

***Note:** The IntentsController only returns the static spec, not actual intent functionality, so the response for the POST requests above will show the static spec content as currently implemented.*

### **4\. Simulate LLM fetching a document by ID**

Simulate a call for the getDoc intent.

curl \-s \-X POST http://localhost:8080/api/intents/getDoc \\  
  \-H "Content-Type: application/json" \\  
  \-d '{"doc\_id":"doc-123"}'

## ---

**🗒️ Change Log**

| Version | Description |
| :---- | :---- |
| **0.1** | Initial draft of OpenMCPSpec schema and MCP Docs Server example. |

## **👥 Contributors**

* Chaitanya Pinnamaraju  
* Microsoft Copilot

## **🛣️ Roadmap**

* **Version 0.2:** Further development and refinement.  
* **Version 0.3:** Feature completion and stabilization.  
* **Long-Term:** Production readiness and expanded features.

---

