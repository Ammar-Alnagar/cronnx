# Crates and Dependencies

This document provides an overview of the external crates and dependencies used in the Cronnx project, along with their purposes and configuration.

## Core Dependencies

### Web Framework
- **axum**: 0.7.x
  - Purpose: Modern, ergonomic web framework built on tokio
  - Features: `["macros"]` for route macros
  - Used for: HTTP server, routing, request/response handling

- **tokio**: 1.0+
  - Purpose: Asynchronous runtime for Rust
  - Features: `["full"]` for complete async ecosystem
  - Used for: Async/await support, task spawning, async channels

### Serialization
- **serde**: 1.0+
  - Purpose: Serialization/deserialization framework
  - Features: `["derive"]` for derive macros
  - Used for: JSON serialization, config parsing

- **serde_json**: 1.0+
  - Purpose: JSON serialization support
  - Used for: Request/response body parsing

- **serde_yaml**: 0.9+
  - Purpose: YAML serialization support
  - Used for: Configuration file parsing

- **base64**: 0.21+
  - Purpose: Base64 encoding/decoding
  - Used for: Image data encoding in API requests

### ML & Math
- **ort**: 2.0.0-rc.10
  - Purpose: ONNX Runtime bindings for Rust
  - Features: `["ndarray"]` for ndarray integration
  - Used for: Model loading, inference execution
  - Optional features: `["cuda"]` for GPU acceleration

- **ndarray**: 0.15+
  - Purpose: N-dimensional array library
  - Used for: Tensor manipulation, mathematical operations

### Image Processing
- **image**: 0.24+
  - Purpose: Image processing library
  - Used for: Image loading, resizing, format conversion
  - Supports: PNG, JPEG, GIF, and many other formats

### Error Handling
- **thiserror**: 1.0+
  - Purpose: Custom error types with derive macros
  - Used for: Domain-specific error types with proper error chaining

- **anyhow**: 1.0+
  - Purpose: Error handling for applications
  - Used for: High-level error handling in main functions

### Logging & Observability
- **tracing**: 0.1+
  - Purpose: Structured, async-aware logging
  - Used for: Application logging, request tracing

- **tracing-subscriber**: 0.3+
  - Purpose: Tracing subscriber implementations
  - Features: `["env-filter"]` for environment-based filtering
  - Used for: Log formatting and filtering

- **metrics**: 0.21+
  - Purpose: Metrics collection interface
  - Used for: Application metrics collection

- **metrics-exporter-prometheus**: 0.12+
  - Purpose: Prometheus metrics exporter
  - Used for: Metrics exposure in Prometheus format

### HTTP Middleware
- **tower**: 0.4+
  - Purpose: Service abstraction for async Rust
  - Features: `["util"]` for utility types
  - Used for: Middleware implementation

- **tower-http**: 0.5+
  - Purpose: HTTP-specific tower middleware
  - Features: `["trace", "cors"]` for tracing and CORS
  - Used for: Request tracing, HTTP utilities

## Development Dependencies

### Testing
- **tempfile**: 3.0+
  - Purpose: Temporary file/directory creation
  - Used for: Unit tests requiring temporary files
  - Only in dev-dependencies

## Dependency Tree

```
cronnx
├── axum (0.7) - HTTP framework
│   ├── tokio (1.0) - async runtime
│   └── serde (1.0) - serialization
├── ort (2.0) - ONNX Runtime
│   └── ndarray (0.15) - tensor operations
├── image (0.24) - image processing
├── tracing (0.1) - structured logging
├── metrics (0.21) - metrics collection
└── thiserror (1.0) - error handling
```

## Optional Features

### GPU Support
- **ort** with `cuda` feature enables GPU acceleration
- Adds dependency on CUDA libraries
- Significantly improves inference performance on supported hardware

### Development Features
- **tempfile** only used in tests
- No runtime dependency in production builds

## Version Management

### Stability
- Most dependencies use specific version ranges for stability
- ONNX Runtime (ort) uses a release candidate version (2.0.0-rc.10)
- Major versions are locked to prevent breaking changes

### Updates
- Dependencies should be updated carefully to avoid breaking changes
- ONNX Runtime updates may require code changes due to API changes
- Image processing updates may affect normalization behavior

## Security Considerations

### Audit
- Regular dependency audits recommended
- Pay attention to security advisories for ONNX Runtime
- Monitor image processing library for vulnerabilities

### Transitive Dependencies
- Many dependencies have their own dependency trees
- Use `cargo tree` to analyze full dependency graph
- Consider using `cargo-deny` for security checking