# Crates Guide

This document explains the key Rust crates (libraries) used in the **Cronnx** inference engine, why we chose them, and what role they play in the system.

## Core Runtime

### `tokio`

- **Version**: `1.0` (features = `["full"]`)
- **Role**: The Asynchronous Runtime.
- **Why**: Tokio is the standard for async Rust. It provides the event loop, non-blocking I/O, timers, and channels (`mpsc`, `oneshot`) that power our high-concurrency architecture.

### `axum`

- **Version**: `0.7`
- **Role**: Web Framework.
- **Why**: Built by the Tokio team, Axum is ergonomic, modern, and fully compatible with the Tower ecosystem. It uses an extractor pattern (e.g., `Json`, `State`) that makes handlers type-safe and easy to test.

## Machine Learning & Math

### `ort`

- **Version**: `2.0`
- **Role**: ONNX Runtime Bindings.
- **Why**: Accesses Microsoft's ONNX Runtime (C++) SDK. This allows us to run standard `.onnx` models exported from PyTorch, TensorFlow, etc., with high performance and hardware acceleration (CUDA/TensorRT).

### `ndarray`

- **Version**: `0.15`
- **Role**: N-Dimensional Arrays (Tensors).
- **Why**: The "NumPy of Rust". It provides the `Array4` type we use for image tensors. It handles memory layout (NCHW vs NHWC) and efficient slicing, used heavily during batching and preprocessing.

### `image`

- **Version**: `0.24`
- **Role**: Image Processing.
- **Why**: The standard library for loading, resizing, and manipulating images in Rust. We use it to load JPEGs from raw bytes and resize them to model input dimensions (e.g., 224x224).

## Serialization & Config

### `serde` & `serde_json`

- **Version**: `1.0`
- **Role**: Serialization/Deserialization.
- **Why**: Effectively maps Rust structs to JSON (for API responses) and YAML (for configuration). The `derive` feature automatically generates the code to convert our `PredictRequest` and `PredictResponse` structs.

### `base64`

- **Version**: `0.21`
- **Role**: Encoding/Decoding.
- **Why**: ML APIs often receive images as Base64 strings inside JSON payloads rather than multipart uploads. This crate handles the decoding to raw bytes.

## Observability

### `tracing` & `tracing-subscriber`

- **Version**: `0.1` / `0.3`
- **Role**: Logging & Instrumentation.
- **Why**: Instead of messy `println!`, Tracing allows "Structured Logging". We can attach context (like `model_name`) to a span, and all logs within that span automatically inherit it.

### `metrics` & `metrics-exporter-prometheus`

- **Version**: `0.21` / `0.12`
- **Role**: Metrics Collection.
- **Why**: Defines a vendor-neutral API for counting events and timing durations. The exporter ships these metrics to a Prometheus-compatible endpoint, essential for production monitoring dashboards.

### `tower` & `tower-http`

- **Version**: `0.4` / `0.5`
- **Role**: Middleware.
- **Why**: Middleware wraps our HTTP service to add functionality like Timeout, Tracing (logging every request), CORS, and Limiters without touching business logic.

## Utilities

### `thiserror`

- **Version**: `1.0`
- **Role**: Error Definition.
- **Why**: Makes defining custom `enum` errors (like `InferenceError`) trivial. It works well for libraries or internal modules where you want structured error types.

### `anyhow`

- **Version**: `1.0`
- **Role**: Error Handling (Application level).
- **Why**: Used in `main.rs` to handle "any" error that might occur during startup. It provides nice formatting and stack traces for fatal crashes.
