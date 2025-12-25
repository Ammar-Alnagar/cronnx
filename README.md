# Cronnx: Production-Grade ML Inference Engine in Rust

**Cronnx** is a high-performance, asynchronous Machine Learning inference server built in Rust. It demonstrates how to take a raw ONNX model and serve it via a robust HTTP API with features like dynamic batching, multi-model support, and production observability.

This repository contains a comprehensive step-by-step guide and implementation to build Cronnx from scratch.

## Key Features

- **High Performance**: Low-latency inference using `ort` (ONNX Runtime bindings) and `ndarray`.
- **Asynchronous Core**: Built on `tokio` and `axum` to handle thousands of concurrent connections.
- **Dynamic Batching**: Automatically groups incoming requests into batches for efficient GPU/CPU utilization (simulating NVIDIA Triton features).
- **Multi-Model Support**: Hot-swappable model registry supporting V1/V2 deployments and A/B testing.
- **Observability**: Built-in Prometheus metrics (`requests`, `latency`, `batch_size`) and structured tracing.
- **Production Ready**: Dockerized with multi-stage builds and GPU support (CUDA).

## Implementation Guide

The project is documented in 6 progressive phases. You can find the detailed implementation guides in the `docs/` directory:

| Phase                                            | Description                                                                    | Key Tech                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------- |
| [**01_foundation**](docs/01_foundation.md)       | **Core Inference**: Setting up project, ONNX loading, and image preprocessing. | `ort`, `ndarray`, `image` |
| [**02_async_api**](docs/02_async_api.md)         | **HTTP Server**: Creating an Async Web API with Request/Response DTOs.         | `axum`, `serde`, `tokio`  |
| [**03_batching**](docs/03_batching.md)           | **Dynamic Batching**: Implementing queue-based batching for high throughput.   | `mpsc`, `tokio::select!`  |
| [**04_multimodel**](docs/04_multimodel.md)       | **Model Registry**: Supporting multiple models via URL routing.                | `RwLock`, `HashMap`       |
| [**05_observability**](docs/05_observability.md) | **Metrics & Tracing**: Adding Prometheus metrics and structured logs.          | `tracing`, `metrics`      |
| [**06_deployment**](docs/06_deployment.md)       | **Prod & GPU**: Dockerization and enabling CUDA acceleration.                  | `Docker`, `CUDA`          |

## Quick Start

### Prerequisites

- Rust 1.75+
- `libssl-dev` (or equivalent)

### Running the Server

1.  **Download Model**:

    ```bash
    mkdir -p models
    wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O models/mobilenetv2-7.onnx
    ```

2.  **Configuration**:
    Create a `config.yaml` file:

    ```yaml
    server:
      port: 3000
      host: "0.0.0.0"
    models:
      - name: "mobilenet"
        path: "models/mobilenetv2-7.onnx"
        batch_size: 16
        batch_timeout_ms: 10
    ```

3.  **Run**:

    ```bash
    cargo run --release
    ```

4.  **Test Inference**:
    ```bash
    # POST a base64 encoded image
    IMAGE=$(base64 -w 0 data/test_image.jpg)
    curl -X POST http://localhost:3000/predict/mobilenet \
         -H "Content-Type: application/json" \
         -d "{\"image\": \"$IMAGE\"}"
    ```

##  Project Structure

```
cronnx/
├── Cargo.toml          # Dependencies
├── config.yaml         # App Configuration
├── docs/               # The Implementation Guides
├── models/             # Directory for .onnx files
└── src/
    ├── main.rs         # Entry point & setup
    ├── config.rs       # Config structs
    ├── error.rs        # Custom Error handling
    ├── server/         # Axum Handlers & Routes
    ├── model/          # ONNX Loading & Registry
    ├── batching/       # Dynamic Batching Logic
    └── preprocessing/  # Image resizing & normalization
```

##  License

MIT
