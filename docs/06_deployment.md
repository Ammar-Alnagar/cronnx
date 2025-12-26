# Phase 6: Advanced Features & Deployment

## 1. Phase Introduction

We have a functioning, monitored, multi-model server. The final steps are **Optimization** and **Containerization**.

**Goals:**

1.  Enable GPU acceleration (CUDA) for 10-100x faster inference.
2.  Optimize the ONNX graph (fusing nodes).
3.  Package the application into a minimal Docker image.
4.  Document comprehensive testing and deployment procedures.

## 2. GPU Acceleration

ONNX Runtime supports "Execution Providers". We can configure `loader.rs` to try CUDA (NVIDIA) or TensorRT.

**Update `Cargo.toml`:**

```toml
ort = { version = "2.0.0-rc.10", features = ["ndarray", "cuda"] } # Add cuda feature
```

**Update `src/model/loader.rs`:**

```rust
use ort::{SessionBuilder, ExecutionProviderDispatch};

pub fn load_model(path: impl AsRef<Path>) -> Result<Session, InferenceError> {
    // Basic Builder
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?;

    // Try to enable CUDA, fallback to CPU silently (or log it)
    let builder = match SessionBuilder::with_execution_providers(
        builder,
        [ExecutionProviderDispatch::CUDA(Default::default())]
    ) {
        Ok(b) => {
            println!("CUDA Execution Provider registered");
            b
        },
        Err(e) => {
            println!("CUDA failed, using CPU: {}", e);
            // Re-create builder or just continue if method allows (builder pattern in ort might consume)
            // Re-initialization often safer if the previous consume failed,
            // but usually we just add the provider to the priority list.
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
        }
    };

    let session = builder.commit_from_file(path)?;
    Ok(session)
}
```

## 3. Docker Deployment

Rust binaries are static, but modern Rust apps often link to `glibc` (unless `musl` is used). Also, `ort` might link to shared `libonnxruntime.so` depending on build.

**File: `Dockerfile`**

```dockerfile
# Stage 1: Builder
FROM rust:1.75-slim-bookworm as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/cronnx .

# Create directories for config and models
RUN mkdir -p /app/config /app/models

EXPOSE 3000

CMD ["./cronnx"]
```

## 4. Configuration Management

### 4.1 Environment Variables

The application supports configuration through environment variables:

```bash
# Set log level
RUST_LOG=info

# Enable tracing
RUST_LOG=trace

# Set port (through config file)
# The config file is the primary configuration method
```

### 4.2 Production Configuration

Example production `config.yaml`:

```yaml
server:
  port: 3000
  host: "0.0.0.0"

models:
  - name: "mobilenet_v2"
    path: "/app/models/mobilenetv2-7.onnx"
    batch_size: 32
    batch_timeout_ms: 5
  - name: "resnet50"
    path: "/app/models/resnet50.onnx"
    batch_size: 16
    batch_timeout_ms: 10
```

## 5. Testing Procedures

### 5.1 Unit Testing

Run all unit tests:

```bash
# Run all tests
cargo test

# Run specific test modules
cargo test error
cargo test preprocessing
cargo test batching
cargo test server
cargo test model
```

### 5.2 Integration Testing

Test the complete system flow:

```bash
# Run integration tests
cargo test --test integration_tests

# Run with specific filters
cargo test end_to_end
```

### 5.3 Performance Testing

Use tools like `wrk` or `hey` for load testing:

```bash
# Example load test
hey -n 1000 -c 10 -m POST -D request.json http://localhost:3000/predict/mobilenet
```

## 6. Monitoring and Observability

### 6.1 Metrics Collection

The application exposes Prometheus metrics at `/metrics`:

- `requests_received` - Counter of requests by model
- `request_latency_seconds` - Histogram of request processing time
- `batch_size` - Histogram of batch sizes processed

### 6.2 Log Collection

Structured logs are available via the tracing system:

```bash
# View logs
RUST_LOG=info cargo run

# Detailed tracing
RUST_LOG=trace cargo run
```

## 7. Kubernetes Deployment

Example Kubernetes deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cronnx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cronnx
  template:
    metadata:
      labels:
        app: cronnx
    spec:
      containers:
      - name: cronnx
        image: cronnx:latest
        ports:
        - containerPort: 3000
        env:
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: config
        configMap:
          name: cronnx-config
---
apiVersion: v1
kind: Service
metadata:
  name: cronnx
spec:
  selector:
    app: cronnx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
  type: LoadBalancer
```

## 8. Final Verification

### 8.1 Build and Run

1.  **Build Image**: `docker build -t cronnx:v1 .`
2.  **Run**: `docker run -p 3000:3000 -v ./models:/app/models -v ./config.yaml:/app/config.yaml cronnx:v1`
3.  **Test**: `curl http://localhost:3000/health`

### 8.2 API Testing

```bash
# Health check
curl http://localhost:3000/health

# Model prediction
curl -X POST http://localhost:3000/predict/mobilenet_v2 \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image"}'

# Metrics
curl http://localhost:3000/metrics
```

## 9. Course Completion

Congratulations! You have built a Production-Grade ML Inference Engine in Rust.

**You have mastered:**

- [x] **Safe Systems Programming**: Handling errors and memory safely.
- [x] **Concurrent Architecture**: Using `tokio` channels for dynamic batching.
- [x] **API Design**: Building robust APIs with `Axum`.
- [x] **ML Integration**: Using `ort` and `ndarray`.
- [x] **DevOps**: Observability and Containerization.
- [x] **Testing**: Comprehensive test coverage across all modules.
- [x] **Documentation**: Complete documentation for all features.

This foundation is scalable. You can now deploy this to Kubernetes, add Redis for caching, or swap ONNX Runtime for `candle` or `torch-tvm` using the same architecture.