# Phase 6: Advanced Features & Deployment

## 1. Phase Introduction

We have a functioning, monitored, multi-model server. The final steps are **Optimization** and **Containerization**.

**Goals:**

1.  Enable GPU acceleration (CUDA) for 10-100x faster inference.
2.  Optimize the ONNX graph (fusing nodes).
3.  Package the application into a minimal Docker image.

## 2. GPU Acceleration

ONNX Runtime supports "Execution Providers". We can configure `loader.rs` to try CUDA (NVIDIA) or TensorRT.

**Update `Cargo.toml`:**

```toml
ort = { version = "2.0", features = ["ndarray", "cuda"] } # Add cuda feature
```

**Update `src/model/loader.rs`**:

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

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
# Create dummy main to build deps
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm src/main.rs

# Build app
COPY . .
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime deps (OpenSSL, etc)
RUN apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/ml-inference-engine .
# Copy config and models (In production, mount models via volume)
COPY config.yaml .
COPY models ./models

EXPOSE 3000

CMD ["./ml-inference-engine"]
```

## 4. Final Verification

1.  **Build Image**: `docker build -t inference-engine:v1 .`
2.  **Run**: `docker run -p 3000:3000 inference-engine:v1`
3.  **Test**: Curl localhost:3000.

## 5. Course Completion

Congratulations! You have built a Production-Grade ML Inference Engine in Rust.

**You have mastered:**

- [x] **Safe Systems Programming**: Handling errors and memory safely.
- [x] **Concurrent Architecture**: Using `tokio` channels for dynamic batching.
- [x] **API Design**: Building robust APIs with `Axum`.
- [x] **ML Integration**: Using `ort` and `ndarray`.
- [x] **DevOps**: Observability and Containerization.

This foundation is scalable. You can now deploy this to Kubernetes, add Redis for caching, or swap ONNX Runtime for `candle` or `torch-tvm` using the same architecture.
