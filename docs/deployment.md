# Deployment Documentation

## Overview

This document provides comprehensive instructions for deploying Cronnx in various environments, from development to production. It covers containerization, configuration, monitoring, and scaling considerations.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ or CentOS 7+)
- **Architecture**: x86_64
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB free space + model file sizes
- **CPU**: Modern x86-64 processor with SSE4.1 support

### Software Dependencies
- **Docker**: Version 20.10+ (for containerized deployment)
- **Kubernetes**: Version 1.20+ (for orchestration)
- **ONNX Models**: Pre-trained models in `.onnx` format
- **Rust Toolchain**: For building from source (optional)

## Containerization

### Docker Build

```dockerfile
# Dockerfile
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

### Building the Image

```bash
# Build the Docker image
docker build -t cronnx:latest .

# Tag for registry
docker tag cronnx:latest your-registry/cronnx:latest

# Push to registry
docker push your-registry/cronnx:latest
```

### Running with Docker

```bash
# Basic run command
docker run -p 3000:3000 cronnx:latest

# With models volume
docker run -p 3000:3000 \
  -v /path/to/models:/app/models \
  -v /path/to/config.yaml:/app/config.yaml \
  cronnx:latest

# With environment variables
docker run -p 3000:3000 \
  -e RUST_LOG=info \
  -v ./models:/app/models \
  cronnx:latest
```

## Configuration

### Application Configuration

#### config.yaml
```yaml
server:
  port: 3000
  host: "0.0.0.0"

models:
  - name: "mobilenet_v2"
    path: "/app/models/mobilenetv2-7.onnx"
    batch_size: 16
    batch_timeout_ms: 10
  - name: "resnet50"
    path: "/app/models/resnet50.onnx"
    batch_size: 8
    batch_timeout_ms: 20
```

#### Configuration Parameters

**Server Configuration:**
- `port`: HTTP server port (default: 3000)
- `host`: Network interface to bind (default: "0.0.0.0")

**Model Configuration:**
- `name`: Unique identifier for the model
- `path`: Path to the ONNX model file
- `batch_size`: Maximum batch size for this model
- `batch_timeout_ms`: Timeout before processing partial batch (ms)

### Environment Variables

```bash
# Logging level
RUST_LOG=info          # info, debug, trace, warn, error

# Memory settings (if needed)
RUST_BACKTRACE=1       # Enable backtraces

# Runtime settings
HTTP_PROXY=proxy_url   # Proxy settings if needed
HTTPS_PROXY=proxy_url  # HTTPS proxy settings
```

## Kubernetes Deployment

### Kubernetes Manifests

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cronnx
  labels:
    app: cronnx
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
        image: your-registry/cronnx:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 3000
        env:
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: config
        configMap:
          name: cronnx-config
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: cronnx-service
spec:
  selector:
    app: cronnx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
  type: LoadBalancer  # or ClusterIP for internal access
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cronnx-config
data:
  config.yaml: |
    server:
      port: 3000
      host: "0.0.0.0"
    models:
      - name: "mobilenet_v2"
        path: "/app/models/mobilenetv2-7.onnx"
        batch_size: 16
        batch_timeout_ms: 10
```

### Helm Chart (Optional)

Create a Helm chart for easier deployment:

```bash
# Install Helm chart
helm install cronnx ./charts/cronnx

# Upgrade deployment
helm upgrade cronnx ./charts/cronnx --set image.tag=latest
```

## GPU Acceleration

### CUDA Support

To enable GPU acceleration:

1. **Update Cargo.toml**:
```toml
ort = { version = "2.0.0-rc.10", features = ["ndarray", "cuda"] }
```

2. **Use CUDA-enabled base image**:
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder
# ... rest of Dockerfile
```

3. **Kubernetes with GPU**:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1
```

### GPU Requirements
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- Proper GPU drivers

## Monitoring and Observability

### Metrics Collection

#### Prometheus Integration
- **Endpoint**: `http://your-server:3000/metrics`
- **Metrics available**:
  - `requests_received_total`
  - `request_latency_seconds`
  - `batch_size`

#### Grafana Dashboard
Create a Grafana dashboard to visualize:
- Request rate and latency
- Batch size distribution
- Error rates
- System resource usage

### Logging

#### Log Levels
- `error`: Critical errors only
- `warn`: Warnings and errors
- `info`: Normal operational logs
- `debug`: Detailed debugging info
- `trace`: Very detailed tracing

#### Log Collection
- Forward logs to ELK stack
- Use structured logging with JSON format
- Set up log rotation

### Health Checks

#### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10
```

#### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Scaling Strategies

### Horizontal Scaling
- Deploy multiple replicas in Kubernetes
- Use load balancer for traffic distribution
- Monitor batch size across instances

### Vertical Scaling
- Increase CPU and memory resources
- Optimize batch sizes for available resources
- Monitor resource utilization

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cronnx-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cronnx
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Security Considerations

### Network Security
- Use HTTPS with TLS termination
- Implement rate limiting
- Restrict network access with network policies

### Container Security
- Run as non-root user
- Use read-only root filesystem where possible
- Scan images for vulnerabilities

### Input Validation
- Validate all image inputs
- Implement size limits
- Sanitize base64 encoded data

## Backup and Recovery

### Configuration Backup
- Version control for config files
- Regular backup of model files
- Backup of application state

### Model Management
- Version control for models
- A/B testing with model routing
- Model rollback procedures

## Troubleshooting

### Common Issues

#### Application Won't Start
- Check if config file exists and is valid
- Verify model files exist and are accessible
- Check file permissions

#### High Memory Usage
- Reduce batch sizes
- Limit concurrent requests
- Monitor for memory leaks

#### Slow Response Times
- Increase batch timeout
- Optimize batch sizes
- Check for resource constraints

### Debugging Commands

```bash
# Check application logs
kubectl logs -f deployment/cronnx

# Get application metrics
kubectl port-forward service/cronnx-service 3000:80
curl http://localhost:3000/metrics

# Execute in running container
kubectl exec -it deployment/cronnx -- /bin/bash
```

## Performance Optimization

### Batch Size Tuning
- Start with batch_size=16 for most models
- Monitor batch_size metrics
- Adjust based on model and hardware

### Resource Allocation
- Monitor CPU and memory usage
- Set appropriate resource limits
- Use HPA for automatic scaling

### Model Optimization
- Use ONNX model optimization
- Quantize models for faster inference
- Consider model-specific optimizations

## Production Checklist

- [ ] Configuration validated in staging
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Performance benchmarks established
- [ ] Rollback procedures documented
- [ ] Capacity planning completed
- [ ] Load testing performed

This deployment documentation provides comprehensive guidance for deploying Cronnx in production environments with proper monitoring, scaling, and security considerations.