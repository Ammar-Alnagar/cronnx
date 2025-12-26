# API Documentation

## HTTP Endpoints

### Health Check
```
GET /health
```

**Description**: Checks if the server is running and responding to requests.

**Response**:
- Status: `200 OK`
- Body: `"OK"`

### Model Prediction
```
POST /predict/{model_name}
```

**Description**: Performs inference on an image using the specified model.

**Path Parameters**:
- `model_name` (string): The name of the model to use for inference.

**Request Body**:
```json
{
  "image": "base64_encoded_image_data"
}
```

**Request Parameters**:
- `image` (string): Base64-encoded image data in any format supported by the `image` crate (PNG, JPEG, etc.).

**Response**:
- Status: `200 OK`
- Body:
```json
{
  "predictions": [
    {
      "class_id": 282,
      "confidence": 0.9876
    }
  ],
  "inference_time_ms": 45.2
}
```

**Response Parameters**:
- `predictions` (array): Top 5 predictions with highest confidence.
  - `class_id` (integer): The ImageNet class ID.
  - `confidence` (number): Confidence score between 0 and 1.
- `inference_time_ms` (number): Total time taken for inference in milliseconds.

**Error Responses**:
- `400 Bad Request`: Invalid image data or malformed request.
- `500 Internal Server Error`: Model not found or internal processing error.

### Metrics Endpoint
```
GET /metrics
```

**Description**: Exposes Prometheus-formatted metrics for monitoring.

**Response**:
- Status: `200 OK`
- Body: Prometheus metrics in text format.

**Available Metrics**:
- `requests_received_total{model="model_name"}`: Counter of total requests received per model.
- `request_latency_seconds_bucket{model="model_name", le="0.005"}`: Histogram of request processing time per model.
- `batch_size_bucket{le="1"}`: Histogram of batch sizes processed.
- `request_latency_seconds_sum{model="model_name"}`: Total latency across all requests per model.
- `request_latency_seconds_count{model="model_name"}`: Total count of requests per model.

## Configuration

### Config File Format (config.yaml)

```yaml
server:
  port: 3000          # Port to listen on
  host: "0.0.0.0"     # Host to bind to

models:
  - name: "model_name"        # Unique identifier for the model
    path: "path/to/model.onnx" # Path to the ONNX model file
    batch_size: 16            # Maximum batch size for this model
    batch_timeout_ms: 10      # Timeout before processing partial batch (ms)
```

## Error Handling

### HTTP Error Codes

- `200 OK`: Successful request
- `400 Bad Request`: Client error (invalid input, malformed request)
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Server error (model loading failure, processing error)

### Error Response Format

```json
{
  "error": "Error message describing the issue"
}
```

## Data Types

### Input Image Format
- The image should be base64-encoded
- Any format supported by the `image` crate is accepted (PNG, JPEG, GIF, etc.)
- Images are automatically resized to 224x224 pixels
- Images are normalized using ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

### Output Prediction Format
- Predictions are returned as the top 5 classes with highest confidence
- Confidence values are floats between 0 and 1
- Class IDs correspond to ImageNet class labels

## Performance Considerations

### Batching
- Requests are batched based on `batch_size` and `batch_timeout_ms` configuration
- Larger batch sizes improve throughput but may increase latency
- Optimal batch size depends on your hardware and model

### Concurrency
- The server handles multiple requests concurrently using async/await
- Each model has its own batching queue
- Requests for different models are processed independently

## Security Considerations

### Input Validation
- Images are validated during preprocessing
- Invalid image data returns 400 Bad Request
- Large images are resized to prevent memory exhaustion

### Resource Limits
- Batching queues have configurable size limits
- Request timeouts prevent hanging requests
- Memory usage is monitored during inference

## Examples

### Python Client Example
```python
import base64
import requests

# Load and encode image
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    'http://localhost:3000/predict/mobilenet_v2',
    json={'image': image_data}
)

# Process response
result = response.json()
print(f"Top prediction: {result['predictions'][0]}")
```

### cURL Example
```bash
# Send image for prediction
curl -X POST http://localhost:3000/predict/mobilenet_v2 \
     -H "Content-Type: application/json" \
     -d '{"image": "$(base64 -w 0 image.jpg)"}'
```

### Check Metrics
```bash
# Get current metrics
curl http://localhost:3000/metrics
```