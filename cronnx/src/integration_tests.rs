#[cfg(test)]
mod end_to_end_tests {
    use crate::{
        batching::queue::{Batcher, BatcherConfig, InferenceJob},
        model::registry::ModelRegistry,
        server::{handlers, routes, types::AppState},
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use ndarray::Array4;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_full_system_flow() {
        // Test the complete flow: request -> handler -> registry -> batcher -> response
        let registry = ModelRegistry::new();
        
        // Create a channel to simulate the batcher
        let (tx, mut rx) = mpsc::channel(10);
        registry.register("test_model".to_string(), tx);
        
        // Create app state
        let state = Arc::new(AppState { registry });
        
        // Create a mock request (using minimal valid base64 image)
        let request_json = json!({
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        });
        
        let response = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("test_model".to_string()),
            axum::Json(serde_json::from_value(request_json).unwrap()),
        ).await;
        
        // The handler should send the job to the queue, which we can verify
        // The response will likely be an error since there's no actual batcher running,
        // but the important part is that the job gets sent to the queue
        let job_received = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        assert!(job_received.is_ok());
        assert!(job_received.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_model_registry_integration() {
        // Test that the registry properly connects models to their queues
        let registry = ModelRegistry::new();
        
        // Create multiple model channels
        let (tx1, mut rx1) = mpsc::channel(10);
        let (tx2, mut rx2) = mpsc::channel(10);
        
        registry.register("model1".to_string(), tx1);
        registry.register("model2".to_string(), tx2);
        
        // Verify both models can receive jobs
        let job1_tensor = Array4::<f32>::zeros((1, 3, 224, 224));
        let (result_tx1, _result_rx1) = tokio::sync::oneshot::channel();
        let job1 = InferenceJob {
            input: job1_tensor,
            result_sender: result_tx1,
        };
        
        let job2_tensor = Array4::<f32>::zeros((1, 3, 224, 224));
        let (result_tx2, _result_rx2) = tokio::sync::oneshot::channel();
        let job2 = InferenceJob {
            input: job2_tensor,
            result_sender: result_tx2,
        };
        
        // Send jobs to different models
        if let Some(sender1) = registry.get("model1") {
            let _ = sender1.send(job1).await;
        }
        
        if let Some(sender2) = registry.get("model2") {
            let _ = sender2.send(job2).await;
        }
        
        // Verify jobs are received by the correct channels
        let received1 = tokio::time::timeout(std::time::Duration::from_millis(100), rx1.recv()).await;
        let received2 = tokio::time::timeout(std::time::Duration::from_millis(100), rx2.recv()).await;

        assert!(received1.is_ok() && received1.unwrap().is_some());
        assert!(received2.is_ok() && received2.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_error_propagation() {
        // Test that errors propagate correctly through the system
        let registry = ModelRegistry::new();
        let state = Arc::new(AppState { registry });
        
        // Try to predict with a non-existent model
        let request_json = json!({
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        });
        
        let result = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("nonexistent_model".to_string()),
            axum::Json(serde_json::from_value(request_json).unwrap()),
        ).await;
        
        // Should return an error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_check_endpoint() {
        // Test the health check endpoint
        let response = handlers::health_check().await;
        assert_eq!(response, "OK");
    }

    #[test]
    fn test_config_parsing_simulation() {
        // While we can't easily test config file parsing without files,
        // we can test the config structures
        use crate::config::{AppConfig, ModelConfig, ServerConfig};
        
        let model_config = ModelConfig {
            name: "test_model".to_string(),
            path: "models/test.onnx".to_string(),
            batch_size: 16,
            batch_timeout_ms: 5,
        };
        
        let server_config = ServerConfig {
            port: 3000,
            host: "0.0.0.0".to_string(),
        };
        
        let app_config = AppConfig {
            server: server_config,
            models: vec![model_config],
        };
        
        assert_eq!(app_config.models[0].name, "test_model");
        assert_eq!(app_config.server.port, 3000);
    }

    #[tokio::test]
    async fn test_tensor_processing_pipeline() {
        // Test the complete pipeline: image -> preprocess -> tensor -> batch -> result
        use crate::preprocessing::image::process_bytes;
        use image::{RgbImage, ImageFormat};
        use std::io::Cursor;
        
        // Create a test image
        let img = RgbImage::new(10, 10);
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, ImageFormat::Png).unwrap();
        
        // Process the image to tensor
        let tensor_result = process_bytes(&buffer);
        assert!(tensor_result.is_ok());
        
        let tensor = tensor_result.unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
        
        // Simulate sending through the batching system
        let (tx, mut rx) = mpsc::channel(1);
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        
        let job = InferenceJob {
            input: tensor,
            result_sender: result_tx,
        };
        
        let send_result = tx.send(job).await;
        assert!(send_result.is_ok());
        
        let received_job = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        assert!(received_job.is_ok());
        assert!(received_job.unwrap().is_some());
    }
}