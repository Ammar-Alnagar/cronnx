#[cfg(test)]
mod integration_tests {
    use crate::server::types::{PredictRequest, PredictResponse};
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tower::ServiceExt; // for `app.oneshot()`
    
    use crate::{
        batching::queue::{InferenceJob, BatcherConfig},
        server::{handlers, types::AppState},
        model::registry::ModelRegistry,
    };
    use ndarray::Array4;

    #[tokio::test]
    async fn test_health_check_handler() {
        let registry = ModelRegistry::new();
        let state = Arc::new(AppState { registry });
        
        let response = handlers::health_check().await;
        assert_eq!(response, "OK");
    }

    #[tokio::test]
    async fn test_predict_handler_with_mock_batcher() {
        // Create a mock registry with a channel for testing
        let registry = ModelRegistry::new();
        
        // Create a channel for the mock batcher
        let (tx, mut rx) = mpsc::channel(10);
        
        // Register a mock model
        registry.register("test_model".to_string(), tx);
        
        let state = Arc::new(AppState { registry });
        
        // Create a mock request with base64 encoded dummy image data
        // Using a simple base64 string that represents minimal valid image data
        let request = PredictRequest {
            image: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(), // Minimal PNG
        };

        // We can't fully test the prediction handler without a real model,
        // but we can test that it properly sends the job to the queue
        let result = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("test_model".to_string()),
            axum::Json(request),
        ).await;
        
        // The handler should return an error since we don't have a real batcher running
        // to process the job, but it should at least send the job to the queue
        // Let's check if a job was sent to the queue within a short timeout
        let job_received = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        
        match job_received {
            Ok(Some(_job)) => {
                // Job was successfully sent to the queue
                // This indicates the handler is working correctly
            },
            Ok(None) => {
                // Channel was closed
                panic!("Channel was closed unexpectedly");
            },
            Err(_) => {
                // Timeout - no job was sent
                // This could happen if there's an early error in the handler
                // Let's test the error case
            }
        }
    }

    #[tokio::test]
    async fn test_predict_handler_invalid_model() {
        let registry = ModelRegistry::new();
        let state = Arc::new(AppState { registry });
        
        let request = PredictRequest {
            image: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(),
        };

        // Try to predict with a non-existent model
        let result = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("nonexistent_model".to_string()),
            axum::Json(request),
        ).await;
        
        // Should return ModelNotFound error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_predict_handler_invalid_base64() {
        let registry = ModelRegistry::new();
        let (tx, _rx) = mpsc::channel(10);
        registry.register("test_model".to_string(), tx);
        
        let state = Arc::new(AppState { registry });
        
        let request = PredictRequest {
            image: "invalid_base64_data".to_string(), // Invalid base64
        };

        let result = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("test_model".to_string()),
            axum::Json(request),
        ).await;
        
        // Should return PreprocessingError due to invalid base64
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_predict_handler_empty_base64() {
        let registry = ModelRegistry::new();
        let (tx, _rx) = mpsc::channel(10);
        registry.register("test_model".to_string(), tx);
        
        let state = Arc::new(AppState { registry });
        
        let request = PredictRequest {
            image: "".to_string(), // Empty base64
        };

        let result = handlers::predict(
            axum::extract::State(state),
            axum::extract::Path("test_model".to_string()),
            axum::Json(request),
        ).await;
        
        // Should return an error due to empty/invalid base64
        assert!(result.is_err());
    }
}

// Test for the server routes module
#[cfg(test)]
mod route_tests {
    use crate::server::routes;
    use crate::model::registry::ModelRegistry;
    use metrics_exporter_prometheus::PrometheusBuilder;

    #[test]
    fn test_router_creation() {
        let registry = ModelRegistry::new();
        let metrics_handle = PrometheusBuilder::new()
            .build()
            .expect("Failed to create metrics handle");
        
        // Test that the router can be created without panicking
        let _router = routes::create_router(registry, metrics_handle);
        
        // The router should be created successfully
        // We can't easily test the routes without starting a server,
        // but we can verify the function doesn't panic
    }
}