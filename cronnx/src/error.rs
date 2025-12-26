use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ndarray::ShapeError;
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not found at path: {0}")]
    ModelNotFound(String),

    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Input shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ShapeError),
}

impl IntoResponse for InferenceError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            InferenceError::ModelNotFound(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            InferenceError::ShapeMismatch { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            InferenceError::ImageError(_) => {
                (StatusCode::BAD_REQUEST, "Invalid image data".to_string())
            }
            InferenceError::PreprocessingError(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            InferenceError::ShapeError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };

        let body = Json(json!({
            "error": error_message
        }));

        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found_error() {
        let error = InferenceError::ModelNotFound("test_path".to_string());
        assert_eq!(error.to_string(), "Model not found at path: test_path");
    }

    #[test]
    fn test_shape_mismatch_error() {
        let error = InferenceError::ShapeMismatch {
            expected: vec![1, 3, 224, 224],
            got: vec![1, 3, 256, 256],
        };
        assert_eq!(
            error.to_string(),
            "Input shape mismatch: expected [1, 3, 224, 224], got [1, 3, 256, 256]"
        );
    }

    #[test]
    fn test_preprocessing_error() {
        let error = InferenceError::PreprocessingError("Invalid format".to_string());
        assert_eq!(error.to_string(), "Preprocessing error: Invalid format");
    }

    #[test]
    fn test_shape_error_conversion() {
        let shape_error = ShapeError::from_kind(ndarray::ErrorKind::OutOfBounds);
        let inference_error = InferenceError::from(shape_error);
        match inference_error {
            InferenceError::ShapeError(_) => {} // Expected
            _ => panic!("Expected ShapeError"),
        }
    }

    #[test]
    fn test_ort_error_conversion() {
        // Test that OrtError can be converted from ort::Error
        let ort_error = ort::Error::new("test error");
        let inference_error = InferenceError::from(ort_error);
        match inference_error {
            InferenceError::OrtError(_) => {} // Expected
            _ => panic!("Expected OrtError"),
        }
    }

    #[test]
    fn test_image_error_conversion() {
        // Test that ImageError can be converted from image::ImageError
        let image_error =
            image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        let inference_error = InferenceError::from(image_error);
        match inference_error {
            InferenceError::ImageError(_) => {} // Expected
            _ => panic!("Expected ImageError"),
        }
    }

    #[test]
    fn test_into_response_model_not_found() {
        let error = InferenceError::ModelNotFound("test".to_string());
        let response = error.into_response();
        // We can't easily test the response content without more complex mocking,
        // but we can ensure the method runs without panicking
        assert!(response.status().is_client_error() || response.status().is_server_error());
    }

    #[test]
    fn test_into_response_shape_mismatch() {
        let error = InferenceError::ShapeMismatch {
            expected: vec![1, 3, 224, 224],
            got: vec![1, 3, 256, 256],
        };
        let response = error.into_response();
        assert!(response.status().is_client_error());
    }
}
