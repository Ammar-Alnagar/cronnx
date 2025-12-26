use crate::model::registry::ModelRegistry;
use serde::{Deserialize, Serialize};

/// Shared Application State
#[derive(Clone)]
pub struct AppState {
    pub registry: ModelRegistry,
}

// --- DTOs (Data Transfer Objects) ---

// Image Classification
#[derive(Deserialize)]
pub struct ImageClassifyRequest {
    /// Base64 encoded image data
    pub image: String,
}

#[derive(Serialize)]
pub struct ImageClassifyResponse {
    pub predictions: Vec<Prediction>,
    pub inference_time_ms: f64,
}

#[derive(Serialize)]
pub struct Prediction {
    pub class_id: usize,
    pub confidence: f32,
}

// Text Generation
#[derive(Deserialize)]
pub struct TextGenerateRequest {
    pub prompt: String,
    pub max_length: Option<usize>,
}

#[derive(Serialize)]
pub struct TextGenerateResponse {
    pub generated_text: String,
    pub inference_time_ms: f64,
}

// Text Classification
#[derive(Deserialize)]
pub struct TextClassifyRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct TextClassifyResponse {
    pub predictions: Vec<TextPrediction>,
    pub inference_time_ms: f64,
}

#[derive(Serialize)]
pub struct TextPrediction {
    pub label: String,
    pub confidence: f32,
}

// Text Embedding
#[derive(Deserialize)]
pub struct TextEmbedRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct TextEmbedResponse {
    pub embedding: Vec<f32>,
    pub inference_time_ms: f64,
}

// Decoding
#[derive(Deserialize)]
pub struct DecodeRequest {
    pub encoded_data: Vec<f32>,
}

#[derive(Serialize)]
pub struct DecodeResponse {
    pub decoded_text: String,
    pub inference_time_ms: f64,
}

// Encoding
#[derive(Deserialize)]
pub struct EncodeRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct EncodeResponse {
    pub encoded_data: Vec<f32>,
    pub inference_time_ms: f64,
}

// Regression
#[derive(Deserialize)]
pub struct RegressionRequest {
    pub features: Vec<f32>,
}

#[derive(Serialize)]
pub struct RegressionResponse {
    pub prediction: f32,
    pub inference_time_ms: f64,
}

// General Prediction
#[derive(Deserialize)]
pub struct GeneralPredictionRequest {
    pub data: serde_json::Value,
}

#[derive(Serialize)]
pub struct GeneralPredictionResponse {
    pub result: serde_json::Value,
    pub inference_time_ms: f64,
}
