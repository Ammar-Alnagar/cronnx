use axum::{
    extract::{Path, State},
    Json,
};
use base64::{engine::general_purpose, Engine as _};
use std::sync::Arc;
use std::time::Instant;

use crate::config::TaskType;
use crate::error::InferenceError;
use crate::server::types::*;
use ndarray::Axis;
use ort::value::Value;

pub async fn health_check() -> &'static str {
    "OK"
}

pub async fn image_classification_predict(
    State(state): State<Arc<AppState>>,
    Path(model_name): Path<String>,
    Json(payload): Json<ImageClassifyRequest>,
) -> Result<Json<ImageClassifyResponse>, InferenceError> {
    // 1. Decode Base64
    let image_bytes = general_purpose::STANDARD
        .decode(&payload.image)
        .map_err(|e| InferenceError::PreprocessingError(format!("Base64 decode failed: {}", e)))?;

    // 2. Preprocess
    let start = Instant::now();
    let input_tensor = crate::preprocessing::image::process_bytes(&image_bytes)?;

    // 3. Inference
    let session_arc = state
        .registry
        .get(&TaskType::ImageClassification, &model_name)
        .ok_or_else(|| InferenceError::ModelNotFound(model_name.clone()))?;
    let mut session_guard = session_arc.lock().unwrap();
    let input_name = session_guard.inputs[0].name.clone();
    let shape = input_tensor.shape().to_vec();
    let data = input_tensor.into_raw_vec().into_boxed_slice();
    let input_value = Value::from_array((shape, data))?;
    let outputs = session_guard.run(ort::inputs![input_name => input_value])?;

    let duration = start.elapsed();

    // 4. Post-process
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let output = ndarray::ArrayViewD::from_shape(dims.as_slice(), data)?;
    let probabilities = output.index_axis(Axis(0), 0);

    let mut preds: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top 5
    let predictions = preds
        .iter()
        .take(5)
        .map(|(id, prob)| Prediction {
            class_id: *id,
            confidence: *prob,
        })
        .collect();

    Ok(Json(ImageClassifyResponse {
        predictions,
        inference_time_ms: duration.as_secs_f64() * 1000.0,
    }))
}

// Stub implementations for other task types
pub async fn text_generation_predict(
    State(state): State<Arc<AppState>>,
    Path(model_name): Path<String>,
    Json(payload): Json<TextGenerateRequest>,
) -> Result<Json<TextGenerateResponse>, InferenceError> {
    // Preprocess text
    let start = Instant::now();
    let input_tensor = crate::preprocessing::text::preprocess_text(&payload.prompt)?;

    // Inference
    let session_arc = state
        .registry
        .get(&TaskType::TextGeneration, &model_name)
        .ok_or_else(|| InferenceError::ModelNotFound(model_name.clone()))?;
    let mut session_guard = session_arc.lock().unwrap();
    let input_name = session_guard.inputs[0].name.clone();
    let shape = input_tensor.shape().to_vec();
    let data = input_tensor.into_raw_vec().into_boxed_slice();
    let input_value = Value::from_array((shape, data))?;
    let outputs = session_guard.run(ort::inputs![input_name => input_value])?;

    let duration = start.elapsed();

    // Post-process: generate text based on top class
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let output = ndarray::ArrayViewD::from_shape(dims.as_slice(), data)?;
    let probabilities = output.index_axis(Axis(0), 0);

    let mut preds: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_class = preds[0].0;
    let generated_text = format!(
        "{} (generated based on class {})",
        payload.prompt, top_class
    );

    Ok(Json(TextGenerateResponse {
        generated_text,
        inference_time_ms: duration.as_secs_f64() * 1000.0,
    }))
}

pub async fn text_classification_predict(
    State(state): State<Arc<AppState>>,
    Path(model_name): Path<String>,
    Json(payload): Json<TextClassifyRequest>,
) -> Result<Json<TextClassifyResponse>, InferenceError> {
    // Preprocess text
    let start = Instant::now();
    let input_tensor = crate::preprocessing::text::preprocess_text(&payload.text)?;

    // Inference
    let session_arc = state
        .registry
        .get(&TaskType::TextClassification, &model_name)
        .ok_or_else(|| InferenceError::ModelNotFound(model_name.clone()))?;
    let mut session_guard = session_arc.lock().unwrap();
    let input_name = session_guard.inputs[0].name.clone();
    let shape = input_tensor.shape().to_vec();
    let data = input_tensor.into_raw_vec().into_boxed_slice();
    let input_value = Value::from_array((shape, data))?;
    let outputs = session_guard.run(ort::inputs![input_name => input_value])?;

    let duration = start.elapsed();

    // Post-process
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let output = ndarray::ArrayViewD::from_shape(dims.as_slice(), data)?;
    let probabilities = output.index_axis(Axis(0), 0);

    let mut preds: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Map class to label (simple mapping)
    let predictions: Vec<TextPrediction> = preds
        .iter()
        .take(2)
        .map(|(class, conf)| {
            let label = match class {
                0 => "negative",
                1 => "positive",
                _ => "neutral",
            }
            .to_string();
            TextPrediction {
                label,
                confidence: *conf,
            }
        })
        .collect();

    Ok(Json(TextClassifyResponse {
        predictions,
        inference_time_ms: duration.as_secs_f64() * 1000.0,
    }))
}

pub async fn text_embedding_predict(
    State(_state): State<Arc<AppState>>,
    Path(_model_name): Path<String>,
    Json(_payload): Json<TextEmbedRequest>,
) -> Result<Json<TextEmbedResponse>, InferenceError> {
    Ok(Json(TextEmbedResponse {
        embedding: vec![0.1, 0.2, 0.3],
        inference_time_ms: 0.0,
    }))
}

pub async fn decoding_predict(
    State(_state): State<Arc<AppState>>,
    Path(_model_name): Path<String>,
    Json(_payload): Json<DecodeRequest>,
) -> Result<Json<DecodeResponse>, InferenceError> {
    Ok(Json(DecodeResponse {
        decoded_text: "Decoded text placeholder".to_string(),
        inference_time_ms: 0.0,
    }))
}

pub async fn encoding_predict(
    State(_state): State<Arc<AppState>>,
    Path(_model_name): Path<String>,
    Json(_payload): Json<EncodeRequest>,
) -> Result<Json<EncodeResponse>, InferenceError> {
    Ok(Json(EncodeResponse {
        encoded_data: vec![0.1, 0.2, 0.3],
        inference_time_ms: 0.0,
    }))
}

pub async fn regression_predict(
    State(_state): State<Arc<AppState>>,
    Path(_model_name): Path<String>,
    Json(_payload): Json<RegressionRequest>,
) -> Result<Json<RegressionResponse>, InferenceError> {
    Ok(Json(RegressionResponse {
        prediction: 42.0,
        inference_time_ms: 0.0,
    }))
}

pub async fn general_prediction_predict(
    State(_state): State<Arc<AppState>>,
    Path(_model_name): Path<String>,
    Json(_payload): Json<GeneralPredictionRequest>,
) -> Result<Json<GeneralPredictionResponse>, InferenceError> {
    Ok(Json(GeneralPredictionResponse {
        result: serde_json::json!({"prediction": "placeholder"}),
        inference_time_ms: 0.0,
    }))
}
