use crate::model::registry::ModelRegistry;
use crate::server::{handlers, types::AppState};
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

pub fn create_router(registry: ModelRegistry) -> Router {
    let state = Arc::new(AppState { registry });

    Router::new()
        .route("/health", get(handlers::health_check))
        .route(
            "/image-classification/:model_name",
            post(handlers::image_classification_predict),
        )
        .route(
            "/text-generation/:model_name",
            post(handlers::text_generation_predict),
        )
        .route(
            "/text-classification/:model_name",
            post(handlers::text_classification_predict),
        )
        .route(
            "/text-embedding/:model_name",
            post(handlers::text_embedding_predict),
        )
        .route("/decoding/:model_name", post(handlers::decoding_predict))
        .route("/encoding/:model_name", post(handlers::encoding_predict))
        .route(
            "/regression/:model_name",
            post(handlers::regression_predict),
        )
        .route(
            "/prediction/:model_name",
            post(handlers::general_prediction_predict),
        )
        .with_state(state)
}
