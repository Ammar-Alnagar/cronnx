use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
}

#[derive(Deserialize, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
}

#[derive(Deserialize, Clone)]
pub struct ModelConfig {
    pub task_type: TaskType,
    pub name: String,
    pub path: String,
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
}

#[derive(Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub enum TaskType {
    ImageClassification,
    TextGeneration,
    TextClassification,
    TextEmbedding,
    Decoding,
    Encoding,
    Regression,
    Prediction,
}
