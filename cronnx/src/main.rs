use cronnx::{config, model, server};
use std::fs;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Init
    model::loader::init_ort()?;

    // 2. Load Config
    let config_content = fs::read_to_string("config.yaml")?;
    let config: config::AppConfig = serde_yaml::from_str(&config_content)?;

    let registry = model::registry::ModelRegistry::new();

    // 3. Initialize Models
    for model_conf in config.models {
        println!(
            "Loading model: {} for task {:?}",
            model_conf.name, model_conf.task_type
        );
        let session = model::loader::load_model(&model_conf.path)?;
        let session = std::sync::Arc::new(std::sync::Mutex::new(session));

        registry.register(model_conf.task_type, model_conf.name, session);
    }

    // 4. Create Router
    let app = server::routes::create_router(registry);

    // 5. Bind & Serve
    let listener =
        TcpListener::bind(format!("{}:{}", config.server.host, config.server.port)).await?;
    println!(
        "Server listening on http://{}:{}",
        config.server.host, config.server.port
    );

    axum::serve(listener, app).await?;

    Ok(())
}
