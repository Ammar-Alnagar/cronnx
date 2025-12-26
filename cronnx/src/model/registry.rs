use crate::config::TaskType;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Type alias for the complex registry type
type ModelMap = HashMap<(TaskType, String), Arc<Mutex<Session>>>;

/// The Registry maps a (task_type, model_name) to its session.
/// We use RwLock to allow concurrent reads (lookups).
#[derive(Clone)]
pub struct ModelRegistry {
    // key: (task_type, model_name), value: Session
    sessions: Arc<RwLock<ModelMap>>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register(&self, task_type: TaskType, name: String, session: Arc<Mutex<Session>>) {
        let mut map = self.sessions.write().unwrap();
        map.insert((task_type, name), session);
    }

    pub fn get(&self, task_type: &TaskType, name: &str) -> Option<Arc<Mutex<Session>>> {
        let map = self.sessions.read().unwrap();
        map.get(&(task_type.clone(), name.to_string())).cloned()
    }
}
