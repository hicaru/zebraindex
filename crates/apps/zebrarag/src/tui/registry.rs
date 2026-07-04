use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

pub use zrag_remote_embed::RemoteProvider;

#[derive(Debug, Deserialize)]
pub struct ModelsRegistry {
    #[serde(rename = "models")]
    pub entries: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ModelSource {
    #[default]
    Local,
    Remote(RemoteProvider),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub parameters: String,
    pub technologies: Vec<String>,
    pub description: String,
    #[serde(default, skip_deserializing)]
    pub source: ModelSource,
    /// Weight formats detected for this model (safetensors / gguf) by searching
    /// its HF repo + local cache. Populated at load; not read from models.toml.
    #[serde(default, skip_deserializing)]
    pub formats: Vec<zrag_embed::model_registry::WeightsFormat>,
    /// Precision-level variants detected for this model (populated async by
    /// `detect_weight_variants`, streamed into the list as they resolve).
    #[serde(default, skip_deserializing)]
    pub variants: Vec<zrag_embed::model_registry::WeightVariant>,
}

pub fn is_model_downloaded(model_id: &str) -> bool {
    zrag_embed::is_model_cached(model_id)
}

impl ModelEntry {
    pub fn is_downloaded(&self) -> bool {
        match self.source {
            ModelSource::Local => is_model_downloaded(&self.model_id),
            ModelSource::Remote(_) => false,
        }
    }
}

/// Menu placeholder for a remote provider; selecting it starts the API-key flow.
pub fn remote_sentinel(provider: RemoteProvider) -> ModelEntry {
    ModelEntry {
        model_id: provider.as_str().to_string(),
        parameters: String::from("remote"),
        technologies: vec![String::from("API")],
        description: format!(
            "Embeddings via {} API — no local download required",
            provider.label()
        ),
        source: ModelSource::Remote(provider),
        formats: Vec::new(),
        variants: Vec::new(),
    }
}

/// One sentinel entry per supported remote provider, in declaration order.
pub fn remote_sentinels() -> Vec<ModelEntry> {
    RemoteProvider::ALL
        .iter()
        .map(|&provider| remote_sentinel(provider))
        .collect()
}

pub fn registry_path() -> Result<PathBuf> {
    Ok(zrag_common::paths::data_dir()?.join("models.toml"))
}

pub fn parse(content: &str) -> Result<ModelsRegistry> {
    toml::from_str(content).map_err(Into::into)
}

pub fn load() -> Result<Option<ModelsRegistry>> {
    let path = registry_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    parse(&content).map(Some)
}

pub fn sort_by_hardware(entries: &mut [ModelEntry]) {
    entries.sort_by_key(|entry| {
        let rank = match entry.source {
            ModelSource::Remote(_) => 2,
            ModelSource::Local if is_model_downloaded(&entry.model_id) => 1,
            ModelSource::Local => 0,
        };
        std::cmp::Reverse(rank)
    });
}
