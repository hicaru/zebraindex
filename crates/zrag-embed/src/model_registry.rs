use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::pooling::PoolingStrategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub model_id: String,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config_path: PathBuf,
    pub dim: usize,
    pub max_length: usize,
    pub pooling: PoolingStrategyEnum,
    pub query_prefix: Option<String>,
    pub passage_prefix: Option<String>,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub compute_dtype: Option<ComputeDTypeHint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDTypeHint {
    F32,
    F16,
    BF16,
}

impl ComputeDTypeHint {
    pub fn from_torch_dtype(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "float32" | "torch.float32" | "f32" | "fp32" | "float" => Some(Self::F32),
            "float16" | "torch.float16" | "f16" | "fp16" | "half" => Some(Self::F16),
            "bfloat16" | "torch.bfloat16" | "bf16" => Some(Self::BF16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PoolingStrategyEnum {
    Mean,
    Cls,
}

impl From<PoolingStrategyEnum> for PoolingStrategy {
    fn from(v: PoolingStrategyEnum) -> Self {
        match v {
            PoolingStrategyEnum::Mean => PoolingStrategy::Mean,
            PoolingStrategyEnum::Cls => PoolingStrategy::Cls,
        }
    }
}

pub struct ResolvedModel {
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub torch_dtype: Option<ComputeDTypeHint>,
}

impl ModelConfig {
    #[inline]
    pub fn ffn_size(&self) -> usize {
        self.intermediate_size.unwrap_or(self.hidden_size * 4)
    }
}

impl<'de> serde::Deserialize<'de> for ModelConfig {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(de)?;
        fn u64_or<E: serde::de::Error>(
            v: &serde_json::Value,
            primary: &str,
            alt: &str,
        ) -> std::result::Result<u64, E> {
            v.get(primary)
                .or_else(|| v.get(alt))
                .and_then(|v| v.as_u64())
                .ok_or_else(|| E::custom(format!("missing field `{}` or `{}`", primary, alt)))
        }
        Ok(ModelConfig {
            hidden_size: u64_or(&v, "hidden_size", "n_embd")? as usize,
            num_hidden_layers: u64_or(&v, "num_hidden_layers", "n_layer")? as usize,
            intermediate_size: v
                .get("intermediate_size")
                .or_else(|| v.get("n_inner"))
                .or_else(|| v.get("inner_dim"))
                .or_else(|| v.get("dim_feedforward"))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            max_position_embeddings: u64_or(&v, "max_position_embeddings", "n_positions")? as usize,
            num_attention_heads: u64_or(&v, "num_attention_heads", "n_head")? as usize,
            torch_dtype: v
                .get("torch_dtype")
                .and_then(|v| v.as_str())
                .and_then(ComputeDTypeHint::from_torch_dtype),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerCfg {
    #[serde(default)]
    pub model_max_length: Option<serde_json::Value>,
}

impl TokenizerCfg {
    pub fn effective_max_length(&self) -> Option<usize> {
        let v = self.model_max_length.as_ref()?;
        if let Some(u) = v.as_u64() {
            return (u <= i64::MAX as u64).then_some(u as usize);
        }
        if let Some(f) = v.as_f64()
            && f.is_finite()
            && f >= 1.0
            && f <= i64::MAX as f64
        {
            return Some(f as usize);
        }
        None
    }
}

const SAFETENSORS_SEARCH: &[&str] = &["", "safetensors"];
const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer.json", "onnx/tokenizer.json"];

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader).with_context(|| format!("parsing {}", path.display()))
}

#[derive(Debug, Clone, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: Option<bool>,
    #[serde(default)]
    pooling_mode_mean_tokens: Option<bool>,
    #[serde(default)]
    pooling_mode_max_tokens: Option<bool>,
    #[serde(default)]
    pooling_mode_lasttoken: Option<bool>,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: Option<bool>,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct StConfig {
    #[serde(default)]
    prompts: Option<StPrompts>,
}

#[derive(Debug, Clone, Deserialize)]
struct StPrompts {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    passage: Option<String>,
}

fn read_pooling_from_metadata(dir: &Path) -> PoolingStrategyEnum {
    let pooling_path = dir.join("1_Pooling").join("config.json");
    let Ok(cfg) = read_json::<PoolingConfig>(&pooling_path) else {
        return PoolingStrategyEnum::Mean;
    };

    if cfg.pooling_mode_cls_token == Some(true) {
        return PoolingStrategyEnum::Cls;
    }
    if cfg.pooling_mode_mean_tokens == Some(true) {
        return PoolingStrategyEnum::Mean;
    }

    let unsupported = cfg.pooling_mode_max_tokens == Some(true)
        || cfg.pooling_mode_lasttoken == Some(true)
        || cfg.pooling_mode_mean_sqrt_len_tokens == Some(true)
        || cfg.pooling_mode_weightedmean_tokens == Some(true);
    if unsupported {
        tracing::warn!(
            path = %pooling_path.display(),
            "model declares an unsupported pooling mode \
             (max / lasttoken / weightedmean / mean_sqrt_len); falling back to Mean",
        );
    }
    PoolingStrategyEnum::Mean
}

fn read_prefixes_from_metadata(dir: &Path) -> (Option<String>, Option<String>) {
    let st_path = dir.join("config_sentence_transformers.json");
    let Ok(cfg) = read_json::<StConfig>(&st_path) else {
        return (None, None);
    };
    match cfg.prompts {
        Some(p) => (p.query, p.passage),
        None => (None, None),
    }
}

fn guess_prefixes_from_model_id(model_id: &str) -> (Option<String>, Option<String>) {
    let name = model_id.to_ascii_lowercase();

    if name.contains("e5") {
        return (Some("query: ".into()), Some("passage: ".into()));
    }
    if name.contains("bge") && name.contains("en") {
        return (
            Some("Represent this sentence for searching relevant passages: ".into()),
            None,
        );
    }
    if name.contains("mxbai") || name.contains("mixedbread") {
        return (
            Some("Represent this sentence for searching relevant passages: ".into()),
            None,
        );
    }
    if name.contains("nomic-embed") {
        return (
            Some("search_query: ".into()),
            Some("search_document: ".into()),
        );
    }
    if name.contains("instructor") {
        return (
            Some("Represent the question for retrieving supporting documents: ".into()),
            Some("Represent the document for retrieval: ".into()),
        );
    }

    (None, None)
}

pub fn resolve_profile(
    model_id: &str,
    query_prefix_override: Option<&str>,
    passage_prefix_override: Option<&str>,
) -> Result<ModelProfile> {
    let files = resolve_model_files(model_id)?;
    let cfg: ModelConfig = read_json(&files.config_path)?;

    let tok_cfg_limit: usize = files
        .tokenizer_config_path
        .as_deref()
        .and_then(|p| read_json::<TokenizerCfg>(p).ok())
        .and_then(|t| t.effective_max_length())
        .unwrap_or(usize::MAX);

    let max_length = cfg.max_position_embeddings.min(tok_cfg_limit);

    let source = if tok_cfg_limit < cfg.max_position_embeddings {
        "tokenizer_config"
    } else {
        "config.max_position_embeddings"
    };
    tracing::info!(
        max_length,
        source,
        config_limit = cfg.max_position_embeddings,
        tok_cfg_limit,
        "resolved max_length",
    );

    let metadata_dir = files.config_path.parent().unwrap_or(Path::new("."));
    let pooling = read_pooling_from_metadata(metadata_dir);

    let (mut auto_query, mut auto_passage) = read_prefixes_from_metadata(metadata_dir);

    if auto_query.is_none() && auto_passage.is_none() {
        let (guess_q, guess_p) = guess_prefixes_from_model_id(model_id);
        if guess_q.is_some() || guess_p.is_some() {
            tracing::info!(model = model_id, "applied heuristic prefix fallback");
            auto_query = guess_q;
            auto_passage = guess_p;
        }
    }

    let query_prefix = match query_prefix_override {
        Some(p) => {
            tracing::info!(prefix = p, "applying CLI query_prefix override");
            Some(p.to_owned())
        }
        None => auto_query,
    };

    let passage_prefix = match passage_prefix_override {
        Some(p) => {
            tracing::info!(prefix = p, "applying CLI passage_prefix override");
            Some(p.to_owned())
        }
        None => auto_passage,
    };

    Ok(ModelProfile {
        model_id: model_id.to_string(),
        weights_path: files.weights_path,
        tokenizer_path: files.tokenizer_path,
        config_path: files.config_path,
        dim: 0,
        max_length,
        pooling,
        query_prefix,
        passage_prefix,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        intermediate_size: cfg.ffn_size(),
        num_attention_heads: cfg.num_attention_heads,
        compute_dtype: cfg.torch_dtype,
    })
}

pub fn resolve_model_files(model_id: &str) -> Result<ResolvedModel> {
    let p = Path::new(model_id);
    if p.exists() {
        resolve_local(p)
    } else {
        resolve_hf(model_id)
    }
}

fn find_safetensors_in(dir: &Path) -> Result<PathBuf> {
    for sub in SAFETENSORS_SEARCH {
        let scan = if sub.is_empty() {
            dir.to_path_buf()
        } else {
            dir.join(sub)
        };
        if !scan.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&scan)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == "model.safetensors" {
                tracing::info!(path = %path.display(), "selected safetensors weights");
                return Ok(path);
            }
        }
    }
    anyhow::bail!("no model.safetensors found in {}", dir.display())
}

fn resolve_local(p: &Path) -> Result<ResolvedModel> {
    let dir = if p.is_dir() {
        p
    } else {
        anyhow::bail!("{} is not a directory", p.display());
    };

    let weights_path = find_safetensors_in(dir)?;
    let tokenizer_path = find_tokenizer_in(dir)?;

    let config_path = dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!(
            "missing config.json in {} (the HF clone step should have placed it there)",
            dir.display()
        );
    }

    let tok_cfg = dir.join("tokenizer_config.json");
    let tokenizer_config_path = if tok_cfg.exists() {
        Some(tok_cfg)
    } else {
        None
    };

    tracing::info!(
        weights = %weights_path.display(),
        tokenizer = %tokenizer_path.display(),
        config = %config_path.display(),
        "using local model files"
    );

    Ok(ResolvedModel {
        weights_path,
        tokenizer_path,
        config_path,
        tokenizer_config_path,
    })
}

fn find_tokenizer_in(dir: &Path) -> Result<PathBuf> {
    for c in TOKENIZER_CANDIDATES {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no tokenizer.json found in {} (download it from the model's HF repo, \
         e.g. https://huggingface.co/<owner>/<name>/resolve/main/tokenizer.json)",
        dir.display()
    )
}

fn split_model_id(model_id: &str) -> Result<(&str, &str)> {
    let mut parts = model_id.splitn(2, '/');
    match (parts.next(), parts.next()) {
        (Some(o), Some(n)) if !o.is_empty() && !n.is_empty() => Ok((o, n)),
        _ => anyhow::bail!(
            "invalid model_id: expected 'owner/name', got '{}'",
            model_id
        ),
    }
}

/// Whether `model_id`'s files are already present in the local HF cache, so it
/// can be loaded without a network download.
pub fn is_model_cached(model_id: &str) -> bool {
    zrag_common::paths::models_dir().is_ok_and(|cache_dir| {
        hf_hub::Cache::new(cache_dir)
            .model(model_id.to_string())
            .get("config.json")
            .is_some()
    })
}

/// Whether a file in a HuggingFace repo is needed to load and run the local
/// embedding engine. Everything else — `.bin`/`.onnx` weights, README assets,
/// `flax/`/`tf/`/`pytorch` export dirs, etc. — is skipped to avoid downloading
/// the entire repo: a typical embed model ships several GiB of redundant
/// weight formats alongside the single `*.safetensors` the loader actually
/// uses.
fn is_needed_hf_file(rfilename: &str) -> bool {
    // Weights: any `*.safetensors` (single or sharded) plus the shard index.
    if rfilename.ends_with(".safetensors") || rfilename.ends_with(".safetensors.index.json") {
        return true;
    }
    // Configs and tokenizer files consumed by `resolve_local` / the loader.
    // `tokenizer.json` is self-contained (embeds its vocab), so `vocab.txt` /
    // `special_tokens_map.json` are intentionally not pulled.
    matches!(
        rfilename,
        "config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "config_sentence_transformers.json"
            | "onnx/tokenizer.json"
            | "1_Pooling/config.json"
    )
}

fn resolve_hf(model_id: &str) -> Result<ResolvedModel> {
    // Validate the "owner/name" shape up front for a clear early error.
    split_model_id(model_id)?;

    let cache_dir = zrag_common::paths::models_dir()?;
    std::fs::create_dir_all(&cache_dir)?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.clone())
        .build()
        .context("building HF API client")?;
    let repo = api.model(model_id.to_string());

    // List the repo and fetch every file into the managed cache. hf-hub skips
    // files already present for the current revision, so repeat runs are cheap
    // and incremental. A network failure here is non-fatal: fall through to the
    // cache lookup so an already-downloaded model still resolves offline.
    match repo.info() {
        Ok(info) => {
            tracing::info!(model = model_id, sha = %info.sha, "syncing HF repo");
            let mut fetched = 0usize;
            let mut skipped = 0usize;
            for sibling in &info.siblings {
                if !is_needed_hf_file(&sibling.rfilename) {
                    skipped += 1;
                    continue;
                }
                repo.get(&sibling.rfilename)
                    .with_context(|| format!("downloading {} for {model_id}", sibling.rfilename))?;
                fetched += 1;
            }
            tracing::info!(model = model_id, fetched, skipped, "synced only needed HF files");
        }
        Err(e) => tracing::debug!(error = %e, "HF repo info failed (offline?); using cache"),
    }

    // Resolve the snapshot directory from the local cache (no network).
    let config_path = hf_hub::Cache::new(cache_dir)
        .model(model_id.to_string())
        .get("config.json")
        .ok_or_else(|| {
            anyhow::anyhow!("model '{model_id}' is not cached and could not be downloaded")
        })?;
    let snapshot_dir = config_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("no parent dir for {}", config_path.display()))?;

    resolve_local(snapshot_dir)
}
