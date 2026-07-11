use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use candle_core::quantized::{GgmlDType, gguf_file};
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
    #[serde(default)]
    pub format: WeightsFormat,
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

    /// Canonical lowercase precision id ("f32" | "f16" | "bf16").
    pub fn precision_id(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
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

/// On-disk weights container. Drives whether the engine loads a dense
/// `VarBuilder` (safetensors) or candle's quantized `VarBuilder` (GGUF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WeightsFormat {
    #[default]
    Safetensors,
    /// GGUF: quantized weights spanning F32/F16/BF16 and Q4_0…Q8K.
    Gguf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVariant {
    pub format: WeightsFormat,
    /// Canonical precision id: "f32" | "f16" | "bf16" | "q4_k_m" | "q8_0" | ...
    pub precision: String,
    /// Repo filename for GGUF variants (drives file selection at load time).
    pub file: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeChoice {
    pub label: String,
    pub cli_value: String,
    pub description: String,
    pub detail: String,
    pub device_tag: String,
    pub recommended: bool,
    pub warning: Option<String>,
}

pub struct ResolvedModel {
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub format: WeightsFormat,
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
        format: files.format,
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

/// Collect every weights file with the given extension across the search paths.
/// Shared by [`find_safetensors_in`] and [`find_gguf_in`] so the directory-scan
/// logic lives in exactly one place.
fn weights_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for sub in SAFETENSORS_SEARCH {
        let scan = if sub.is_empty() {
            dir.to_path_buf()
        } else {
            dir.join(sub)
        };
        if !scan.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&scan) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some(ext) {
                found.push(path);
            }
        }
    }
    found
}

fn find_safetensors_in(dir: &Path) -> Result<PathBuf> {
    for path in weights_with_ext(dir, "safetensors") {
        if path.file_name().and_then(|s| s.to_str()) == Some("model.safetensors") {
            tracing::info!(path = %path.display(), "selected safetensors weights");
            return Ok(path);
        }
    }
    anyhow::bail!("no model.safetensors found in {}", dir.display())
}

/// Locate a GGUF checkpoint, preferring a specific quant variant when given.
pub(crate) fn find_gguf_in(dir: &Path, variant: Option<&str>) -> Result<PathBuf> {
    let all = weights_with_ext(dir, "gguf");
    if let Some(v) = variant {
        if let Some(p) = all.iter().find(|p| {
            quant_from_filename(&p.file_name().unwrap_or_default().to_string_lossy())
                .is_some_and(|q| q == v)
        }) {
            tracing::info!(path = %p.display(), variant = v, "selected GGUF quant variant");
            return Ok(p.clone());
        }
        tracing::warn!(variant = v, "requested GGUF variant not found; using first available");
    }
    match all.into_iter().next() {
        Some(path) => {
            tracing::info!(path = %path.display(), "selected GGUF weights");
            Ok(path)
        }
        None => anyhow::bail!("no *.gguf found in {}", dir.display()),
    }
}

fn gguf_meta_u32(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<usize> {
    md.get(key).and_then(|v| v.to_u32().ok()).map(|n| n as usize)
}

fn gguf_meta_f64(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<f64> {
    md.get(key).and_then(|v| v.to_f32().ok()).map(|f| f as f64)
}

fn gguf_meta_str<'a>(
    md: &'a std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<&'a str> {
    md.get(key)
        .and_then(|v| v.to_string().ok())
        .map(std::string::String::as_str)
}

/// Derive a `config.json` from a GGUF checkpoint's metadata (BERT-family embedding
/// models that ship no config — common with llama.cpp conversions) and write it to a
/// stable temp path, so the rest of the pipeline — which reads `config.json` via the
/// various `Deserialize` impls — works unchanged.
///
/// Keys follow llama.cpp's BERT convention; both the bare and `bert.`-prefixed forms
/// are accepted. The JSON is shaped to satisfy [`ModelConfig`] and candle's
/// `BertConfig` / `JinaConfig` / `PositionEmbeddingPeek`.
fn gguf_metadata_config_path(weights_path: &Path) -> Result<PathBuf> {
    let mut file = std::fs::File::open(weights_path)
        .with_context(|| format!("opening GGUF {}", weights_path.display()))?;
    let content = gguf_file::Content::read(&mut file)?;
    let md = &content.metadata;

    let hidden_size = ["bert.embedding_length", "embedding_length"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .ok_or_else(|| anyhow!("GGUF has no bert.embedding_length"))?;
    let num_hidden_layers = ["bert.block_count", "block_count"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .ok_or_else(|| anyhow!("GGUF has no bert.block_count"))?;
    let num_attention_heads = ["bert.attention.head_count", "attention.head_count"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .ok_or_else(|| anyhow!("GGUF has no bert.attention.head_count"))?;
    let intermediate_size = ["bert.feed_forward_length", "feed_forward_length"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .unwrap_or(hidden_size * 4);
    let max_position_embeddings = [
        "bert.max_position_embeddings",
        "max_position_embeddings",
        "bert.context_length",
        "context_length",
    ]
    .into_iter()
    .find_map(|k| gguf_meta_u32(md, k))
    .ok_or_else(|| anyhow!("GGUF has no max_position/context length"))?;
    let vocab_size = ["bert.vocab_size", "vocab_size"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .unwrap_or(30522);
    let type_vocab_size = ["bert.type_vocab_size", "type_vocab_size"]
        .into_iter()
        .find_map(|k| gguf_meta_u32(md, k))
        .unwrap_or(2);
    let layer_norm_eps = ["bert.layer_norm_eps", "attention.layer_norm_eps", "layer_norm_eps"]
        .into_iter()
        .find_map(|k| gguf_meta_f64(md, k))
        .unwrap_or(1e-12);
    let position_embedding_type = ["bert.position_embedding_type", "position_embedding_type"]
        .into_iter()
        .find_map(|k| gguf_meta_str(md, k))
        .unwrap_or("absolute");

    let config = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "max_position_embeddings": max_position_embeddings,
        "type_vocab_size": type_vocab_size,
        "initializer_range": 0.02,
        "layer_norm_eps": layer_norm_eps,
        "pad_token_id": 0,
        "position_embedding_type": position_embedding_type,
    });

    let stem = weights_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let parent = weights_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("dir");
    let out = std::env::temp_dir()
        .join(format!("zrag-gguf-config-{parent}-{stem}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&config)?)?;
    tracing::info!(
        gguf = %weights_path.display(),
        config = %out.display(),
        "derived config.json from GGUF metadata (checkpoint shipped none)",
    );
    Ok(out)
}

fn resolve_local(p: &Path) -> Result<ResolvedModel> {
    let dir = if p.is_dir() {
        p
    } else {
        anyhow::bail!("{} is not a directory", p.display());
    };

    // Prefer an explicit GGUF checkpoint (quantized Q4_0…Q8K); fall back to the
    // dense safetensors file when no `.gguf` is present.
    let (weights_path, format) = match find_gguf_in(dir, None) {
        Ok(path) => (path, WeightsFormat::Gguf),
        Err(_) => (find_safetensors_in(dir)?, WeightsFormat::Safetensors),
    };
    let tokenizer_path = find_tokenizer_in(dir)?;

    let candidate = dir.join("config.json");
    let config_path = if candidate.exists() {
        candidate
    } else if format == WeightsFormat::Gguf {
        // llama.cpp GGUF conversions often ship no config.json; derive one from the
        // checkpoint's metadata so the rest of the pipeline loads unchanged.
        gguf_metadata_config_path(&weights_path)?
    } else {
        anyhow::bail!(
            "missing config.json in {} (the HF clone step should have placed it there)",
            dir.display()
        );
    };

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
        format,
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
        let cached = hf_hub::Cache::new(cache_dir.clone());
        // A safetensors model is cached once config.json lands; a config-less GGUF
        // checkpoint (common with llama.cpp conversions) counts once any *.gguf is
        // present in its snapshot dir.
        if cached
            .model(model_id.to_string())
            .get("config.json")
            .is_some()
        {
            return true;
        }
        // GGUF check: scan snapshot dirs for any *.gguf file.
        if let Some((org, name)) = model_id.split_once('/') {
            let snapshots = cache_dir
                .join(format!("models--{org}--{name}"))
                .join("snapshots");
            if let Ok(revs) = std::fs::read_dir(&snapshots) {
                for rev in revs.flatten() {
                    if let Ok(files) = std::fs::read_dir(rev.path()) {
                        for f in files.flatten() {
                            if f.path()
                                .extension()
                                .and_then(|s| s.to_str())
                                == Some("gguf")
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    })
}

// ── Weight-variant detection (precision-level) ───────────────────────────

/// Detect weight variants (precision-level) for `model_id`: cache-first
/// (offline-friendly, instant), network-second. Returns one entry per
/// distinct (format, precision) pair. Never empty — falls back to a
/// safetensors default so the dtype picker always has at least one row.
pub fn detect_weight_variants(model_id: &str) -> Vec<WeightVariant> {
    let mut out = Vec::new();
    let Ok(cache_dir) = zrag_common::paths::models_dir() else {
        return out;
    };

    // 1. Cache FIRST — offline-friendly and instant.
    collect_cached_variants(&cache_dir, model_id, &mut out);

    // 2. Network only when the cache told us nothing.
    if out.is_empty()
        && let Some(siblings) = hf_repo_siblings(&cache_dir, model_id)
    {
        let st_precision = safetensors_precision(&cache_dir, model_id);
        for name in &siblings {
            if name.ends_with(".safetensors") {
                push_unique(
                    &mut out,
                    WeightVariant {
                        format: WeightsFormat::Safetensors,
                        precision: st_precision.clone(),
                        file: None,
                        size_bytes: None,
                    },
                );
            } else if name.ends_with(".gguf") {
                push_unique(
                    &mut out,
                    WeightVariant {
                        format: WeightsFormat::Gguf,
                        precision: quant_from_filename(name).unwrap_or_else(|| "gguf".into()),
                        file: Some(name.clone()),
                        size_bytes: None,
                    },
                );
            }
        }
    }

    if out.is_empty() {
        out.push(WeightVariant {
            format: WeightsFormat::Safetensors,
            precision: "f32".into(),
            file: None,
            size_bytes: None,
        });
    }
    out
}

/// List sibling filenames for a HF repo via the sync API; `None` if unreachable.
fn hf_repo_siblings(cache_dir: &Path, model_id: &str) -> Option<Vec<String>> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .ok()?;
    let info = api.model(model_id.to_string()).info().ok()?;
    Some(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

/// `"model-Q4_K_M.gguf"` -> `"q4_k_m"`; `"foo.f16.gguf"` -> `"f16"`;
/// `"model.gguf"` -> `None`.
pub fn quant_from_filename(name: &str) -> Option<String> {
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)[-_.](q\d(?:_[a-z0-9]+)*|iq\d[_a-z0-9]*|f16|f32|bf16)\.gguf$",
        )
        .expect("gguf-quant regex is a compile-time constant")
    });
    RE.captures(name).map(|c| c[1].to_ascii_lowercase())
}

/// Dominant tensor dtype from the GGUF header (header-only read, ~KBs).
fn gguf_precision_from_header(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let content = gguf_file::Content::read(&mut file).ok()?;
    let mut counts: HashMap<GgmlDType, usize> = HashMap::new();
    for info in content.tensor_infos.values() {
        *counts.entry(info.ggml_dtype).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(dt, _)| normalize_ggml_dtype(dt))
}

/// Map a candle `GgmlDType` to a canonical precision id.
fn normalize_ggml_dtype(dt: GgmlDType) -> String {
    match dt {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
    }
    .to_string()
}

/// Deduplicate by (format, precision) — multiple shards share one entry.
fn push_unique(out: &mut Vec<WeightVariant>, v: WeightVariant) {
    if !out.iter().any(|e| e.format == v.format && e.precision == v.precision) {
        out.push(v);
    }
}

/// Scan snapshot dirs for `.gguf`/`.safetensors`, reading GGUF headers for
/// plain-named files and `config.json` `torch_dtype` for safetensors.
fn collect_cached_variants(cache_dir: &Path, model_id: &str, out: &mut Vec<WeightVariant>) {
    let Some((org, name)) = model_id.split_once('/') else {
        return;
    };
    let snapshots = cache_dir
        .join(format!("models--{org}--{name}"))
        .join("snapshots");
    let Ok(revs) = std::fs::read_dir(&snapshots) else {
        return;
    };
    let st_precision = cached_safetensors_precision(&snapshots);
    for rev in revs.flatten() {
        let Ok(files) = std::fs::read_dir(rev.path()) else {
            continue;
        };
        for f in files.flatten() {
            let fname = f.file_name();
            let fname = fname.to_string_lossy();
            if fname.ends_with(".safetensors") {
                push_unique(
                    out,
                    WeightVariant {
                        format: WeightsFormat::Safetensors,
                        precision: st_precision.clone(),
                        file: None,
                        size_bytes: f.metadata().ok().map(|m| m.len()),
                    },
                );
            } else if fname.ends_with(".gguf") {
                let precision = quant_from_filename(&fname)
                    .or_else(|| gguf_precision_from_header(&f.path()))
                    .unwrap_or_else(|| "gguf".into());
                push_unique(
                    out,
                    WeightVariant {
                        format: WeightsFormat::Gguf,
                        precision,
                        file: Some(fname.into_owned()),
                        size_bytes: f.metadata().ok().map(|m| m.len()),
                    },
                );
            }
        }
    }
}

/// Read `torch_dtype` from the first `config.json` in any snapshot revision.
fn cached_safetensors_precision(snapshots: &Path) -> String {
    let Ok(revs) = std::fs::read_dir(snapshots) else {
        return "f32".into();
    };
    for rev in revs.flatten() {
        let cfg = rev.path().join("config.json");
        if let Ok(text) = std::fs::read_to_string(&cfg)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(dt) = v.get("torch_dtype").and_then(|d| d.as_str())
            && let Some(hint) = ComputeDTypeHint::from_torch_dtype(dt)
        {
            return hint.precision_id().into();
        }
    }
    "f32".into()
}

/// Safetensors precision for a repo (network path): defaults to `"f16"` since
/// most HF embed models ship f16 and `config.json` is not yet downloaded.
fn safetensors_precision(_cache_dir: &Path, _model_id: &str) -> String {
    "f16".into()
}

/// Whether a file in a HuggingFace repo is needed to load and run the local
/// embedding engine. Everything else — `.bin`/`.onnx` weights, README assets,
/// `flax/`/`tf/`/`pytorch` export dirs, etc. — is skipped to avoid downloading
/// the entire repo: a typical embed model ships several GiB of redundant
/// weight formats alongside the single `*.safetensors` the loader actually
/// uses.
fn is_needed_hf_file(rfilename: &str) -> bool {
    // Weights: any `*.safetensors` (single or sharded) plus the shard index, and
    // single-file GGUF checkpoints (quantized Q4_0…Q8K).
    if rfilename.ends_with(".safetensors")
        || rfilename.ends_with(".safetensors.index.json")
        || rfilename.ends_with(".gguf")
    {
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

#[cfg(test)]
mod gguf_config_tests {
    use super::*;
    use candle_core::quantized::gguf_file::{self, Value};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device};

    #[test]
    fn derives_config_from_gguf_metadata() -> Result<()> {
        let dev = Device::Cpu;
        let w = candle_core::Tensor::arange(0u32, 64, &dev)?
            .to_dtype(DType::F32)?
            .reshape((8, 8))?;
        let qt = QTensor::quantize(&w, GgmlDType::F32)?;

        let vals = [
            Value::U32(768),
            Value::U32(12),
            Value::U32(12),
            Value::U32(3072),
            Value::U32(512),
            Value::U32(30522),
            Value::F32(1e-12_f32),
        ];
        let metadata: [(&str, &Value); 7] = [
            ("bert.embedding_length", &vals[0]),
            ("bert.block_count", &vals[1]),
            ("bert.attention.head_count", &vals[2]),
            ("bert.feed_forward_length", &vals[3]),
            ("bert.max_position_embeddings", &vals[4]),
            ("bert.vocab_size", &vals[5]),
            ("bert.layer_norm_eps", &vals[6]),
        ];

        let gguf = std::env::temp_dir().join("zrag-registry-cfg-test.gguf");
        {
            let mut f = std::fs::File::create(&gguf)?;
            gguf_file::write(&mut f, &metadata, &[("t.weight", &qt)])?;
        }

        let cfg_path = gguf_metadata_config_path(&gguf)?;
        let cfg: ModelConfig = read_json(&cfg_path)?;
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.max_position_embeddings, 512);
        assert_eq!(cfg.intermediate_size, Some(3072));

        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(&cfg_path);
        Ok(())
    }

    #[test]
    fn quant_from_filename_parses_quant_suffix() {
        assert_eq!(
            quant_from_filename("m-Q4_K_M.gguf"),
            Some("q4_k_m".into())
        );
        assert_eq!(quant_from_filename("m.f16.gguf"), Some("f16".into()));
        assert_eq!(quant_from_filename("m-BF16.gguf"), Some("bf16".into()));
        assert_eq!(
            quant_from_filename("m-IQ2_XXS.gguf"),
            Some("iq2_xxs".into())
        );
        assert_eq!(quant_from_filename("model.gguf"), None);
        assert_eq!(quant_from_filename("Q8_0.gguf"), None); // no separator prefix
    }

    #[test]
    fn normalize_ggml_dtype_covers_all_variants() {
        assert_eq!(normalize_ggml_dtype(GgmlDType::F32), "f32");
        assert_eq!(normalize_ggml_dtype(GgmlDType::Q4K), "q4_k");
        assert_eq!(normalize_ggml_dtype(GgmlDType::Q8_0), "q8_0");
        assert_eq!(normalize_ggml_dtype(GgmlDType::BF16), "bf16");
    }

    #[test]
    fn detect_weight_variants_never_empty() {
        // Even a bogus model_id returns at least one fallback variant.
        let variants = detect_weight_variants("nonexistent/model");
        assert!(!variants.is_empty());
        assert!(variants.iter().any(|v| v.precision == "f32"));
    }
}
