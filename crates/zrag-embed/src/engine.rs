use std::borrow::Cow;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use arrow::array::{FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::Config as BertConfig;
use candle_transformers::models::jina_bert::Config as JinaConfig;
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use crate::bert::BertModel;
use crate::jina_bert::JinaBertModel;

use crate::batch::{BATCH_BUCKETS, BATCH_CEILING, SEQ_BUCKETS, next_bucket};
use crate::model_registry::{
    ComputeDTypeHint, ModelProfile, PoolingStrategyEnum, WeightsFormat, read_json, resolve_profile,
};
use crate::qlayers::{Vb, dense_varbuilder_from_gguf, warn_on_slow_gguf_dtypes};
use candle_transformers::quantized_var_builder as qvb;
use crate::pooling::{PoolingStrategy, pool_on_device};
use crate::tokenizer::{Tokenized, Tokenizer};
use zrag_hw::{Hardware, candle_device, probe};

#[derive(Debug, Clone)]
pub struct Pooled {
    pub data: Vec<f32>,
    pub dim: usize,
    pub batch: usize,
}

impl Pooled {
    pub fn empty(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            batch: 0,
        }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks(self.dim)
    }

    pub fn into_float32_array(self) -> Float32Array {
        Float32Array::from(self.data)
    }

    pub fn into_fixed_size_list(self) -> FixedSizeListArray {
        FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            self.dim as i32,
            Arc::new(Float32Array::from(self.data)),
            None,
        )
    }
}

pub fn apply_prefix<'a>(text: &'a str, prefix: Option<&str>) -> Cow<'a, str> {
    match prefix {
        Some(p) => Cow::Owned(format!("{p}{text}")),
        None => Cow::Borrowed(text),
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadOverrides<'a> {
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
    /// Raw precision string: "f16", "f32", "bf16", "q4_k_m", etc. Flows
    /// through IPC untouched and is resolved into a dense DType or GGUF
    /// quant selector in [`EmbedEngine::load_with`].
    pub model_dtype: Option<&'a str>,
}

pub fn parse_model_dtype(raw: &str) -> Option<DType> {
    match raw.to_ascii_lowercase().as_str() {
        "f16" | "float16" | "half" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f32" | "float32" | "float" => Some(DType::F32),
        _ => None,
    }
}

/// A user-requested precision: either a dense candle DType or a GGUF quant
/// selector string (e.g. "q4_k_m") matched against variant filenames.
pub enum RequestedPrecision {
    Dense(DType),
    /// GGUF quant selector, e.g. "q4_k_m" — matched against variant filenames.
    Quant(String),
}

/// Parse a raw precision string into either a dense [`DType`] or a GGUF
/// quant selector. Returns `None` for unrecognized values (caller falls
/// back to profile/defaults).
pub fn parse_requested_precision(raw: &str) -> Option<RequestedPrecision> {
    if let Some(dt) = parse_model_dtype(raw) {
        return Some(RequestedPrecision::Dense(dt));
    }
    let l = raw.to_ascii_lowercase();
    (l.starts_with('q') || l.starts_with("iq")).then_some(RequestedPrecision::Quant(l))
}

struct Scratch {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    valid_counts: Vec<usize>,
    pooled_flat: Vec<f32>,
}

impl Scratch {
    fn with_capacity(max_batch: usize, max_seq: usize, dim: usize) -> Self {
        let total = max_batch * max_seq;
        Self {
            input_ids: Vec::with_capacity(total),
            attention_mask: Vec::with_capacity(total),
            valid_counts: Vec::with_capacity(max_batch),
            pooled_flat: Vec::with_capacity(max_batch * dim),
        }
    }

    fn prepare(&mut self, batch: usize, seq: usize) {
        let total = batch * seq;
        self.input_ids.clear();
        self.input_ids.resize(total, 0);
        self.attention_mask.clear();
        self.attention_mask.resize(total, 0);
        self.valid_counts.clear();
        self.valid_counts.resize(batch, 0);
    }
}

enum Model {
    Bert(BertModel),
    Jina(JinaBertModel),
}

impl Model {
    fn forward(
        &self,
        ids: &Tensor,
        token_type_ids: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Bert(model) => model.forward(ids, token_type_ids, mask),
            Self::Jina(model) => model.forward(ids, token_type_ids, mask),
        }
    }
}

struct State {
    model: Model,
    device: candle_core::Device,
    scratch: Scratch,
}

#[derive(Debug, Deserialize)]
struct PositionEmbeddingPeek {
    #[serde(default)]
    position_embedding_type: Option<String>,
}

/// Immutable shape/pooling config the worker needs per request. Captured once
/// at load (post-warmup) and owned by the worker thread.
#[derive(Clone, Copy)]
struct WorkerCfg {
    dim: usize,
    max_length: usize,
    pooling: PoolingStrategyEnum,
}

/// One embedding job: raw texts for the worker to tokenize + forward.
/// Moving tokenization onto the worker thread eliminates all synchronous
/// CPU work from the tokio reactor — the async caller just sends Strings
/// and awaits the one-shot reply.
struct EmbedRequest {
    texts: Arc<[String]>,
    reply: oneshot::Sender<Result<Pooled>>,
}

#[inline]
fn dtype_from_hint(hint: ComputeDTypeHint, device: &zrag_hw::Device) -> DType {
    match (hint, device) {
        (ComputeDTypeHint::F32, _) => DType::F32,
        (ComputeDTypeHint::F16, _) | (ComputeDTypeHint::BF16, zrag_hw::Device::Metal) => DType::F16,
        (ComputeDTypeHint::BF16, _) => DType::BF16,
    }
}

/// Build the embedding model from `profile`.
///
/// `quant` selects GGUF behaviour for the Jina path (which owns all its layers
/// and so can run native-quantized):
///   * `true`  — keep Q4_0…Q8K weights quantized in memory (fused `vec_dot`),
///   * `false` — dequantize to F32 (the precision fallback after a NaN warmup).
///
/// BERT always runs dense: it delegates its encoder to upstream candle's
/// `BertEncoder` (dense-only), so a GGUF checkpoint is dequantized to F32.
/// For safetensors `quant` is irrelevant — `dtype` selects the matmul precision.
fn load_model(
    profile: &ModelProfile,
    dtype: DType,
    device: &candle_core::Device,
    quant: bool,
) -> Result<Model> {
    let is_jina = read_json::<PositionEmbeddingPeek>(&profile.config_path)?
        .position_embedding_type
        .as_deref()
        == Some("alibi");

    if is_jina {
        let vb = match (profile.format, quant) {
            (WeightsFormat::Safetensors, _) => Vb::Dense(unsafe {
                VarBuilder::from_mmaped_safetensors(&[&profile.weights_path], dtype, device)?
            }),
            (WeightsFormat::Gguf, true) => {
                warn_on_slow_gguf_dtypes(&profile.weights_path, device);
                Vb::Quant(qvb::VarBuilder::from_gguf(&profile.weights_path, device)?)
            }
            (WeightsFormat::Gguf, false) => {
                Vb::Dense(dense_varbuilder_from_gguf(&profile.weights_path, device)?)
            }
        };
        let config: JinaConfig = read_json(&profile.config_path)?;
        Ok(Model::Jina(JinaBertModel::load(vb, &config)?))
    } else {
        let vb = match profile.format {
            WeightsFormat::Safetensors => unsafe {
                VarBuilder::from_mmaped_safetensors(&[&profile.weights_path], dtype, device)?
            },
            WeightsFormat::Gguf => {
                // Upstream BertEncoder is dense-only, so a GGUF checkpoint is
                // dequantized to F32: no fused quant kernels / RAM savings here.
                tracing::warn!(
                    "GGUF BERT dequantized to F32 (upstream BertEncoder is dense-only); \
                     no fused quant kernels or RAM savings for this architecture"
                );
                dense_varbuilder_from_gguf(&profile.weights_path, device)?
            }
        };
        let config: BertConfig = read_json(&profile.config_path)?;
        Ok(Model::Bert(BertModel::load(vb, &config)?))
    }
}

fn warmup_model(
    model: &Model,
    tokenizer: &Tokenizer,
    device: &candle_core::Device,
    profile: &mut ModelProfile,
    dtype: DType,
    hw: &Hardware,
) -> Result<()> {
    let enc = tokenizer.encode("a")?;
    if enc.ids.is_empty() {
        anyhow::bail!("warmup produced no tokens");
    }
    let ids = Tensor::from_slice(&enc.ids, (1, enc.ids.len()), device)?;
    let token_type_ids = Tensor::zeros_like(&ids)?;
    let mask = Tensor::ones_like(&ids)?;
    let out = model.forward(&ids, &token_type_ids, Some(&mask))?;
    if profile.dim == 0 {
        profile.dim = out
            .dims()
            .get(2)
            .copied()
            .ok_or_else(|| anyhow!("warmup output missing embedding dim"))?;
        tracing::info!(dim = profile.dim, "probed embedding dim");
    }
    let flat = out
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if flat.iter().any(|v| v.is_nan()) {
        anyhow::bail!(
            "model produced NaN at load (dtype {dtype:?}, device {:?}) — \
             this precision is unsupported for this model/device",
            hw.device
        );
    }
    Ok(())
}

pub struct EmbedEngine {
    /// Hands jobs to the tokenizer thread of the embed pipeline (a CPU thread
    /// that runs `encode_batch` and forwards encoded batches to the GPU thread
    /// over an internal channel). The reactor never blocks on tokenization or
    /// GPU work.
    tx: mpsc::UnboundedSender<EmbedRequest>,
    /// Clone source for [`Self::device`]. The model lives on the worker; this
    /// is a cheap, uncontended handle used only to build the GPU rerank scorer
    /// on the search path.
    device: Mutex<candle_core::Device>,
    /// Shared with the worker thread. Kept on the engine for lightweight,
    /// non-GPU operations: `count_tokens`, `truncates`, BPT estimation.
    tokenizer: Arc<Tokenizer>,
    profile: ModelProfile,
    hardware: Arc<Hardware>,
    optimal_batch: usize,
}

#[inline]
fn dtype_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F16 | DType::BF16 => 2,
        _ => 4,
    }
}

fn run_dummy_forward(
    model: &Model,
    device: &candle_core::Device,
    batch: usize,
    seq: usize,
    strategy: &PoolingStrategy,
) {
    let shape = (batch, seq);
    let Ok(ids) = Tensor::zeros(shape, DType::U32, device) else {
        return;
    };
    let Ok(zeros) = Tensor::zeros(shape, DType::U32, device) else {
        return;
    };
    let Ok(mask) = Tensor::ones(shape, DType::U32, device) else {
        return;
    };
    if let Ok(out) = model.forward(&ids, &zeros, Some(&mask)) {
        let _ = pool_on_device(&out, &mask, strategy, batch);
    }
}

fn time_passes(
    model: &Model,
    device: &candle_core::Device,
    batch: usize,
    seq: usize,
    n: usize,
    strategy: &PoolingStrategy,
) -> f64 {
    let mut total = std::time::Duration::ZERO;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        run_dummy_forward(model, device, batch, seq, strategy);
        total += t0.elapsed();
    }
    total.as_secs_f64() / n.max(1) as f64
}

/// Empirically select the optimal batch size for this model/device, then
/// pre-compile backend shaders/kernels at every (optimal_batch, seq) bucket.
///
/// Behaviour per device:
/// - Metal: times `candidate` vs `candidate/2` and picks the faster one.
///   Memory capacity says `candidate` is safe, but unified-memory bandwidth can
///   make a larger batch slower (observed: batch=32 → 4.4 chunks/s vs batch=16
///   → 7.5 chunks/s on Apple Silicon with jina F16 at seq=512).
/// - CUDA: uses `candidate` directly — cuBLAS GEMM scales well with batch and
///   the memory-capacity formula is accurate for discrete GPU VRAM.
/// - CPU: uses `candidate` directly — no JIT compilation, formula is accurate.
fn calibrate_and_prewarm(
    model: &Model,
    device: &candle_core::Device,
    candidate: usize,
    seq_len: usize,
    hw_device: &zrag_hw::Device,
    strategy: &PoolingStrategy,
) -> usize {
    let optimal = if candidate > 1 {
        match hw_device {
            zrag_hw::Device::Metal => {
                let half = crate::batch::BATCH_BUCKETS
                    .iter()
                    .rev()
                    .copied()
                    .find(|&b| b <= candidate / 2)
                    .unwrap_or(1);
                let t_full = time_passes(model, device, candidate, seq_len, 2, strategy);
                let t_half = time_passes(model, device, half, seq_len, 2, strategy);
                let tp_full = candidate as f64 / t_full.max(f64::MIN_POSITIVE);
                let tp_half = half as f64 / t_half.max(f64::MIN_POSITIVE);
                if tp_full >= tp_half * 1.05 { candidate } else { half }
            }
            zrag_hw::Device::Cuda | zrag_hw::Device::Cpu => candidate,
        }
    } else {
        candidate
    };

    // Pre-compile backend kernels/shaders for every (optimal, seq) bucket.
    for &seq in crate::batch::SEQ_BUCKETS
        .iter()
        .filter(|&&s| s <= seq_len)
    {
        run_dummy_forward(model, device, optimal, seq, strategy);
    }
    optimal
}

impl EmbedEngine {
    pub fn load(model_id: &str) -> Result<Self> {
        let hw = Arc::new(probe());
        Self::load_with(model_id, hw, &LoadOverrides::default())
    }

    pub fn load_with_device(model_id: &str, hw: Arc<Hardware>) -> Result<Self> {
        Self::load_with(model_id, hw, &LoadOverrides::default())
    }

    pub fn load_with(model_id: &str, hw: Arc<Hardware>, opts: &LoadOverrides<'_>) -> Result<Self> {
        let mut profile = resolve_profile(model_id, opts.query_prefix, opts.passage_prefix)?;

        tracing::info!(path = %profile.weights_path.display(), "loading safetensors model");

        let device = candle_device(&hw);
        let requested = opts.model_dtype.and_then(|raw| {
            match parse_requested_precision(raw) {
                Some(p) => Some(p),
                None => {
                    tracing::warn!(raw, "unrecognized model dtype; falling back to default");
                    None
                }
            }
        });

        // Select the GGUF quant variant file when a quant is requested.
        if let Some(RequestedPrecision::Quant(ref v)) = requested
            && profile.format == WeightsFormat::Gguf
            && let Some(dir) = profile.weights_path.parent()
            && let Ok(path) = crate::model_registry::find_gguf_in(dir, Some(v))
        {
            profile.weights_path = path;
        }

        // Quant-first unless the user explicitly chose a dense dtype.
        let quant_first = !matches!(requested, Some(RequestedPrecision::Dense(_)));
        let requested_dtype: DType = match requested {
            Some(RequestedPrecision::Dense(dt)) => dt,
            _ => profile
                .compute_dtype
                .filter(|_| !matches!(hw.device, zrag_hw::Device::Cpu))
                .map(|hint| dtype_from_hint(hint, &hw.device))
                .unwrap_or(DType::F32),
        };
        let mut tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

        let seq_cap = crate::batch::attention_safe_seq_cap(&profile, &hw);
        if seq_cap < profile.max_length {
            tracing::info!(
                seq_cap,
                model_max = profile.max_length,
                device = ?hw.device,
                "capping embed seq length for memory safety"
            );
            profile.max_length = seq_cap;
        }

        // Override the shipped tokenizer truncation (sentence-transformers repos
        // often pin 128, far below the resolved context window) so encoding uses
        // the full resolved/memory-capped length.
        tokenizer.set_truncation(profile.max_length)?;

        // Warmup forward: validates the model is NaN-free for this dtype/device
        // and probes the embedding dim when the config didn't provide it. A
        // reduced-precision dtype that overflows (e.g. a candle mask bug) falls
        // back to F32 once instead of silently wasting an entire index.
        let mut model = load_model(&profile, requested_dtype, &device, quant_first)?;
        let mut final_dtype = requested_dtype;
        if let Err(e) = warmup_model(
            &model,
            &tokenizer,
            &device,
            &mut profile,
            requested_dtype,
            &hw,
        ) {
            if requested_dtype == DType::F32 {
                return Err(e);
            }
            tracing::warn!(
                error = %e,
                requested_dtype = ?requested_dtype,
                format = ?profile.format,
                "model warmup failed; retrying embed load with a dense F32 fallback"
            );
            profile.dim = 0;
            // `quant = false` dequantizes a GGUF Jina model to dense F32 (the
            // native-quant fallback) and, for safetensors, simply reloads at F32
            // (the reduced-precision/BF16 fallback) — one path covers both.
            model = load_model(&profile, DType::F32, &device, false)?;
            warmup_model(&model, &tokenizer, &device, &mut profile, DType::F32, &hw)?;
            final_dtype = DType::F32;
        }

        // Wrap after all &mut calls (set_truncation) are done — worker gets a
        // clone, engine keeps one for count_tokens / truncates / BPT checks.
        let tokenizer = Arc::new(tokenizer);

        tracing::info!(
            dim = profile.dim,
            max_len = profile.max_length,
            device = ?hw.device,
            model = %model_id,
            "embed engine ready"
        );

        // Calibrate: time `candidate` vs `candidate/2` forward passes on Metal to
        // find the empirically fastest batch size (memory capacity is not the only
        // constraint — bandwidth limits can make a larger batch slower). Then
        // pre-compile Metal shaders at every (optimal, seq) bucket used in production.
        let candidate = crate::batch::recommended_batch_size(
            &profile,
            &hw,
            dtype_bytes(final_dtype),
        );
        let pooling = PoolingStrategy::from(profile.pooling);
        let optimal_batch = calibrate_and_prewarm(
            &model,
            &device,
            candidate,
            profile.max_length,
            &hw.device,
            &pooling,
        );
        tracing::info!(candidate_batch = candidate, optimal_batch, device = ?hw.device, "calibrated embed batch size");

        let scratch = Scratch::with_capacity(BATCH_CEILING, profile.max_length, profile.dim);

        // Spawn the single embedding worker. It owns the model, device, and
        // scratch for its whole life; callers submit jobs over `tx` and await a
        // one-shot reply. This keeps GPU work off the tokio reactor (no
        // `block_in_place`) and removes the global model mutex — only this
        // thread ever runs a forward pass.
        let device_handle = device.clone();
        let cfg = WorkerCfg {
            dim: profile.dim,
            max_length: profile.max_length,
            pooling: profile.pooling,
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<EmbedRequest>();
        let worker_tokenizer = Arc::clone(&tokenizer);
        let mut state = State {
            model,
            device,
            scratch,
        };

        // --- Tokenization ↔ GPU pipeline (double-buffer) ---
        // The indexer submits the next batch before awaiting the current one
        // (see `indexer.rs`). A single worker still serves those two batches
        // serially: tokenize N → forward N → tokenize N+1 → forward N+1.
        // Splitting the worker into a *tokenizer* thread (CPU) and a *GPU*
        // thread lets the CPU tokenize batch N+1 while the GPU runs the forward
        // pass for batch N, so the phases overlap. Only the GPU thread ever
        // touches the model/device, preserving the single-GPU-writer invariant.
        // Shutdown cascades: engine `tx` drops → `rx` closes → tokenizer loop
        // ends → `enc_tx` drops → GPU `blocking_recv` returns None.
        let (enc_tx, mut enc_rx) = mpsc::unbounded_channel::<(EmbedRequest, Vec<Tokenized>)>();

        // GPU thread: owns model/device/scratch, runs the forward pass only.
        std::thread::Builder::new()
            .name("zrag-embed-gpu".into())
            .spawn(move || {
                while let Some((req, encs)) = enc_rx.blocking_recv() {
                    let enc_refs: Vec<&Tokenized> = encs.iter().collect();
                    let _ = req.reply.send(embed_on_state(&mut state, &enc_refs, &cfg));
                }
            })
            .map_err(|e| anyhow!("spawn embed gpu worker: {e}"))?;

        // Tokenizer thread: owns the request channel, tokenizes, hands off to
        // the GPU thread. Encode already parallelizes across rayon internally.
        std::thread::Builder::new()
            .name("zrag-embed-tok".into())
            .spawn(move || {
                while let Some(req) = rx.blocking_recv() {
                    let refs: Vec<&str> = req.texts.iter().map(String::as_str).collect();
                    let encs = match worker_tokenizer.encode_batch(refs) {
                        Ok(e) => e,
                        Err(e) => {
                            let _ = req.reply.send(Err(e));
                            continue;
                        }
                    };
                    if enc_tx.send((req, encs)).is_err() {
                        break; // GPU worker gone; nothing more to do.
                    }
                }
            })
            .map_err(|e| anyhow!("spawn embed tok worker: {e}"))?;

        Ok(Self {
            tx,
            device: Mutex::new(device_handle),
            tokenizer,
            profile,
            hardware: hw,
            optimal_batch,
        })
    }

    pub fn dim(&self) -> usize {
        self.profile.dim
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn hardware(&self) -> &Hardware {
        &self.hardware
    }

    pub fn device(&self) -> Result<candle_core::Device> {
        let device = self
            .device
            .lock()
            .map_err(|_| anyhow!("embed device lock poisoned"))?;
        Ok(device.clone())
    }

    pub fn recommended_batch_size(&self) -> usize {
        self.optimal_batch
    }

    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Tokenized>> {
        self.tokenizer.encode_batch(texts.to_vec())
    }

    /// Count tokens in a single text without retaining ids/mask buffers.
    #[inline]
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }

    /// True when the tokenizer truncates (token counts would be capped → unreliable bpt).
    #[inline]
    pub fn truncates(&self) -> bool {
        self.tokenizer.truncation_max_length().is_some()
    }

    /// Queue raw texts on the embedding worker and return immediately.
    /// The worker tokenizes and runs the forward pass on its own OS thread;
    /// the tokio reactor never blocks on tokenization or GPU work.
    /// Await the returned receiver to collect the pooled result.
    pub fn submit_texts(&self, texts: Arc<[String]>) -> Result<oneshot::Receiver<Result<Pooled>>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest { texts, reply })
            .map_err(|_| anyhow!("embed worker thread is gone"))?;
        Ok(rx)
    }

    /// Embed a batch of texts: send to worker, await result. The worker
    /// handles tokenization + forward pass. No synchronous work on the reactor.
    pub async fn embed_batch_pooled_async(&self, texts: &[&str]) -> Result<Pooled> {
        if texts.is_empty() {
            return Ok(Pooled::empty(self.dim()));
        }
        let owned: Arc<[String]> = texts.iter().map(|s| (*s).to_string()).collect();
        self.submit_texts(owned)?
            .await
            .map_err(|_| anyhow!("embed worker dropped without replying"))?
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let pooled = self.embed_batch_pooled_async(texts).await?;
        Ok(pooled.rows().map(<[f32]>::to_vec).collect())
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, self.profile.query_prefix.as_deref());
        let pooled = self.embed_batch_pooled_async(&[&*input]).await?;
        if pooled.data.is_empty() {
            anyhow::bail!("no embedding produced");
        }
        Ok(pooled.data)
    }

    pub async fn embed_passage_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, self.profile.passage_prefix.as_deref());
        let pooled = self.embed_batch_pooled_async(&[&*input]).await?;
        if pooled.data.is_empty() {
            anyhow::bail!("no embedding produced");
        }
        Ok(pooled.data)
    }
}

/// Run one embedding job against the worker-owned `State`. Numerically
/// identical to the previous in-mutex path — only the call site (a dedicated
/// thread instead of `block_in_place`) changed.
fn embed_on_state(state: &mut State, encs: &[&Tokenized], cfg: &WorkerCfg) -> Result<Pooled> {
    let real_batch = encs.len();
    if real_batch == 0 {
        return Ok(Pooled::empty(cfg.dim));
    }
    if real_batch > BATCH_CEILING {
        anyhow::bail!(
            "batch size {} exceeds BATCH_CEILING ({})",
            real_batch,
            BATCH_CEILING,
        );
    }

    let max_len = cfg.max_length;
    let real_seq = encs
        .iter()
        .map(|e| e.ids.len().min(max_len))
        .max()
        .unwrap_or(1)
        .max(1);
    let seq = next_bucket(SEQ_BUCKETS, real_seq, max_len);
    let batch = next_bucket(BATCH_BUCKETS, real_batch, BATCH_CEILING);

    let State {
        model,
        device,
        scratch,
    } = state;
    scratch.prepare(batch, seq);
    fill_scratch(scratch, encs, seq);

    run_and_pool(
        model,
        device,
        scratch,
        Shape {
            batch,
            seq,
            real_batch,
        },
        &PoolingStrategy::from(cfg.pooling),
    )?;

    Ok(Pooled {
        data: std::mem::take(&mut scratch.pooled_flat),
        dim: cfg.dim,
        batch: real_batch,
    })
}

fn fill_scratch(scratch: &mut Scratch, encs: &[&Tokenized], seq: usize) {
    for (i, tok) in encs.iter().enumerate() {
        let len = tok.ids.len().min(seq);
        scratch.valid_counts[i] = len;
        let base = i * seq;
        scratch.input_ids[base..base + len].copy_from_slice(&tok.ids[..len]);
        scratch.attention_mask[base..base + len].copy_from_slice(&tok.mask[..len]);
    }
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    batch: usize,
    seq: usize,
    real_batch: usize,
}

fn run_and_pool(
    model: &Model,
    device: &candle_core::Device,
    scratch: &mut Scratch,
    shape: Shape,
    strategy: &PoolingStrategy,
) -> Result<()> {
    let Shape {
        batch,
        seq,
        real_batch,
        ..
    } = shape;
    let total = batch * seq;

    let ids_src = scratch
        .input_ids
        .get(..total)
        .ok_or_else(|| anyhow!("input_ids shorter than batch*seq"))?;
    let mask_src = scratch
        .attention_mask
        .get(..total)
        .ok_or_else(|| anyhow!("attention_mask shorter than batch*seq"))?;

    let ids = Tensor::from_slice(ids_src, (batch, seq), device)?;
    let token_type_ids = Tensor::zeros_like(&ids)?;
    let mask = Tensor::from_slice(mask_src, (batch, seq), device)?;

    let output = model.forward(&ids, &token_type_ids, Some(&mask))?;
    scratch.pooled_flat = pool_on_device(&output, &mask, strategy, real_batch)?;

    Ok(())
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    /// Loads the model named by `ZRAG_TEST_MODEL`, or `None` (skip) when unset
    /// or unresolvable — clean CI has no weights, the daemon host does.
    fn test_engine() -> Option<EmbedEngine> {
        let model = std::env::var("ZRAG_TEST_MODEL").ok()?;
        match EmbedEngine::load(&model) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("skipping worker test: cannot load {model}: {e}");
                None
            }
        }
    }

    /// The embed math is moved verbatim into `embed_on_state`, so equality with
    /// the prior sync path holds by construction. What the worker rewrite can
    /// still break is the *plumbing*: this asserts the worker is deterministic
    /// and that many concurrent jobs all resolve to the same vector with no
    /// deadlock or cross-request scratch races.
    #[test]
    fn worker_embed_is_deterministic_and_concurrent() {
        // Load the engine *outside* any runtime — exactly as the daemon does
        // (hf-hub blocks internally on load, which panics inside a runtime).
        let Some(engine) = test_engine() else {
            return;
        };
        let engine = Arc::new(engine);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("build test runtime");

        rt.block_on(async {
            let query = "fn parse_config(path: &str) -> Result<Config>";

            let base = engine.embed_query_async(query).await.expect("embed base");
            assert_eq!(base.len(), engine.dim(), "embedding width must equal dim");
            assert!(
                base.iter().any(|v| *v != 0.0),
                "embedding must be non-trivial"
            );
            let norm = base.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embedding must be L2-normalized, got norm {norm}"
            );

            let again = engine.embed_query_async(query).await.expect("embed again");
            assert_eq!(base, again, "worker embeddings must be deterministic");

            let mut handles = Vec::with_capacity(16);
            for _ in 0..16 {
                let e = Arc::clone(&engine);
                let q = query.to_string();
                handles.push(tokio::spawn(async move { e.embed_query_async(&q).await }));
            }
            for h in handles {
                let v = h.await.expect("join").expect("concurrent embed");
                assert_eq!(v, base, "concurrent embedding must equal the serial result");
            }
        });
    }
}
