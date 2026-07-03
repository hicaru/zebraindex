//! Dense/quantized weight-source adapter.
//!
//! Model code (e.g. [`crate::jina_bert`]) is written once against the [`Vb`]
//! abstraction and runs unchanged against either:
//!   * safetensors — a dense `candle_nn::VarBuilder`, or
//!   * GGUF        — candle's quantized `VarBuilder` (keeps Q4_0…Q8K weights
//!     quantized in memory so the fused `vec_dot` kernels in `k_quants` apply).
//!
//! The free constructors ([`linear`], [`linear_no_bias`], [`embedding`],
//! [`layer_norm`]) deliberately mirror `candle_nn`'s names and argument order, so
//! call sites need no edits when switching a model from dense to quantized.
//!
//! Architectures that delegate their layers to upstream candle (e.g. `BertEncoder`,
//! which only accepts the dense `VarBuilder`) use [`dense_varbuilder_from_gguf`] to
//! dequantize the whole GGUF into a dense builder instead — correct for every
//! Q-type at the cost of holding F32 weights in memory.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn as nn;
use candle_transformers::quantized_nn as qnn;
use candle_transformers::quantized_var_builder as qvb;

/// A weight source: dense (safetensors) or quantized (GGUF).
pub enum Vb<'a> {
    Dense(nn::VarBuilder<'a>),
    Quant(qvb::VarBuilder),
}

impl<'a> Vb<'a> {
    /// Push a path segment, mirroring `VarBuilder::pp`.
    pub fn pp<S: ToString>(&self, s: S) -> Vb<'a> {
        match self {
            Vb::Dense(v) => Vb::Dense(v.pp(s)),
            Vb::Quant(v) => Vb::Quant(v.pp(s)),
        }
    }

    /// The device tensors are materialized on.
    pub fn device(&self) -> &candle_core::Device {
        match self {
            Vb::Dense(v) => v.device(),
            Vb::Quant(v) => v.device(),
        }
    }
}

/// A linear layer backed by either a dense matmul or a quantized one.
///
/// Both variants implement `candle_core::Module`, so [`QLinear::forward`] dispatches
/// uniformly; GGUF `F32`/`F16`/`BF16` tensors cost nothing extra (candle keeps them
/// dense inside `QMatMul`), and the real Q-types get the fused kernels for free.
#[derive(Debug)]
pub enum QLinear {
    Dense(nn::Linear),
    Quant(qnn::Linear),
}

impl QLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Dense(l) => l.forward(x),
            QLinear::Quant(l) => l.forward(x),
        }
    }
}

/// An embedding table: dense, or quantized (dequantized once on load by candle).
#[derive(Debug)]
pub enum QEmbedding {
    Dense(nn::Embedding),
    Quant(qnn::Embedding),
}

impl QEmbedding {
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            QEmbedding::Dense(e) => e.forward(ids),
            QEmbedding::Quant(e) => e.forward(ids),
        }
    }
}

// --- constructors mirroring `candle_nn` (identical names + arg order) ---

/// `candle_nn::linear` equivalent, dispatching on the weight source.
pub fn linear(in_d: usize, out_d: usize, vb: Vb) -> Result<QLinear> {
    match vb {
        Vb::Dense(v) => Ok(QLinear::Dense(nn::linear(in_d, out_d, v)?)),
        Vb::Quant(v) => Ok(QLinear::Quant(qnn::linear(in_d, out_d, v)?)),
    }
}

/// `candle_nn::linear_no_bias` equivalent.
pub fn linear_no_bias(in_d: usize, out_d: usize, vb: Vb) -> Result<QLinear> {
    match vb {
        Vb::Dense(v) => Ok(QLinear::Dense(nn::linear_no_bias(in_d, out_d, v)?)),
        Vb::Quant(v) => Ok(QLinear::Quant(qnn::linear_no_bias(in_d, out_d, v)?)),
    }
}

/// `candle_nn::embedding` equivalent.
pub fn embedding(rows: usize, dim: usize, vb: Vb) -> Result<QEmbedding> {
    match vb {
        Vb::Dense(v) => Ok(QEmbedding::Dense(nn::embedding(rows, dim, v)?)),
        Vb::Quant(v) => Ok(QEmbedding::Quant(qnn::Embedding::new(rows, dim, v)?)),
    }
}

/// `candle_nn::layer_norm` equivalent. Returns a `candle_nn::LayerNorm` for both
/// paths (the quantized loader dequantizes the small norm weights on load), so
/// model structs need no enum field for layer norms.
pub fn layer_norm(size: usize, eps: f64, vb: Vb) -> Result<nn::LayerNorm> {
    match vb {
        Vb::Dense(v) => nn::layer_norm(size, eps, v),
        Vb::Quant(v) => qnn::layer_norm(size, eps, v),
    }
}

/// Dequantize every tensor in a GGUF file into a dense `candle_nn::VarBuilder`.
///
/// Used for architectures whose layers are owned by upstream candle (e.g.
/// [`crate::bert::BertModel`] delegates to `candle_transformers::models::bert`
/// `BertEncoder`, which only accepts the dense `VarBuilder`) and as the precision
/// fallback when a quantized build produces bad values at warmup. Every Q-type is
/// supported; weights land in memory as F32.
pub fn dense_varbuilder_from_gguf<P: AsRef<Path>>(
    path: P,
    device: &candle_core::Device,
) -> Result<nn::VarBuilder<'static>> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
    for name in content.tensor_infos.keys() {
        let qtensor = content.tensor(&mut file, name, device)?;
        tensors.insert(name.clone(), qtensor.dequantize(device)?);
    }
    tracing::info!(
        tensors = tensors.len(),
        gguf = %path.display(),
        "dequantized GGUF -> dense VarBuilder (F32) for an upstream/dense model",
    );
    Ok(nn::VarBuilder::from_tensors(tensors, DType::F32, device))
}
