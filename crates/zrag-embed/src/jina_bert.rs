//! JinaBERT qk-post-norm geGLU variant used by
//! `jinaai/jina-embeddings-v2-base-code`.
//!
//! This intentionally does not wrap candle's bundled JinaBERT implementation:
//! the code embedding model uses different tensor names and adds Q/K layer
//! normalization around self-attention.

use std::{cell::RefCell, collections::HashMap};

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Activation, Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear,
    linear_no_bias,
};
use candle_transformers::models::jina_bert::Config;

use crate::bert::extended_attention_mask;

#[derive(Debug)]
struct JinaEmbeddings {
    word: Embedding,
    token_type: Embedding,
    layer_norm: LayerNorm,
}

impl JinaEmbeddings {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            word: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?,
            token_type: embedding(
                cfg.type_vocab_size,
                cfg.hidden_size,
                vb.pp("token_type_embeddings"),
            )?,
            layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embeddings =
            (self.word.forward(input_ids)? + self.token_type.forward(token_type_ids)?)?;
        self.layer_norm.forward(&embeddings)
    }
}

#[derive(Debug)]
struct JinaSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    layer_norm_q: LayerNorm,
    layer_norm_k: LayerNorm,
    n_heads: usize,
    head_dim: usize,
}

impl JinaSelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        if cfg.num_attention_heads == 0 {
            candle_core::bail!("JinaBERT num_attention_heads must be non-zero");
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            candle_core::bail!(
                "JinaBERT hidden_size {} is not divisible by num_attention_heads {}",
                cfg.hidden_size,
                cfg.num_attention_heads,
            );
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            query: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("query"))?,
            key: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("key"))?,
            value: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("value"))?,
            layer_norm_q: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_q"))?,
            layer_norm_k: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_k"))?,
            n_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = xs.dims3()?;
        xs.reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let query = self.query.forward(xs)?.apply(&self.layer_norm_q)?;
        let key = self.key.forward(xs)?.apply(&self.layer_norm_k)?;
        let value = self.value.forward(xs)?;

        let query = self.transpose_for_scores(&query)?;
        let key = self.transpose_for_scores(&key)?;
        let value = self.transpose_for_scores(&value)?;

        let scores = query.matmul(&key.t()?)?;
        let scores = (scores / (self.head_dim as f64).sqrt())?;
        let scores = scores.broadcast_add(bias)?.broadcast_add(attention_mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.matmul(&value)?;
        context
            .transpose(1, 2)?
            .contiguous()?
            .flatten_from(D::Minus2)
    }
}

#[derive(Debug)]
struct JinaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl JinaSelfOutput {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            dense: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?,
            layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, context: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let output = self.dense.forward(context)?;
        self.layer_norm.forward(&(output + residual)?)
    }
}

#[derive(Debug)]
struct JinaAttention {
    self_attention: JinaSelfAttention,
    output: JinaSelfOutput,
}

impl JinaAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            self_attention: JinaSelfAttention::load(vb.pp("self"), cfg)?,
            output: JinaSelfOutput::load(vb.pp("output"), cfg)?,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let context = self.self_attention.forward(xs, bias, attention_mask)?;
        self.output.forward(&context, xs)
    }
}

#[derive(Debug)]
struct JinaMlp {
    up_gated_layer: Linear,
    down_layer: Linear,
    act: Activation,
    intermediate_size: usize,
}

impl JinaMlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            up_gated_layer: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size * 2,
                vb.pp("up_gated_layer"),
            )?,
            down_layer: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_layer"))?,
            act: Activation::Gelu,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = self.up_gated_layer.forward(xs)?;
        let up = gated.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate = gated.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let hidden = (up * gate.apply(&self.act)?)?;
        self.down_layer.forward(&hidden)
    }
}

#[derive(Debug)]
struct JinaLayer {
    attention: JinaAttention,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    mlp: JinaMlp,
}

impl JinaLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            attention: JinaAttention::load(vb.pp("attention"), cfg)?,
            layer_norm_1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_1"))?,
            layer_norm_2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_2"))?,
            mlp: JinaMlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(xs, bias, attention_mask)?;
        let hidden = self.layer_norm_1.forward(&(xs + attention_output)?)?;
        let mlp_output = self.mlp.forward(&hidden)?;
        self.layer_norm_2.forward(&(hidden + mlp_output)?)
    }
}

/// Sequence lengths above this are rebuilt per call instead of cached:
/// bias is `n_heads × seq × seq`, so large buckets can hold hundreds of
/// MiB or more on device. The hot bucketed embed path up to 1024 stays cached;
/// oversized requests keep today's build-per-call behavior.
const MAX_CACHED_SEQ: usize = 1024;

#[derive(Debug)]
struct AlibiCache {
    entries: RefCell<HashMap<(usize, DType), Tensor>>,
    /// Last bias above MAX_CACHED_SEQ. Single slot = the pre-map behavior:
    /// constant oversized seqs hit; only alternating oversized seqs rebuild.
    oversized: RefCell<Option<((usize, DType), Tensor)>>,
}

impl Default for AlibiCache {
    fn default() -> Self {
        Self {
            entries: RefCell::new(HashMap::with_capacity(crate::batch::SEQ_BUCKETS.len() + 1)),
            oversized: RefCell::new(None),
        }
    }
}

impl AlibiCache {
    fn get(&self, n_heads: usize, seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        // `n_heads` and `device` are fixed for the lifetime of the owning model,
        // so the cache key only needs the bucketed sequence length and dtype.
        let key = (seq_len, dtype);

        if seq_len <= MAX_CACHED_SEQ {
            if let Some(bias) = self.entries.borrow().get(&key) {
                return Ok(bias.clone());
            }

            let bias = alibi_bias(n_heads, seq_len, device)?.to_dtype(dtype)?;
            self.entries.borrow_mut().insert(key, bias.clone());
            return Ok(bias);
        }

        if let Some((cached_key, bias)) = self.oversized.borrow().as_ref()
            && *cached_key == key
        {
            return Ok(bias.clone());
        }

        let bias = alibi_bias(n_heads, seq_len, device)?.to_dtype(dtype)?;
        *self.oversized.borrow_mut() = Some((key, bias.clone()));
        Ok(bias)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    #[cfg(test)]
    fn has_oversized(&self) -> bool {
        self.oversized.borrow().is_some()
    }
}

#[derive(Debug)]
pub struct JinaBertModel {
    embeddings: JinaEmbeddings,
    layers: Vec<JinaLayer>,
    n_heads: usize,
    alibi_cache: AlibiCache,
}

impl JinaBertModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embeddings = JinaEmbeddings::load(vb.pp("embeddings"), cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        (0..cfg.num_hidden_layers).try_for_each(|index| -> Result<()> {
            layers.push(JinaLayer::load(
                vb.pp(format!("encoder.layer.{index}")),
                cfg,
            )?);
            Ok(())
        })?;
        Ok(Self {
            embeddings,
            layers,
            n_heads: cfg.num_attention_heads,
            alibi_cache: AlibiCache::default(),
        })
    }

    /// Signature identical to `crate::bert::BertModel::forward`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden = self.embeddings.forward(input_ids, token_type_ids)?;
        let dtype = hidden.dtype();
        let seq_len = hidden.dim(1)?;
        let mask_storage;
        let mask = match attention_mask {
            Some(mask) => mask,
            None => {
                mask_storage = input_ids.ones_like()?;
                &mask_storage
            }
        };
        let extended_mask = extended_attention_mask(mask, dtype)?;
        let bias = self
            .alibi_cache
            .get(self.n_heads, seq_len, dtype, hidden.device())?;
        self.layers.iter().try_fold(hidden, |state, layer| {
            layer.forward(&state, &bias, &extended_mask)
        })
    }
}

fn alibi_bias(n_heads: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
    if n_heads == 0 {
        candle_core::bail!("ALiBi requires at least one attention head");
    }
    let Ok(seq_len_i64) = i64::try_from(seq_len) else {
        candle_core::bail!("ALiBi sequence length {seq_len} exceeds i64::MAX");
    };
    let positions = Tensor::arange(0, seq_len_i64, &Device::Cpu)?.to_dtype(DType::F32)?;
    let distances = positions
        .reshape((1, seq_len))?
        .broadcast_sub(&positions.reshape((seq_len, 1))?)?
        .abs()?
        .broadcast_left(n_heads)?;

    let Some(n_heads2) = n_heads.checked_next_power_of_two() else {
        candle_core::bail!("ALiBi head count {n_heads} cannot be rounded to a power of two");
    };
    let mut slopes = Vec::with_capacity(n_heads2);
    slopes.extend((1..=n_heads2).map(|head| {
        let exponent = (head * 8) as f32 / n_heads2 as f32;
        -1f32 / exponent.exp2()
    }));

    let slopes = if n_heads2 == n_heads {
        slopes
    } else {
        let mut reordered = Vec::with_capacity(n_heads);
        reordered.extend(
            slopes
                .iter()
                .skip(1)
                .step_by(2)
                .chain(slopes.iter().step_by(2))
                .take(n_heads)
                .copied(),
        );
        reordered
    };
    let slopes = Tensor::new(slopes, &Device::Cpu)?.reshape((1, (), 1, 1))?;
    distances
        .to_dtype(DType::F32)?
        .broadcast_mul(&slopes)?
        .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn ensure_same_values(label: &str, expected: &[f32], actual: &Tensor) -> Result<()> {
        let actual = tensor_values(actual)?;
        if expected != actual {
            candle_core::bail!("{label} ALiBi bias differs from fresh bias");
        }
        Ok(())
    }

    #[test]
    fn alibi_cache_matches_fresh_bias_for_miss_and_hit() -> Result<()> {
        let device = Device::Cpu;
        let cache = AlibiCache::default();
        let n_heads = 12;

        for seq_len in [8, 13] {
            let fresh = alibi_bias(n_heads, seq_len, &device)?.to_dtype(DType::F32)?;
            let fresh_values = tensor_values(&fresh)?;
            let cached = cache.get(n_heads, seq_len, DType::F32, &device)?;
            ensure_same_values("cached", &fresh_values, &cached)?;

            let hit = cache.get(n_heads, seq_len, DType::F32, &device)?;
            ensure_same_values("cache hit", &fresh_values, &hit)?;
        }

        Ok(())
    }

    #[test]
    fn alibi_cache_keeps_alternating_buckets() -> Result<()> {
        let device = Device::Cpu;
        let cache = AlibiCache::default();
        let n_heads = 12;

        cache.get(n_heads, 64, DType::F32, &device)?;
        cache.get(n_heads, 128, DType::F32, &device)?;

        let fresh = alibi_bias(n_heads, 64, &device)?.to_dtype(DType::F32)?;
        let fresh_values = tensor_values(&fresh)?;
        let hit = cache.get(n_heads, 64, DType::F32, &device)?;
        ensure_same_values("alternating bucket cache hit", &fresh_values, &hit)?;

        if cache.len() != 2 {
            candle_core::bail!("expected 2 cached ALiBi buckets, got {}", cache.len());
        }

        Ok(())
    }

    #[test]
    fn alibi_cache_uses_single_oversized_slot() -> Result<()> {
        let device = Device::Cpu;
        let cache = AlibiCache::default();
        let n_heads = 1;
        let seq_len = MAX_CACHED_SEQ + 1;

        let fresh = alibi_bias(n_heads, seq_len, &device)?.to_dtype(DType::F32)?;
        let fresh_values = tensor_values(&fresh)?;
        let first = cache.get(n_heads, seq_len, DType::F32, &device)?;
        ensure_same_values("oversized first", &fresh_values, &first)?;

        if cache.len() != 0 {
            candle_core::bail!("expected oversized ALiBi bias to stay out of the bounded map");
        }
        if !cache.has_oversized() {
            candle_core::bail!("expected oversized ALiBi bias to use the overflow slot");
        }

        let second = cache.get(n_heads, seq_len, DType::F32, &device)?;
        ensure_same_values("oversized cache hit", &fresh_values, &second)?;

        if cache.len() != 0 {
            candle_core::bail!("expected oversized ALiBi bias to stay out of the bounded map");
        }

        Ok(())
    }
}
