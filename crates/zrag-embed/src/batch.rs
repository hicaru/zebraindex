use std::sync::OnceLock;

use zrag_hw::{Device, Hardware};

use crate::model_registry::ModelProfile;

const F32: usize = 4;
const WEIGHT_OVERHEAD: usize = 2;
const ATTN_TENSORS: usize = 4;
const FFN_TENSORS: usize = 2;
const PIPELINE_LIVE: usize = 2;
/// Live (count, heads, seq, seq) buffers in one attention op chain (matmul →
/// /sqrt → +bias → +mask → softmax → matmul); candle is out-of-place.
const ATTN_LIVE_BUFFERS: usize = 6;

pub const BATCH_CEILING: usize = 64;

pub const TYPICAL_SEQ_LEN: usize = 512;

/// Sequence-length buckets the engine pads to. Backends may cache compiled
/// graphs per `(batch, seq)` shape, so we want a small finite set; 4096
/// bridges the gap between 2048 and the 8192 maximum so a 2049-token chunk
/// does not pad 6143 zeros through every layer.
pub const SEQ_BUCKETS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// Batch buckets the engine pads to. Same cache-shape rationale as
/// [`SEQ_BUCKETS`].
pub const BATCH_BUCKETS: &[usize] = &[1, 4, 8, 16, 32, 64];

/// Round `n` up to the next entry in `buckets`, clamped to `cap`.
///
/// Returns `cap` if `n` exceeds the largest bucket — callers are expected to
/// ensure `cap` is itself the model's hard limit, not an arbitrary cutoff.
#[inline]
pub fn next_bucket(buckets: &[usize], n: usize, cap: usize) -> usize {
    for &b in buckets {
        if b >= n {
            return b.min(cap);
        }
    }
    cap
}

/// Round `n` down to the previous sequence bucket.
#[inline]
pub fn prev_bucket(n: usize) -> usize {
    SEQ_BUCKETS
        .iter()
        .rev()
        .copied()
        .find(|&b| b <= n)
        .unwrap_or(TYPICAL_SEQ_LEN)
}

const DEFAULT_METAL_MEM_FRAC: (usize, usize) = (7, 10);
const DEFAULT_CUDA_MEM_FRAC: (usize, usize) = (6, 10);
const DEFAULT_CPU_MEM_FRAC: (usize, usize) = (5, 10);

// Metal throughput ceiling for seq length. JinaBERT/BERT attention is O(seq²),
// and candle-metal ALiBi kernels are not well-optimised beyond 512. Keep chunks
// at the same 512-token budget used by `recommended_batch_size`. Override with
// ZRAG_EMBED_SEQ_CAP when longer context is needed.
const DEFAULT_METAL_SEQ_CAP: usize = TYPICAL_SEQ_LEN;

/// Memory-capacity estimate for the maximum safe batch size. This is an upper
/// bound only — the engine calibrates empirically at load time and may choose a
/// smaller value if the larger batch is not faster on the current hardware.
pub fn recommended_batch_size(profile: &ModelProfile, hw: &Hardware, dtype_bytes: usize) -> usize {
    let effective_seq = (profile.max_length).min(TYPICAL_SEQ_LEN);
    let per_sample = effective_seq
        .saturating_mul(profile.num_hidden_layers)
        .saturating_mul(
            ATTN_TENSORS.saturating_mul(profile.hidden_size)
                + FFN_TENSORS.saturating_mul(profile.intermediate_size),
        )
        .saturating_mul(dtype_bytes)
        .saturating_mul(PIPELINE_LIVE)
        .max(1);

    let (usable_num, usable_den) = usable_fraction(&hw.device);

    let budget = hw.mem_avail as usize * usable_num / usable_den;

    let weight_bytes = std::fs::metadata(&profile.weights_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let weight_overhead = weight_bytes.saturating_mul(WEIGHT_OVERHEAD);

    let inference_budget = budget.saturating_sub(weight_overhead);

    let raw = (inference_budget / per_sample).max(1);
    let pow2 = prev_power_of_two(raw);
    pow2.min(BATCH_CEILING)
}

/// Largest sequence length whose single-row attention transient fits the
/// device memory budget. Chunks longer than this are split before embedding so
/// one long chunk cannot allocate a multi-GB `(heads, seq, seq)` score tensor
/// per op. `ZRAG_EMBED_SEQ_CAP` overrides; otherwise fully derived from the
/// hardware and model profile.
pub fn attention_safe_seq_cap(profile: &ModelProfile, hw: &Hardware) -> usize {
    if let Some(cap) = env_seq_cap() {
        return clamp_seq_cap(cap, profile.max_length);
    }

    let (usable_num, usable_den) = usable_fraction(&hw.device);
    let budget = (hw.mem_avail as usize).saturating_mul(usable_num) / usable_den;
    let weight_bytes = std::fs::metadata(&profile.weights_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let inference_budget = budget.saturating_sub(weight_bytes.saturating_mul(WEIGHT_OVERHEAD));
    let per_seq2 = ATTN_LIVE_BUFFERS
        .saturating_mul(profile.num_attention_heads.max(1))
        .saturating_mul(F32)
        .max(1);
    let cap = (inference_budget / per_seq2).max(1).isqrt();

    let mem_cap = clamp_seq_cap(cap, profile.max_length);

    // On Metal, clamp to DEFAULT_METAL_SEQ_CAP (512) so actual forward-pass seq
    // length matches the budget assumed by `recommended_batch_size`. Longer
    // sequences make attention O(seq²) slower and overflow the batch budget.
    // Override with ZRAG_EMBED_SEQ_CAP when longer context is intentional.
    if hw.device == Device::Metal {
        let floor = TYPICAL_SEQ_LEN.min(profile.max_length);
        mem_cap.min(DEFAULT_METAL_SEQ_CAP).max(floor)
    } else {
        mem_cap
    }
}

/// Clamp a derived sequence cap to `[min(TYPICAL_SEQ_LEN, max_length), max_length]`.
///
/// Flooring at `TYPICAL_SEQ_LEN.min(max_length)` keeps the floor from ever
/// exceeding the ceiling, so models whose resolved `max_length` is below
/// `TYPICAL_SEQ_LEN` collapse to their own limit instead of panicking on
/// `clamp(min > max)`.
#[inline]
fn clamp_seq_cap(cap: usize, max_length: usize) -> usize {
    let floor = TYPICAL_SEQ_LEN.min(max_length);
    prev_bucket(cap).clamp(floor, max_length)
}

static FRAC_OVERRIDE: OnceLock<Option<(usize, usize)>> = OnceLock::new();
static SEQ_CAP_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();

fn usable_fraction(device: &Device) -> (usize, usize) {
    let cached = FRAC_OVERRIDE.get_or_init(|| {
        let s = std::env::var("ZRAG_BATCH_MEM_FRAC").ok()?;
        let f = s.parse::<f64>().ok()?;
        (0.05..=0.95)
            .contains(&f)
            .then_some(((f * 100.0) as usize, 100))
    });
    if let Some(v) = cached {
        return *v;
    }
    match device {
        Device::Metal => DEFAULT_METAL_MEM_FRAC,
        Device::Cuda => DEFAULT_CUDA_MEM_FRAC,
        Device::Cpu => DEFAULT_CPU_MEM_FRAC,
    }
}

fn env_seq_cap() -> Option<usize> {
    *SEQ_CAP_OVERRIDE.get_or_init(|| parse_seq_cap(std::env::var("ZRAG_EMBED_SEQ_CAP").ok()))
}

fn parse_seq_cap(value: Option<String>) -> Option<usize> {
    value?.parse::<usize>().ok().filter(|&cap| cap > 0)
}

#[inline]
fn prev_power_of_two(n: usize) -> usize {
    debug_assert!(n > 0);
    1usize << (usize::BITS - 1 - n.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrag_hw::{CpuCaps, Device, Hardware};

    fn hw(device: Device, mem_avail_gib: u64) -> Hardware {
        Hardware {
            device,
            cpus: 8,
            mem_total: mem_avail_gib << 30,
            mem_avail: mem_avail_gib << 30,
            cpu: CpuCaps::default(),
        }
    }

    fn profile(
        hidden: usize,
        layers: usize,
        ffn: usize,
        heads: usize,
        seq: usize,
        weights: &str,
    ) -> ModelProfile {
        ModelProfile {
            model_id: "test".into(),
            weights_path: std::path::PathBuf::from(weights),
            tokenizer_path: std::path::PathBuf::new(),
            config_path: std::path::PathBuf::new(),
            dim: hidden,
            max_length: seq,
            pooling: crate::model_registry::PoolingStrategyEnum::Mean,
            query_prefix: None,
            passage_prefix: None,
            hidden_size: hidden,
            num_hidden_layers: layers,
            intermediate_size: ffn,
            num_attention_heads: heads,
            compute_dtype: None,
            format: crate::model_registry::WeightsFormat::default(),
        }
    }

    #[test]
    fn nomic_on_8gib_metal_is_safe() {
        let p = profile(768, 12, 3072, 12, 512, "/nonexistent");
        let b = recommended_batch_size(&p, &hw(Device::Metal, 8), 2);
        assert!((1..=32).contains(&b), "expected [1..=32], got {b}");
    }

    #[test]
    fn bge_small_on_8gib_metal_has_room() {
        let p = profile(384, 6, 1536, 6, 512, "/nonexistent");
        let b = recommended_batch_size(&p, &hw(Device::Metal, 8), 2);
        assert!(b >= 8, "expected >= 8, got {b}");
    }

    #[test]
    fn pathological_clamps_to_one() {
        let p = profile(4096, 32, 16384, 32, 4096, "/nonexistent");
        let b = recommended_batch_size(&p, &hw(Device::Metal, 1), 4);
        assert_eq!(b, 1);
    }

    #[test]
    fn next_bucket_zero_returns_smallest() {
        assert_eq!(next_bucket(SEQ_BUCKETS, 0, 8192), 64);
        assert_eq!(next_bucket(BATCH_BUCKETS, 0, 64), 1);
    }

    #[test]
    fn next_bucket_exact_match_returns_same() {
        assert_eq!(next_bucket(SEQ_BUCKETS, 512, 8192), 512);
        assert_eq!(next_bucket(BATCH_BUCKETS, 8, 64), 8);
    }

    #[test]
    fn next_bucket_one_over_2048_picks_4096() {
        assert_eq!(next_bucket(SEQ_BUCKETS, 2049, 8192), 4096);
    }

    #[test]
    fn next_bucket_above_max_clamps_to_cap() {
        assert_eq!(next_bucket(SEQ_BUCKETS, 12_000, 8192), 8192);
        assert_eq!(next_bucket(BATCH_BUCKETS, 200, 64), 64);
    }

    #[test]
    fn prev_bucket_boundaries() {
        assert_eq!(prev_bucket(0), TYPICAL_SEQ_LEN);
        assert_eq!(prev_bucket(63), TYPICAL_SEQ_LEN);
        assert_eq!(prev_bucket(64), 64);
        assert_eq!(prev_bucket(513), 512);
        assert_eq!(prev_bucket(4096), 4096);
        assert_eq!(prev_bucket(12_000), 8192);
    }

    #[test]
    fn attention_safe_seq_cap_keeps_model_max_on_large_metal() {
        // Metal is capped at DEFAULT_METAL_SEQ_CAP (512) regardless of available
        // memory, to align with `recommended_batch_size`'s TYPICAL_SEQ_LEN budget.
        let p = profile(768, 12, 3072, 12, 8192, "/nonexistent");
        assert_eq!(attention_safe_seq_cap(&p, &hw(Device::Metal, 64)), 512);
    }

    #[test]
    fn attention_safe_seq_cap_shrinks_to_memory_bucket() {
        // Both cases are now capped at DEFAULT_METAL_SEQ_CAP (512) on Metal.
        let p = profile(768, 12, 3072, 12, 8192, "/nonexistent");
        assert_eq!(attention_safe_seq_cap(&p, &hw(Device::Metal, 13)), 512);
        assert_eq!(attention_safe_seq_cap(&p, &hw(Device::Metal, 4)), 512);
    }

    #[test]
    fn attention_safe_seq_cap_never_below_typical_seq_len() {
        let p = profile(768, 12, 3072, 12, 8192, "/nonexistent");
        assert_eq!(
            attention_safe_seq_cap(&p, &hw(Device::Metal, 0)),
            TYPICAL_SEQ_LEN
        );
    }

    #[test]
    fn attention_safe_seq_cap_handles_max_below_typical() {
        let p = profile(384, 6, 1536, 6, 128, "/nonexistent");
        assert_eq!(attention_safe_seq_cap(&p, &hw(Device::Metal, 8)), 128);
        assert_eq!(attention_safe_seq_cap(&p, &hw(Device::Metal, 0)), 128);
    }

    #[test]
    fn seq_cap_override_parser_honors_positive_integer() {
        assert_eq!(parse_seq_cap(Some("4096".into())), Some(4096));
        assert_eq!(parse_seq_cap(Some("0".into())), None);
        assert_eq!(parse_seq_cap(Some("not-a-number".into())), None);
        assert_eq!(parse_seq_cap(None), None);
    }

    #[test]
    fn next_bucket_cap_below_natural_bucket_clamps() {
        // model max_length is 256: a 200-token batch buckets to 256 — but if
        // the model's max is 200 we should clamp to 200, not 256.
        assert_eq!(next_bucket(SEQ_BUCKETS, 200, 200), 200);
    }
}
