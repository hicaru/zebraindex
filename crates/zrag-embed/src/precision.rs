//! Hardware-aware dtype recommendation: turns a list of [`WeightVariant`]s
//! detected for a model into a ranked list of [`DTypeChoice`]s the TUI can
//! render, each tagged with its device affinity and any caveat.

use zrag_hw::Hardware;

pub use crate::model_registry::{DTypeChoice, WeightVariant, WeightsFormat};

/// Build the dtype choice list for the picker: one row per variant, an F32
/// fallback row guaranteed present, and exactly one row marked recommended.
pub fn build_dtype_choices(variants: &[WeightVariant], hw: &Hardware) -> Vec<DTypeChoice> {
    let gpu = !matches!(hw.device, zrag_hw::Device::Cpu);
    let fast = hw.cpu.fast_quant();
    let hw_label = hw.cpu.label();

    let mut out: Vec<DTypeChoice> = variants
        .iter()
        .map(|v| choice_from_variant(v, gpu, fast, hw_label))
        .collect();

    // F32 is always loadable — `dense_varbuilder_from_gguf` dequantizes any
    // Q-type, and it's already the warmup fallback in `engine::load_with`.
    ensure_f32_present(&mut out, hw_label);

    mark_recommended(&mut out, gpu, fast);
    out
}

fn choice_from_variant(
    v: &WeightVariant,
    gpu: bool,
    fast: bool,
    hw_label: &str,
) -> DTypeChoice {
    let (tag, warning) = device_tag_and_warning(v, gpu, fast, hw_label);
    let (description, detail) = describe(&v.precision, v.format);
    DTypeChoice {
        label: v.precision.to_uppercase(),
        cli_value: v.precision.clone(),
        description,
        detail,
        device_tag: tag,
        recommended: false,
        warning,
    }
}

fn device_tag_and_warning(
    v: &WeightVariant,
    gpu: bool,
    fast: bool,
    hw_label: &str,
) -> (String, Option<String>) {
    let is_quant = v.format == WeightsFormat::Gguf && v.precision.starts_with('q');
    if is_quant {
        let tag = if fast {
            format!("[CPU {hw_label}]")
        } else {
            "[CPU]".into()
        };
        let warn = if fast {
            slow_qtype_warning(&v.precision)
        } else {
            Some(
                "no AVX2+FMA fast path: quantized matmul falls back to slow scalar".into(),
            )
        };
        (tag, warn)
    } else if v.precision == "f16" || v.precision == "bf16" {
        let warn = (!gpu).then(|| "half precision on CPU is slower than F32".into());
        ("[GPU]".into(), warn)
    } else {
        ("[CPU]".into(), None)
    }
}

fn slow_qtype_warning(precision: &str) -> Option<String> {
    match precision {
        "q8_1" => Some(
            "Q8_1 has no fused SIMD/CUDA matmul kernel in candle 0.10.2; \
             scalar-only"
                .into(),
        ),
        "q8_k" => Some(
            "Q8K has no fused CUDA kernel in candle 0.10.2; \
             dequant+cuBLAS fallback"
                .into(),
        ),
        _ => None,
    }
}

fn describe(precision: &str, format: WeightsFormat) -> (String, String) {
    match (format, precision) {
        (WeightsFormat::Gguf, p) if p.starts_with('q') => (
            format!("{p} GGUF quant"),
            "Quantized weights. Lower memory, fast matmul on AVX2+FMA/NEON.".into(),
        ),
        (_, "f16") => (
            "Half precision.".into(),
            "Fast on GPU. ~2× smaller than F32.".into(),
        ),
        (_, "bf16") => (
            "Brain float16.".into(),
            "Best GPU precision (avoids F16 range loss).".into(),
        ),
        (_, "f32") => (
            "Full precision.".into(),
            "Universal fallback. Highest accuracy.".into(),
        ),
        _ => (precision.into(), String::new()),
    }
}

fn ensure_f32_present(out: &mut Vec<DTypeChoice>, _hw_label: &str) {
    if !out.iter().any(|c| c.cli_value == "f32") {
        out.push(DTypeChoice {
            label: "F32".into(),
            cli_value: "f32".into(),
            description: "Full precision.".into(),
            detail: "Universal fallback. Highest accuracy.".into(),
            device_tag: "[CPU]".into(),
            recommended: false,
            warning: None,
        });
    }
}

fn mark_recommended(out: &mut [DTypeChoice], gpu: bool, fast: bool) {
    let target = if gpu {
        out.iter()
            .position(|c| c.cli_value == "bf16")
            .or_else(|| out.iter().position(|c| c.cli_value == "f16"))
            .or_else(|| out.iter().position(|c| c.cli_value == "f32"))
    } else if fast {
        best_quant_rank(out).or_else(|| out.iter().position(|c| c.cli_value == "f32"))
    } else {
        out.iter().position(|c| c.cli_value == "f32")
    };
    if let Some(i) = target {
        if let Some(c) = out.get_mut(i) {
            c.recommended = true;
        }
    }
}

/// Preference order for quantized picks (lower index = preferred).
fn best_quant_rank(out: &[DTypeChoice]) -> Option<usize> {
    const PREF: &[&str] = &[
        "q4_k_m", "q4_k_s", "q5_k_m", "q6_k", "q5_k_s", "q4_k", "q5_k", "q8_0", "q4_0", "q5_0",
        "q8_1", "q8_k", "q4_1", "q5_1", "q2_k", "q3_k",
    ];
    for pref in PREF {
        if let Some(i) = out.iter().position(|c| c.cli_value == *pref) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::{WeightVariant, WeightsFormat};
    use zrag_hw::{CpuCaps, Device, Hardware};

    fn variants(precisions: &[&str], fmt: WeightsFormat) -> Vec<WeightVariant> {
        precisions
            .iter()
            .map(|p| WeightVariant {
                format: fmt,
                precision: (*p).into(),
                file: Some(format!("model-{p}.gguf")),
                size_bytes: None,
            })
            .collect()
    }

    fn hw_cpu_fast() -> Hardware {
        Hardware {
            device: Device::Cpu,
            cpus: 8,
            mem_total: 0,
            mem_avail: 0,
            cpu: CpuCaps {
                avx2: true,
                fma: true,
                neon: false,
                built_with_simd: true,
            },
        }
    }

    fn hw_cpu_scalar() -> Hardware {
        Hardware {
            device: Device::Cpu,
            cpus: 8,
            mem_total: 0,
            mem_avail: 0,
            cpu: CpuCaps::default(),
        }
    }

    fn hw_metal() -> Hardware {
        Hardware {
            device: Device::Metal,
            cpus: 8,
            mem_total: 0,
            mem_avail: 0,
            cpu: CpuCaps::default(),
        }
    }

    #[test]
    fn cpu_fast_quant_recommends_best_quant() {
        let v = variants(&["q4_k_m", "q8_0", "f32"], WeightsFormat::Gguf);
        let choices = build_dtype_choices(&v, &hw_cpu_fast());
        let rec = choices.iter().find(|c| c.recommended).unwrap();
        assert_eq!(rec.cli_value, "q4_k_m");
    }

    #[test]
    fn cpu_scalar_recommends_f32_with_warning() {
        let v = variants(&["q4_k_m", "f32"], WeightsFormat::Gguf);
        let choices = build_dtype_choices(&v, &hw_cpu_scalar());
        let rec = choices.iter().find(|c| c.recommended).unwrap();
        assert_eq!(rec.cli_value, "f32");
        // The q4 row should carry the scalar warning.
        let q4 = choices.iter().find(|c| c.cli_value == "q4_k_m").unwrap();
        assert!(q4.warning.as_ref().unwrap().contains("scalar"));
    }

    #[test]
    fn metal_recommends_f16() {
        let v = variants(&["f16", "f32"], WeightsFormat::Safetensors);
        let choices = build_dtype_choices(&v, &hw_metal());
        let rec = choices.iter().find(|c| c.recommended).unwrap();
        assert_eq!(rec.cli_value, "f16");
    }

    #[test]
    fn f32_always_present() {
        let v = variants(&["q4_k_m"], WeightsFormat::Gguf);
        let choices = build_dtype_choices(&v, &hw_cpu_fast());
        assert!(choices.iter().any(|c| c.cli_value == "f32"));
    }

    #[test]
    fn exactly_one_recommended() {
        let v = variants(&["q4_k_m", "q8_0", "f32"], WeightsFormat::Gguf);
        let choices = build_dtype_choices(&v, &hw_cpu_fast());
        assert_eq!(choices.iter().filter(|c| c.recommended).count(), 1);
    }
}
