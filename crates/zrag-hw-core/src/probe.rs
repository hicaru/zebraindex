use sysinfo::System;

use crate::device::{Device, Hardware};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuCaps {
    pub avx2: bool,
    pub fma: bool,
    pub neon: bool,
    /// Binary was compiled with the SIMD path enabled (target-feature).
    pub built_with_simd: bool,
}

impl CpuCaps {
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            neon: false,
            built_with_simd: cfg!(target_feature = "avx2"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        // NEON is baseline in the AArch64 ABI (v8.0+ mandates Advanced SIMD),
        // but we still check the compile-time flag so fast_quant() doesn't lie
        // on a theoretical aarch64 binary built without NEON kernels.
        Self {
            neon: true,
            built_with_simd: cfg!(target_feature = "neon"),
            ..Default::default()
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        Self::default()
    }

    /// Fast quantized (GGUF Q-type) matmul is actually usable on this host+binary.
    pub fn fast_quant(&self) -> bool {
        (self.avx2 && self.fma && self.built_with_simd) || self.neon
    }

    pub fn label(&self) -> &'static str {
        match (self.neon, self.avx2 && self.fma, self.built_with_simd) {
            (true, _, _) => "NEON",
            (_, true, true) => "AVX2+FMA",
            (_, true, false) => "AVX2 (binary built w/o SIMD!)",
            _ => "scalar",
        }
    }
}

pub fn probe(supported: &[Device]) -> Hardware {
    let cpu = CpuCaps::detect();
    warn_if_simd_mismatch(&cpu);
    let mut sys = System::new();
    sys.refresh_memory();

    let cpus = System::physical_core_count().unwrap_or(1);
    let mem_total = sys.total_memory();
    let mem_avail = sys.available_memory();

    let device = select_device(supported);

    Hardware {
        device,
        cpus,
        mem_total,
        mem_avail,
        cpu,
    }
}

fn select_device(supported: &[Device]) -> Device {
    for d in [&Device::Metal, &Device::Cuda] {
        if supported.contains(d) && hardware_supports(d) {
            return *d;
        }
    }
    Device::Cpu
}

fn hardware_supports(d: &Device) -> bool {
    match d {
        Device::Cpu => true,
        Device::Metal => cfg!(target_os = "macos"),
        Device::Cuda => cfg!(any(target_os = "linux", target_os = "windows")),
    }
}

/// Warn when the host CPU has AVX2+FMA but the running binary was built without
/// them, so candle's quantized matmul silently falls back to the slow scalar path.
///
/// candle 0.10.2 gates its quantized SIMD kernels behind `#[cfg(target_feature)]`
/// at compile time (no runtime dispatch), so a feature-less build quietly forfeits
/// the large speedups Q4_0…Q8K matmuls otherwise get. This covers AMD Ryzen/EPYC
/// CPUs, whose AVX2+FMA units run the exact same kernels as Intel's.
pub fn warn_if_simd_mismatch(cpu: &CpuCaps) {
    if cpu.avx2 && cpu.fma && !cpu.built_with_simd {
        tracing::warn!(
            "CPU supports AVX2+FMA but this binary was built without it; quantized \
             matmul (GGUF Q-types) will use the slow scalar path. Rebuild with \
             RUSTFLAGS=\"-C target-feature=+avx2,+fma\" (or -C target-cpu=native) \
             for a large speedup."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_caps_have_label() {
        let caps = CpuCaps::detect();
        eprintln!("CPU caps: {caps:?} ({})", caps.label());
        assert!(!caps.label().is_empty());
    }

    #[test]
    fn fast_quant_requires_runtime_and_binary_simd_or_neon() {
        assert!(!CpuCaps::default().fast_quant());
        assert!(CpuCaps {
            avx2: true,
            fma: true,
            neon: false,
            built_with_simd: true,
        }
        .fast_quant());
        assert!(CpuCaps {
            neon: true,
            ..Default::default()
        }
        .fast_quant());
    }
}
