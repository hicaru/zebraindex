use sysinfo::System;

use crate::device::{Device, Hardware};

pub fn probe(supported: &[Device]) -> Hardware {
    warn_if_simd_mismatch();
    let mut sys = System::new();
    sys.refresh_cpu_all();
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
#[cfg(target_arch = "x86_64")]
pub fn warn_if_simd_mismatch() {
    if is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && !cfg!(target_feature = "avx2")
    {
        tracing::warn!(
            "CPU supports AVX2+FMA but this binary was built without it; quantized \
             matmul (GGUF Q-types) will use the slow scalar path. Rebuild with \
             RUSTFLAGS=\"-C target-feature=+avx2,+fma\" (or -C target-cpu=native) \
             for a large speedup."
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn warn_if_simd_mismatch() {}
