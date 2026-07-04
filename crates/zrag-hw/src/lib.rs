pub use zrag_hw_core::{CpuCaps, Device, Hardware};

pub fn supported_devices() -> Vec<Device> {
    let mut devs = Vec::with_capacity(3);
    #[cfg(all(target_os = "macos", feature = "metal"))]
    devs.push(Device::Metal);
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    devs.push(Device::Cuda);
    devs.push(Device::Cpu);
    devs
}

pub fn probe() -> Hardware {
    zrag_hw_core::probe(&supported_devices())
}

pub fn candle_device(hw: &Hardware) -> candle_core::Device {
    match hw.device {
        #[cfg(feature = "metal")]
        Device::Metal => candle_core::Device::new_metal(0).unwrap_or_else(|e| {
            tracing::warn!("Metal init failed: {e}, falling back to CPU");
            candle_core::Device::Cpu
        }),
        #[cfg(feature = "cuda")]
        Device::Cuda => candle_core::Device::new_cuda(0).unwrap_or_else(|e| {
            tracing::warn!(
                "CUDA init failed ({e}); falling back to CPU. candle 0.10.2 ships \
                 only NVIDIA (CUDA) and Apple (Metal) GPU backends — AMD/Intel GPUs \
                 are not supported and run on the CPU (AVX2+FMA) path."
            );
            candle_core::Device::Cpu
        }),
        _ => candle_core::Device::Cpu,
    }
}

/// Like [`probe`] but corrects [`Hardware::device`] when a GPU backend init
/// fails (e.g. Metal not available on this system). The returned
/// [`Hardware`] always reflects the *effective* device so the TUI header and
/// dtype recommendations don't lie about what the engine will actually use.
pub fn probe_effective() -> Hardware {
    let mut hw = probe();
    let dev = candle_device(&hw);
    if matches!(dev, candle_core::Device::Cpu) && hw.device != Device::Cpu {
        tracing::warn!(
            claimed = %hw.device.as_str(),
            "backend init failed; hardware downgraded to CPU",
        );
        hw.device = Device::Cpu;
    }
    hw
}
