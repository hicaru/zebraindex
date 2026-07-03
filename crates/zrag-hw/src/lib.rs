pub use zrag_hw_core::{Device, Hardware};

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
