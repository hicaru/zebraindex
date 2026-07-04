use crate::probe::CpuCaps;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Device {
    #[default]
    Cpu,
    Metal,
    Cuda,
}

impl Device {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Hardware {
    pub device: Device,
    pub cpus: usize,
    pub mem_total: u64,
    pub mem_avail: u64,
    pub cpu: CpuCaps,
}
