use anyhow::{Context, Result};

pub fn env_or_arg(name: &str, arg_index: usize, help: &'static str) -> Result<String> {
    std::env::var(name)
        .or_else(|_| {
            std::env::args()
                .nth(arg_index)
                .ok_or(std::env::VarError::NotPresent)
        })
        .with_context(|| help)
}

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
pub fn start_heap_profiler() -> dhat::Profiler {
    dhat::Profiler::new_heap()
}
