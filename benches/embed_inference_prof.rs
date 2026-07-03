mod prof_common;

use anyhow::Result;
use prof_common::env_or_arg;
use zrag_embed::EmbedEngine;

const WARMUP: usize = 3;
const ITERS: usize = 50;

const TEXTS: &[&str] = &[
    "fn main() { println!(\"hello world\"); }",
    "pub struct EmbedEngine { tx: mpsc::UnboundedSender<EmbedRequest> }",
    "impl EmbedEngine { pub fn load(model_id: &str) -> Result<Self> { todo!() } }",
    "SELECT id, name FROM users WHERE active = true ORDER BY created_at DESC LIMIT 100;",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "async function fetchUser(id: string): Promise<User> { return await api.get(`/users/${id}`); }",
    "class Node<T> { value: T; next: Option<Box<Node<T>>>; }",
    "import numpy as np\nX = np.random.randn(1000, 128)\nU, S, Vt = np.linalg.svd(X, full_matrices=False)",
];

fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = prof_common::start_heap_profiler();

    let model_id = env_or_arg(
        "MODEL_ID",
        1,
        "set MODEL_ID env var or pass model_id as first arg",
    )?;

    eprintln!("loading model: {model_id}");
    let engine = EmbedEngine::load(&model_id)?;
    eprintln!(
        "loaded. dim={}, format={:?}, device={:?}",
        engine.dim(),
        engine.profile().format,
        engine.hardware().device,
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    for _ in 0..WARMUP {
        rt.block_on(engine.embed_batch_pooled_async(TEXTS))?;
    }
    eprintln!("warmup done, running {ITERS} profiling iters...");

    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        rt.block_on(engine.embed_batch_pooled_async(TEXTS))?;
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "done. {ITERS} iters in {:.2?} → {:.1} ms/iter",
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / ITERS as f64,
    );
    Ok(())
}
