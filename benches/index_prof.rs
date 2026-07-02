mod prof_common;

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Result;
use prof_common::env_or_arg;
use zrag_common::ids::project_id;
use zrag_embed::{AnyEmbedEngine, EmbedEngine};
use zrag_pipeline::indexer::index_project;
use zrag_pipeline::progress::{IpcReporter, Reporter};
use zrag_protocol::response::IndexPhase;
use zrag_store::Db;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = prof_common::start_heap_profiler();

    let model_id = env_or_arg(
        "MODEL_ID",
        1,
        "set MODEL_ID env var or pass model_id as first arg",
    )?;

    let project_root = env_or_arg(
        "PROJECT_ROOT",
        2,
        "set PROJECT_ROOT env var or pass project root as second arg",
    )?;

    let root = Path::new(&project_root).canonicalize()?;
    eprintln!("model:   {model_id}");
    eprintln!("project: {}", root.display());

    let engine = tokio::task::spawn_blocking({
        let m = model_id.clone();
        move || EmbedEngine::load(&m)
    })
    .await??;
    let engine = AnyEmbedEngine::Local(engine);
    eprintln!("model loaded. dim={}\n", engine.dim());

    let pid = project_id(&root);
    let db = Db::open(&pid).await?;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let reporter = Reporter::Ipc(IpcReporter::new(tx));

    let progress_task = tokio::spawn(async move {
        let t0 = Instant::now();
        let mut phase_start = Instant::now();
        let mut last_phase: Option<IndexPhase> = None;

        while let Some(p) = rx.recv().await {
            let changed = last_phase != Some(p.phase);
            if changed {
                if let Some(lp) = last_phase {
                    eprintln!("  └─ {:?} done in {:.2?}", lp, phase_start.elapsed());
                }
                eprintln!("\n[{:>8.2?}] Phase::{:?}", t0.elapsed(), p.phase);
                phase_start = Instant::now();
                last_phase = Some(p.phase);
            }
            if p.total > 0 {
                eprintln!("  {:>6}/{} {}", p.current, p.total, p.message);
            } else if !p.message.is_empty() {
                eprintln!("  {}", p.message);
            }
        }
        if let Some(lp) = last_phase {
            eprintln!("  └─ {:?} done in {:.2?}", lp, phase_start.elapsed());
        }
        t0.elapsed()
    });

    let cancel = AtomicBool::new(false);
    let stats = index_project(&root, &engine, &db, &reporter, None, &cancel, false).await?;
    drop(reporter);

    let total_elapsed = progress_task.await?;

    eprintln!("\n═══ IndexStats ═══════════════════");
    eprintln!("  total files:     {}", stats.total_files);
    eprintln!("  total chunks:    {}", stats.total_chunks);
    eprintln!("  new chunks:      {}", stats.new_chunks);
    eprintln!("  reindexed files: {}", stats.reindexed_files);
    eprintln!("  duration_ms:     {}", stats.duration_ms);
    eprintln!("  wall time:       {:.2?}", total_elapsed);
    Ok(())
}
