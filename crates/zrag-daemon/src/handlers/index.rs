use std::sync::atomic::Ordering;

use anyhow::Result;
use tokio::io::AsyncWrite;
use tokio::sync::mpsc;

use zrag_pipeline::progress::Reporter;
use zrag_protocol::codec::write_frame;
use zrag_protocol::request::IndexReq;
use zrag_protocol::response::{ErrorBody, IndexStats, Response};

use crate::state::DaemonState;

/// Streaming index. Writes any number of `Response::IndexProgress` frames
/// followed by exactly one terminal `Response::Index(Ok|Err)`.
pub async fn handle_streaming<W>(writer: &mut W, req: &IndexReq, state: &DaemonState) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return write_frame(
                writer,
                &Response::Index(Err(ErrorBody {
                    message: e.to_string(),
                })),
            )
            .await;
        }
    };

    let _lock = project.indexing_lock.lock().await;
    project.cancel.store(false, Ordering::Relaxed);

    let root = std::path::Path::new(&req.project_root).to_path_buf();
    let engine = state.primary_engine();
    let db = project.db.clone();
    let override_method = req
        .search_method
        .as_deref()
        .and_then(zrag_ann::SearchMethod::parse);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let reporter = Reporter::Ipc(zrag_pipeline::progress::IpcReporter::new(tx));

    let proj = project.clone();
    let refresh = req.refresh;

    let mut handle = tokio::spawn(async move {
        zrag_pipeline::indexer::index_project(
            &root,
            &engine,
            &db,
            &reporter,
            override_method,
            &proj.cancel,
            refresh,
        )
        .await
    });

    let final_result = loop {
        tokio::select! {
            biased;
            Some(p) = rx.recv() => {
                if let Err(e) = write_frame(writer, &Response::IndexProgress(p)).await {
                    tracing::debug!("progress write error: {}", e);
                    handle.abort();
                    return Err(e);
                }
            }
            joined = &mut handle => {
                break joined;
            }
        }
    };

    // Drain any progress events that landed after indexing returned.
    while let Ok(p) = rx.try_recv() {
        let _ = write_frame(writer, &Response::IndexProgress(p)).await;
    }

    let final_result = match final_result {
        Ok(r) => r,
        Err(e) => Err(anyhow::anyhow!("indexing task failed: {e}")),
    };

    let terminal = match final_result {
        Ok(stats) => {
            *project.search_params.write().await = None;
            *project.dsl_index.write().await = None;
            if let Ok(root) = std::path::Path::new(&req.project_root).canonicalize() {
                let pid = zrag_common::ids::project_id(&root);
                state.ann.invalidate(&pid).await;
                if let Some(manager) = state.watch.get() {
                    let _ = manager.watch(root, pid).await;
                }
            }
            Response::Index(Ok(IndexStats {
                total_chunks: stats.total_chunks,
                total_files: stats.total_files,
                new_chunks: stats.new_chunks,
                reindexed_files: stats.reindexed_files,
                duration_ms: stats.duration_ms,
                paused: stats.paused,
            }))
        }
        Err(e) => Response::Index(Err(ErrorBody {
            message: e.to_string(),
        })),
    };

    write_frame(writer, &terminal).await
}
