use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
use tokio::sync::{Mutex, mpsc};

use zrag_pipeline::manifest::{foundry_roots, is_index_candidate};
use zrag_pipeline::progress::{Reporter, SilentReporter};

use crate::state::DaemonState;

/// Quiet window before a burst of edits (editor save storms, `git pull`) turns
/// into a single reindex.
const DEBOUNCE: Duration = Duration::from_secs(2);

type FsDebouncer = Debouncer<RecommendedWatcher, RecommendedCache>;

/// One watched project root and the id its events route to.
struct Watched {
    root: PathBuf,
    pid: [u8; 32],
    foundry_roots: std::collections::HashSet<PathBuf>,
}

pub struct WatchManager {
    debouncer: Mutex<FsDebouncer>,
    watched: Mutex<Vec<Watched>>,
}

impl WatchManager {
    /// Build the debouncer, spawn the event loop, return the shared manager.
    pub fn start(state: Arc<DaemonState>) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::unbounded_channel::<DebounceEventResult>();
        // The notify callback is sync; `unbounded_send` is non-blocking, so it
        // bridges cleanly into the async event loop.
        let debouncer = new_debouncer(DEBOUNCE, None, move |res| {
            let _ = tx.send(res);
        })?;

        let manager = Arc::new(Self {
            debouncer: Mutex::new(debouncer),
            watched: Mutex::new(Vec::with_capacity(8)),
        });

        tokio::spawn(event_loop(rx, Arc::clone(&manager), state));
        Ok(manager)
    }

    /// Recursively watch a project root and remember its id. Idempotent per pid.
    pub async fn watch(&self, root: PathBuf, pid: [u8; 32]) -> Result<()> {
        {
            let watched = self.watched.lock().await;
            if watched.iter().any(|w| w.pid == pid) {
                return Ok(());
            }
        }
        self.debouncer
            .lock()
            .await
            .watch(&root, RecursiveMode::Recursive)?;
        let roots = foundry_roots(&root);
        self.watched.lock().await.push(Watched {
            root,
            pid,
            foundry_roots: roots,
        });
        Ok(())
    }

    /// Stop watching a project (used by remove_project).
    pub async fn unwatch(&self, pid: &[u8; 32]) {
        let removed = {
            let mut watched = self.watched.lock().await;
            watched
                .iter()
                .position(|w| &w.pid == pid)
                .map(|i| watched.swap_remove(i))
        };
        if let Some(w) = removed {
            let _ = self.debouncer.lock().await.unwatch(&w.root);
        }
    }
}

/// Drain debounced batches and schedule one reindex per affected project.
async fn event_loop(
    mut rx: mpsc::UnboundedReceiver<DebounceEventResult>,
    manager: Arc<WatchManager>,
    state: Arc<DaemonState>,
) {
    while let Some(res) = rx.recv().await {
        let events = match res {
            Ok(events) => events,
            Err(errs) => {
                tracing::debug!(count = errs.len(), "watch errors");
                continue;
            }
        };

        // Lock watched once, filter all paths against it, de-dupe pids before
        // cloning any root PathBuf. Few projects are watched so a Vec beats a
        // HashSet.
        let watched = manager.watched.lock().await;
        let mut dirty: Vec<([u8; 32], PathBuf)> = Vec::with_capacity(watched.len());
        for w in watched.iter() {
            let hit = events.iter().flat_map(|e| e.paths.iter()).any(|p| {
                p.starts_with(&w.root) && is_index_candidate(&w.root, p, &w.foundry_roots)
            });
            if hit && !dirty.iter().any(|(p, _)| p == &w.pid) {
                dirty.push((w.pid, w.root.clone()));
            }
        }
        drop(watched);

        for (pid, root) in dirty {
            schedule_reindex(Arc::clone(&state), pid, root);
        }
    }
}

/// Single-flight reindex with a trailing run. `swap(true)` coalesces events that
/// arrive while a run is queued; resetting the flag *after* taking the lock means
/// an edit landing mid-run schedules exactly one more pass.
fn schedule_reindex(state: Arc<DaemonState>, pid: [u8; 32], root: PathBuf) {
    tokio::spawn(async move {
        let project = {
            let reg = state.registry.read().await;
            match reg.get(&pid) {
                Some(p) => Arc::clone(p),
                None => return,
            }
        };

        if project.reindex_scheduled.swap(true, Ordering::AcqRel) {
            return; // already queued/running — coalesced
        }

        let engine = state.primary_engine();
        let _guard = project.indexing_lock.lock().await;
        project.reindex_scheduled.store(false, Ordering::Release);
        project.cancel.store(false, Ordering::Relaxed);

        let reporter = Reporter::Silent(SilentReporter);
        match zrag_pipeline::indexer::index_project(
            &root,
            &engine,
            &project.db,
            &reporter,
            None,
            &project.cancel,
            false,
        )
        .await
        {
            Ok(stats) => {
                *project.search_params.write().await = None;
                *project.dsl_index.write().await = None;
                state.ann.invalidate(&pid).await;
                tracing::info!(
                    reindexed = stats.reindexed_files,
                    total_files = stats.total_files,
                    total_chunks = stats.total_chunks,
                    "auto-reindex done"
                );
            }
            Err(e) => tracing::warn!("auto-reindex failed: {e}"),
        }
    });
}
