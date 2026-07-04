use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock, watch};

use zrag_ann::AnnCache;
use zrag_common::ids;
use zrag_dsl::ProjectIndex;
use zrag_embed::{AnyEmbedEngine, EmbedEngine, LoadOverrides};
use zrag_remote_embed::{RemoteEmbedEngine, RemoteProvider};
use zrag_rerank::TurboReranker;
use zrag_rerank::gpu::TurboScorerCache;
use zrag_store::Db;

pub struct RerankerCache {
    by_dim: RwLock<HashMap<usize, Arc<TurboReranker>>>,
}

impl Default for RerankerCache {
    fn default() -> Self {
        Self {
            by_dim: RwLock::new(HashMap::with_capacity(2)),
        }
    }
}

impl RerankerCache {
    pub async fn get(&self, dim: usize) -> anyhow::Result<Arc<TurboReranker>> {
        if let Some(reranker) = self.by_dim.read().await.get(&dim) {
            return Ok(Arc::clone(reranker));
        }

        let mut cache = self.by_dim.write().await;
        if let Some(reranker) = cache.get(&dim) {
            return Ok(Arc::clone(reranker));
        }

        let reranker = Arc::new(TurboReranker::new(dim)?);
        cache.insert(dim, Arc::clone(&reranker));
        Ok(reranker)
    }
}

pub struct LoadedProject {
    pub db: Db,
    pub dsl_index: RwLock<Option<Arc<ProjectIndex>>>,
    pub search_params: RwLock<Option<Arc<zrag_ann::SearchParams>>>,
    pub indexing_lock: Mutex<()>,
    pub cancel: AtomicBool,
    /// Set while an auto-reindex is queued or running, so overlapping FS events
    /// coalesce into one trailing run instead of stacking tasks.
    pub reindex_scheduled: AtomicBool,
}

pub struct DaemonState {
    pub primary_model: Arc<str>,
    primary_engine: Arc<AnyEmbedEngine>,
    pub engines: RwLock<HashMap<Arc<str>, Arc<AnyEmbedEngine>>>,
    pub load_lock: Mutex<()>,
    pub hardware: Arc<zrag_hw::Hardware>,
    pub model_dtype: Option<String>,
    pub remote_api_key: Option<Arc<str>>,
    pub remote_dim_hint: Option<usize>,
    pub registry: RwLock<HashMap<[u8; 32], Arc<LoadedProject>>>,
    pub ann: Arc<AnnCache>,
    pub turbo: Arc<TurboScorerCache>,
    pub reranker: RerankerCache,
    pub started_at_ns: u64,
    pub started_at: Instant,
    pub shutdown_tx: watch::Sender<bool>,
    pub shutdown_rx: watch::Receiver<bool>,
    pub watch: OnceLock<Arc<crate::watch::WatchManager>>,
    _pid_lock: File,
}

impl DaemonState {
    pub fn new(
        engine: AnyEmbedEngine,
        model_id: Arc<str>,
        hardware: Arc<zrag_hw::Hardware>,
        model_dtype: Option<String>,
        remote_api_key: Option<Arc<str>>,
        remote_dim_hint: Option<usize>,
        pid_lock: File,
    ) -> Self {
        let started_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let primary = Arc::new(engine);
        let mut engines = HashMap::with_capacity(1);
        engines.insert(Arc::clone(&model_id), Arc::clone(&primary));

        Self {
            primary_model: model_id,
            primary_engine: primary,
            engines: RwLock::new(engines),
            load_lock: Mutex::new(()),
            hardware,
            model_dtype,
            remote_api_key,
            remote_dim_hint,
            registry: RwLock::new(HashMap::with_capacity(4)),
            ann: Arc::new(AnnCache::default()),
            turbo: Arc::new(TurboScorerCache::default()),
            reranker: RerankerCache::default(),
            started_at_ns,
            started_at: Instant::now(),
            shutdown_tx,
            shutdown_rx,
            watch: OnceLock::new(),
            _pid_lock: pid_lock,
        }
    }

    pub fn primary_engine(&self) -> Arc<AnyEmbedEngine> {
        Arc::clone(&self.primary_engine)
    }

    pub fn device_str(&self) -> &str {
        self.hardware.device.as_str()
    }

    pub async fn engine_for_model(&self, model_id: &str) -> anyhow::Result<Arc<AnyEmbedEngine>> {
        if let Some(engine) = self.engines.read().await.get(model_id) {
            return Ok(Arc::clone(engine));
        }

        let _load_guard = self.load_lock.lock().await;
        if let Some(engine) = self.engines.read().await.get(model_id) {
            return Ok(Arc::clone(engine));
        }

        let result = if let Some((provider, remote_model)) = RemoteProvider::from_model_id(model_id)
        {
            tracing::info!(
                provider = %provider.label(),
                model = remote_model,
                "daemon: loading remote embed engine for search"
            );
            let api_key = self
                .remote_api_key
                .as_ref()
                .map(Arc::clone)
                .ok_or_else(|| anyhow::anyhow!("remote API key is not available"))?;
            tracing::info!(
                provider = %provider.label(),
                model = remote_model,
                "daemon: fetching remote model info"
            );
            let info = zrag_remote_embed::fetch_model_info(provider, &api_key, remote_model).await?;
            tracing::info!(
                provider = %provider.label(),
                model = remote_model,
                context_length = info.context_length,
                "daemon: model info fetched, connecting remote engine"
            );
            let engine = RemoteEmbedEngine::connect(provider, api_key, &info, self.remote_dim_hint)
                .await
                .map(AnyEmbedEngine::Remote);
            match &engine {
                Ok(e) => tracing::info!(
                    provider = %provider.label(),
                    model = remote_model,
                    dim = e.dim(),
                    "daemon: remote embed engine loaded"
                ),
                Err(e) => tracing::warn!(
                    provider = %provider.label(),
                    model = remote_model,
                    error = %e,
                    "daemon: remote embed engine load failed"
                ),
            }
            engine
        } else {
            let hw = Arc::clone(&self.hardware);
            let owned = model_id.to_owned();
            let model_dtype = self.model_dtype.as_deref().map(String::from);
            tokio::task::spawn_blocking(move || {
                EmbedEngine::load_with(&owned, hw, &LoadOverrides {
                    model_dtype: model_dtype.as_deref(),
                    ..Default::default()
                })
                .map(AnyEmbedEngine::Local)
            })
            .await?
        };

        let arc = Arc::new(result?);
        self.engines
            .write()
            .await
            .insert(Arc::from(model_id), Arc::clone(&arc));
        Ok(arc)
    }

    pub async fn load_or_open(&self, project_root: &str) -> anyhow::Result<Arc<LoadedProject>> {
        let root = std::path::Path::new(project_root).canonicalize()?;
        let pid = ids::project_id(&root);

        {
            let reg = self.registry.read().await;
            if let Some(proj) = reg.get(&pid) {
                return Ok(Arc::clone(proj));
            }
        }

        let db = Db::open(&pid).await?;
        let project = Arc::new(LoadedProject {
            db,
            dsl_index: RwLock::new(None),
            search_params: RwLock::new(None),
            indexing_lock: Mutex::new(()),
            cancel: AtomicBool::new(false),
            reindex_scheduled: AtomicBool::new(false),
        });

        {
            let mut reg = self.registry.write().await;
            reg.insert(pid, Arc::clone(&project));
        }

        // Lazily start watching this project now that it's loaded.
        if let Some(manager) = self.watch.get() {
            let _ = manager.watch(root, pid).await;
        }

        Ok(project)
    }
}
