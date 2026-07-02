use std::borrow::Cow;
use std::io::Write as _;
use std::sync::Arc;

use anyhow::{Context, Result};
use fs2::FileExt;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;
use zrag_embed::{AnyEmbedEngine, EmbedEngine, LoadOverrides};
use zrag_remote_embed::{RemoteEmbedEngine, RemoteProvider};

pub mod handlers;
pub mod listener;
pub mod registry;
pub mod state;
pub mod watch;

use state::DaemonState;

pub const DAEMON_DEFAULT_LOG_FILTER: &str = concat!(
    "info,zrag_daemon=debug,zrag_embed=debug,",
    "zrag_pipeline=debug,zrag_dsl=debug,zrag_store=debug,",
    "zrag_rerank=trace",
);

pub struct DaemonConfig<'a> {
    pub model: Cow<'a, str>,
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
    pub model_dtype: Option<&'a str>,
    pub remote_api_key: Option<&'a str>,
    pub remote_dim_hint: Option<usize>,
}

pub fn run_daemon(config: &DaemonConfig<'_>) -> Result<()> {
    let pid_path = zrag_common::paths::daemon_pid()?;
    let mut pid_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(&pid_path)
        .with_context(|| format!("opening {}", pid_path.display()))?;

    if pid_file.try_lock_exclusive().is_err() {
        // Lock held — a daemon is already running. Show a friendly
        // message with the existing PID instead of a cryptic OS error.
        let existing_pid = std::fs::read_to_string(&pid_path).unwrap_or_else(|_| String::from("?"));
        let socket = zrag_common::paths::daemon_socket()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| String::from("?"));
        eprintln!(
            "Daemon is already running (PID {}).\n\
             Socket: {}\n\
             Use CLI commands directly: `zebrarag index`, `zebrarag search`, etc.",
            existing_pid.trim(),
            socket,
        );
        return Ok(());
    }

    pid_file.set_len(0)?;
    write!(pid_file, "{}", std::process::id())?;
    pid_file.flush()?;

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(DAEMON_DEFAULT_LOG_FILTER)),
        )
        .with_writer(std::io::stderr)
        .try_init();
    // Cap the `log` bridge: dependency debug/trace records are dropped
    // before they reach the tracing dispatcher (matching the main.rs policy).
    log::set_max_level(log::LevelFilter::Warn);

    let socket_path = zrag_common::paths::daemon_socket()?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    tracing::info!("loading model: {}", config.model);
    let hw = Arc::new(zrag_hw::probe());
    tracing::info!(device = ?hw.device, "hardware detected");

    let opts = LoadOverrides {
        query_prefix: config.query_prefix,
        passage_prefix: config.passage_prefix,
        model_dtype: config.model_dtype.and_then(zrag_embed::parse_model_dtype),
    };
    let is_remote_model = RemoteProvider::from_model_id(config.model.as_ref()).is_some();
    let preloaded_engine = if is_remote_model {
        None
    } else {
        Some(AnyEmbedEngine::Local(EmbedEngine::load_with(
            &config.model,
            Arc::clone(&hw),
            &opts,
        )?))
    };
    let remote_api_key = config.remote_api_key.map(Arc::<str>::from);
    let remote_dim_hint = config.remote_dim_hint;

    // Build the runtime explicitly: a fixed, named worker pool (one per core)
    // keeps the reactor from over-provisioning and makes thread profiles
    // legible (`zrag-rt-*` vs anonymous `tokio-runtime-worker`). The blocking
    // pool is left at its default bound — Lance drives its own IO through
    // `spawn_blocking`, so capping it risks starving storage operations.
    let worker_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .thread_name("zrag-rt")
        .enable_all()
        .build()?;
    rt.block_on(async {
        let model_id: Arc<str> = Arc::from(config.model.as_ref());
        let engine = if let Some(engine) = preloaded_engine {
            engine
        } else if let Some((provider, remote_model)) =
            RemoteProvider::from_model_id(config.model.as_ref())
        {
            let api_key = remote_api_key.as_ref().map(Arc::clone).ok_or_else(|| {
                anyhow::anyhow!("remote API key is required for {}", provider.label())
            })?;
            let info = zrag_remote_embed::fetch_model_info(provider, &api_key, remote_model).await?;
            let remote =
                RemoteEmbedEngine::connect(provider, Arc::clone(&api_key), &info, remote_dim_hint)
                    .await?;
            tracing::info!(
                dim = remote.dim(),
                model = remote.model_id(),
                "remote embed engine ready"
            );
            AnyEmbedEngine::Remote(remote)
        } else {
            anyhow::bail!(
                "unsupported remote model '{}': missing provider prefix",
                config.model
            );
        };
        let model_dtype = config.model_dtype.map(String::from);
        let state = Arc::new(DaemonState::new(
            engine,
            model_id,
            hw,
            model_dtype,
            remote_api_key,
            remote_dim_hint,
            pid_file,
        ));

        match watch::WatchManager::start(Arc::clone(&state)) {
            Ok(manager) => {
                let _ = state.watch.set(manager);
            }
            Err(e) => tracing::warn!("file watcher disabled: {e}"),
        }

        let listener = UnixListener::bind(&socket_path)?;
        tracing::info!("daemon listening on {}", socket_path.display());
        listener::run(listener, state).await?;
        let _ = std::fs::remove_file(&pid_path);
        let _ = std::fs::remove_file(&socket_path);
        Ok(())
    })
}
