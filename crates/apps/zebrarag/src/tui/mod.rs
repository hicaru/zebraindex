mod actions;
mod app;
mod config;
mod event;
mod registry;
mod tasks;
mod widgets;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use app::{App, AppMessage, DaemonStatus, Modal, Screen, SetupPhase};
use tasks::{fetch_registry, resolve_startup, spawn_daemon_monitor, spawn_refresh_projects};

pub fn run_tui(
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        terminal::enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut app = App {
            model: model.map(Arc::from),
            query_prefix: query_prefix.map(Arc::from),
            passage_prefix: passage_prefix.map(Arc::from),
            model_dtype: model_dtype.map(Arc::from),
            local_hardware: Some(zrag_hw::probe_effective()),
            ..App::default()
        };

        let (tx, mut rx) = mpsc::channel::<AppMessage>(4096);

        if model.is_some() {
            app.screen = Screen::Main;
            spawn_daemon_monitor(&mut app, &tx);
        } else {
            let tx_r = tx.clone();
            tokio::spawn(async move { resolve_startup(tx_r).await });
        }

        let mut tick: u16 = 0;

        loop {
            terminal.draw(|f| widgets::draw(f, &app, tick))?;

            while let Ok(msg) = rx.try_recv() {
                dispatch(&mut app, msg, &tx).await;
            }

            if crossterm::event::poll(Duration::from_millis(50))?
                && let crossterm::event::Event::Key(key) = crossterm::event::read()?
            {
                let action = event::map_key(&key, &app);
                actions::handle_action(&mut app, action, &tx).await;
            }

            tick = tick.wrapping_add(1);

            if app.should_quit {
                break;
            }
        }

        terminal::disable_raw_mode()?;
        execute!(std::io::stdout(), LeaveAlternateScreen)?;
        Ok(())
    })
}

async fn dispatch(app: &mut App, msg: AppMessage, tx: &mpsc::Sender<AppMessage>) {
    match msg {
        AppMessage::ConfigResolved { model: None, .. } => {
            app.screen = Screen::Setup(SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }
        AppMessage::ConfigResolved {
            model: Some(m),
            search_method,
            model_dtype,
            remote_provider,
            remote_api_key,
            remote_dim_hint,
        } => {
            if let Some(dt) = &model_dtype {
                // Metadata-only save: the API key was already retrieved from the
                // keyring by resolve_startup — re-saving it would trigger a
                // redundant keychain write prompt on macOS.
                let _ = config::save(
                    config::SaveConfig {
                        model: &m,
                        search_method: search_method.as_deref(),
                        dtype: Some(dt),
                        remote_provider: remote_provider.as_deref(),
                        remote_dim_hint,
                    },
                    None,
                );
            }
            app.model = Some(Arc::from(m.as_str()));
            if let Some(s) = &search_method {
                app.search_method = zrag_ann::SearchMethod::parse(s);
            }
            if let Some(d) = model_dtype {
                app.model_dtype = Some(Arc::from(d));
            }
            app.remote_provider = remote_provider
                .as_deref()
                .and_then(|provider| zrag_remote_embed::RemoteProvider::try_from(provider).ok());
            app.remote_api_key = remote_api_key.map(Arc::from);
            app.remote_dim_hint = remote_dim_hint;
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::RegistryLoaded(mut entries) => {
            registry::sort_by_hardware(&mut entries);
            let shared: Arc<[registry::ModelEntry]> = entries.into();
            app.setup_registry = Some(Arc::clone(&shared));
            app.screen = Screen::Setup(SetupPhase::ModelSelection {
                entries: shared,
                selected: 0,
            });
        }
        AppMessage::VariantsDetected { index, variants } => {
            // Update the live model list: derive formats from the cache-first
            // variant detection and store the precision-level variants.
            if let Screen::Setup(SetupPhase::ModelSelection { entries, .. }) =
                &mut app.screen
            {
                let mut vec: Vec<registry::ModelEntry> = entries.to_vec();
                if let Some(entry) = vec.get_mut(index) {
                    let mut seen = Vec::new();
                    for v in &variants {
                        if !seen.contains(&v.format) {
                            seen.push(v.format);
                        }
                    }
                    entry.formats = seen;
                    entry.variants = variants;
                }
                let new_entries: Arc<[registry::ModelEntry]> = Arc::from(vec);
                app.setup_registry = Some(Arc::clone(&new_entries));
                *entries = new_entries;
            }
        }
        AppMessage::RegistryError(msg) => {
            app.screen = Screen::Setup(SetupPhase::Error {
                message: msg,
                can_retry: true,
            });
        }
        AppMessage::ModelDownloaded(model_id) => {
            let _ = tx.send(AppMessage::SetupComplete { model: model_id }).await;
        }
        AppMessage::ModelDownloadError(msg) => {
            app.screen = Screen::Setup(SetupPhase::Error {
                message: msg,
                can_retry: false,
            });
        }
        AppMessage::RemoteModelsLoaded {
            provider,
            api_key,
            models,
        } => {
            // If the key matches what's already in memory, it came from the
            // keyring (or was pre-filled from it) — no need to re-store.
            let from_keyring = app
                .remote_api_key
                .as_deref()
                .is_some_and(|stored| stored == api_key.as_ref());
            app.screen = Screen::Setup(SetupPhase::RemoteModelSelection {
                provider,
                api_key,
                models: Arc::from(models),
                selected: 0,
                from_keyring,
            });
        }
        AppMessage::RemoteModelsError(msg) => {
            app.screen = Screen::Setup(SetupPhase::Error {
                message: msg,
                can_retry: true,
            });
        }
        AppMessage::SetupComplete { model } => {
            app.model = Some(Arc::clone(&model));
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(mut c) = guard.take() {
                    let _ = c.request(&zrag_protocol::request::Request::Stop).await;
                }
            });
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::ProjectRemoved => {
            app.modal = None;
            if app.selected_project < app.projects.len() {
                app.projects.remove(app.selected_project);
                if app.selected_project >= app.projects.len() && !app.projects.is_empty() {
                    app.selected_project = app.projects.len() - 1;
                }
            }
            spawn_refresh_projects(tx);
        }
        AppMessage::ProjectRemoveError(e) => {
            app.modal = Some(Modal::Error { message: e });
        }
        AppMessage::IndexComplete => {
            app.modal = None;
            spawn_refresh_projects(tx);
        }
        AppMessage::IndexProgress {
            phase,
            current,
            total,
            message,
            is_reindex,
        } => {
            let (started_at, project_root, mut files, mut chunks) = match &app.modal {
                Some(Modal::Indexing {
                    started_at,
                    project_root,
                    files,
                    chunks,
                    ..
                }) => (*started_at, project_root.clone(), *files, *chunks),
                _ => (std::time::Instant::now(), String::new(), 0, 0),
            };
            if phase == zrag_protocol::response::IndexPhase::Dsl {
                files = total;
            }
            if phase == zrag_protocol::response::IndexPhase::Tokenize {
                chunks = total;
            }
            app.modal = Some(Modal::Indexing {
                project_root,
                phase,
                current,
                total,
                message,
                is_reindex,
                started_at,
                files,
                chunks,
            });
        }
        AppMessage::IndexPaused => {
            app.modal = None;
            spawn_refresh_projects(tx);
        }
        AppMessage::IndexError(e) => {
            app.modal = Some(Modal::Error { message: e });
        }
        AppMessage::DaemonEnvLoaded {
            cpus: env_cpus,
            mem_total_mb: env_mem,
            model_dim,
        } => {
            if let DaemonStatus::Running {
                ref mut cpus,
                ref mut mem_total_mb,
                ..
            } = app.daemon_status
            {
                *cpus = env_cpus;
                *mem_total_mb = env_mem;
            }
            if app.model.as_deref().is_some_and(|model| {
                zrag_remote_embed::RemoteProvider::from_model_id(model).is_some()
            }) && model_dim > 0
                && app.remote_dim_hint != Some(model_dim as usize)
            {
                app.remote_dim_hint = Some(model_dim as usize);
                let _ = config::update_dim_hint(model_dim as usize);
            }
        }
        other => app.apply_message(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn index_progress_preserves_files_and_chunks_across_phases() {
        let mut app = App::default();
        let (tx, _rx) = mpsc::channel(4096);

        let total_code = 318u64;
        let total_chunks = 6668u64;

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Dsl,
                current: total_code,
                total: total_code,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (f, c) = modal_files_chunks(&app);
        assert_eq!(f, total_code, "Dsl should set files");
        assert_eq!(c, 0, "Dsl should not change chunks");

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Gather,
                current: total_code,
                total: total_code,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (f, c) = modal_files_chunks(&app);
        assert_eq!(f, total_code, "Gather should preserve files");
        assert_eq!(c, 0, "Gather should not set chunks");

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Start,
                current: 0,
                total: total_chunks,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (f, c) = modal_files_chunks(&app);
        assert_eq!(f, total_code, "Start should preserve files");
        assert_eq!(c, 0, "Start should not set chunks");

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Tokenize,
                current: 0,
                total: total_chunks,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (f, c) = modal_files_chunks(&app);
        assert_eq!(f, total_code, "Tokenize should preserve files");
        assert_eq!(c, total_chunks, "Tokenize should set chunks = total");

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Tokenize,
                current: 1234,
                total: total_chunks,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (_f, c) = modal_files_chunks(&app);
        assert_eq!(c, total_chunks, "Tokenize progress should keep chunks");

        dispatch(
            &mut app,
            AppMessage::IndexProgress {
                phase: zrag_protocol::response::IndexPhase::Embed,
                current: 4827,
                total: total_chunks,
                message: String::new(),
                is_reindex: false,
            },
            &tx,
        )
        .await;
        let (f, c) = modal_files_chunks(&app);
        assert_eq!(f, total_code, "Embed should preserve files");
        assert_eq!(c, total_chunks, "Embed should preserve chunks");
    }

    fn modal_files_chunks(app: &App) -> (u64, u64) {
        match &app.modal {
            Some(Modal::Indexing { files, chunks, .. }) => (*files, *chunks),
            _ => (0, 0),
        }
    }
}
