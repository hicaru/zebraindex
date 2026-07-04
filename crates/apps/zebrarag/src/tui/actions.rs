use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::app::{self, DEFAULT_DIM};
use super::config;
use super::event;
use super::registry::ModelSource;
use super::tasks::{
    ClientCtx, IndexMode, build_change_method_modal, cancel_index, do_index, do_remove_project,
    do_search, download_model, fetch_registry, spawn_daemon_monitor, spawn_fetch_remote_models,
};

fn spawn_reindex(app: &mut app::App, tx: &mpsc::Sender<app::AppMessage>, mode: IndexMode) {
    let Some(project) = app.projects.get(app.selected_project) else {
        return;
    };
    let root = project.root_path.clone();
    let mut ctx = ClientCtx::from_app(app);
    ctx.search_method = project
        .search_method
        .as_deref()
        .and_then(zrag_ann::SearchMethod::parse);
    app.modal = Some(app::Modal::Indexing {
        project_root: root.clone(),
        phase: zrag_protocol::response::IndexPhase::Start,
        current: 0,
        total: 0,
        message: String::with_capacity(64),
        is_reindex: true,
        started_at: std::time::Instant::now(),
        files: 0,
        chunks: 0,
    });
    let tx_c = tx.clone();
    tokio::spawn(async move {
        do_index(root, mode, ctx, tx_c).await;
    });
}

pub async fn handle_action(
    app: &mut app::App,
    action: event::Action,
    tx: &mpsc::Sender<app::AppMessage>,
) {
    match action {
        event::Action::Quit => app.should_quit = true,

        event::Action::SetupNext => {
            match &mut app.screen {
                app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected })
                    if *selected + 1 < entries.len() =>
                {
                    *selected += 1;
                }
                app::Screen::Setup(app::SetupPhase::DTypeSelection { choices, selected, .. })
                    if *selected + 1 < choices.len() =>
                {
                    *selected += 1;
                }
                app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                    methods,
                    selected,
                    ..
                }) if *selected + 1 < methods.len() => {
                    *selected += 1;
                }
                app::Screen::Setup(app::SetupPhase::RemoteModelSelection {
                    models,
                    selected,
                    ..
                }) if *selected + 1 < models.len() => {
                    *selected += 1;
                }
                _ => {}
            }
            if let Some(app::Modal::ChangeIndexMethod {
                methods, selected, ..
            }) = &mut app.modal
                && *selected + 1 < methods.len()
            {
                *selected += 1;
            }
        }
        event::Action::SetupPrev => {
            match &mut app.screen {
                app::Screen::Setup(
                    app::SetupPhase::ModelSelection { selected, .. }
                    | app::SetupPhase::DTypeSelection { selected, .. }
                    | app::SetupPhase::IndexMethodSelection { selected, .. }
                    | app::SetupPhase::RemoteModelSelection { selected, .. },
                ) if *selected > 0 => *selected -= 1,
                _ => {}
            }
            if let Some(app::Modal::ChangeIndexMethod { selected, .. }) = &mut app.modal
                && *selected > 0
            {
                *selected -= 1;
            }
        }
        event::Action::SetupAutoRecommend => {
            if let app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                methods,
                selected,
                ..
            }) = &mut app.screen
                && let Some(pos) = methods.iter().position(|(_, r)| *r)
            {
                *selected = pos;
            }
            if let Some(app::Modal::ChangeIndexMethod {
                methods, selected, ..
            }) = &mut app.modal
                && let Some(pos) = methods.iter().position(|(_, r)| *r)
            {
                *selected = pos;
            }
        }
        event::Action::SetupConfirm => match &app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected }) => {
                let Some(entry) = entries.get(*selected) else {
                    return;
                };
                match entry.source {
                    ModelSource::Remote(provider) => {
                        // Check for a stored key before asking — the key may
                        // already be in memory (from startup or a previous
                        // provider selection) or in the OS keyring.
                        let stored = app.remote_api_key.clone().or_else(|| {
                            zrag_common::secrets::retrieve(provider.as_str()).map(Arc::from)
                        });
                        if let Some(key) = stored {
                            // Key found: pre-fill the entry so the user sees it
                            // masked and can press Enter to continue or type a
                            // new key to override.
                            app.screen = app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                                provider,
                                input: key.to_string(),
                                error: None,
                                from_keyring: true,
                            });
                        } else {
                            app.screen = app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                                provider,
                                input: String::with_capacity(128),
                                error: None,
                                from_keyring: false,
                            });
                        }
                    }
                    ModelSource::Local => {
                        let model_id: Arc<str> = Arc::from(entry.model_id.as_str());
                        if entry.is_downloaded() {
                            let hw = app.local_hardware.clone().unwrap_or_default();
                            let variants = entry.variants.clone();
                            let choices: Arc<[zrag_embed::precision::DTypeChoice]> = Arc::from(
                                zrag_embed::precision::build_dtype_choices(&variants, &hw),
                            );
                            let sel = choices.iter().position(|c| c.recommended).unwrap_or(0);
                            app.screen = app::Screen::Setup(app::SetupPhase::DTypeSelection {
                                model_id,
                                choices,
                                selected: sel,
                                hw_label: hw.cpu.label(),
                            });
                        } else {
                            let id = Arc::clone(&model_id);
                            app.screen =
                                app::Screen::Setup(app::SetupPhase::DownloadingModel { model_id });
                            let tx_c = tx.clone();
                            tokio::spawn(async move { download_model(id, tx_c).await });
                        }
                    }
                }
            }
            app::Screen::Setup(app::SetupPhase::DTypeSelection { model_id, choices, selected, .. }) => {
                let dtype_label = choices[*selected].cli_value.clone();
                app.model_dtype = Some(Arc::from(dtype_label.as_str()));

                let default_hw = zrag_hw::Hardware::default();
                let hw = app.local_hardware.as_ref().unwrap_or(&default_hw);
                let max_chunks = app
                    .projects
                    .iter()
                    .map(|p| p.total_chunks as usize)
                    .max()
                    .unwrap_or(5_000);
                let recommended = zrag_ann::recommend(max_chunks, DEFAULT_DIM, hw);
                let methods: Arc<[(zrag_ann::SearchMethod, bool)]> =
                    Arc::from(zrag_ann::SearchMethod::ALL.map(|m| (m, m == recommended)));
                let rec_idx = methods.iter().position(|(_, r)| *r).unwrap_or(0);
                app.screen = app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                    model_id: Arc::clone(model_id),
                    methods,
                    selected: rec_idx,
                });
            }
            app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                model_id,
                methods,
                selected,
            }) => {
                let (method, _) = methods[*selected];
                app.search_method = Some(method);
                let save_model = Arc::clone(model_id);
                let launch_model = Arc::clone(model_id);
                let complete_model = Arc::clone(model_id);

                if let Err(e) = config::save(
                    config::SaveConfig {
                        model: &save_model,
                        search_method: Some(method.as_str()),
                        dtype: app.model_dtype.as_deref(),
                        remote_provider: None,
                        remote_dim_hint: None,
                    },
                    None,
                ) {
                    app.screen = app::Screen::Setup(app::SetupPhase::Error {
                        message: format!("Failed to save config: {e}"),
                        can_retry: false,
                    });
                    return;
                }

                app.screen = app::Screen::Setup(app::SetupPhase::Launching {
                    model_id: launch_model,
                });

                let _ = tx
                    .send(app::AppMessage::SetupComplete {
                        model: complete_model,
                    })
                    .await;
            }
            app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                provider,
                input,
                from_keyring,
                ..
            }) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    if let app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                        ref mut error, ..
                    }) = app.screen
                    {
                        *error = Some(String::from("API key cannot be empty"));
                    }
                    return;
                }
                let api_key: Arc<str> = Arc::from(trimmed);
                let provider = *provider;
                let handle = spawn_fetch_remote_models(provider, Arc::clone(&api_key), tx.clone());
                app.screen = app::Screen::Setup(app::SetupPhase::FetchingRemoteModels {
                    provider,
                    api_key,
                    cancel: Arc::new(handle.abort_handle()),
                    from_keyring: *from_keyring,
                });
            }
            app::Screen::Setup(app::SetupPhase::RemoteModelSelection {
                provider,
                api_key,
                models,
                selected,
                from_keyring,
            }) => {
                let Some(model) = models.get(*selected) else {
                    return;
                };
                let model_str = format!("{}{}", provider.model_prefix(), model.id);
                let model_arc: Arc<str> = Arc::from(model_str.as_str());
                // When the key was retrieved from the keyring (unchanged),
                // skip re-storing it — avoids a redundant keychain write prompt.
                let key_for_save = if *from_keyring {
                    None
                } else {
                    Some(api_key.as_ref())
                };
                if let Err(e) = config::save(
                    config::SaveConfig {
                        model: &model_str,
                        search_method: None,
                        dtype: None,
                        remote_provider: Some(provider.as_str()),
                        remote_dim_hint: None,
                    },
                    key_for_save,
                ) {
                    app.screen = app::Screen::Setup(app::SetupPhase::Error {
                        message: format!("Failed to save config: {e}"),
                        can_retry: false,
                    });
                    return;
                }
                app.remote_provider = Some(*provider);
                app.remote_api_key = Some(Arc::clone(api_key));
                app.remote_dim_hint = None;
                app.screen = app::Screen::Setup(app::SetupPhase::Launching {
                    model_id: Arc::clone(&model_arc),
                });
                let _ = tx
                    .send(app::AppMessage::SetupComplete { model: model_arc })
                    .await;
            }
            _ => {}
        },
        event::Action::SetupBack => match &app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { .. }) => {
                if app.model.is_some() {
                    app.should_run.store(true, Ordering::Relaxed);
                    app.screen = app::Screen::Main;
                    spawn_daemon_monitor(app, tx);
                } else {
                    app.should_quit = true;
                }
            }
            app::Screen::Setup(
                app::SetupPhase::DTypeSelection { .. } | app::SetupPhase::ApiKeyEntry { .. },
            ) => {
                if let Some(entries) = app.setup_registry.clone() {
                    app.screen = app::Screen::Setup(app::SetupPhase::ModelSelection {
                        entries,
                        selected: 0,
                    });
                } else {
                    app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
                    let tx_c = tx.clone();
                    tokio::spawn(async move { fetch_registry(tx_c).await });
                }
            }
            app::Screen::Setup(app::SetupPhase::FetchingRemoteModels {
                cancel,
                provider,
                api_key,
                from_keyring,
            }) => {
                cancel.abort();
                app.screen = app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                    provider: *provider,
                    input: api_key.to_string(),
                    error: None,
                    from_keyring: *from_keyring,
                });
            }
            app::Screen::Setup(app::SetupPhase::RemoteModelSelection {
                provider,
                api_key,
                from_keyring,
                ..
            }) => {
                app.screen = app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                    provider: *provider,
                    input: api_key.to_string(),
                    error: None,
                    from_keyring: *from_keyring,
                });
            }
            _ => {}
        },
        event::Action::SetupRetry => {
            app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }

        event::Action::SwitchPanel => {
            app.active_panel = match app.active_panel {
                app::ActivePanel::Projects => app::ActivePanel::Search,
                app::ActivePanel::Search => app::ActivePanel::Projects,
            };
        }
        event::Action::FocusSearch => {
            app.active_panel = app::ActivePanel::Search;
        }
        event::Action::ToggleSearchMode => {
            app.search_input.mode = match app.search_input.mode {
                zrag_protocol::request::SearchMode::Query => {
                    zrag_protocol::request::SearchMode::Passage
                }
                zrag_protocol::request::SearchMode::Passage => {
                    zrag_protocol::request::SearchMode::Query
                }
            };
        }
        event::Action::SelectPrevProject => {
            if app.selected_project > 0 {
                app.selected_project -= 1;
            }
        }
        event::Action::SelectNextProject => {
            if app.selected_project < app.projects.len() {
                app.selected_project += 1;
            }
        }
        event::Action::ScrollUp => {
            if app.results_scroll > 0 {
                app.results_scroll -= 1;
            }
        }
        event::Action::ScrollDown => {
            let vis = app.results_visible_height.get();
            let max = app.results_total_lines.saturating_sub(vis);
            app.results_scroll = (app.results_scroll + 1).min(max);
        }
        event::Action::PageUp => {
            let step = app.results_visible_height.get().max(1);
            app.results_scroll = app.results_scroll.saturating_sub(step);
        }
        event::Action::PageDown => {
            let vis = app.results_visible_height.get();
            let max = app.results_total_lines.saturating_sub(vis);
            let step = vis.max(1);
            app.results_scroll = (app.results_scroll + step).min(max);
        }
        event::Action::Input(c) => {
            if let app::Screen::Setup(app::SetupPhase::ApiKeyEntry {
                ref mut input,
                ref mut error,
                ..
            }) = app.screen
            {
                input.push(c);
                *error = None;
            } else if let Some(app::Modal::AddProject {
                ref mut path_input,
                ref mut error,
                ..
            }) = app.modal
            {
                path_input.push(c);
                *error = None;
            } else {
                app.search_input.text.push(c);
            }
        }
        event::Action::Backspace => {
            if let app::Screen::Setup(app::SetupPhase::ApiKeyEntry { ref mut input, .. }) =
                app.screen
            {
                input.pop();
            } else if let Some(app::Modal::AddProject {
                ref mut path_input, ..
            }) = app.modal
            {
                path_input.pop();
            } else {
                app.search_input.text.pop();
            }
        }
        event::Action::SubmitSearch => {
            if app.search_input.text.is_empty() || app.searching {
                return;
            }
            let mode = app.search_input.mode;
            let query = std::mem::take(&mut app.search_input.text);
            app.searching = true;
            app.search_error = None;
            let root = app.selected_project_root().map(|s| s.to_string());
            let ctx = ClientCtx::from_app(app);
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                do_search(query, mode, root, ctx, tx_clone).await;
            });
        }
        event::Action::StopDaemon => {
            app.should_run.store(false, Ordering::Relaxed);
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(mut c) = guard.take() {
                    let _ = c.request(&zrag_protocol::request::Request::Stop).await;
                }
            });
        }
        event::Action::RestartDaemon => {
            app.should_run.store(true, Ordering::Relaxed);
            {
                let mut guard = app.client.lock().await;
                *guard = None;
            }
            app.daemon_status = app::DaemonStatus::Starting;
        }
        event::Action::ChangeModel => {
            if let Some(handle) = app.monitor_handle.take() {
                handle.abort();
            }
            app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }

        event::Action::OpenProjectDetail => {
            if app.selected_project == app.projects.len() {
                app.modal = Some(app::Modal::AddProject {
                    path_input: String::with_capacity(256),
                    error: None,
                });
            } else if !app.projects.is_empty() {
                app.modal = Some(app::Modal::ProjectDetail {
                    selected_button: app::DetailButton::default(),
                });
            }
        }
        event::Action::DetailButtonNext => match &mut app.modal {
            Some(app::Modal::ProjectDetail { selected_button }) => {
                *selected_button = match selected_button {
                    app::DetailButton::Remove => app::DetailButton::Reindex,
                    app::DetailButton::Reindex => app::DetailButton::Back,
                    app::DetailButton::Back => app::DetailButton::Remove,
                };
            }
            Some(app::Modal::ChangeIndexMethod {
                selected_button,
                canonical_path,
                ..
            }) if canonical_path.is_some() => {
                *selected_button = match selected_button {
                    app::IndexMethodButton::Confirm => app::IndexMethodButton::Cancel,
                    app::IndexMethodButton::Cancel => app::IndexMethodButton::Confirm,
                };
            }
            _ => {}
        },
        event::Action::DetailButtonPrev => match &mut app.modal {
            Some(app::Modal::ProjectDetail { selected_button }) => {
                *selected_button = match selected_button {
                    app::DetailButton::Remove => app::DetailButton::Back,
                    app::DetailButton::Reindex => app::DetailButton::Remove,
                    app::DetailButton::Back => app::DetailButton::Reindex,
                };
            }
            Some(app::Modal::ChangeIndexMethod {
                selected_button,
                canonical_path,
                ..
            }) if canonical_path.is_some() => {
                *selected_button = match selected_button {
                    app::IndexMethodButton::Confirm => app::IndexMethodButton::Cancel,
                    app::IndexMethodButton::Cancel => app::IndexMethodButton::Confirm,
                };
            }
            _ => {}
        },
        event::Action::DetailConfirm => match app.modal.take() {
            Some(app::Modal::ProjectDetail { selected_button }) => match selected_button {
                app::DetailButton::Remove => {
                    app.modal = Some(app::Modal::ConfirmRemove);
                }
                app::DetailButton::Reindex => {
                    spawn_reindex(app, tx, IndexMode::Reindex);
                }
                app::DetailButton::Back => {}
            },
            Some(app::Modal::ChangeIndexMethod {
                project_root,
                canonical_path,
                is_reindex,
                methods,
                selected,
                selected_button,
                ..
            }) => {
                if let Some(cp) = &canonical_path {
                    match selected_button {
                        app::IndexMethodButton::Cancel => {
                            app.modal = Some(app::Modal::AddProject {
                                path_input: cp.to_string(),
                                error: None,
                            });
                            return;
                        }
                        app::IndexMethodButton::Confirm => {}
                    }
                }
                let (method, _) = methods[selected];
                app.search_method = Some(method);
                if let Err(e) = config::update_search_method(method.as_str()) {
                    app.modal = Some(app::Modal::Error {
                        message: format!("Failed to save config: {e}"),
                    });
                    return;
                }
                if let Some(root) = &project_root {
                    let root_s = root.to_string();
                    app.modal = Some(app::Modal::Indexing {
                        project_root: root_s.clone(),
                        phase: zrag_protocol::response::IndexPhase::Start,
                        current: 0,
                        total: 0,
                        message: String::with_capacity(64),
                        is_reindex,
                        started_at: std::time::Instant::now(),
                        files: 0,
                        chunks: 0,
                    });
                    let ctx = ClientCtx::from_app(app);
                    let tx_c = tx.clone();
                    let mode = if is_reindex {
                        IndexMode::ForceReindex
                    } else {
                        IndexMode::Initial
                    };
                    tokio::spawn(async move {
                        do_index(root_s, mode, ctx, tx_c).await;
                    });
                }
            }
            other => {
                app.modal = other;
            }
        },
        event::Action::DetailBack => {
            app.modal = None;
        }
        event::Action::DetailForceReindex => {
            if matches!(app.modal, Some(app::Modal::ProjectDetail { .. })) {
                spawn_reindex(app, tx, IndexMode::ForceReindex);
            }
        }
        event::Action::ConfirmRemoveYes => {
            app.modal = None;
            if let Some(p) = app.projects.get(app.selected_project)
                && let Ok(pid) = <[u8; 32]>::try_from(p.project_id.as_slice())
            {
                let root = p.root_path.clone();
                let ctx = ClientCtx::from_app(app);
                let tx_c = tx.clone();
                tokio::spawn(async move {
                    do_remove_project(root, pid, ctx, tx_c).await;
                });
            }
        }
        event::Action::ConfirmRemoveNo => {
            if app.projects.get(app.selected_project).is_some() {
                app.modal = Some(app::Modal::ProjectDetail {
                    selected_button: app::DetailButton::Remove,
                });
            } else {
                app.modal = None;
            }
        }
        event::Action::SubmitPath => {
            if let Some(app::Modal::AddProject { ref path_input, .. }) = app.modal {
                let trimmed = path_input.trim();
                if trimmed.is_empty() {
                    if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                        *error = Some(String::from("Path cannot be empty"));
                    }
                } else {
                    let path = std::path::Path::new(trimmed);
                    if !path.is_dir() {
                        if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                            *error = Some(String::from("Directory does not exist"));
                        }
                    } else if let Ok(canonical) = path.canonicalize() {
                        let canonical_path = canonical.to_string_lossy().into_owned();
                        let already_indexed =
                            app.projects.iter().any(|p| p.root_path == canonical_path);
                        let canonical_arc: Arc<str> = Arc::from(canonical_path);
                        let default_hw = zrag_hw::Hardware::default();
                        let hw = app.local_hardware.as_ref().unwrap_or(&default_hw);
                        app.modal = Some(build_change_method_modal(
                            Some(Arc::clone(&canonical_arc)),
                            Some(canonical_arc),
                            false,
                            Some(already_indexed),
                            app.search_method,
                            &app.projects,
                            hw,
                        ));
                    } else if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                        *error = Some(String::from("Cannot resolve path"));
                    }
                }
            }
        }
        event::Action::CancelIndex => {
            let root = match &app.modal {
                Some(app::Modal::Indexing { project_root, .. }) => Some(project_root.clone()),
                _ => None,
            };
            if let Some(project_root) = root {
                let ctx = ClientCtx::from_app(app);
                tokio::spawn(async move {
                    cancel_index(project_root, ctx).await;
                });
            }
        }
        event::Action::None => {}
    }
}
