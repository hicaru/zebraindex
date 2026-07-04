use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::Mutex;
use zrag_ipc_client::Client;
use zrag_protocol::request::SearchMode;
use zrag_protocol::response::SearchResults;
use zrag_store::ProjectRow;

use super::registry::{ModelEntry, RemoteProvider};

pub const DEFAULT_DIM: usize = 768;
pub const PREVIEW_LINES: usize = 6;

pub enum Screen {
    Setup(SetupPhase),
    Main,
}

impl Default for Screen {
    fn default() -> Self {
        Self::Setup(SetupPhase::default())
    }
}

#[derive(Default)]
pub enum SetupPhase {
    #[default]
    Resolving,
    FetchingRegistry,
    ModelSelection {
        entries: Arc<[ModelEntry]>,
        selected: usize,
    },
    DownloadingModel {
        model_id: Arc<str>,
    },
    DTypeSelection {
        model_id: Arc<str>,
        choices: Arc<[zrag_embed::precision::DTypeChoice]>,
        selected: usize,
        hw_label: &'static str,
    },
    IndexMethodSelection {
        model_id: Arc<str>,
        methods: Arc<[(zrag_ann::SearchMethod, bool)]>,
        selected: usize,
    },
    ApiKeyEntry {
        provider: RemoteProvider,
        input: String,
        error: Option<String>,
        from_keyring: bool,
    },
    FetchingRemoteModels {
        provider: RemoteProvider,
        api_key: Arc<str>,
        cancel: Arc<tokio::task::AbortHandle>,
        from_keyring: bool,
    },
    RemoteModelSelection {
        provider: RemoteProvider,
        api_key: Arc<str>,
        models: Arc<[zrag_remote_embed::RemoteModelInfo]>,
        selected: usize,
        from_keyring: bool,
    },
    Launching {
        model_id: Arc<str>,
    },
    Error {
        message: String,
        can_retry: bool,
    },
}

#[derive(Default)]
pub enum DaemonStatus {
    #[default]
    Unknown,
    Starting,
    Running {
        device: String,
        uptime_secs: u64,
        cpus: u32,
        mem_total_mb: u64,
    },
    Stopped,
    Error(String),
}

#[derive(Default)]
pub enum ActivePanel {
    #[default]
    Projects,
    Search,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum DetailButton {
    #[default]
    Remove,
    Reindex,
    Back,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum IndexMethodButton {
    #[default]
    Confirm,
    Cancel,
}

pub enum Modal {
    ProjectDetail {
        selected_button: DetailButton,
    },
    ConfirmRemove,
    Error {
        message: String,
    },
    Indexing {
        project_root: String,
        phase: zrag_protocol::response::IndexPhase,
        current: u64,
        total: u64,
        message: String,
        is_reindex: bool,
        started_at: std::time::Instant,
        files: u64,
        chunks: u64,
    },
    AddProject {
        path_input: String,
        error: Option<String>,
    },
    ChangeIndexMethod {
        project_root: Option<Arc<str>>,
        canonical_path: Option<Arc<str>>,
        is_reindex: bool,
        already_indexed: Option<bool>,
        methods: Arc<[(zrag_ann::SearchMethod, bool)]>,
        selected: usize,
        selected_button: IndexMethodButton,
    },
}

pub enum AppMessage {
    DaemonStatusUpdate(DaemonStatus),
    DaemonEnvLoaded {
        cpus: u32,
        mem_total_mb: u64,
        model_dim: u32,
    },
    ProjectsLoaded(Vec<ProjectRow>),
    SearchDone(SearchResults),
    SearchError(String),
    ConfigResolved {
        model: Option<String>,
        search_method: Option<String>,
        model_dtype: Option<String>,
        remote_provider: Option<String>,
        remote_api_key: Option<String>,
        remote_dim_hint: Option<usize>,
    },
    RegistryLoaded(Vec<ModelEntry>),
    VariantsDetected {
        index: usize,
        variants: Vec<zrag_embed::model_registry::WeightVariant>,
    },
    RegistryError(String),
    ModelDownloaded(Arc<str>),
    ModelDownloadError(String),
    RemoteModelsLoaded {
        provider: RemoteProvider,
        api_key: Arc<str>,
        models: Vec<zrag_remote_embed::RemoteModelInfo>,
    },
    RemoteModelsError(String),
    SetupComplete {
        model: Arc<str>,
    },
    ProjectRemoved,
    ProjectRemoveError(String),
    IndexComplete,
    IndexPaused,
    IndexProgress {
        phase: zrag_protocol::response::IndexPhase,
        current: u64,
        total: u64,
        message: String,
        is_reindex: bool,
    },
    IndexError(String),
}

pub struct SearchInput {
    pub text: String,
    pub mode: SearchMode,
}

pub struct App {
    pub screen: Screen,
    pub setup_registry: Option<Arc<[ModelEntry]>>,
    pub daemon_status: DaemonStatus,
    pub projects: Vec<ProjectRow>,
    pub selected_project: usize,
    pub active_panel: ActivePanel,
    pub modal: Option<Modal>,
    pub search_input: SearchInput,
    pub search_results: Option<SearchResults>,
    pub search_error: Option<String>,
    pub searching: bool,
    pub results_scroll: usize,
    pub results_total_lines: usize,
    pub results_visible_height: Cell<usize>,
    pub should_quit: bool,
    pub client: Arc<Mutex<Option<Client>>>,
    pub model: Option<Arc<str>>,
    pub query_prefix: Option<Arc<str>>,
    pub passage_prefix: Option<Arc<str>>,
    pub model_dtype: Option<Arc<str>>,
    pub remote_provider: Option<RemoteProvider>,
    pub remote_api_key: Option<Arc<str>>,
    pub remote_dim_hint: Option<usize>,
    pub search_method: Option<zrag_ann::SearchMethod>,
    pub local_hardware: Option<zrag_hw::Hardware>,
    pub env_cpus: u32,
    pub env_mem_total_mb: u64,
    pub should_run: Arc<AtomicBool>,
    pub monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            screen: Screen::default(),
            setup_registry: None,
            daemon_status: DaemonStatus::default(),
            projects: Vec::with_capacity(8),
            selected_project: 0,
            active_panel: ActivePanel::default(),
            modal: None,
            search_input: SearchInput {
                text: String::with_capacity(256),
                mode: SearchMode::Query,
            },
            search_results: None,
            search_error: None,
            searching: false,
            results_scroll: 0,
            results_total_lines: 0,
            results_visible_height: Cell::new(0),
            should_quit: false,
            client: Arc::new(Mutex::new(None)),
            model: None,
            query_prefix: None,
            passage_prefix: None,
            model_dtype: None,
            remote_provider: None,
            remote_api_key: None,
            remote_dim_hint: None,
            search_method: None,
            local_hardware: None,
            env_cpus: 0,
            env_mem_total_mb: 0,
            should_run: Arc::new(AtomicBool::new(true)),
            monitor_handle: None,
        }
    }
}

impl App {
    pub fn selected_project_root(&self) -> Option<&str> {
        self.projects
            .get(self.selected_project)
            .map(|p| p.root_path.as_str())
    }

    pub fn effective_hardware(&self) -> (&str, u32, u64) {
        match &self.daemon_status {
            DaemonStatus::Running {
                device,
                cpus,
                mem_total_mb,
                ..
            } if *cpus > 0 => (device.as_str(), *cpus, *mem_total_mb),
            DaemonStatus::Running { device, .. } => {
                (device.as_str(), self.env_cpus, self.env_mem_total_mb)
            }
            _ => {
                let hw = self.local_hardware.as_ref();
                (
                    hw.map(|h| h.device.as_str()).unwrap_or("--"),
                    hw.map(|h| h.cpus as u32).unwrap_or(self.env_cpus),
                    hw.map(|h| h.mem_total / (1024 * 1024))
                        .unwrap_or(self.env_mem_total_mb),
                )
            }
        }
    }

    pub fn apply_message(&mut self, msg: AppMessage) {
        match msg {
            AppMessage::DaemonStatusUpdate(status) => self.daemon_status = status,
            AppMessage::DaemonEnvLoaded {
                cpus, mem_total_mb, ..
            } => {
                self.env_cpus = cpus;
                self.env_mem_total_mb = mem_total_mb;
            }
            AppMessage::ProjectsLoaded(projects) => {
                self.projects = projects;
                let max = self.projects.len();
                if self.selected_project > max {
                    self.selected_project = max;
                }
            }
            AppMessage::SearchDone(results) => {
                let total = 1 + results
                    .hits
                    .iter()
                    .map(|hit| {
                        let n = hit.content.lines().count();
                        1 + n.min(PREVIEW_LINES) + if n > PREVIEW_LINES { 1 } else { 0 }
                    })
                    .sum::<usize>();
                self.results_total_lines = total;
                self.search_results = Some(results);
                self.search_error = None;
                self.searching = false;
                self.results_scroll = 0;
            }
            AppMessage::SearchError(e) => {
                self.search_error = Some(e);
                self.searching = false;
            }
            _ => {}
        }
    }
}
