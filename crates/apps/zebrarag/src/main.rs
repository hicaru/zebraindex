use std::borrow::Cow;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
#[cfg(feature = "tokio-console")]
use tracing_subscriber::prelude::*;

#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cli;
mod dsl;
mod mcp;
mod tui;

#[derive(Parser)]
#[command(name = "zebrarag", version, about = "Zebra semantic code indexer")]
struct TopLevel {
    #[arg(long, help = "Run as MCP server (stdio)")]
    mcp: bool,

    #[arg(short, long, global = true)]
    model: Option<String>,

    #[arg(long, global = true)]
    query_prefix: Option<String>,

    #[arg(long, global = true)]
    passage_prefix: Option<String>,

    #[arg(long, global = true)]
    model_dtype: Option<String>,

    #[command(subcommand)]
    command: Option<TopCommand>,
}

#[derive(Subcommand)]
enum TopCommand {
    #[command(about = "Run the daemon process")]
    Daemon {
        #[arg(short, long)]
        model: String,
        #[arg(long)]
        query_prefix: Option<String>,
        #[arg(long)]
        passage_prefix: Option<String>,
        #[arg(long)]
        model_dtype: Option<String>,
    },
    #[command(about = "DSL graph dump for debugging")]
    Dsl {
        #[command(subcommand)]
        command: dsl::DslCommands,
        #[arg(short, long, help = "Project root path")]
        root: std::path::PathBuf,
    },
    #[command(flatten)]
    Cli(cli::CliCommand),
}

fn env_filter(default_level: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level))
}

fn cap_log_bridge() {
    // Installing a tracing subscriber also installs the global LogTracer with
    // no max-level cap, so every dependency `log!` is bridged + filtered at
    // runtime. Cap it so warn+ records from dependencies are dropped before
    // reaching the tracing dispatcher.
    log::set_max_level(log::LevelFilter::Warn);
}

fn init_tracing(default_level: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter(default_level))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();
    cap_log_bridge();
}

#[cfg(feature = "tokio-console")]
fn init_tokio_console_tracing(default_level: &str) {
    let console_default = format!("{default_level},tokio=trace,runtime=trace");
    let console_layer = console_subscriber::ConsoleLayer::builder().spawn();

    let _ = tracing_subscriber::registry()
        .with(env_filter(&console_default))
        .with(console_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(false),
        )
        .try_init();
    cap_log_bridge();
}

fn main() -> Result<()> {
    let top = TopLevel::parse();

    if top.mcp {
        init_tracing("warn");
        return mcp::run_mcp();
    }

    let model = top.model.as_deref();
    let query_prefix = top.query_prefix.as_deref();
    let passage_prefix = top.passage_prefix.as_deref();
    let model_dtype = top.model_dtype.as_deref();

    match top.command {
        None => {
            init_tracing("warn");
            tui::run_tui(model, query_prefix, passage_prefix, model_dtype)
        }
        Some(TopCommand::Daemon {
            model,
            query_prefix,
            passage_prefix,
            model_dtype,
        }) => {
            #[cfg(feature = "tokio-console")]
            init_tokio_console_tracing(zrag_daemon::DAEMON_DEFAULT_LOG_FILTER);

            // Resolve the key from the provider env var first, then the OS
            // keyring (where the TUI persists it once at setup) so launching the
            // daemon directly works without re-supplying the key.
            let remote_key = zrag_remote_embed::RemoteProvider::from_model_id(&model).and_then(
                |(provider, _)| {
                    std::env::var(provider.env_var())
                        .ok()
                        .or_else(|| zrag_common::secrets::retrieve(provider.as_str()))
                },
            );
            let env_remote_dim_hint = std::env::var("ZEBRA_REMOTE_DIM_HINT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok());
            let remote_api_key = remote_key.as_deref();
            let config = zrag_daemon::DaemonConfig {
                model: Cow::Owned(model),
                query_prefix: query_prefix.as_deref(),
                passage_prefix: passage_prefix.as_deref(),
                model_dtype: model_dtype.as_deref(),
                remote_api_key,
                remote_dim_hint: env_remote_dim_hint,
            };
            zrag_daemon::run_daemon(&config)
        }
        Some(TopCommand::Dsl { command, root }) => {
            init_tracing("warn");
            dsl::run_dsl(&root, command)
        }
        Some(TopCommand::Cli(cmd)) => {
            init_tracing("warn");
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cli::run(
                cmd,
                model,
                query_prefix,
                passage_prefix,
                model_dtype,
            ))
        }
    }
}
