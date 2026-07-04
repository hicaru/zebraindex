use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use super::super::app::{IndexMethodButton, SetupPhase};
use super::super::registry::{ModelEntry, ModelSource, RemoteProvider};
use super::common::{centered_rect, render_bar, render_button_row, spinner_ch};

pub fn draw(f: &mut Frame, phase: &SetupPhase, tick: u16) {
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        f.area(),
    );
    match phase {
        SetupPhase::Resolving => draw_spinner(f, "Checking configuration...", tick),
        SetupPhase::FetchingRegistry => draw_spinner(f, "Downloading model catalog...", tick),
        SetupPhase::ModelSelection { entries, selected } => {
            draw_model_selection(f, entries, *selected);
        }
        SetupPhase::DTypeSelection {
            model_id, choices, selected, hw_label,
        } => {
            draw_dtype_selection(f, model_id, choices, *selected, hw_label);
        }
        SetupPhase::DownloadingModel { model_id } => draw_download(f, model_id, tick),
        SetupPhase::IndexMethodSelection {
            methods, selected, ..
        } => draw_method_selection(
            f,
            methods,
            *selected,
            None,
            false,
            false,
            IndexMethodButton::default(),
        ),
        SetupPhase::ApiKeyEntry {
            provider,
            input,
            error,
            from_keyring,
        } => draw_api_key_entry(f, *provider, input, error.as_deref(), *from_keyring),
        SetupPhase::FetchingRemoteModels { provider, .. } => {
            draw_spinner(f, &format!("Fetching {} models...", provider.label()), tick);
        }
        SetupPhase::RemoteModelSelection {
            provider,
            models,
            selected,
            ..
        } => draw_remote_model_selection(f, *provider, models, *selected),
        SetupPhase::Launching { model_id, .. } => draw_launching(f, model_id, tick),
        SetupPhase::Error {
            message, can_retry, ..
        } => draw_error(f, message, *can_retry),
    }
}

fn draw_spinner(f: &mut Frame, message: &str, tick: u16) {
    let area = centered_rect(50, 20, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebrarag setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(message),
        ]),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block);
    f.render_widget(para, area);
}

fn draw_download(f: &mut Frame, model_id: &str, tick: u16) {
    let area = centered_rect(55, 25, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebrarag setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_id, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("Downloading from HuggingFace..."),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Please wait, this may take a few minutes.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_launching(f: &mut Frame, model_id: &str, tick: u16) {
    let area = centered_rect(50, 20, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebrarag setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("Starting daemon..."),
        ]),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_id, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block);
    f.render_widget(para, area);
}

fn draw_model_selection(f: &mut Frame, entries: &[ModelEntry], selected: usize) {
    let area = centered_rect(80, 80, f.area());
    f.render_widget(Clear, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let mut items: Vec<ListItem> = Vec::with_capacity(entries.len());
    for entry in entries.iter() {
        let tag = match entry.source {
            ModelSource::Remote(_) => Span::styled(" [remote]", Style::default().fg(Color::Yellow)),
            ModelSource::Local if entry.is_downloaded() => {
                Span::styled(" [downloaded]", Style::default().fg(Color::Green))
            }
            ModelSource::Local => Span::raw(""),
        };

        // Highlight GGUF-capable models; safetensors-only (the default) stays clean.
        let format_tag = {
            use zrag_embed::model_registry::WeightsFormat;
            let quants: Vec<String> = entry
                .variants
                .iter()
                .filter(|v| v.format == WeightsFormat::Gguf)
                .map(|v| v.precision.to_uppercase())
                .collect();
            if quants.is_empty() {
                if entry.formats.contains(&WeightsFormat::Gguf) {
                    Span::styled(" [gguf]", Style::default().fg(Color::Magenta))
                } else {
                    Span::raw("")
                }
            } else {
                Span::styled(
                    format!(" [gguf: {}]", quants.join(",")),
                    Style::default().fg(Color::Magenta),
                )
            }
        };

        let line1 = Line::from(vec![
            Span::styled(&entry.model_id, Style::default()),
            Span::styled(
                format!("  ({})", entry.parameters),
                Style::default().fg(Color::DarkGray),
            ),
            tag,
            format_tag,
        ]);

        let line2 = Line::from(vec![
            Span::raw("    "),
            Span::styled(&entry.description, Style::default().fg(Color::Gray)),
        ]);

        let techs = entry.technologies.join(", ");
        let line3 = Line::from(vec![
            Span::raw("    "),
            Span::styled(techs, Style::default().fg(Color::DarkGray)),
        ]);

        items.push(ListItem::new(vec![line1, line2, line3]));
    }

    let block = Block::default()
        .title(" Select Embedding Model ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let mut state = ListState::default().with_selected(Some(selected));
    f.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_symbol("> ")
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        layout[0],
        &mut state,
    );

    let help = Paragraph::new("  j/k: navigate   Enter: select   q: quit")
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, layout[1]);
}

fn draw_dtype_selection(
    f: &mut Frame,
    model_id: &str,
    choices: &[zrag_embed::precision::DTypeChoice],
    selected: usize,
    hw_label: &str,
) {
    let area = centered_rect(70, 70, f.area());
    f.render_widget(Clear, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let mut items: Vec<ListItem> = Vec::with_capacity(choices.len());
    for (i, opt) in choices.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "> " } else { "  " };
        let style = if is_sel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let rec = if opt.recommended {
            Span::styled(" ★ recommended", Style::default().fg(Color::Green))
        } else {
            Span::raw("")
        };
        let warn_color = if opt.warning.is_some() { Color::Yellow } else { Color::DarkGray };

        let line1 = Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(opt.label.clone(), style),
            rec,
            Span::raw("  "),
            Span::styled(opt.description.clone(), Style::default().fg(Color::Gray)),
            Span::raw("  "),
            Span::styled(opt.device_tag.clone(), Style::default().fg(Color::DarkGray)),
        ]);

        let second = opt.warning.clone().unwrap_or_else(|| opt.detail.clone());
        let line2 = Line::from(vec![
            Span::raw("    "),
            Span::styled(second, Style::default().fg(warn_color)),
        ]);

        items.push(ListItem::new(vec![line1, line2]));
    }

    let block = Block::default()
        .title(format!(" Select Precision  |  {}  |  Hardware: {} ", model_id, hw_label))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let list_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(5)])
        .split(layout[0]);

    let info = Paragraph::new(vec![Line::from(Span::styled(
        "  Select precision for model weights. [CPU] = best for CPU  [GPU] = best for Metal/CUDA",
        Style::default().fg(Color::DarkGray),
    ))]);
    f.render_widget(info, list_area[0]);
    f.render_widget(List::new(items).block(block), list_area[1]);

    let help = Paragraph::new("  j/k: navigate   Enter: select   Esc: back")
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, layout[1]);
}

pub fn draw_method_selection_modal(
    f: &mut Frame,
    methods: &[(zrag_ann::SearchMethod, bool)],
    selected: usize,
    canonical_path: Option<&str>,
    already_indexed: bool,
    is_add: bool,
    selected_button: IndexMethodButton,
) {
    draw_method_selection(
        f,
        methods,
        selected,
        canonical_path,
        already_indexed,
        is_add,
        selected_button,
    );
}

fn draw_method_selection(
    f: &mut Frame,
    methods: &[(zrag_ann::SearchMethod, bool)],
    selected: usize,
    canonical_path: Option<&str>,
    already_indexed: bool,
    is_add: bool,
    selected_button: IndexMethodButton,
) {
    let area = centered_rect(90, 85, f.area());
    f.render_widget(Clear, area);

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(outer[0]);

    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if is_add {
            let header_h = if canonical_path.is_some_and(|p| p.len() > 40) {
                9u16
            } else {
                8
            };
            vec![Constraint::Length(header_h), Constraint::Min(5)]
        } else {
            vec![Constraint::Min(5)]
        })
        .split(cols[0]);

    let method_area = if is_add { left_rows[1] } else { left_rows[0] };

    if let Some(path) = canonical_path {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed("?"));

        let status = if already_indexed {
            "Already indexed (will re-index)"
        } else {
            "Not indexed"
        };

        let info_block = Block::default()
            .title(" Project ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let info_lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("  Name:   "),
                Span::styled(name, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::raw("  Path:   "),
                Span::styled(path, Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::raw("  Status: "),
                Span::styled(status, Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
        ];

        f.render_widget(
            Paragraph::new(info_lines)
                .block(info_block)
                .wrap(Wrap { trim: false }),
            left_rows[0],
        );
    }

    let mut items: Vec<ListItem> = Vec::with_capacity(methods.len());
    for (i, &(method, recommended)) in methods.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "> " } else { "  " };
        let style = if is_sel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let mut spans = vec![
            Span::styled(prefix, style),
            Span::styled(method.label(), style),
        ];
        if recommended {
            spans.push(Span::styled(" *", Style::default().fg(Color::Green)));
        }
        items.push(ListItem::new(Line::from(spans)));
    }
    let list_block = Block::default()
        .title(" Index Method ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(List::new(items).block(list_block), method_area);

    let (method, _) = methods[selected];
    let stats = method.stats();

    let detail_block = Block::default()
        .title(" Method Detail ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = detail_block.inner(cols[1]);
    f.render_widget(detail_block, cols[1]);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(6),
            Constraint::Min(6),
            Constraint::Length(4),
        ])
        .split(inner);

    let title_text = vec![
        Line::from(Span::styled(
            method.label(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            method.description(),
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(Paragraph::new(title_text), rows[0]);

    let bars = vec![
        render_bar("Accuracy", stats.accuracy),
        render_bar("Search Speed", stats.search_speed),
        render_bar("Build Speed", stats.build_speed),
        render_bar("Compression", stats.compression),
    ];
    f.render_widget(Paragraph::new(bars), rows[1]);

    let mut param_lines: Vec<Line<'_>> = Vec::with_capacity(stats.params.len());
    for &(name, value) in stats.params {
        param_lines.push(Line::from(vec![
            Span::styled(format!(" {:<14}", name), Style::default().fg(Color::White)),
            Span::styled(value, Style::default().fg(Color::Cyan)),
        ]));
    }
    let param_block = Block::default()
        .title(" Parameters ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(param_lines).block(param_block), rows[2]);

    let notes = vec![
        Line::from(vec![
            Span::styled("  Best for: ", Style::default().fg(Color::DarkGray)),
            Span::styled(stats.best_for, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Storage:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(stats.storage_note, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  RAM:      ", Style::default().fg(Color::DarkGray)),
            Span::styled(stats.ram_note, Style::default().fg(Color::White)),
        ]),
    ];
    f.render_widget(Paragraph::new(notes), rows[3]);

    if is_add {
        let buttons = render_button_row(&[
            ("Confirm", selected_button == IndexMethodButton::Confirm),
            ("Cancel", selected_button == IndexMethodButton::Cancel),
        ]);
        let help_text =
            "  j/k: navigate   Tab: switch   Enter: confirm   a: auto-recommend   Esc: back";
        let help_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Length(1)])
            .split(outer[1]);
        f.render_widget(Paragraph::new(vec![Line::from(""), buttons]), help_area[0]);
        f.render_widget(
            Paragraph::new(help_text).block(Block::default().borders(Borders::ALL)),
            help_area[1],
        );
    } else {
        let help =
            Paragraph::new("  j/k: navigate   Enter: select   a: auto-recommend   Esc: back")
                .block(Block::default().borders(Borders::ALL));
        f.render_widget(help, outer[1]);
    }
}

fn draw_error(f: &mut Frame, message: &str, can_retry: bool) {
    let area = centered_rect(50, 25, f.area());
    f.render_widget(Clear, area);

    let keys = if can_retry {
        "  r: retry   q: quit"
    } else {
        "  q: quit"
    };

    let block = Block::default()
        .title(" Error ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", message),
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
        Line::from(Span::styled(keys, Style::default().fg(Color::DarkGray))),
        Line::from(""),
    ];

    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_api_key_entry(
    f: &mut Frame,
    provider: RemoteProvider,
    input: &str,
    error: Option<&str>,
    from_keyring: bool,
) {
    let area = centered_rect(60, 35, f.area());
    f.render_widget(Clear, area);

    let has_stored = from_keyring && !input.is_empty();

    let title = if has_stored {
        format!(" {} — Stored API Key ", provider.label())
    } else {
        format!(" {} — Enter API Key ", provider.label())
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let env_hint = format!("  Tip: set {} to skip this screen", provider.env_var());
    let storage_hint = if zrag_common::secrets::available() {
        "  Key is stored in your OS keyring."
    } else {
        "  ⚠ No OS keyring available; key will be stored as plaintext locally."
    };
    let storage_style = if zrag_common::secrets::available() {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::Yellow)
    };

    let mut lines = Vec::with_capacity(12);
    lines.push(Line::from(""));

    if has_stored {
        // Show the stored key in a readable masked format:
        // first 5 chars visible, middle replaced with bullets, last 4 visible.
        let pretty = pretty_mask_key(input);
        let label = format!("  {} KEY:", provider.label().to_uppercase());
        lines.push(Line::from(vec![
            Span::styled(label, Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled(
                pretty,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Press Enter to use this key, or type a new one:",
            Style::default().fg(Color::Gray),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "  Paste your API key:",
            Style::default().fg(Color::Gray),
        )));
    }

    lines.push(Line::from(""));

    let masked_input = if input.len() > 4 {
        let tail_start = input.len().saturating_sub(4);
        format!("{}{}", "*".repeat(tail_start), &input[tail_start..])
    } else if input.is_empty() {
        String::new()
    } else {
        "*".repeat(input.len())
    };
    let cursor = Span::styled("_", Style::default().fg(Color::Yellow));
    lines.push(Line::from(vec![
        Span::raw("  > "),
        Span::styled(
            masked_input,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        cursor,
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(storage_hint, storage_style)));
    lines.push(Line::from(Span::styled(
        env_hint,
        Style::default().fg(Color::DarkGray),
    )));
    if let Some(err) = error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {err}"),
            Style::default().fg(Color::Red),
        )));
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Format a stored key for display: first 5 chars visible, middle replaced
/// with bullets, last 4 visible.  E.g. `sk-or-v1-••••••••••••••••6g45`.
fn pretty_mask_key(key: &str) -> String {
    if key.len() <= 9 {
        return "*".repeat(key.len());
    }
    let prefix = &key[..5];
    let suffix = &key[key.len().saturating_sub(4)..];
    let bullet_count = key.len().saturating_sub(9);
    format!("{prefix}{}{suffix}", "•".repeat(bullet_count))
}

fn pricing_span(pricing: &Option<zrag_remote_embed::RemoteModelPricing>) -> Span<'static> {
    match pricing {
        Some(p) if p.is_free() => Span::styled(
            "  FREE",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Some(p) => Span::styled(
            format!("  {}", p.format_price()),
            Style::default().fg(Color::Yellow),
        ),
        None => Span::raw(""),
    }
}

fn draw_remote_model_selection(
    f: &mut Frame,
    provider: RemoteProvider,
    models: &[zrag_remote_embed::RemoteModelInfo],
    selected: usize,
) {
    let area = centered_rect(80, 80, f.area());
    f.render_widget(Clear, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let mut items: Vec<ListItem> = Vec::with_capacity(models.len());
    for model in models {
        let ctx = if model.context_length > 0 {
            Span::styled(
                format!("  ({} ctx)", model.context_length),
                Style::default().fg(Color::DarkGray),
            )
        } else {
            Span::raw("")
        };
        let price = pricing_span(&model.pricing);
        let line1 = Line::from(vec![Span::styled(&model.id, Style::default()), ctx, price]);
        let line2 = Line::from(vec![
            Span::raw("    "),
            Span::styled(&model.description, Style::default().fg(Color::Gray)),
        ]);
        items.push(ListItem::new(vec![line1, line2]));
    }

    let block = Block::default()
        .title(format!(" {} — Select Embedding Model ", provider.label()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let mut state = ListState::default().with_selected(Some(selected));
    f.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_symbol("> ")
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        layout[0],
        &mut state,
    );

    let help = Paragraph::new("  j/k: navigate   Enter: select   Esc: back")
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, layout[1]);
}
