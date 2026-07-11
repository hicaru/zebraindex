use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};

use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use arrow::array::{
    BinaryBuilder, FixedSizeBinaryBuilder, ListBuilder, RecordBatch, StringArray, StringBuilder,
    UInt8Array, UInt32Array, UInt32Builder, UInt64Array,
};
use rayon::prelude::*;
use tokio::sync::oneshot;
use tracing::info;

use zrag_common::ids::project_id;
use zrag_dsl::chunking::ChunkStrategy;
use zrag_dsl::{Chunk, DslChunker, EdgeKind, SourceFile, Target, build_index_from_sources};
use zrag_embed::{AnyEmbedEngine, Pooled};
use zrag_recursive_chunk;
use zrag_rerank::TurboReranker;
use zrag_store::Db;
use zrag_tree_sitter::{Language, frontend_for};
use zrag_ts_core::walker::LanguageFrontend;

const APPENDIX_DEPTH: usize = 2;
const APPENDIX_CAP_PER_CHUNK: usize = 32;
const CHUNK_OVERLAP: usize = 200;
const MIN_CHUNK_FLOOR: usize = 512;

use crate::manifest::{FileSnapshot, SourceKind, detect_changes, walk_source_files};
use crate::pdf_chunk::pack_pdf_pages;
use crate::progress::{ProgressReporter, Reporter};

fn content_chunk_id(chunk: &Chunk<'_>) -> [u8; 16] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(chunk.file.as_bytes());
    hasher.update(chunk.qualified.as_bytes());
    hasher.update(&chunk.sub_chunk_idx.to_le_bytes());
    hasher.update(chunk.body.as_bytes());
    let hash = hasher.finalize();
    let mut chunk_id = [0u8; 16];
    chunk_id.copy_from_slice(&hash.as_bytes()[..16]);
    chunk_id
}

#[derive(Debug, Clone, Copy)]
struct AdaptiveChunkSizing {
    chunk_size: usize,
    min_chunk_size: usize,
}

/// Pure byte-sizing math: given a measured bytes-per-token ratio, decide whether
/// `body_len` bytes exceed the model's token budget and, if so, the byte limits
/// for recursive splitting. Split when the estimated token count
/// (`body_len / bpt`) exceeds `max_len`, i.e. `body_len > max_len * bpt`.
fn sizing_for(body_len: usize, max_len: usize, bpt: usize) -> Option<AdaptiveChunkSizing> {
    let chunk_size = max_len.saturating_mul(bpt);
    (body_len > chunk_size).then(|| AdaptiveChunkSizing {
        chunk_size: chunk_size.max(MIN_CHUNK_FLOOR + 1),
        min_chunk_size: (max_len / 2).saturating_mul(bpt).max(MIN_CHUNK_FLOOR),
    })
}

/// `Some(sizing)` when the body should be recursively split; `None` keeps it whole.
fn adaptive_split(body: &str, engine: &AnyEmbedEngine) -> Option<AdaptiveChunkSizing> {
    let max_len = engine.chunk_max_tokens();

    // Fast path: bytes ≤ max_len → tokens ≤ bytes ≤ max_len → always fits.
    if body.len() <= max_len {
        return None;
    }

    sizing_for(body.len(), max_len, engine.chars_per_token())
}

#[inline]
fn generate_sub_chunks<'a>(
    chunk: &Chunk<'a>,
    sizing: &AdaptiveChunkSizing,
    lang: Option<&tree_sitter::Language>,
    kind_label: &'static str,
    out: &mut Vec<(Chunk<'a>, &'static str, u32)>,
    terminal_kinds: &[u16],
    fidx: u32,
) {
    let sub_chunks = zrag_recursive_chunk::split_text(
        &chunk.body,
        &zrag_recursive_chunk::ChunkConfig {
            chunk_size: sizing.chunk_size,
            min_chunk_size: sizing.min_chunk_size,
            chunk_overlap: CHUNK_OVERLAP,
        },
        lang,
        terminal_kinds,
    );
    let total = sub_chunks.len() as u32;
    for (i, sub) in sub_chunks.iter().enumerate() {
        let body: Cow<'a, str> = match &chunk.body {
            Cow::Borrowed(s) => Cow::Borrowed(&s[sub.byte_start..sub.byte_end]),
            Cow::Owned(s) => Cow::Owned(s[sub.byte_start..sub.byte_end].to_string()),
        };
        let sc = Chunk {
            file: Arc::clone(&chunk.file),
            rel_file: Arc::clone(&chunk.rel_file),
            start_line: chunk.start_line + sub.start_line - 1,
            end_line: chunk.start_line + sub.end_line - 1,
            sym_id: chunk.sym_id,
            sub_chunk_idx: i as u32,
            total_sub_chunks: total,
            chunk_strategy: ChunkStrategy::Recursive,
            body,
            qualified: chunk.qualified.clone(),
            kind: chunk.kind,
        };
        out.push((sc, kind_label, fidx));
    }
}

pub struct IndexStats {
    pub total_chunks: usize,
    pub total_files: usize,
    pub new_chunks: usize,
    pub reindexed_files: usize,
    pub duration_ms: u64,
    pub paused: bool,
}

/// Absolute path as stored in the chunks table for a project-relative file.
///
/// The files table keys rows by relative path (`src/foo.rs`); chunks store the
/// absolute path (`/abs/root/src/foo.rs`) via the same `root.join(rel).display()`
/// construction used at insert time. Deletes must use this form or no rows match.
#[inline]
fn chunk_file_path(root: &Path, rel: &str) -> String {
    root.join(rel).display().to_string()
}

pub async fn index_project(
    root: &Path,
    engine: &AnyEmbedEngine,
    db: &Db,
    reporter: &Reporter,
    override_method: Option<zrag_ann::SearchMethod>,
    cancel: &AtomicBool,
    refresh: bool,
) -> Result<IndexStats> {
    let start = std::time::Instant::now();
    let pid = project_id(root);

    let root_str = root.to_string_lossy();
    info!("indexing {}", root_str);

    let phase_start = std::time::Instant::now();
    let snapshots = tokio::task::block_in_place(|| walk_source_files(root));
    info!(
        files = snapshots.len(),
        ms = phase_start.elapsed().as_millis() as u64,
        "walk_source_files"
    );

    let files_table = db.files_table().await?;
    let previous = files_table.list().await.unwrap_or_default();

    let changes = detect_changes(&snapshots, &previous);
    info!(
        "changes: {} added, {} modified, {} removed, {} unchanged",
        changes.added.len(),
        changes.modified.len(),
        changes.removed.len(),
        changes.unchanged.len(),
    );

    let mut chunks_table = db.chunks_table(engine.dim()).await?;
    let chunks_len = chunks_table.len().await?;
    let project_row = db.projects_table().await?.get(&pid).await?;
    let stale_index_version = project_row
        .as_ref()
        .is_some_and(|row| row.index_version < zrag_store::projects_table::INDEX_FORMAT_VERSION);
    let missing_project_row_with_chunks = project_row.is_none() && chunks_len > 0;

    let force_rebuild = refresh
        || stale_index_version
        || missing_project_row_with_chunks
        || (!previous.is_empty() && chunks_len == 0);

    if force_rebuild && !refresh {
        if stale_index_version || missing_project_row_with_chunks {
            info!("index format changed, forcing full reindex");
        } else {
            info!("self-heal: empty index detected, forcing full reindex");
        }
    }

    let need_reindex: Vec<String>;
    // Relative paths for the files table (and change detection keys).
    let rel_delete: Vec<&str>;

    if force_rebuild {
        need_reindex = snapshots.keys().cloned().collect();
        rel_delete = previous.iter().map(|r| r.file_path.as_str()).collect();
    } else {
        need_reindex = changes
            .added
            .iter()
            .chain(changes.modified.iter())
            .cloned()
            .collect();
        // Delete prior chunks for EVERY file we're about to reindex (added too),
        // so a resumed run can't duplicate chunks left behind by a paused run.
        rel_delete = changes
            .removed
            .iter()
            .map(String::as_str)
            .chain(need_reindex.iter().map(String::as_str))
            .collect();
    }

    if !rel_delete.is_empty() {
        // chunks.file_path is absolute; files.file_path is relative.
        let abs_delete: Vec<String> = rel_delete
            .iter()
            .map(|rel| chunk_file_path(root, rel))
            .collect();
        let abs_delete_refs: Vec<&str> = abs_delete.iter().map(String::as_str).collect();
        chunks_table.delete_for_files(&abs_delete_refs).await?;
        files_table.delete_for_paths(&rel_delete).await?;
        info!(
            files = rel_delete.len(),
            "deleted file rows + chunks for paths"
        );
    }

    if need_reindex.is_empty() {
        // True no-op: nothing removed and nothing to embed.
        if rel_delete.is_empty() {
            reporter.finish_with_message("nothing to reindex");
            let elapsed = start.elapsed();
            return Ok(IndexStats {
                total_chunks: chunks_table.len().await.unwrap_or(0),
                total_files: snapshots.len(),
                new_chunks: 0,
                reindexed_files: 0,
                duration_ms: elapsed.as_millis() as u64,
                paused: false,
            });
        }

        // Removals only — chunks already deleted above; refresh project metadata
        // so total_chunks / total_files stay honest for search overfetch.
        let total_in_db = chunks_table.len().await?;
        if total_in_db > 0 {
            let _ = chunks_table.optimize().await;
        }

        let fallback_hw;
        let hw = if let Some(hw) = engine.hardware() {
            hw
        } else {
            fallback_hw = zrag_hw::probe_effective();
            &fallback_hw
        };
        let previous_params: Option<zrag_ann::SearchParams> = project_row
            .as_ref()
            .and_then(|r| r.search_params.as_deref())
            .and_then(|s| toml::from_str(s).ok());
        let mut params =
            zrag_ann::choose_method(total_in_db, engine.dim(), hw, previous_params.as_ref());
        if let Some(m) = override_method {
            params.method = m;
        }

        let languages: HashSet<Language> = snapshots
            .values()
            .filter_map(|s| match s.kind {
                SourceKind::Code(l) => Some(l),
                SourceKind::Tsv | SourceKind::Psv | SourceKind::Text | SourceKind::Pdf => None,
            })
            .collect();
        let languages: Vec<&Language> = languages.iter().collect();
        upsert_project(
            db,
            &pid,
            root_str,
            total_in_db,
            snapshots.len(),
            &languages,
            engine,
            &params,
        )
        .await?;

        info!(
            removed = changes.removed.len(),
            remaining_files = snapshots.len(),
            remaining_chunks = total_in_db,
            "index: removals applied, no files to re-embed"
        );
        reporter.finish_with_message(&format!(
            "removed {} file(s), {} passages remain",
            changes.removed.len(),
            total_in_db
        ));
        let elapsed = start.elapsed();
        return Ok(IndexStats {
            total_chunks: total_in_db,
            total_files: snapshots.len(),
            new_chunks: 0,
            reindexed_files: 0,
            duration_ms: elapsed.as_millis() as u64,
            paused: false,
        });
    }

    // Single FS walk: reuse the snapshots we already loaded to drive the DSL
    // parser. Avoids walking the tree (and re-reading every file) a second
    // time inside `zrag_dsl::build_index`. Text-kind snapshots have no
    // tree-sitter frontend, so they're filtered out here and chunked
    // separately below.
    let phase_start = std::time::Instant::now();
    let total_code = snapshots
        .values()
        .filter(|s| matches!(s.kind, SourceKind::Code(_)))
        .count() as u64;
    reporter.set_phase(
        zrag_protocol::response::IndexPhase::Dsl,
        0,
        total_code,
        "parsing code files",
    );
    let dsl_sources = snapshots.iter().filter_map(|(rel, snap)| match snap.kind {
        SourceKind::Code(lang) => Some(SourceFile {
            full_path: root.join(rel).display().to_string(),
            content: snap.contents.as_str(),
            language: lang,
        }),
        SourceKind::Tsv | SourceKind::Psv | SourceKind::Text | SourceKind::Pdf => None,
    });
    let dsl_index = tokio::task::block_in_place(|| {
        build_index_from_sources(root_str.to_string(), dsl_sources, |processed| {
            reporter.set_phase(
                zrag_protocol::response::IndexPhase::Dsl,
                processed as u64,
                total_code,
                "parsing code files",
            );
        })
    });
    info!(
        symbols = dsl_index.symbols.len(),
        edges = dsl_index.edges.len(),
        files = dsl_index.files.len(),
        ms = phase_start.elapsed().as_millis() as u64,
        "dsl parse"
    );
    reporter.set_phase(
        zrag_protocol::response::IndexPhase::Gather,
        dsl_index.files.len() as u64,
        dsl_index.files.len() as u64,
        "generating chunks",
    );
    let chunker = DslChunker::new(&dsl_index);
    info!(
        "dsl-chunker created, building terminal_cache for {} files",
        dsl_index.files.len()
    );

    let mut terminal_cache: FxHashMap<Language, Vec<u16>> =
        FxHashMap::with_capacity_and_hasher(4, rustc_hash::FxBuildHasher);
    for lang in dsl_index.files.iter().map(|f| f.language) {
        if terminal_cache.contains_key(&lang) {
            continue;
        }
        let frontend = frontend_for(lang);
        let ts_lang = frontend.language();
        let names = frontend.config().terminal_node_kinds;
        let mut ids = Vec::with_capacity(names.len());
        for name in names {
            let id = ts_lang.id_for_node_kind(name, true);
            if id != 0 {
                ids.push(id);
            }
        }
        terminal_cache.insert(lang, ids);
    }
    info!(
        "terminal_cache built with {} languages, starting chunk generation for {} files",
        terminal_cache.len(),
        need_reindex.len()
    );

    let phase_start = std::time::Instant::now();
    // TODO(perf): replace eager all_pending Vec with bounded streaming.
    //
    // Plan: use crossbeam_channel::bounded(2 * batch_size) to stream chunks
    // from a producer thread (rayon par_iter flat_map → tx.send) to the
    // consumer loop.  The consumer collects chunks into a local Vec and builds
    // remaining / chunk_sym_set / appendix_map after the producer drains.
    //
    // Deeper refactor (larger win): compute appendix BFS lazily per-sym_id
    // instead of precomputing for all sym_ids, then the embed loop can start
    // as soon as the first batch of chunks arrives — no need to materialize
    // every chunk before embedding.  Requires: cache appendix results in a
    // FxHashMap that fills on first access per sym_id.
    let all_pending: Vec<(Chunk<'_>, &'static str, u32)> = tokio::task::block_in_place(|| {
        need_reindex
            .par_iter()
            .enumerate()
            .flat_map(|(fidx, rel)| {
                let fidx = fidx as u32;
                let snap = match snapshots.get(rel) {
                    Some(s) => s,
                    None => return Vec::new(),
                };
                match snap.kind {
                    SourceKind::Code(lang) => {
                        let full_path = root.join(rel);
                        let label = full_path.display().to_string();
                        let chunks = chunker.chunks_for_file(&label, &snap.contents);
                        let frontend = frontend_for(lang);
                        let ts_lang = frontend.language();
                        let terminal_ids = terminal_cache
                            .get(&lang)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        let mut out = Vec::with_capacity(chunks.len());
                        for c in chunks {
                            match adaptive_split(&c.body, engine) {
                                Some(sizing) => generate_sub_chunks(
                                    &c,
                                    &sizing,
                                    Some(&ts_lang),
                                    lang.as_str(),
                                    &mut out,
                                    terminal_ids,
                                    fidx,
                                ),
                                None => out.push((c, lang.as_str(), fidx)),
                            }
                        }
                        out
                    }
                    SourceKind::Tsv | SourceKind::Psv => {
                        let full_path = root.join(rel).display().to_string();
                        // Pack rows up to the same byte budget `adaptive_split` uses
                        // so multi-row chunks fit the model and aren't re-split.
                        let budget = engine.chunk_max_tokens().saturating_mul(engine.chars_per_token());
                        let rows = zrag_dsl::chunking::chunk_tabular_file(
                            rel,
                            &full_path,
                            &snap.contents,
                            budget,
                        );
                        let label = snap.kind.label();
                        let mut out = Vec::with_capacity(rows.len());
                        for chunk in rows {
                            match adaptive_split(&chunk.body, engine) {
                                Some(sizing) => generate_sub_chunks(
                                    &chunk,
                                    &sizing,
                                    None,
                                    label,
                                    &mut out,
                                    &[],
                                    fidx,
                                ),
                                None => out.push((chunk, label, fidx)),
                            }
                        }
                        out
                    }
                    SourceKind::Text | SourceKind::Pdf => {
                        let full_path = root.join(rel).display().to_string();
                        let label = snap.kind.label();
                        // PDF: page-aware packing with heading-aware boundaries.
                        // Text: one whole-file Document chunk (existing behaviour).
                        let budget = engine.chunk_max_tokens().saturating_mul(engine.chars_per_token());
                        let packed: Vec<Chunk<'_>> = match &snap.pdf_pages {
                            Some(metas) => {
                                pack_pdf_pages(rel, &full_path, &snap.contents, metas, budget)
                            }
                            None => vec![zrag_dsl::chunking::chunk_text_file(
                                rel,
                                &full_path,
                                &snap.contents,
                            )],
                        };
                        let mut out = Vec::with_capacity(packed.len());
                        for chunk in packed {
                            match adaptive_split(&chunk.body, engine) {
                                Some(sizing) => generate_sub_chunks(
                                    &chunk,
                                    &sizing,
                                    None,
                                    label,
                                    &mut out,
                                    &[],
                                    fidx,
                                ),
                                None => out.push((chunk, label, fidx)),
                            }
                        }
                        out
                    }
                }
            })
            .collect()
    });

    info!(
        chunks = all_pending.len(),
        files = need_reindex.len(),
        ms = phase_start.elapsed().as_millis() as u64,
        "chunking"
    );

    let mut remaining: Vec<u32> = vec![0u32; need_reindex.len()];
    for (_, _, fidx) in &all_pending {
        if let Some(slot) = remaining.get_mut(*fidx as usize) {
            *slot = slot.saturating_add(1);
        }
    }

    // Files that produced no chunks (empty / no symbols) are "done"
    // immediately — record them now so they aren't re-walked on every resume.
    let zero_chunk: Vec<&str> = need_reindex
        .iter()
        .enumerate()
        .filter(|(i, _)| remaining.get(*i).copied().unwrap_or(0) == 0)
        .map(|(_, p)| p.as_str())
        .collect();
    if !zero_chunk.is_empty() {
        upsert_files(&files_table, &snapshots, &zero_chunk).await?;
    }

    info!("building chunk_sym_set from {} chunks", all_pending.len());
    let chunk_sym_set: HashSet<u32> = all_pending
        .iter()
        .filter(|(c, _, _)| c.sym_id != u32::MAX)
        .map(|(c, _, _)| c.sym_id)
        .collect();
    info!(
        "chunk_sym_set built with {} symbols, precomputing appendix BFS",
        chunk_sym_set.len()
    );

    let appendix_map: FxHashMap<u32, Vec<u32>> = chunk_sym_set
        .iter()
        .map(|&sym_id| {
            let mut visited: HashSet<u32> = HashSet::with_capacity(16);
            let mut queue: VecDeque<(u32, usize)> = VecDeque::with_capacity(16);
            let mut out: Vec<u32> = Vec::with_capacity(APPENDIX_CAP_PER_CHUNK);
            visited.insert(sym_id);
            queue.push_back((sym_id, 0));
            while let Some((cur, depth)) = queue.pop_front() {
                if depth >= APPENDIX_DEPTH {
                    continue;
                }
                let Some(edges) = dsl_index.forward_edges.get(&cur) else {
                    continue;
                };
                for edge in edges.iter().filter(|e| e.kind == EdgeKind::Call) {
                    let Target::Resolved(rid) = edge.to else {
                        continue;
                    };
                    if !visited.insert(rid) {
                        continue;
                    }
                    if !chunk_sym_set.contains(&rid) {
                        continue;
                    }
                    if out.len() < APPENDIX_CAP_PER_CHUNK {
                        out.push(rid);
                    }
                    queue.push_back((rid, depth + 1));
                }
            }
            (sym_id, out)
        })
        .collect();

    let total_chunks = all_pending.len();
    reporter.start(total_chunks as u64);
    reporter.set_phase(
        zrag_protocol::response::IndexPhase::Tokenize,
        0,
        total_chunks as u64,
        "tokenizing chunks",
    );

    let batch_size = engine.recommended_batch_size().max(1);
    let fallback_hw;
    let hw = if let Some(hw) = engine.hardware() {
        hw
    } else {
        fallback_hw = zrag_hw::probe_effective();
        &fallback_hw
    };
    info!(
        batch_size,
        device = ?hw.device,
        mem_avail_mb = hw.mem_avail >> 20,
        max_length = engine.max_length(),
        chunk_max_tokens = engine.chunk_max_tokens(),
        remote = engine.is_remote(),
        "computed embed batch_size",
    );

    let mut total_embedded = 0usize;
    let reranker = TurboReranker::new(engine.dim())?;

    let total_chunks = all_pending.len();
    let mut paused = false;

    reporter.set_phase(
        zrag_protocol::response::IndexPhase::Tokenize,
        total_chunks as u64,
        total_chunks as u64,
        if engine.is_remote() {
            "remote mode: skipping local tokenization"
        } else {
            "tokenizing chunks"
        },
    );
    reporter.set_phase(
        zrag_protocol::response::IndexPhase::Embed,
        0,
        total_chunks as u64,
        "embedding chunks",
    );

    // Coalesce embed-batch RecordBatches and flush them with a single Lance
    // `add` (one manifest commit) once ~CHUNK_FLUSH_ROWS rows accumulate,
    // instead of a merge_insert per embed batch. The indexer deletes each
    // (re)indexed file's prior chunks before this loop, so freshly-hashed
    // chunk_ids can't collide with surviving rows — append is duplicate-free.
    // 256 keeps pending_batches bounded at ~1 MB (vs 4096 which never flushed
    // mid-loop for projects under 4096 chunks, causing a single end-of-loop burst).
    //
    // Tuning: try 128, 256, 512, 1024 on your fixture and compare
    //   - DHAT total bytes / peak live
    //   - Lance ObjectWriter::new total bytes
    //   - Lance commit count (logged by append_batches)
    //   - Wall time and embed throughput (chunks/s)
    // Higher values → fewer commits, larger per-commit write bursts.
    const CHUNK_FLUSH_ROWS: usize = 256;
    let mut pending_batches: Vec<RecordBatch> = Vec::with_capacity(4);
    let mut pending_rows = 0usize;
    let mut pending_file_idxs: Vec<u32> = Vec::with_capacity(CHUNK_FLUSH_ROWS);

    let dim = engine.dim();
    let schema = Arc::new(zrag_store::schema::chunks_schema(dim));
    let file_types: Vec<u8> = need_reindex
        .iter()
        .map(|rel| {
            snapshots
                .get(rel)
                .map(|s| s.file_type.into())
                .unwrap_or_default()
        })
        .collect();

    let embed_start = std::time::Instant::now();
    if !paused {
        info!(
            total_chunks,
            batch_size,
            "embed loop: starting",
        );
        let mut cursor = 0usize;
        // GPU pipelining: submit batch N+1 BEFORE awaiting batch N. Since
        // tokenization now runs on the worker thread (not the tokio reactor),
        // submit is instant — it just sends Strings through the channel.
        // The worker tokenizes batch N+1 while the tokio task processes
        // batch N's results (turbo codes, arrow builders).
        let mut pending_rx: Option<oneshot::Receiver<Result<Pooled>>> = None;
        while cursor < all_pending.len() {
            if cancel.load(Ordering::Relaxed) {
                paused = true;
                break;
            }

            let end = cursor.saturating_add(batch_size).min(all_pending.len());
            let batch_items = all_pending
                .get(cursor..end)
                .ok_or_else(|| anyhow::anyhow!("invalid embed batch range"))?;
            let text_refs: Vec<&str> = batch_items
                .iter()
                .map(|(chunk, _, _)| chunk.body.as_ref())
                .collect();

            // Submit current batch to GPU worker if not already prefetched
            if pending_rx.is_none() && !engine.is_remote() {
                pending_rx = Some(engine.submit_texts_pooled(&text_refs)?);
            }

            // Prefetch next batch: tokenize + queue on worker BEFORE awaiting
            // current. This tokenization (CPU) overlaps with the GPU finishing
            // the current batch, and the worker starts the next forward pass
            // immediately after the current one completes.
            let prefetch_end = end.saturating_add(batch_size).min(all_pending.len());
            let next_rx = if prefetch_end > end && !engine.is_remote() {
                let next_items = all_pending
                    .get(end..prefetch_end)
                    .ok_or_else(|| anyhow::anyhow!("invalid prefetch range"))?;
                let next_refs: Vec<&str> = next_items
                    .iter()
                    .map(|(chunk, _, _)| chunk.body.as_ref())
                    .collect();
                Some(engine.submit_texts_pooled(&next_refs)?)
            } else {
                None
            };

            // Await current batch result (GPU may still be finishing)
            let batch_started = std::time::Instant::now();
            let max_body_len = text_refs.iter().map(|s| s.len()).max().unwrap_or(0);
            info!(
                cursor,
                items = end - cursor,
                max_body_chars = max_body_len,
                "embed loop: calling engine",
            );
            let embs = match pending_rx.take() {
                Some(rx) => rx
                    .await
                    .map_err(|_| anyhow::anyhow!("embed worker dropped without replying"))??,
                None => engine.embed_texts_pooled_async(&text_refs).await?,
            };
            pending_rx = next_rx;
            info!(
                ms = batch_started.elapsed().as_millis() as u64,
                "embed loop: engine returned",
            );

            let n = embs.batch;
            if n != batch_items.len() {
                anyhow::bail!(
                    "embed row count mismatch: got {}, expected {}",
                    n,
                    batch_items.len(),
                );
            }

            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            // Quantize the whole batch in parallel: TurboQuantizer::encode
            // is the #1 app-CPU cost and was previously run serially on the
            // async task while rayon cores sat idle. `&reranker` is Sync;
            // results are index-aligned with rows, preserving order.
            let turbo_start = std::time::Instant::now();
            let turbo_codes: Vec<Option<Vec<u8>>> = (0..n)
                .into_par_iter()
                .map(|i| match reranker.encode(embs.row(i)) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        tracing::debug!("turbo encode failed: {}", e);
                        None
                    }
                })
                .collect();
            info!(
                ms = turbo_start.elapsed().as_millis() as u64,
                "embed loop: turbo encoded",
            );

            let mut chunk_id_builder = FixedSizeBinaryBuilder::with_capacity(n, 16);
            let mut file_path_builder = StringBuilder::with_capacity(n, n * 64);
            let mut language_builder = StringBuilder::with_capacity(n, n * 8);
            let mut file_type_builder = UInt8Array::builder(n);
            let mut symbol_qualified_builder = StringBuilder::with_capacity(n, n * 64);
            let mut symbol_kind_builder = StringBuilder::with_capacity(n, n * 16);
            let mut sym_id_builder = UInt32Array::builder(n);
            let mut sub_chunk_idx_builder = UInt32Array::builder(n);
            let mut total_sub_chunks_builder = UInt32Array::builder(n);
            let mut chunk_strategy_builder = UInt8Array::builder(n);
            let mut parent_sym_id_builder = UInt32Array::builder(n);
            let mut appendix_sym_ids_builder =
                ListBuilder::with_capacity(UInt32Builder::with_capacity(n * 4), n);
            let mut start_line_builder = UInt32Array::builder(n);
            let mut end_line_builder = UInt32Array::builder(n);
            let mut content_builder = StringBuilder::with_capacity(n, n * 64);
            let mut turbo_code_builder = BinaryBuilder::with_capacity(n, n * 32);
            let mut indexed_at_builder = UInt64Array::builder(n);

            for (i, (chunk, lang, fidx)) in batch_items.iter().enumerate() {
                pending_file_idxs.push(*fidx);
                let file_type = file_types.get(*fidx as usize).copied().unwrap_or_default();
                let emb = embs.row(i);

                if emb.iter().any(|v| v.is_nan()) {
                    anyhow::bail!(
                        "NaN embedding for {}:{}-{}",
                        chunk.file,
                        chunk.start_line,
                        chunk.end_line,
                    );
                }

                let chunk_id = content_chunk_id(chunk);

                let parent_sym_id = if chunk.sym_id == u32::MAX {
                    None
                } else {
                    dsl_index
                        .symbols
                        .get(chunk.sym_id as usize)
                        .and_then(|s| s.parent)
                };
                let appendix_ids: &[u32] = if chunk.sym_id == u32::MAX {
                    &[]
                } else {
                    appendix_map.get(&chunk.sym_id).map_or(&[], Vec::as_slice)
                };

                chunk_id_builder.append_value(chunk_id)?;
                file_path_builder.append_value(&chunk.file);
                language_builder.append_value(lang);
                file_type_builder.append_value(file_type);
                symbol_qualified_builder.append_value(&chunk.qualified);
                symbol_kind_builder.append_value(chunk.kind.as_str());
                sym_id_builder.append_value(chunk.sym_id);
                sub_chunk_idx_builder.append_value(chunk.sub_chunk_idx);
                total_sub_chunks_builder.append_value(chunk.total_sub_chunks);
                chunk_strategy_builder.append_value(chunk.chunk_strategy as u8);
                match parent_sym_id {
                    Some(p) => parent_sym_id_builder.append_value(p),
                    None => parent_sym_id_builder.append_null(),
                }
                if appendix_ids.is_empty() {
                    appendix_sym_ids_builder.append_null();
                } else {
                    for id in appendix_ids {
                        appendix_sym_ids_builder.values().append_value(*id);
                    }
                    appendix_sym_ids_builder.append(true);
                }
                start_line_builder.append_value(chunk.start_line);
                end_line_builder.append_value(chunk.end_line);
                content_builder.append_value(&chunk.body);
                match turbo_codes.get(i).and_then(Option::as_ref) {
                    Some(t) => turbo_code_builder.append_value(t),
                    None => turbo_code_builder.append_null(),
                }
                indexed_at_builder.append_value(now_ns);

                total_embedded = total_embedded.saturating_add(1);
            }

            let embedding_arr = embs.into_fixed_size_list();
            let record = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    std::sync::Arc::new(chunk_id_builder.finish()),
                    std::sync::Arc::new(file_path_builder.finish()),
                    std::sync::Arc::new(language_builder.finish()),
                    std::sync::Arc::new(file_type_builder.finish()),
                    std::sync::Arc::new(symbol_qualified_builder.finish()),
                    std::sync::Arc::new(symbol_kind_builder.finish()),
                    std::sync::Arc::new(sym_id_builder.finish()),
                    std::sync::Arc::new(sub_chunk_idx_builder.finish()),
                    std::sync::Arc::new(total_sub_chunks_builder.finish()),
                    std::sync::Arc::new(chunk_strategy_builder.finish()),
                    std::sync::Arc::new(parent_sym_id_builder.finish()),
                    std::sync::Arc::new(appendix_sym_ids_builder.finish()),
                    std::sync::Arc::new(start_line_builder.finish()),
                    std::sync::Arc::new(end_line_builder.finish()),
                    std::sync::Arc::new(content_builder.finish()),
                    std::sync::Arc::new(turbo_code_builder.finish()),
                    std::sync::Arc::new(indexed_at_builder.finish()),
                    std::sync::Arc::new(embedding_arr),
                ],
            )?;

            pending_rows += record.num_rows();
            pending_batches.push(record);
            if pending_rows >= CHUNK_FLUSH_ROWS {
                info!(
                    rows = pending_rows,
                    "embed loop: flushing to LanceDB",
                );
                let flush_start = std::time::Instant::now();
                chunks_table
                    .append_batches(std::mem::take(&mut pending_batches))
                    .await?;
                pending_rows = 0;
                checkpoint_completed(
                    &files_table,
                    &snapshots,
                    &need_reindex,
                    &mut remaining,
                    &std::mem::take(&mut pending_file_idxs),
                )
                .await?;
                info!(
                    ms = flush_start.elapsed().as_millis() as u64,
                    "embed loop: flush + checkpoint done",
                );
            }

            tracing::info!(
                items = n,
                cursor,
                total = total_chunks,
                ms = batch_started.elapsed().as_millis() as u64,
                "embed loop: batch complete",
            );
            reporter.inc(n as u64);

            cursor = end;
        }
    } // if !paused

    info!(
        chunks = total_embedded,
        ms = embed_start.elapsed().as_millis() as u64,
        "embed phase"
    );

    // Flush pending batches regardless of pause — preserves embedded chunks
    // and checkpoints completed files so a resumed run skips them.
    if !pending_batches.is_empty() {
        chunks_table
            .append_batches(std::mem::take(&mut pending_batches))
            .await?;
        checkpoint_completed(
            &files_table,
            &snapshots,
            &need_reindex,
            &mut remaining,
            &std::mem::take(&mut pending_file_idxs),
        )
        .await?;
    }

    if !paused && total_chunks > 0 && total_embedded == 0 {
        anyhow::bail!("no embeddings produced from {total_chunks} chunks");
    }

    let fallback_hw;
    let hw = if let Some(hw) = engine.hardware() {
        hw
    } else {
        fallback_hw = zrag_hw::probe_effective();
        &fallback_hw
    };
    let previous_row = db.projects_table().await?.get(&pid).await.ok().flatten();
    let previous_params: Option<zrag_ann::SearchParams> = previous_row
        .as_ref()
        .and_then(|r| r.search_params.as_deref())
        .and_then(|s| toml::from_str(s).ok());
    let total_in_db = chunks_table.len().await?;
    let mut params =
        zrag_ann::choose_method(total_in_db, engine.dim(), hw, previous_params.as_ref());
    if let Some(m) = override_method {
        params.method = m;
    }
    info!(
        "search method: {:?} (n={}, dim={}, ram_avail={} MB)",
        params.method,
        total_in_db,
        engine.dim(),
        hw.mem_avail >> 20
    );

    if total_in_db > 0 {
        reporter.set_phase(
            zrag_protocol::response::IndexPhase::BuildIndex,
            0,
            0,
            "building search index",
        );
        let lance_start = std::time::Instant::now();
        chunks_table.optimize().await?;
        let opt_ms = lance_start.elapsed().as_millis();
        chunks_table.ensure_fts_indexes().await?;
        let fts_ms = lance_start.elapsed().as_millis() - opt_ms;
        chunks_table.build_index(&params).await?;
        let idx_ms = lance_start.elapsed().as_millis() - opt_ms - fts_ms;
        info!(opt_ms, fts_ms, idx_ms, "lance post-index ops");
    }

    let languages: HashSet<Language> = snapshots
        .values()
        .filter_map(|s| match s.kind {
            SourceKind::Code(l) => Some(l),
            SourceKind::Tsv | SourceKind::Psv | SourceKind::Text | SourceKind::Pdf => None,
        })
        .collect();
    let languages: Vec<&Language> = languages.iter().collect();
    upsert_project(
        db,
        &pid,
        root_str,
        total_in_db,
        snapshots.len(),
        &languages,
        engine,
        &params,
    )
    .await?;

    let msg = if paused {
        format!(
            "paused — {} passages embedded, run index again to resume",
            total_embedded
        )
    } else {
        format!("embedded {} passages", total_embedded)
    };
    reporter.finish_with_message(&msg);

    let elapsed = start.elapsed();
    info!(
        "indexing complete: {} chunks, {} files in {:.1}s",
        total_embedded,
        snapshots.len(),
        elapsed.as_secs_f64(),
    );

    Ok(IndexStats {
        total_chunks: total_embedded,
        total_files: snapshots.len(),
        new_chunks: total_embedded,
        reindexed_files: need_reindex.len(),
        duration_ms: elapsed.as_millis() as u64,
        paused,
    })
}

async fn upsert_files(
    files_table: &zrag_store::files_table::FilesTable,
    snapshots: &std::collections::HashMap<String, FileSnapshot>,
    changed_paths: &[&str],
) -> Result<()> {
    // Build one RecordBatch covering every changed file and merge it in a
    // single call, instead of one merge_insert (and one commit) per file.
    let rows: Vec<(&str, &FileSnapshot)> = changed_paths
        .iter()
        .filter_map(|p| snapshots.get(*p).map(|s| (*p, s)))
        .collect();
    if rows.is_empty() {
        return Ok(());
    }

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let n = rows.len();
    let mut blake3_builder = FixedSizeBinaryBuilder::with_capacity(n, 32);
    let mut paths: Vec<&str> = Vec::with_capacity(n);
    let mut mtimes: Vec<u64> = Vec::with_capacity(n);
    let mut sizes: Vec<u64> = Vec::with_capacity(n);
    let mut langs: Vec<&str> = Vec::with_capacity(n);
    for (path, snap) in &rows {
        blake3_builder.append_value(snap.blake3)?;
        paths.push(path);
        mtimes.push(snap.mtime_ns as u64);
        sizes.push(snap.size_bytes);
        langs.push(snap.kind.label());
    }

    let record = RecordBatch::try_new(
        std::sync::Arc::new(zrag_store::schema::files_schema()),
        vec![
            std::sync::Arc::new(StringArray::from(paths)),
            std::sync::Arc::new(blake3_builder.finish()),
            std::sync::Arc::new(UInt64Array::from(mtimes)),
            std::sync::Arc::new(UInt64Array::from(sizes)),
            std::sync::Arc::new(StringArray::from(langs)),
            std::sync::Arc::new(arrow::array::ListArray::new_null(
                std::sync::Arc::new(arrow::datatypes::Field::new(
                    "item",
                    arrow::datatypes::DataType::FixedSizeBinary(16),
                    false,
                )),
                n,
            )),
            std::sync::Arc::new(UInt64Array::from(vec![now_ns; n])),
        ],
    )?;

    files_table.upsert(record).await?;
    Ok(())
}

/// Decrement per-file chunk counts for a just-committed flush. A file reaching
/// zero has all its chunks durably in `chunks_table`, so record its row now —
/// the checkpoint a resumed run relies on.
async fn checkpoint_completed(
    files_table: &zrag_store::files_table::FilesTable,
    snapshots: &std::collections::HashMap<String, FileSnapshot>,
    need_reindex: &[String],
    remaining: &mut [u32],
    flushed_file_idxs: &[u32],
) -> Result<()> {
    let mut completed: Vec<&str> = Vec::with_capacity(flushed_file_idxs.len());
    for &fidx in flushed_file_idxs {
        if let Some(slot) = remaining.get_mut(fidx as usize) {
            *slot = slot.saturating_sub(1);
            if *slot == 0
                && let Some(path) = need_reindex.get(fidx as usize)
            {
                completed.push(path.as_str());
            }
        }
    }
    if !completed.is_empty() {
        upsert_files(files_table, snapshots, &completed).await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn upsert_project(
    db: &Db,
    pid: &[u8; 32],
    root: std::borrow::Cow<'_, str>,
    total_chunks: usize,
    total_files: usize,
    languages: &[&Language],
    engine: &AnyEmbedEngine,
    choice: &zrag_ann::SearchParams,
) -> Result<()> {
    let projects_table = db.projects_table().await?;
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let mut project_id_builder = FixedSizeBinaryBuilder::with_capacity(1, 32);
    project_id_builder.append_value(pid)?;

    let lang_values = StringArray::from(languages.iter().map(|l| l.as_str()).collect::<Vec<_>>());
    let languages_arr = arrow::array::ListArray::new(
        Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::Utf8,
            false,
        )),
        arrow::buffer::OffsetBuffer::from_lengths([languages.len()]),
        Arc::new(lang_values),
        None,
    );

    let search_method = StringArray::from(vec![choice.method.as_str()]);
    let search_params = StringArray::from(vec![toml::to_string(&choice)?]);

    let record = RecordBatch::try_new(
        Arc::new(zrag_store::schema::projects_schema()),
        vec![
            Arc::new(project_id_builder.finish()),
            Arc::new(StringArray::from(vec![root.to_string()])),
            Arc::new(languages_arr),
            Arc::new(StringArray::from(vec![
                engine.persisted_model_id().as_ref(),
            ])),
            Arc::new(UInt32Array::from(vec![engine.dim() as u32])),
            Arc::new(UInt64Array::from(vec![total_chunks as u64])),
            Arc::new(UInt64Array::from(vec![total_files as u64])),
            Arc::new(UInt64Array::from(vec![now_ns])),
            Arc::new(UInt64Array::from(vec![now_ns])),
            Arc::new(UInt32Array::from(vec![
                zrag_store::projects_table::INDEX_FORMAT_VERSION,
            ])),
            Arc::new(search_method),
            Arc::new(search_params),
        ],
    )?;

    projects_table.upsert(record).await?;
    Ok(())
}

#[cfg(test)]
mod tests_indexing {
    use super::{MIN_CHUNK_FLOOR, chunk_file_path, content_chunk_id, sizing_for};
    use std::borrow::Cow;
    use std::path::Path;
    use zrag_dsl::chunking::{Chunk, ChunkStrategy};
    use zrag_ts_core::types::Kind;

    #[test]
    fn chunk_delete_paths_are_absolute_under_root() {
        let root = Path::new("/tmp/proj");
        assert_eq!(chunk_file_path(root, "src/a.rs"), "/tmp/proj/src/a.rs");
        assert_eq!(chunk_file_path(root, "lib/b.ts"), "/tmp/proj/lib/b.ts");
    }

    #[test]
    fn sizing_for_none_when_body_fits() {
        // body_len <= max_len * bpt → fits in one chunk.
        assert!(sizing_for(2048, 512, 4).is_none()); // exactly at the limit
        assert!(sizing_for(100, 512, 4).is_none()); // well under
    }

    #[test]
    fn sizing_for_splits_and_uses_measured_bpt() {
        // Sparse markdown: bpt high → larger byte-chunks than the old `* 4`.
        let s = sizing_for(170_000_000, 512, 7).expect("huge sparse body must split");
        assert_eq!(s.chunk_size, 512 * 7); // 3584, not the old 2048
        assert_eq!(s.min_chunk_size, 256 * 7); // (max_len/2)*bpt = 1792
        assert!(
            s.chunk_size > s.min_chunk_size,
            "chunker requires chunk_size > min"
        );
    }

    #[test]
    fn sizing_for_dense_code_smaller_chunks() {
        // Dense code: bpt low → tighter byte-chunks that still fit max_len tokens.
        let s = sizing_for(50_000, 512, 2).expect("body exceeds 1024 bytes → split");
        assert_eq!(s.chunk_size, 1024);
        assert_eq!(s.min_chunk_size, 512);
    }

    #[test]
    fn sizing_for_enforces_floors() {
        // Tiny max_len * bpt would drop below the floor; floors must clamp it
        // while preserving chunk_size > min_chunk_size.
        let s = sizing_for(10_000, 10, 1).expect("10_000 > 10 → split");
        assert_eq!(s.chunk_size, MIN_CHUNK_FLOOR + 1); // 513
        assert_eq!(s.min_chunk_size, MIN_CHUNK_FLOOR); // 512
        assert!(s.chunk_size > s.min_chunk_size);
    }

    #[test]
    fn sizing_for_saturates_without_panic() {
        // chunk_size saturates to usize::MAX → nothing can exceed it → None, no overflow.
        assert!(sizing_for(10, usize::MAX, 4).is_none());
        assert!(sizing_for(usize::MAX, 512, usize::MAX).is_none());
        assert!(sizing_for(usize::MAX - 1, usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn content_chunk_id_ignores_line_shifts() {
        let first = Chunk {
            file: "/src/lib.rs".into(),
            rel_file: "src/lib.rs".into(),
            start_line: 10,
            end_line: 20,
            sym_id: 7,
            sub_chunk_idx: 0,
            total_sub_chunks: 1,
            chunk_strategy: ChunkStrategy::Symbol,
            body: Cow::Borrowed("fn top_k() { 1 }"),
            qualified: "top_k".into(),
            kind: Kind::Function,
        };
        let shifted = Chunk {
            start_line: 15,
            end_line: 25,
            ..first.clone()
        };
        assert_eq!(content_chunk_id(&first), content_chunk_id(&shifted));
    }

    #[test]
    fn test_generate_sub_chunks_metadata() {
        let parent = Chunk {
            file: "/src/main.rs".into(),
            rel_file: "src/main.rs".into(),
            start_line: 10,
            end_line: 20,
            sym_id: 42,
            sub_chunk_idx: 0,
            total_sub_chunks: 1,
            chunk_strategy: ChunkStrategy::Symbol,
            body: Cow::Borrowed("line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"),
            qualified: "foo::bar".into(),
            kind: Kind::Function,
        };

        let mock_sub_chunks = [
            zrag_recursive_chunk::SubChunk {
                byte_start: 0,
                byte_end: 20,
                start_line: 1,
                end_line: 3,
            },
            zrag_recursive_chunk::SubChunk {
                byte_start: 20,
                byte_end: 48,
                start_line: 3,
                end_line: 7,
            },
        ];

        let mut out: Vec<(Chunk<'_>, &str, u32)> = Vec::new();
        let total = mock_sub_chunks.len() as u32;
        for (i, sub) in mock_sub_chunks.iter().enumerate() {
            let sc = Chunk {
                file: parent.file.clone(),
                rel_file: parent.rel_file.clone(),
                start_line: parent.start_line + sub.start_line - 1,
                end_line: parent.start_line + sub.end_line - 1,
                sym_id: parent.sym_id,
                sub_chunk_idx: i as u32,
                total_sub_chunks: total,
                chunk_strategy: ChunkStrategy::Recursive,
                body: Cow::Owned(parent.body[sub.byte_start..sub.byte_end].to_string()),
                qualified: parent.qualified.clone(),
                kind: parent.kind,
            };
            out.push((sc, "rust", 0));
        }

        assert_eq!(out.len(), 2);

        let s0 = &out[0].0;
        assert_eq!(s0.sub_chunk_idx, 0);
        assert_eq!(s0.total_sub_chunks, 2);
        assert_eq!(s0.chunk_strategy, ChunkStrategy::Recursive);
        assert_eq!(s0.start_line, 10);
        assert_eq!(s0.end_line, 12);
        assert_eq!(s0.body, parent.body[..20]);

        let s1 = &out[1].0;
        assert_eq!(s1.sub_chunk_idx, 1);
        assert_eq!(s1.total_sub_chunks, 2);
        assert_eq!(s1.chunk_strategy, ChunkStrategy::Recursive);
        assert_eq!(s1.start_line, 12);
        assert_eq!(s1.end_line, 16);
        assert_eq!(s1.body, parent.body[20..48]);
    }
}
