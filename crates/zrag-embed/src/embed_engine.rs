//! Experimental TEI-backed embedding engine scaffold.
//!
//! This module is intentionally kept parallel to the existing `engine` module
//! while the migration proceeds. The current production API still re-exports
//! `engine::EmbedEngine`; this type exists so the TEI dependencies and feature
//! wiring can be validated before deleting the custom Candle backend.

use std::marker::PhantomData;

use text_embeddings_backend::{Backend, DType, ModelType, Pool};
use text_embeddings_core::infer::Infer;

/// TEI-backed embedding engine scaffold.
#[derive(Debug, Default)]
pub struct EmbedEngineV2 {
    _infer: PhantomData<Infer>,
    _backend: PhantomData<Backend>,
    _dtype: PhantomData<DType>,
    _model_type: PhantomData<ModelType>,
    _pool: PhantomData<Pool>,
}

impl EmbedEngineV2 {
    /// Build a scaffold value without starting TEI worker tasks.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _infer: PhantomData,
            _backend: PhantomData,
            _dtype: PhantomData,
            _model_type: PhantomData,
            _pool: PhantomData,
        }
    }
}
