use anyhow::Result;
use tokenizers::{Tokenizer as HfTokenizer, TruncationParams};

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Tokenized> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        // TODO(perf): these .to_vec() calls clone the Encoding's internal
        // Vec<u32> buffers.  Upstream tokenizers 0.23 exposes only getters
        // (get_ids / get_attention_mask); there is no take_ids / take_attention_mask.
        // File a PR on huggingface/tokenizers adding these move-out APIs, then
        // replace .to_vec() with .take_ids() / .take_attention_mask() here and
        // in encode_batch below.  Saves 2× batch × seq × 4 bytes per call.
        let ids = encoding.get_ids().to_vec();
        let mask = encoding.get_attention_mask().to_vec();
        Ok(Tokenized { ids, mask })
    }

    /// Batch encode. Runs in parallel via rayon inside `tokenizers`.
    pub fn encode_batch(&self, texts: Vec<&str>) -> Result<Vec<Tokenized>> {
        let encs = self
            .inner
            .encode_batch(texts, false)
            .map_err(|e| anyhow::anyhow!("encode_batch: {}", e))?;

        let mut out = Vec::with_capacity(encs.len());
        // TODO(perf): same .to_vec() issue as encode() above — waiting on
        // upstream tokenizers take_ids / take_attention_mask API.
        for enc in encs {
            let ids = enc.get_ids().to_vec();
            let mask = enc.get_attention_mask().to_vec();
            out.push(Tokenized { ids, mask });
        }
        Ok(out)
    }

    /// Force truncation at `max_length`, overriding whatever `tokenizer.json`
    /// shipped. Sentence-transformers repos often pin a 128-token truncation
    /// below the model's real context window; this keeps long inputs from
    /// silently losing context.
    pub fn set_truncation(&mut self, max_length: usize) -> Result<()> {
        let params = TruncationParams {
            max_length,
            ..Default::default()
        };
        self.inner
            .with_truncation(Some(params))
            .map_err(|e| anyhow::anyhow!("set truncation: {e}"))?;
        Ok(())
    }

    /// Read the tokenizer's truncation `max_length` if configured.
    pub fn truncation_max_length(&self) -> Option<usize> {
        self.inner.get_truncation().map(|t| t.max_length)
    }

    /// Token count only; the encoding is dropped without cloning its buffers.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().len())
    }
}

pub struct Tokenized {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
}
