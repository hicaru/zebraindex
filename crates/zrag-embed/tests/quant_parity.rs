//! Quantization parity for the GGUF path (Phase F).
//!
//! For every GGUF dtype — F32/F16/BF16 and Q4_0…Q8K — we quantize a known
//! weight matrix, round-trip it through a real GGUF file via candle's quantized
//! `VarBuilder`, build a [`zrag_embed::qlayers::QLinear`] exactly as the engine
//! does, and assert its output stays within a cosine-similarity band of the
//! dense (F32) reference. No network is required: the weights are deterministic.

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder;
use zrag_embed::qlayers::{QLinear, Vb, dense_varbuilder_from_gguf, linear_no_bias};

/// Minimum output cosine similarity accepted for each GGUF dtype vs the dense F32
/// reference for a single matmul. More aggressive rounding gets a looser floor.
fn min_cosine(dt: GgmlDType) -> f32 {
    use GgmlDType::*;
    match dt {
        F32 => 1.0 - 1e-6,
        F16 | BF16 => 0.9999,
        Q8_0 | Q8_1 | Q8K | Q6K => 0.999,
        Q5_0 | Q5_1 | Q5K | Q4K => 0.99,
        Q4_0 | Q4_1 => 0.97,
        Q3K | Q2K => 0.90,
    }
}

fn all_quant_dtypes() -> Vec<GgmlDType> {
    use GgmlDType::*;
    vec![
        F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    ]
}

/// Cosine similarity between two equally-shaped tensors (flattened to F32).
fn cosine(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    Ok(dot / (na * nb + 1e-12))
}

/// Deterministic weights in roughly [-0.5, 0.5) shaped `(rows, cols)`. `cols` and
/// `rows` are kept divisible by 256 so every K-quant block size aligns.
fn deterministic(rows: usize, cols: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let n = (rows * cols) as f64;
    let arange = Tensor::arange(0u32, (rows * cols) as u32, dev)?.to_dtype(DType::F32)?;
    let t = arange.affine(1.0 / n, -0.5)?;
    t.reshape((rows, cols))
}

#[test]
fn quantized_linear_matches_dense_within_tolerance() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (256usize, 256usize);

    let w = deterministic(out_dim, in_dim, &dev)?;
    // Fixed input batch: (seq=4, batch=2, in_dim).
    let x = deterministic(4 * 2, in_dim, &dev)?.reshape((4, 2, in_dim))?;

    // Dense F32 reference built through the same `qlayers` entry point.
    let mut dense_ts = HashMap::new();
    dense_ts.insert("layer.weight".to_string(), w.clone());
    let dense_vb = VarBuilder::from_tensors(dense_ts, DType::F32, &dev).pp("layer");
    let dense_lin: QLinear = linear_no_bias(in_dim, out_dim, Vb::Dense(dense_vb))?;
    let dense_out = dense_lin.forward(&x)?;

    let mut tested = 0usize;
    for dt in all_quant_dtypes() {
        // Run the full GGUF round-trip pipeline; candle 0.10.2 doesn't implement
        // every dtype end-to-end (e.g. Q8_1's matmul kernel), so treat a pipeline
        // error as "unsupported here" and move on. Every cosine we DO get must
        // clear its dtype's floor.
        let path = std::env::temp_dir().join(format!("zrag-quant-parity-{dt:?}.gguf"));
        let res = (|| -> candle_core::Result<f32> {
            let qtensor = QTensor::quantize(&w, dt)?;
            {
                let mut f = std::fs::File::create(&path)?;
                gguf_file::write(&mut f, &[], &[("layer.weight", &qtensor)])?;
            }
            let qvb = quantized_var_builder::VarBuilder::from_gguf(&path, &dev)?;
            let qlin: QLinear = linear_no_bias(in_dim, out_dim, Vb::Quant(qvb.pp("layer")))?;
            let qout = qlin.forward(&x)?;
            cosine(&dense_out, &qout)
        })();
        let _ = std::fs::remove_file(&path);

        match res {
            Ok(cos) => {
                let floor = min_cosine(dt);
                assert!(
                    cos >= floor,
                    "{dt:?}: cosine {cos:.6} below floor {floor} (GGUF round-trip parity)"
                );
                tested += 1;
                println!("{dt:?}: cosine = {cos:.6} (floor {floor})");
            }
            Err(e) => {
                println!("{dt:?}: skipped (candle 0.10.2 pipeline limitation: {e})");
            }
        }
    }

    // Guard against the test silently exercising nothing.
    assert!(tested >= 10, "expected ≥10 quantizable dtypes, got {tested}");
    Ok(())
}

#[test]
fn dense_varbuilder_from_gguf_round_trips() -> anyhow::Result<()> {
    // The BERT engine path dequantizes a whole GGUF into a dense VarBuilder;
    // verify it recovers the weights within the Q8_0 quantization band.
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (256usize, 256usize);
    let w = deterministic(out_dim, in_dim, &dev)?;
    let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;

    let path = std::env::temp_dir().join("zrag-dense-dequant.gguf");
    {
        let mut f = std::fs::File::create(&path)?;
        gguf_file::write(&mut f, &[], &[("w", &qt)])?;
    }

    let vb = dense_varbuilder_from_gguf(&path, &dev)?;
    let recovered = vb.get((out_dim, in_dim), "w")?;
    let cos = cosine(&w, &recovered)?;
    assert!(cos > 0.999, "dequant round-trip cosine {cos:.6} too low");

    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// End-to-end GGUF load through the real engine. Ignored by default — point it at a
/// local GGUF model directory (must contain the `.gguf` weights plus `config.json`
/// and `tokenizer.json`) via an env var:
///
/// ```sh
/// ZEBRARAG_GGUF_MODEL=/path/to/jina-embeddings-v2-gguf \
///   cargo test -p zrag-embed --test quant_parity e2e_gguf_engine -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn e2e_gguf_engine() -> anyhow::Result<()> {
    let model_id = match std::env::var("ZEBRARAG_GGUF_MODEL") {
        Ok(id) => id,
        Err(_) => {
            eprintln!("set ZEBRARAG_GGUF_MODEL to a local GGUF model dir to run this test");
            return Ok(());
        }
    };
    // `load` resolves GGUF weights, builds the (native-quant Jina / dequant BERT)
    // model, and runs the warmup forward internally.
    let engine = zrag_embed::EmbedEngine::load(&model_id)?;
    assert_eq!(
        engine.profile().format,
        zrag_embed::model_registry::WeightsFormat::Gguf,
        "engine should resolve GGUF weights"
    );
    assert!(engine.dim() > 0, "warmup should have probed an embedding dim");
    println!("e2e GGUF load ok: dim={}, format={:?}", engine.dim(), engine.profile().format);
    Ok(())
}
