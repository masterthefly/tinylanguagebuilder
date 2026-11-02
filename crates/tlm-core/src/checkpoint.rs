//! Saving and loading, using the safetensors file format.
//!
//!   [ 8 bytes: header length N, little-endian ]
//!   [ N bytes: JSON header, padded with spaces so the data is 8-byte aligned ]
//!   [ raw little-endian f32 data, tensor after tensor ]
//!
//! The header maps each tensor name to {"dtype", "shape", "data_offsets"}, plus
//! an optional "__metadata__" map of strings. Loading never executes anything
//! from the file; it is only numbers, and we validate every length and offset.

use crate::bpe::Tokenizer;
use crate::optim::AdamW;
use crate::transformer::{param_names, Transformer, TransformerConfig};
use crate::Matrix;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Build the bytes of a safetensors file in memory.
pub fn safetensors_to_bytes(tensors: &[(String, &Matrix)], metadata: &BTreeMap<String, String>) -> io::Result<Vec<u8>> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), json!(metadata));
    }
    let mut offset = 0usize;
    for (name, m) in tensors {
        let len = m.data.len() * 4;
        header.insert(
            name.clone(),
            json!({ "dtype": "F32", "shape": [m.rows, m.cols], "data_offsets": [offset, offset + len] }),
        );
        offset += len;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).map_err(|e| invalid(e.to_string()))?;
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }

    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for (_, m) in tensors {
        for v in &m.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

/// Write tensors to `path`. The file appears atomically: we write a temporary
/// file and rename it, so a crash mid-save can never leave a broken checkpoint.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, &Matrix)],
    metadata: &BTreeMap<String, String>,
) -> io::Result<()> {
    let out = safetensors_to_bytes(tensors, metadata)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &out)?;
    std::fs::rename(&tmp, path)
}

type Loaded = (BTreeMap<String, Matrix>, BTreeMap<String, String>);

/// Read a safetensors file of f32 tensors (1-D tensors become 1 x N).
/// Returns the tensors by name, and the metadata strings.
pub fn read_safetensors(path: &Path) -> io::Result<Loaded> {
    parse_safetensors(&std::fs::read(path)?)
}

/// Parse safetensors bytes already in memory. Validates everything it reads.
pub fn parse_safetensors(bytes: &[u8]) -> io::Result<Loaded> {
    if bytes.len() < 8 {
        return Err(invalid("file too short to be safetensors"));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let header_end = usize::try_from(header_len)
        .ok()
        .and_then(|n| n.checked_add(8))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| invalid("header length runs past the end of the file"))?;
    let header: Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| invalid(format!("bad header: {e}")))?;
    let body = &bytes[header_end..];

    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, entry) in header.as_object().ok_or_else(|| invalid("header is not a JSON object"))? {
        if name == "__metadata__" {
            for (k, v) in entry.as_object().ok_or_else(|| invalid("bad __metadata__"))? {
                metadata.insert(k.clone(), v.as_str().ok_or_else(|| invalid("metadata values must be strings"))?.to_string());
            }
            continue;
        }
        if entry["dtype"] != "F32" {
            return Err(invalid(format!("tensor {name}: only F32 is supported")));
        }
        let dims: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| invalid(format!("tensor {name}: missing shape")))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| invalid("bad shape")))
            .collect::<io::Result<_>>()?;
        let (rows, cols) = match dims[..] {
            [n] => (1, n),
            [r, c] => (r, c),
            _ => return Err(invalid(format!("tensor {name}: only 1-D and 2-D tensors are supported"))),
        };
        let offsets = entry["data_offsets"].as_array().ok_or_else(|| invalid(format!("tensor {name}: missing offsets")))?;
        let (start, end) = match (offsets.first().and_then(Value::as_u64), offsets.get(1).and_then(Value::as_u64)) {
            (Some(s), Some(e)) => (s as usize, e as usize),
            _ => return Err(invalid(format!("tensor {name}: bad offsets"))),
        };
        // Check everything BEFORE allocating or slicing.
        let expected = rows.checked_mul(cols).and_then(|n| n.checked_mul(4));
        if start > end || end > body.len() || expected != Some(end - start) {
            return Err(invalid(format!("tensor {name}: offsets do not match its shape or the file size")));
        }
        let data = body[start..end].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        tensors.insert(name.clone(), Matrix::from_vec(rows, cols, data));
    }
    Ok((tensors, metadata))
}

fn meta_usize(meta: &BTreeMap<String, String>, key: &str) -> io::Result<usize> {
    meta.get(key)
        .ok_or_else(|| invalid(format!("metadata is missing \"{key}\"")))?
        .parse()
        .map_err(|_| invalid(format!("metadata \"{key}\" is not a number")))
}

// ---- model --------------------------------------------------------------------

/// Save weights + config to `<dir>/model.safetensors`. This file alone is
/// enough to share a model (you also need the tokenizer to use it for text).
pub fn save_model(dir: &Path, model: &Transformer, step: usize) -> io::Result<()> {
    let (tensors, metadata) = model_parts(model, step);
    write_safetensors(&dir.join("model.safetensors"), &tensors, &metadata)
}

/// The same file as `save_model` writes, but as bytes in memory.
pub fn model_to_bytes(model: &Transformer, step: usize) -> io::Result<Vec<u8>> {
    let (tensors, metadata) = model_parts(model, step);
    safetensors_to_bytes(&tensors, &metadata)
}

fn model_parts(model: &Transformer, step: usize) -> (Vec<(String, &Matrix)>, BTreeMap<String, String>) {
    let names = param_names(&model.cfg);
    let tensors: Vec<(String, &Matrix)> = names.into_iter().zip(&model.params).collect();
    let c = model.cfg;
    let metadata = BTreeMap::from([
        ("format".to_string(), "tlm-transformer-v1".to_string()),
        ("vocab_size".to_string(), c.vocab_size.to_string()),
        ("ctx".to_string(), c.ctx.to_string()),
        ("d_model".to_string(), c.d_model.to_string()),
        ("n_layer".to_string(), c.n_layer.to_string()),
        ("n_head".to_string(), c.n_head.to_string()),
        ("step".to_string(), step.to_string()),
    ]);
    (tensors, metadata)
}

/// Returns the model and the training step it was saved at.
pub fn load_model(dir: &Path) -> io::Result<(Transformer, usize)> {
    model_from_loaded(read_safetensors(&dir.join("model.safetensors"))?)
}

/// Like `load_model`, from bytes already in memory.
pub fn model_from_bytes(bytes: &[u8]) -> io::Result<(Transformer, usize)> {
    model_from_loaded(parse_safetensors(bytes)?)
}

fn model_from_loaded((mut tensors, meta): Loaded) -> io::Result<(Transformer, usize)> {
    let cfg = TransformerConfig {
        vocab_size: meta_usize(&meta, "vocab_size")?,
        ctx: meta_usize(&meta, "ctx")?,
        d_model: meta_usize(&meta, "d_model")?,
        n_layer: meta_usize(&meta, "n_layer")?,
        n_head: meta_usize(&meta, "n_head")?,
    };
    let mut params = Vec::new();
    for name in param_names(&cfg) {
        params.push(tensors.remove(&name).ok_or_else(|| invalid(format!("missing tensor {name}")))?);
    }
    let model = Transformer::from_params(cfg, params).map_err(invalid)?;
    Ok((model, meta_usize(&meta, "step").unwrap_or(0)))
}

// ---- optimizer ------------------------------------------------------------------

fn optimizer_parts<'a>(
    names: &[String],
    opt: &'a AdamW,
    step: usize,
) -> (Vec<(String, &'a Matrix)>, BTreeMap<String, String>) {
    let (m, v) = opt.moments();
    let mut tensors: Vec<(String, &Matrix)> = Vec::new();
    for (name, mat) in names.iter().zip(m) {
        tensors.push((format!("m.{name}"), mat));
    }
    for (name, mat) in names.iter().zip(v) {
        tensors.push((format!("v.{name}"), mat));
    }
    let metadata = BTreeMap::from([
        ("updates".to_string(), opt.updates().to_string()),
        ("step".to_string(), step.to_string()),
    ]);
    (tensors, metadata)
}

/// Save Adam's memory (needed only to RESUME training, not to use the model).
pub fn save_optimizer(dir: &Path, cfg: &TransformerConfig, opt: &AdamW, step: usize) -> io::Result<()> {
    let (tensors, metadata) = optimizer_parts(&param_names(cfg), opt, step);
    write_safetensors(&dir.join("optimizer.safetensors"), &tensors, &metadata)
}

/// The same file as `save_optimizer` writes, as bytes in memory.
pub fn optimizer_to_bytes(cfg: &TransformerConfig, opt: &AdamW, step: usize) -> io::Result<Vec<u8>> {
    let (tensors, metadata) = optimizer_parts(&param_names(cfg), opt, step);
    safetensors_to_bytes(&tensors, &metadata)
}

/// Restore Adam's memory into `opt`. Returns the step the file was saved at.
pub fn load_optimizer(dir: &Path, cfg: &TransformerConfig, opt: &mut AdamW) -> io::Result<usize> {
    optimizer_from_loaded(read_safetensors(&dir.join("optimizer.safetensors"))?, cfg, opt)
}

/// Like `load_optimizer`, from bytes already in memory.
pub fn optimizer_from_bytes(bytes: &[u8], cfg: &TransformerConfig, opt: &mut AdamW) -> io::Result<usize> {
    optimizer_from_loaded(parse_safetensors(bytes)?, cfg, opt)
}

fn optimizer_from_loaded((mut tensors, meta): Loaded, cfg: &TransformerConfig, opt: &mut AdamW) -> io::Result<usize> {
    let mut take = |prefix: &str| -> io::Result<Vec<Matrix>> {
        param_names(cfg)
            .into_iter()
            .map(|n| tensors.remove(&format!("{prefix}.{n}")).ok_or_else(|| invalid(format!("missing {prefix}.{n}"))))
            .collect()
    };
    let (m, v) = (take("m")?, take("v")?);
    let updates = meta_usize(&meta, "updates")? as i32;
    opt.restore(updates, m, v).map_err(invalid)?;
    meta_usize(&meta, "step")
}

// ---- tokenizer ------------------------------------------------------------------

pub fn save_tokenizer(dir: &Path, tok: &Tokenizer) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("tokenizer.json"), tok.to_json())
}

pub fn load_tokenizer(dir: &Path) -> io::Result<Tokenizer> {
    let json = std::fs::read_to_string(dir.join("tokenizer.json"))?;
    Tokenizer::from_json(&json).map_err(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    /// A fresh scratch folder that is removed when dropped.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("tlm-test-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn small_model() -> Transformer {
        let cfg = TransformerConfig { vocab_size: 9, ctx: 6, d_model: 8, n_layer: 2, n_head: 2 };
        Transformer::new(cfg, &mut Rng::new(4))
    }

    #[test]
    fn tensors_round_trip_exactly() {
        let dir = TempDir::new("tensors");
        let a = Matrix::from_vec(2, 3, vec![1.0, -2.5, 3.25, 0.0, f32::MIN_POSITIVE, 1e9]);
        let b = Matrix::from_vec(1, 2, vec![7.0, 8.0]);
        let meta = BTreeMap::from([("hello".to_string(), "world".to_string())]);
        let path = dir.0.join("t.safetensors");
        write_safetensors(&path, &[("a".into(), &a), ("b".into(), &b)], &meta).unwrap();
        let (tensors, got_meta) = read_safetensors(&path).unwrap();
        assert_eq!(tensors["a"], a);
        assert_eq!(tensors["b"], b);
        assert_eq!(got_meta, meta);
    }

    #[test]
    fn file_follows_the_safetensors_layout() {
        let dir = TempDir::new("layout");
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let path = dir.0.join("t.safetensors");
        write_safetensors(&path, &[("a".into(), &a)], &BTreeMap::new()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!((8 + n) % 8, 0, "data must start 8-byte aligned");
        let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        assert_eq!(header["a"]["dtype"], "F32");
        assert_eq!(header["a"]["shape"], json!([1, 3]));
        assert_eq!(header["a"]["data_offsets"], json!([0, 12]));
        assert_eq!(bytes.len(), 8 + n + 12);
        assert!(!path.with_extension("tmp").exists(), "temp file must be renamed away");
    }

    #[test]
    fn corrupt_files_are_rejected_not_trusted() {
        let dir = TempDir::new("corrupt");
        let path = dir.0.join("bad.safetensors");

        // Header length claims far more bytes than exist.
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_safetensors(&path).is_err());

        // Offsets point past the end of the data.
        let header = br#"{"a":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8000]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_safetensors(&path).is_err());

        // Shape claims a size that disagrees with the offsets.
        let header = br#"{"a":{"dtype":"F32","shape":[100,100],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_safetensors(&path).is_err());

        // Truncated file.
        std::fs::write(&path, [1, 2, 3]).unwrap();
        assert!(read_safetensors(&path).is_err());
    }

    #[test]
    fn in_memory_bytes_equal_the_file_and_round_trip() {
        let dir = TempDir::new("bytes");
        let model = small_model();
        save_model(&dir.0, &model, 9).unwrap();
        let on_disk = std::fs::read(dir.0.join("model.safetensors")).unwrap();
        let in_memory = model_to_bytes(&model, 9).unwrap();
        assert_eq!(on_disk, in_memory, "file and in-memory export must be identical");

        let (loaded, step) = model_from_bytes(&in_memory).unwrap();
        assert_eq!(step, 9);
        assert_eq!(model.logits(&[3, 1, 2]).data, loaded.logits(&[3, 1, 2]).data);
    }

    #[test]
    fn saved_model_behaves_identically_after_loading() {
        let dir = TempDir::new("model");
        let model = small_model();
        save_model(&dir.0, &model, 123).unwrap();
        let (loaded, step) = load_model(&dir.0).unwrap();
        assert_eq!(step, 123);
        let ids = [1, 4, 2, 8, 0];
        assert_eq!(model.logits(&ids).data, loaded.logits(&ids).data);
    }

    #[test]
    fn resuming_continues_exactly_where_training_stopped() {
        // Train 10 steps straight, versus 5 steps + save + load + 5 steps.
        // The results must be bit-for-bit identical.
        let dir = TempDir::new("resume");
        let inputs = vec![1, 2, 3, 4, 5, 0];
        let targets = vec![2, 3, 4, 5, 0, 1];

        let train = |model: &mut Transformer, opt: &mut AdamW, steps: usize| {
            for _ in 0..steps {
                let (_, g) = model.loss_and_grads(&inputs, &targets, 6, 1);
                opt.step(&mut model.params, &g);
            }
        };

        let mut straight = small_model();
        let mut opt_a = AdamW::new(&straight.params, 1e-2, 0.01);
        train(&mut straight, &mut opt_a, 10);

        let mut first = small_model();
        let mut opt_b = AdamW::new(&first.params, 1e-2, 0.01);
        train(&mut first, &mut opt_b, 5);
        save_model(&dir.0, &first, 5).unwrap();
        save_optimizer(&dir.0, &first.cfg, &opt_b, 5).unwrap();

        let (mut resumed, step) = load_model(&dir.0).unwrap();
        let mut opt_c = AdamW::new(&resumed.params, 1e-2, 0.01);
        assert_eq!(load_optimizer(&dir.0, &resumed.cfg, &mut opt_c).unwrap(), step);
        train(&mut resumed, &mut opt_c, 5);

        for (a, b) in straight.params.iter().zip(&resumed.params) {
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn optimizer_bytes_round_trip_and_resume_exactly() {
        let model = small_model();
        let mut opt = AdamW::new(&model.params, 1e-2, 0.01);
        let (x, y) = (vec![1, 2, 3, 4, 5, 0], vec![2, 3, 4, 5, 0, 1]);
        let mut trained = small_model();
        for _ in 0..4 {
            let (_, g) = trained.loss_and_grads(&x, &y, 6, 1);
            opt.step(&mut trained.params, &g);
        }
        let bytes = optimizer_to_bytes(&trained.cfg, &opt, 4).unwrap();

        let mut restored = AdamW::new(&trained.params, 1e-2, 0.01);
        assert_eq!(optimizer_from_bytes(&bytes, &trained.cfg, &mut restored).unwrap(), 4);
        // Both optimizers must now take the identical next step.
        let (mut a, mut b) = (small_model(), small_model());
        a.params = trained.params.clone();
        b.params = trained.params.clone();
        let (_, g) = trained.loss_and_grads(&x, &y, 6, 1);
        opt.step(&mut a.params, &g);
        restored.step(&mut b.params, &g);
        for (p, q) in a.params.iter().zip(&b.params) {
            assert_eq!(p.data, q.data);
        }
        assert!(optimizer_from_bytes(&bytes[..20], &trained.cfg, &mut restored).is_err());
    }

    #[test]
    fn mismatched_model_file_is_an_error() {
        let dir = TempDir::new("mismatch");
        save_model(&dir.0, &small_model(), 0).unwrap();
        // Tamper with the config: claim 3 layers while the file holds 2.
        let (tensors, mut meta) = read_safetensors(&dir.0.join("model.safetensors")).unwrap();
        meta.insert("n_layer".into(), "3".into());
        let refs: Vec<(String, &Matrix)> = tensors.iter().map(|(k, v)| (k.clone(), v)).collect();
        write_safetensors(&dir.0.join("model.safetensors"), &refs, &meta).unwrap();
        assert!(load_model(&dir.0).is_err());
    }
}
