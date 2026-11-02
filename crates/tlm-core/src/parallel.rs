//! Helpers for data-parallel training: several workers (threads, or browser Web
//! Workers) each hold an identical copy of the model.
//!
//! Every step:
//!   1. all workers draw the SAME batch (same random stream),
//!   2. worker `i` computes gradients for only its share of the batch,
//!   3. the shares are ADDED together (an "all-reduce"),
//!   4. every worker applies the same update, so the copies stay identical.
//!
//! Only the gradients ever travel between workers, flattened into one list of
//! numbers, with the share's loss as the last entry.

use crate::Matrix;

/// Which sequences of a `batch`-sized batch belong to worker `shard` of `n_shards`.
/// The ranges tile the batch exactly, even when it does not divide evenly.
pub fn shard_range(batch: usize, shard: usize, n_shards: usize) -> std::ops::Range<usize> {
    assert!(n_shards > 0 && shard < n_shards, "shard {shard} out of range for {n_shards} shards");
    (shard * batch / n_shards)..((shard + 1) * batch / n_shards)
}

/// Gradients plus this share's loss, as one flat list: [all gradient numbers..., loss].
pub fn grads_to_flat(grads: &[Matrix], loss: f32) -> Vec<f32> {
    let mut flat = Vec::with_capacity(grads.iter().map(|g| g.data.len()).sum::<usize>() + 1);
    for g in grads {
        flat.extend_from_slice(&g.data);
    }
    flat.push(loss);
    flat
}

/// Rebuild gradient matrices shaped like `params` from a flat list.
/// Returns the gradients and the (summed) loss stored in the last entry.
pub fn flat_to_grads(params: &[Matrix], flat: &[f32]) -> Result<(Vec<Matrix>, f32), String> {
    let total: usize = params.iter().map(|p| p.data.len()).sum();
    if flat.len() != total + 1 {
        return Err(format!("expected {} numbers (gradients + loss), got {}", total + 1, flat.len()));
    }
    let mut at = 0;
    let grads = params
        .iter()
        .map(|p| {
            let m = Matrix::from_vec(p.rows, p.cols, flat[at..at + p.data.len()].to_vec());
            at += p.data.len();
            m
        })
        .collect();
    Ok((grads, flat[total]))
}

/// dst += src, element by element (the "reduce" in all-reduce).
pub fn add_flat(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// A cheap fingerprint of all parameters, to verify that replicas have not drifted apart.
pub fn checksum(params: &[Matrix]) -> f64 {
    let mut sum = 0.0f64;
    for (i, p) in params.iter().enumerate() {
        for (j, &v) in p.data.iter().enumerate() {
            // Weight by position so swapped values change the result too.
            sum += v as f64 * (1.0 + ((i * 31 + j) % 17) as f64 * 0.0625);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_tile_the_batch_exactly() {
        for (batch, n) in [(16, 4), (7, 3), (5, 8), (1, 1), (12, 5)] {
            let mut covered = Vec::new();
            for s in 0..n {
                covered.extend(shard_range(batch, s, n));
            }
            assert_eq!(covered, (0..batch).collect::<Vec<_>>(), "batch {batch}, {n} shards");
        }
    }

    #[test]
    fn flatten_round_trip() {
        let params = vec![Matrix::zeros(2, 3), Matrix::zeros(1, 4)];
        let grads = vec![Matrix::from_vec(2, 3, (0..6).map(|i| i as f32).collect()), Matrix::from_vec(1, 4, vec![10.0, 11.0, 12.0, 13.0])];
        let flat = grads_to_flat(&grads, 0.5);
        assert_eq!(flat.len(), 11);
        let (back, loss) = flat_to_grads(&params, &flat).unwrap();
        assert_eq!(back, grads);
        assert_eq!(loss, 0.5);
        assert!(flat_to_grads(&params, &flat[..5]).is_err());
    }

    #[test]
    fn adding_flats_adds_losses_too() {
        let mut a = vec![1.0, 2.0, 0.25];
        add_flat(&mut a, &[10.0, 20.0, 0.5]);
        assert_eq!(a, vec![11.0, 22.0, 0.75]);
    }

    #[test]
    fn checksum_notices_changes() {
        let mut p = vec![Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0])];
        let before = checksum(&p);
        assert_eq!(before, checksum(&p));
        p[0].data.swap(0, 2);
        assert_ne!(before, checksum(&p));
    }
}
