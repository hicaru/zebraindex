use zrag_hw_core::{Device, Hardware};

use crate::method::{SearchMethod, SearchParams};

const FLAT_MAX: usize = 1_024;
const TURBO_MAX: usize = 5_000_000;
const HNSW_RS_MAX: usize = 10_000;
const HYSTERESIS_PCT: u64 = 25;

#[inline]
pub fn recommend(n_chunks: usize, dim: usize, hw: &Hardware) -> SearchMethod {
    let bytes_per_vec = dim.saturating_mul(std::mem::size_of::<f32>());
    let est_full_vec_mem = (n_chunks as u64).saturating_mul(bytes_per_vec as u64);
    let mem_quarter = hw.mem_avail / 4;

    if n_chunks < FLAT_MAX {
        SearchMethod::Flat
    } else if hw.device != Device::Cpu && n_chunks < TURBO_MAX {
        SearchMethod::TurboQuant
    } else if n_chunks < HNSW_RS_MAX && est_full_vec_mem < mem_quarter {
        SearchMethod::Usearch
    } else {
        SearchMethod::IvfHnswSq
    }
}

#[inline]
pub fn choose_method(
    n_chunks: usize,
    dim: usize,
    hw: &Hardware,
    previous: Option<&SearchParams>,
) -> SearchParams {
    let mut method = recommend(n_chunks, dim, hw);

    if let Some(prev) = previous {
        let lo = prev
            .indexed_chunks
            .saturating_sub(prev.indexed_chunks * HYSTERESIS_PCT / 100);
        let hi = prev
            .indexed_chunks
            .saturating_add(prev.indexed_chunks * HYSTERESIS_PCT / 100);
        if (n_chunks as u64) >= lo && (n_chunks as u64) <= hi {
            method = prev.method;
        }
    }

    let num_partitions = (n_chunks as f64).sqrt().round() as u32;
    let num_partitions = num_partitions.clamp(4, 256);

    SearchParams {
        method,
        indexed_chunks: n_chunks as u64,
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        num_partitions,
        nprobes: (num_partitions / 10).max(4),
        refine_factor: 2,
        num_sub_vectors: (dim as u32 / 8).max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrag_hw_core::device::{Device, Hardware};

    fn hw(mem: u64) -> Hardware {
        Hardware {
            device: Device::Cpu,
            cpus: 8,
            mem_total: mem,
            mem_avail: mem,
            cpu: zrag_hw_core::CpuCaps::default(),
        }
    }

    #[test]
    fn flat_for_tiny() {
        let p = choose_method(500, 768, &hw(8 * 1024 * 1024 * 1024), None);
        assert_eq!(p.method, SearchMethod::Flat);
    }

    #[test]
    fn hnsw_rs_for_medium() {
        let p = choose_method(5_000, 768, &hw(32 * 1024 * 1024 * 1024), None);
        assert_eq!(p.method, SearchMethod::Usearch);
    }

    #[test]
    fn ivf_for_large() {
        let p = choose_method(50_000, 768, &hw(8 * 1024 * 1024 * 1024), None);
        assert_eq!(p.method, SearchMethod::IvfHnswSq);
    }

    #[test]
    fn ivf_when_memory_tight() {
        let p = choose_method(5_000, 768, &hw(1024), None);
        assert_eq!(p.method, SearchMethod::IvfHnswSq);
    }

    #[test]
    fn nprobes_at_least_4() {
        let p = choose_method(100, 768, &hw(8 * 1024 * 1024 * 1024), None);
        assert!(p.nprobes >= 4);
    }

    #[test]
    fn hysteresis_stays_on_hnsw_rs() {
        let prev = choose_method(9_900, 768, &hw(32 * 1024 * 1024 * 1024), None);
        assert_eq!(prev.method, SearchMethod::Usearch);
        let now = choose_method(10_100, 768, &hw(32 * 1024 * 1024 * 1024), Some(&prev));
        assert_eq!(now.method, SearchMethod::Usearch);
    }

    #[test]
    fn hysteresis_flips_when_far_enough() {
        let prev = choose_method(9_900, 768, &hw(32 * 1024 * 1024 * 1024), None);
        let now = choose_method(15_000, 768, &hw(32 * 1024 * 1024 * 1024), Some(&prev));
        assert_eq!(now.method, SearchMethod::IvfHnswSq);
    }

    #[test]
    fn num_sub_vectors_computed() {
        let p = choose_method(5_000, 768, &hw(32 * 1024 * 1024 * 1024), None);
        assert_eq!(p.num_sub_vectors, 96); // 768 / 8
    }

    #[test]
    fn recommend_matches_choose_method() {
        let big_hw = hw(32 * 1024 * 1024 * 1024);
        assert_eq!(recommend(500, 768, &big_hw), SearchMethod::Flat);
        assert_eq!(recommend(5_000, 768, &big_hw), SearchMethod::Usearch);
        assert_eq!(recommend(50_000, 768, &big_hw), SearchMethod::IvfHnswSq);
    }
}
