use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
    pub surprise: f32,
}

pub struct ReplayBuffer {
    capacity: usize,
    records: Vec<Record>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            records: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn add(&mut self, r: Record) {
        if self.records.len() < self.capacity {
            self.records.push(r);
            return;
        }
        let (min_idx, _) = self
            .records
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.surprise.partial_cmp(&b.surprise).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        if r.surprise > self.records[min_idx].surprise {
            self.records[min_idx] = r;
        }
    }

    pub fn sample<R: Rng + ?Sized>(&self, batch: usize, rng: &mut R) -> Vec<&Record> {
        if self.records.is_empty() || batch == 0 {
            return Vec::new();
        }
        let weights: Vec<f32> = self.records.iter().map(|r| r.surprise.max(0.0) + 1e-6).collect();
        let dist = WeightedIndex::new(&weights).expect("non-empty weights");
        (0..batch).map(|_| &self.records[dist.sample(rng)]).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Record> {
        self.records.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_evicts_lowest_surprise() {
        let mut b = ReplayBuffer::new(3);
        b.add(Record { input_ids: vec![1], target_ids: vec![2], surprise: 0.1 });
        b.add(Record { input_ids: vec![3], target_ids: vec![4], surprise: 0.5 });
        b.add(Record { input_ids: vec![5], target_ids: vec![6], surprise: 0.9 });
        b.add(Record { input_ids: vec![7], target_ids: vec![8], surprise: 0.7 });
        assert_eq!(b.len(), 3);
        let surprises: Vec<f32> = b.iter().map(|r| r.surprise).collect();
        assert!(!surprises.contains(&0.1f32));
        assert!(surprises.contains(&0.7f32));
    }

    #[test]
    fn sample_distribution_correlates_with_surprise() {
        let mut b = ReplayBuffer::new(100);
        for i in 0..100u32 {
            b.add(Record {
                input_ids: vec![i],
                target_ids: vec![i + 1],
                surprise: i as f32 / 100.0,
            });
        }
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = vec![0u32; 100];
        for _ in 0..10000 {
            let s = b.sample(1, &mut rng);
            let id = s[0].input_ids[0] as usize;
            counts[id] += 1;
        }
        let low: u32 = counts[0..20].iter().sum();
        let high: u32 = counts[80..100].iter().sum();
        assert!(high > 3 * low, "high={high} low={low}");
    }
}
