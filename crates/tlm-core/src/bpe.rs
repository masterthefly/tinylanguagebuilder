//! Byte-level BPE tokenizer.
//!
//! Token ids 0..=255 are the raw bytes. Every merge adds one new id:
//! id 256 = first merge, id 257 = second merge, and so on.
//! Because we start from bytes, ANY text can be encoded (no "unknown" token).

use std::collections::HashMap;

/// Split text into chunks so merges never cross word boundaries.
/// A chunk is: an optional leading space + a run of non-space, non-newline
/// bytes, or a single newline. So " the" is one chunk and "the" another.
fn pretokenize(bytes: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let start = i;
        if bytes[i] == b'\n' {
            i += 1;
        } else {
            if bytes[i] == b' ' {
                i += 1;
            }
            while i < n && bytes[i] != b' ' && bytes[i] != b'\n' {
                i += 1;
            }
        }
        chunks.push(&bytes[start..i]);
    }
    chunks
}

pub struct Tokenizer {
    /// merges[k] = the pair of ids that was merged to create id 256 + k.
    pub merges: Vec<(usize, usize)>,
    /// vocab[id] = the bytes that token stands for.
    pub vocab: Vec<Vec<u8>>,
    /// pair -> (rank, new id), for fast encoding.
    ranks: HashMap<(usize, usize), (usize, usize)>,
}

impl Tokenizer {
    fn from_merges(merges: Vec<(usize, usize)>) -> Self {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut ranks = HashMap::new();
        for (k, &(a, b)) in merges.iter().enumerate() {
            let mut bytes = vocab[a].clone();
            bytes.extend_from_slice(&vocab[b]);
            vocab.push(bytes);
            ranks.insert((a, b), (k, 256 + k));
        }
        Tokenizer { merges, vocab, ranks }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Serialise to JSON. The merge list alone fully defines the tokenizer.
    pub fn to_json(&self) -> String {
        let merges: Vec<[usize; 2]> = self.merges.iter().map(|&(a, b)| [a, b]).collect();
        serde_json::json!({ "type": "byte-bpe", "merges": merges }).to_string()
    }

    /// Parse JSON written by `to_json`. Validates that every merge refers
    /// only to ids that already exist, so a bad file gives an error, not a panic.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let list = value["merges"].as_array().ok_or("tokenizer.json has no \"merges\" array")?;
        let mut merges = Vec::with_capacity(list.len());
        for (k, pair) in list.iter().enumerate() {
            let a = pair[0].as_u64().ok_or("merge entry is not a pair of integers")? as usize;
            let b = pair[1].as_u64().ok_or("merge entry is not a pair of integers")? as usize;
            if a >= 256 + k || b >= 256 + k {
                return Err(format!("merge {k} refers to a token that does not exist yet"));
            }
            merges.push((a, b));
        }
        Ok(Tokenizer::from_merges(merges))
    }

    /// Learn merges from text until the vocabulary reaches `vocab_size`.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        // Count each distinct chunk once; weight merges by how often it occurs.
        // This is what makes training fast: " the" is processed once, not 50,000 times.
        let mut counts: HashMap<&[u8], usize> = HashMap::new();
        for chunk in pretokenize(text.as_bytes()) {
            *counts.entry(chunk).or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<usize>, usize)> = counts
            .into_iter()
            .map(|(chunk, n)| (chunk.iter().map(|&b| b as usize).collect(), n))
            .collect();

        let mut merges = Vec::new();
        while 256 + merges.len() < vocab_size {
            // Count all adjacent pairs, weighted by chunk frequency.
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for (tokens, n) in &words {
                for w in tokens.windows(2) {
                    *pair_counts.entry((w[0], w[1])).or_insert(0) += n;
                }
            }
            // Most frequent pair; ties broken by smallest pair so runs are reproducible.
            let best = pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(&pair, _)| pair);
            let Some(pair) = best else { break }; // nothing left to merge

            let new_id = 256 + merges.len();
            merges.push(pair);
            for (tokens, _) in words.iter_mut() {
                merge_pair(tokens, pair, new_id);
            }
        }
        Tokenizer::from_merges(merges)
    }

    /// Text -> token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cache: HashMap<&[u8], Vec<usize>> = HashMap::new();
        for chunk in pretokenize(text.as_bytes()) {
            if let Some(ids) = cache.get(chunk) {
                out.extend_from_slice(ids);
                continue;
            }
            let mut tokens: Vec<usize> = chunk.iter().map(|&b| b as usize).collect();
            // Repeatedly apply the earliest-learned merge present in the chunk.
            loop {
                let best = tokens
                    .windows(2)
                    .filter_map(|w| self.ranks.get(&(w[0], w[1])))
                    .min_by_key(|&&(rank, _)| rank);
                let Some(&(rank, new_id)) = best else { break };
                merge_pair(&mut tokens, self.merges[rank], new_id);
            }
            out.extend_from_slice(&tokens);
            cache.insert(chunk, tokens);
        }
        out
    }

    /// Token ids -> text. Invalid UTF-8 (possible if you decode a partial
    /// multi-byte character) becomes the replacement character.
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.vocab[id].iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Replace every adjacent occurrence of `pair` in `tokens` with `new_id`.
fn merge_pair(tokens: &mut Vec<usize>, pair: (usize, usize), new_id: usize) {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(tokens[i]);
            i += 1;
        }
    }
    *tokens = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_keeps_leading_space_with_word() {
        let chunks = pretokenize(b"the cat\nsat");
        let as_strs: Vec<&str> = chunks.iter().map(|c| std::str::from_utf8(c).unwrap()).collect();
        assert_eq!(as_strs, vec!["the", " cat", "\n", "sat"]);
    }

    #[test]
    fn first_merge_is_most_frequent_pair() {
        // "aa" is by far the most common pair.
        let t = Tokenizer::train("aaaa aaaa aaaa b", 257);
        assert_eq!(t.merges[0], (b'a' as usize, b'a' as usize));
        assert_eq!(t.vocab[256], b"aa".to_vec());
    }

    #[test]
    fn round_trip_is_lossless() {
        let t = Tokenizer::train("the cat sat on the mat. the dog sat too.", 300);
        for text in ["the cat sat", "unseen words: zebra!", "héllo wörld 😀", "  double  spaces\n\nand newlines", ""] {
            assert_eq!(t.decode(&t.encode(text)), text);
        }
    }

    #[test]
    fn training_compresses_text() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(50);
        let t = Tokenizer::train(&text, 300);
        let ids = t.encode(&text);
        assert!(ids.len() < text.len() / 3, "{} tokens for {} bytes", ids.len(), text.len());
    }

    #[test]
    fn vocab_size_matches_request() {
        let t = Tokenizer::train(&"hello world ".repeat(100), 270);
        assert_eq!(t.vocab_size(), t.merges.len() + 256);
        assert!(t.vocab_size() <= 270);
    }

    #[test]
    fn json_round_trip_keeps_behaviour() {
        let t = Tokenizer::train(&"the cat sat on the mat ".repeat(30), 290);
        let loaded = Tokenizer::from_json(&t.to_json()).unwrap();
        assert_eq!(loaded.merges, t.merges);
        assert_eq!(loaded.encode("the cat"), t.encode("the cat"));
    }

    #[test]
    fn bad_json_is_an_error_not_a_panic() {
        assert!(Tokenizer::from_json("not json").is_err());
        assert!(Tokenizer::from_json(r#"{"merges": [[0, 999]]}"#).is_err());
        assert!(Tokenizer::from_json(r#"{"merges": [[1]]}"#).is_err());
    }

    #[test]
    fn training_is_deterministic() {
        let text = "one fish two fish red fish blue fish ".repeat(20);
        let a = Tokenizer::train(&text, 290);
        let b = Tokenizer::train(&text, 290);
        assert_eq!(a.merges, b.merges);
    }
}
