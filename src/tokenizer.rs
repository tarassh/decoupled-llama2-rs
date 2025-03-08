use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::path::Path;

#[derive(Debug)]
pub struct TokenIndex {
    pub str_val: String, // Changed from char* to String
    pub id: i32,
}

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<String>,     // Changed from char** to Vec<String>
    pub vocab_scores: Vec<f32>, // Changed from float* to Vec<f32>
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: i32,
    pub max_token_length: u32,
    pub byte_pieces: [u8; 512], // Kept as fixed-size array for single-byte strings
}

impl Tokenizer {
    pub fn build_tokenizer<P: AsRef<Path>>(
        tokenizer_path: P,
        vocab_size: i32,
    ) -> Result<Self, std::io::Error> {
        // Initialize vectors with capacity
        let mut vocab = Vec::with_capacity(vocab_size as usize);
        let mut vocab_scores = Vec::with_capacity(vocab_size as usize);

        // Initialize byte_pieces
        let mut byte_pieces = [0u8; 512];
        for i in 0..256 {
            byte_pieces[i * 2] = i as u8;
            byte_pieces[i * 2 + 1] = 0;
        }

        // Open and read the tokenizer file
        let mut file = File::open(tokenizer_path)?;

        // Read max_token_length
        let mut max_token_length_bytes = [0u8; 4];
        file.read_exact(&mut max_token_length_bytes)?;
        let max_token_length = u32::from_le_bytes(max_token_length_bytes);

        // Read vocabulary and scores
        for _ in 0..vocab_size {
            // Read score
            let mut score_bytes = [0u8; 4];
            file.read_exact(&mut score_bytes)?;
            let score = f32::from_le_bytes(score_bytes);
            vocab_scores.push(score);

            // Read string length
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let len = i32::from_le_bytes(len_bytes) as usize;

            // Read string data
            let mut str_bytes = vec![0u8; len];
            file.read_exact(&mut str_bytes)?;

            // Convert bytes to string, handling non-UTF8 data
            let token_str = match String::from_utf8(str_bytes) {
                Ok(s) => s,
                Err(e) => {
                    // Handle non-UTF8 data by using lossy conversion
                    String::from_utf8_lossy(e.as_bytes()).into_owned()
                }
            };

            vocab.push(token_str);
        }

        if vocab.len() != vocab_size as usize {
            return Err(Error::new(
                ErrorKind::Other,
                "Failed to read expected number of tokens",
            ));
        }

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab: Vec::new(), // initialized lazily as in C version
            vocab_size,
            max_token_length,
            byte_pieces,
        })
    }

    // Example usage:
    pub fn new_from_file<P: AsRef<Path>>(path: P, vocab_size: i32) -> Result<Self, std::io::Error> {
        Self::build_tokenizer(path, vocab_size)
    }

    pub fn encode(&mut self, text: &str, bos: i8, eos: i8) -> Vec<i32> {
        let mut tokens = Vec::new();

        // Lazily initialize sorted_vocab if it hasn't been done yet
        if self.sorted_vocab.is_empty() {
            self.sorted_vocab = (0..self.vocab_size as usize)
                .map(|i| TokenIndex {
                    str_val: self.vocab[i].clone(),
                    id: i as i32,
                })
                .collect();

            // Sort the vocabulary
            self.sorted_vocab.sort_by(|a, b| a.str_val.cmp(&b.str_val));
        }

        // Add BOS token if requested
        if bos != 0 {
            tokens.push(1);
        }

        // Add dummy prefix for non-empty text
        if !text.is_empty() {
            if let Ok(idx) = self.str_lookup(" ") {
                tokens.push(idx);
            }
        }

        // Process UTF-8 bytes
        let mut str_buffer = Vec::with_capacity((self.max_token_length as usize * 2) + 3);
        let text_bytes = text.as_bytes();
        let mut i = 0;

        while i < text_bytes.len() {
            str_buffer.clear();

            // Read a complete UTF-8 sequence
            while i < text_bytes.len() {
                let c = text_bytes[i];
                str_buffer.push(c);
                i += 1;

                // If we're at the end or next byte is not a continuation byte
                if i >= text_bytes.len() || (text_bytes[i] & 0xC0) != 0x80 || str_buffer.len() >= 4
                {
                    break;
                }
            }

            // Try to find the sequence in vocabulary
            if let Ok(str_slice) = std::str::from_utf8(&str_buffer) {
                match self.str_lookup(str_slice) {
                    Ok(id) => {
                        // Found in vocab
                        tokens.push(id);
                    }
                    Err(_) => {
                        // Byte fallback: encode each byte separately
                        for &byte in &str_buffer {
                            tokens.push((byte as i32) + 3);
                        }
                    }
                }
            } else {
                // Invalid UTF-8: fallback to bytes
                for &byte in &str_buffer {
                    tokens.push((byte as i32) + 3);
                }
            }
        }

        // Merge tokens according to vocab scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = -1;
            let mut best_idx = -1;

            // Find best consecutive pair to merge
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = format!(
                    "{}{}",
                    &self.vocab[tokens[i] as usize],
                    &self.vocab[tokens[i + 1] as usize]
                );

                if let Ok(id) = self.str_lookup(&merged) {
                    let score = self.vocab_scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_id = id;
                        best_idx = i as i32;
                    }
                }
            }

            if best_idx == -1 {
                break;
            }

            // Perform the merge
            let idx = best_idx as usize;
            tokens[idx] = best_id;
            tokens.remove(idx + 1);
        }

        // Add EOS token if requested
        if eos != 0 {
            tokens.push(2);
        }

        tokens
    }

    // Helper function to look up a string in sorted_vocab
    fn str_lookup(&self, s: &str) -> Result<i32, i32> {
        match self
            .sorted_vocab
            .binary_search_by(|token| token.str_val.as_str().cmp(s))
        {
            Ok(idx) => Ok(self.sorted_vocab[idx].id),
            Err(_) => Err(-1),
        }
    }

    pub fn decode(&self, prev_token: i32, token: i32) -> String {
        if token < 0 || token >= self.vocab_size {
            return String::new();
        }

        let token_str = &self.vocab[token as usize];

        // Handle BOS token (1): strip leading whitespace
        if prev_token == 1 && token_str.starts_with(' ') {
            return token_str[1..].to_string();
        }

        // Handle raw byte tokens in format '<0xXX>'
        if token_str.len() == 6 && token_str.starts_with("<0x") && token_str.ends_with('>') {
            if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                // Return the corresponding byte piece
                return String::from_utf8_lossy(&[self.byte_pieces[(byte_val as usize) * 2]])
                    .into_owned();
            }
        }

        token_str.clone()
    }
}
