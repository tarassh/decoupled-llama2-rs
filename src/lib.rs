use std::time::{SystemTime, UNIX_EPOCH};

mod sampler;
mod tokenizer;
mod transformer;
mod utils;

pub use sampler::*;
pub use tokenizer::*;
pub use transformer::*;
pub use utils::*;

/// Returns the current time in milliseconds since the Unix epoch
pub fn time_in_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn generate(
    transformer: &mut TransformerClient,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    input_prompt: &str,
    steps: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Use empty string if input_prompt is empty
    let input_prompt = if input_prompt.is_empty() {
        ""
    } else {
        input_prompt
    };

    // Encode the prompt into tokens sequence
    let prompt_tokens = tokenizer.encode(input_prompt, 1, 0);
    if prompt_tokens.is_empty() {
        return Err("Expected at least 1 prompt token".into());
    }

    // Start the main loop
    let mut start_time = 0u128; // Used to time our code, only initialized after first iteration
    let mut token = prompt_tokens[0]; // Kick off with the first token in the prompt
    let mut pos = 0i32;

    while pos < steps {
        // Forward the transformer to get logits for the next token
        let logits = transformer.forward(token, pos)?;

        // Advance the state machine
        let next = if (pos as usize) < prompt_tokens.len() - 1 {
            // If we are still processing the input prompt, force the next prompt token
            prompt_tokens[pos as usize + 1]
        } else {
            // Sample the next token from the logits
            sampler.sample(&logits)
        };
        pos += 1;

        // Data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // Decode and print the token
        let piece = tokenizer.decode(token, next);
        print!("{}", piece);
        std::io::stdout().flush()?;
        token = next;

        // Init the timer here because the first iteration can be slower
        if start_time == 0 {
            start_time = time_in_ms();
        }
    }
    println!();

    // Report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end_time = time_in_ms();
        eprintln!(
            "achieved tok/s: {}",
            (pos - 1) as f64 / (end_time - start_time) as f64 * 1000.0
        );
        eprintln!("sent bytes/message: {}", transformer.sent_total / transformer.request_counter);
        eprintln!("received bytes/message: {}", transformer.recv_total / transformer.request_counter);
        eprintln!("sent messages: {}", transformer.request_counter);
    }

    Ok(())
}