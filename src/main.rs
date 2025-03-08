use clap::{Arg, Command};
use std::time::{SystemTime, UNIX_EPOCH};

use decoupled_llama2_rs::{TransformerClient, TransformerServer, Sampler, Tokenizer, generate};


fn main() ->Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("run")
        .version("1.0")
        .about("Runs the model with the specified options")
        .override_usage("run <checkpoint> [options]")
        .after_help("Example: run model.bin -n 256 -i \"Once upon a time\"")
        .arg(Arg::new("checkpoint")
            .help("Path to the checkpoint file")
            .required(true)
            .index(1)
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("temperature")
            .short('t')
            .long("temperature")
            .value_name("float")
            .help("temperature in [0,inf], default 1.0 (0.0 = greedy deterministic, 1.0 = original)")
            .value_parser(clap::value_parser!(f32)))
        .arg(Arg::new("p_value")
            .short('p')
            .long("p_value")
            .value_name("float")
            .help("p value in top-p (nucleus) sampling in [0,1], default 0.9 (1.0 = off)")
            .value_parser(clap::value_parser!(f32)))
        .arg(Arg::new("seed")
            .short('s')
            .long("seed")
            .value_name("int")
            .help("random seed, default: current time")
            .value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("steps")
            .short('n')
            .long("steps")
            .value_name("int")
            .help("number of steps to run for, default 256. 0 = max_seq_len")
            .value_parser(clap::value_parser!(i32)))
        .arg(Arg::new("input_prompt")
            .short('i')
            .long("input_prompt")
            .value_name("string")
            .help("input prompt")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("tokenizer")
            .short('z')
            .long("tokenizer")
            .value_name("string")
            .help("path to tokenizer, default: tokenizer.bin")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("mode")
            .short('m')
            .long("mode")
            .value_name("string")
            .help("mode: client|server, default: client")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("system_prompt")
            .short('y')
            .long("system_prompt")
            .value_name("string")
            .help("(optional) system prompt in chat mode")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("address")
            .short('a')
            .long("address")
            .value_name("string")
            .help("server address to bind to or connect to, default: 127.0.0.1")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("port")
            .short('o')
            .long("port")
            .value_name("int")
            .help("server port, default: 8080")
            .value_parser(clap::value_parser!(u16)))
        .get_matches();

    // Get current time as default seed
    let default_seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    // Get and validate parameters
    let checkpoint = matches
        .get_one::<String>("checkpoint")
        .ok_or("Checkpoint is required")?;

    let mut temperature = matches
        .get_one::<f32>("temperature")
        .copied()
        .unwrap_or(1.0);
    temperature = temperature.max(0.0); // ensure temperature >= 0.0

    let mut p_value = matches.get_one::<f32>("p_value").copied().unwrap_or(0.9);
    p_value = p_value.clamp(0.0, 1.0); // ensure p_value in [0.0, 1.0]

    let mut seed = matches.get_one::<u64>("seed").copied().unwrap_or(0);
    if seed == 0 {
        seed = default_seed;
    }

    let mut steps = matches.get_one::<i32>("steps").copied().unwrap_or(256);
    steps = steps.max(0); // ensure steps >= 0

    let input_prompt = matches
        .get_one::<String>("input_prompt")
        .map(|s| s.as_str())
        .unwrap_or("");
    let tokenizer = matches
        .get_one::<String>("tokenizer")
        .map(|s| s.as_str())
        .unwrap_or("tokenizer.bin");
    let mode = matches
        .get_one::<String>("mode")
        .map(|s| s.as_str())
        .unwrap_or("client");
    let system_prompt = matches
        .get_one::<String>("system_prompt")
        .map(|s| s.as_str())
        .unwrap_or("");

    let address = matches
        .get_one::<String>("address")
        .map(|s| s.as_str())
        .unwrap_or("127.0.0.1");
    let port = matches
        .get_one::<u16>("port")
        .copied()
        .unwrap_or(8080);

    // Debug output
    println!("Checkpoint: {}", checkpoint);
    println!("Temperature: {}", temperature);
    println!("P value: {}", p_value);
    println!("Seed: {}", seed);
    println!("Steps: {}", steps);
    println!("Input prompt: {}", input_prompt);
    println!("Tokenizer: {}", tokenizer);
    println!("Mode: {}", mode);
    println!("System prompt: {}", system_prompt);
    println!("Address: {}", address);
    println!("Port: {}", port);

    match mode {
        "server" => {
            println!("Starting server on {}:{}...", address, port);
            let mut server = TransformerServer::new(checkpoint)?;
            // Use run_blocking to keep the server running until Ctrl+C is pressed
            let server_addr = format!("{}:{}", address, port);
            server.run_blocking(&server_addr)?;
        } 
        "client" => {
            let mut client = TransformerClient::new(checkpoint)?;
            let server_addr = format!("{}:{}", address, port);
            client.connect(&server_addr)?;

            let mut tokenizer = Tokenizer::build_tokenizer(tokenizer, client.config.vocab_size)?;
            
            let mut sampler = Sampler::new(client.config.vocab_size, temperature, p_value, seed);

            generate(&mut client, &mut tokenizer, &mut sampler, &input_prompt, steps)?;
        }
        _ => {
            return Err("Invalid mode".into());
        }
    }

    Ok(())
}
