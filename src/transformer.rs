use crate::utils::{matmul, rmsnorm};
use ctrlc;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::io::{ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::{mem, ptr};

// Configuration for the transformer architecture
#[derive(Debug, Clone)]
pub struct Config {
    pub dim: i32,        // transformer dimension
    pub hidden_dim: i32, // for ffn layers
    pub n_layers: i32,   // number of layers
    pub n_heads: i32,    // number of query heads
    pub n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32,    // max sequence length
}

#[derive(Debug, Clone)]
pub struct TransformerTokenTable {
    pub token_embedding_table: Vec<f32>, // (vocab_size, dim)
}

// Weights for the transformer model
#[derive(Debug, Clone)]
pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    pub wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<f32>, // (layer, hidden_dim, dim)
    pub w2: Vec<f32>, // (layer, dim, hidden_dim)
    pub w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Option<Vec<f32>>,
}

// State for running the transformer
#[derive(Debug)]
pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,      // activation at current time stamp (dim,)
    pub xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,      // query (dim,)
    pub att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,   // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

// Data structures for network communication
#[derive(Serialize, Deserialize, Debug)]
struct ClientRequest {
    token_embedding: Vec<f32>,
    position: i32,
    request_id: u16,
}

#[derive(Serialize, Deserialize, Debug)]
struct ServerResponse {
    logits: Vec<f32>,
    request_id: u16,
}

// Client transformer struct
#[derive(Debug)]
pub struct TransformerClient {
    pub config: Config,
    pub tokens: TransformerTokenTable, // the token embeddings
    connection: Option<TcpStream>,
    pub request_counter: usize,
    pub sent_total: usize,
    pub recv_total: usize,
}

impl TransformerClient {
    pub fn new(checkpoint_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the file
        let mut file = File::open(checkpoint_path)?;

        // Read the config header
        let mut config_bytes = vec![0u8; mem::size_of::<Config>()];
        file.read_exact(&mut config_bytes)?;

        // Convert bytes to Config struct
        let mut config: Config =
            unsafe { ptr::read_unaligned(config_bytes.as_ptr() as *const Config) };

        // Handle shared weights flag (negative vocab_size signals unshared weights)
        config.vocab_size = config.vocab_size.abs();

        // Create memory map
        let mmap = unsafe { Mmap::map(&file)? };

        // Calculate the offset to weights data
        let weights_offset = mem::size_of::<Config>();

        let tokens = memory_map_tokens(&config, &mmap[weights_offset..])?;

        Ok(Self {
            config,
            tokens,
            connection: None,
            request_counter: 0,
            sent_total: 0,
            recv_total: 0,
        })
    }

    // Connect to a server over the network
    pub fn connect(&mut self, server_address: &str) -> Result<(), Box<dyn std::error::Error>> {
        let stream = TcpStream::connect(server_address)?;

        // Set timeouts instead of non-blocking mode
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        // Use blocking mode with timeouts
        stream.set_nonblocking(false)?;

        self.connection = Some(stream);
        self.request_counter = 0;
        self.sent_total = 0;
        self.recv_total = 0;
        Ok(())
    }

    // Disconnect from server
    pub fn disconnect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.connection = None;
        Ok(())
    }

    // Check if connected to a server
    pub fn is_connected(&self) -> bool {
        self.connection.is_some()
    }

    pub fn forward(
        &mut self,
        token: i32,
        pos: i32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let p = &self.config;
        let dim = p.dim as usize;

        // Get the token embedding
        let content_row =
            &self.tokens.token_embedding_table[token as usize * dim..(token as usize + 1) * dim];

        // Create a Vec from the content_row slice
        let mut token_embedding = vec![0.0; dim];
        token_embedding.copy_from_slice(content_row);

        // If connected to a server, send the token embedding and position for processing
        if let Some(stream) = &mut self.connection {
            // Create a request with a unique ID
            let request_id = self.request_counter;
            self.request_counter += 1;

            let request = ClientRequest {
                token_embedding: token_embedding.clone(),
                position: pos,
                request_id: request_id as u16,
            };

            // Serialize the request
            let serialized = bincode::serialize(&request)?;

            // Send the length of the data first (as u64)
            let len = serialized.len() as u64;
            stream.write_all(&len.to_le_bytes())?;

            // Send the actual data
            stream.write_all(&serialized)?;

            // Ensure data is sent immediately
            stream.flush()?;

            self.sent_total += len as usize + mem::size_of::<u64>();

            // Read the response length
            let mut len_bytes = [0u8; 8];
            stream.read_exact(&mut len_bytes)?;
            let response_len = u64::from_le_bytes(len_bytes) as usize;

            // Sanity check for response size
            if response_len > 100_000_000 {
                return Err(format!("Response too large: {} bytes", response_len).into());
            }

            // Read the response data
            let mut response_data = vec![0u8; response_len];
            stream.read_exact(&mut response_data)?;

            self.recv_total += response_len + mem::size_of::<u64>();

            // Deserialize the response
            let response: ServerResponse = bincode::deserialize(&response_data)?;

            // Verify it's the response for our request
            if response.request_id != request_id as u16 {
                return Err(format!(
                    "Received response for wrong request: expected {} got {}",
                    request_id, response.request_id
                )
                .into());
            }

            return Ok(response.logits);
        }

        // If not connected to server, return the embedding directly
        Ok(token_embedding)
    }

    // Non-blocking version using a separate thread
    pub fn forward_async(
        &self,
        token: i32,
        pos: i32,
        callback: impl FnOnce(Result<Vec<f32>, Box<dyn std::error::Error>>) + Send + 'static,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Clone necessary data for the thread
        let p = &self.config;
        let dim = p.dim as usize;

        // Get the token embedding
        let content_row =
            &self.tokens.token_embedding_table[token as usize * dim..(token as usize + 1) * dim];

        let mut token_embedding = vec![0.0; dim];
        token_embedding.copy_from_slice(content_row);

        // Clone the connection if it exists
        if let Some(stream) = &self.connection {
            let mut stream_clone = stream.try_clone()?;
            let request_id = self.request_counter;

            // Spawn a thread to handle the request asynchronously
            thread::spawn(move || {
                let request = ClientRequest {
                    token_embedding: token_embedding.clone(),
                    position: pos,
                    request_id: request_id as u16,
                };

                // Process request in background thread
                let result = (|| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                    // Serialize the request
                    let serialized = bincode::serialize(&request)?;

                    // Send length + data
                    let len = serialized.len() as u64;
                    stream_clone.write_all(&len.to_le_bytes())?;
                    stream_clone.write_all(&serialized)?;
                    stream_clone.flush()?;

                    // Read response length
                    let mut len_bytes = [0u8; 8];
                    stream_clone.read_exact(&mut len_bytes)?;
                    let response_len = u64::from_le_bytes(len_bytes) as usize;

                    // Sanity check for response size
                    if response_len > 100_000_000 {
                        return Err(format!("Response too large: {} bytes", response_len).into());
                    }

                    // Read response data
                    let mut response_data = vec![0u8; response_len];
                    stream_clone.read_exact(&mut response_data)?;

                    // Deserialize response
                    let response: ServerResponse = bincode::deserialize(&response_data)?;

                    // Verify it's the right response
                    if response.request_id != request_id as u16 {
                        return Err(format!(
                            "Received response for wrong request: expected {} got {}",
                            request_id, response.request_id
                        )
                        .into());
                    }

                    Ok(response.logits)
                })();

                // Call the callback with the result
                callback(result);
            });

            return Ok(());
        }

        // If not connected, call callback with just the embedding
        callback(Ok(token_embedding));
        Ok(())
    }
}

// Server transformer struct
#[derive(Debug)]
pub struct TransformerServer {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState,
    listener: Option<TcpListener>,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    running: Arc<Mutex<bool>>,
}

impl TransformerServer {
    pub fn new(checkpoint_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the file
        let mut file = File::open(checkpoint_path)?;

        // Read the config header
        let mut config_bytes = vec![0u8; mem::size_of::<Config>()];
        file.read_exact(&mut config_bytes)?;

        // Convert bytes to Config struct
        let mut config: Config =
            unsafe { ptr::read_unaligned(config_bytes.as_ptr() as *const Config) };

        // Handle shared weights flag (negative vocab_size signals unshared weights)
        let shared_weights = config.vocab_size > 0;
        config.vocab_size = config.vocab_size.abs();

        // Create memory map
        let mmap = unsafe { Mmap::map(&file)? };

        // Calculate the offset to weights data
        let weights_offset = mem::size_of::<Config>();

        // Create weights from the mapped memory
        let weights = memory_map_weights(&config, &mmap[weights_offset..], shared_weights)?;

        // Create run state buffers
        let state = RunState::new(&config);

        Ok(Self {
            config,
            weights,
            state,
            listener: None,
            clients: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
        })
    }

    // Start the server on the specified address
    pub fn start(&mut self, address: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create a TCP listener
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;

        let listener_clone = listener.try_clone()?;
        self.listener = Some(listener);
        *self.running.lock().unwrap() = true;

        // Clone Arc references for the thread
        let clients = Arc::clone(&self.clients);
        let running = Arc::clone(&self.running);

        // Clone the config and weights for client threads to use
        let server_config = self.config.clone();
        let server_weights = self.weights.clone();

        // Spawn a thread to accept new connections
        thread::spawn(move || {
            // Continue accepting connections as long as the server is running
            while *running.lock().unwrap() {
                // Accept new connections
                match listener_clone.accept() {
                    Ok((stream, addr)) => {
                        println!("New connection: {}", addr);

                        // Add the client to our list
                        clients.lock().unwrap().push(stream.try_clone().unwrap());

                        // Clone Arc references for the client handler thread
                        let client_running = Arc::clone(&running);

                        // Clone config and weights for this client
                        let client_config = server_config.clone();
                        let client_weights = server_weights.clone();

                        // Spawn a thread to handle this client
                        thread::spawn(move || {
                            println!("Starting client handler thread");

                            // Make a mutable copy of the stream for this thread
                            match stream.try_clone() {
                                Ok(mut client_stream) => {
                                    // Set the client socket to blocking mode for the handler thread
                                    if let Err(e) = client_stream.set_nonblocking(false) {
                                        println!("Error setting socket to blocking mode: {}", e);
                                        return;
                                    }

                                    // Create a server instance for this client
                                    let config_clone = client_config.clone(); // Clone for RunState::new
                                    let mut server = TransformerServer {
                                        config: client_config,
                                        weights: client_weights,
                                        state: RunState::new(&config_clone),
                                        listener: None,
                                        clients: Arc::new(Mutex::new(Vec::new())),
                                        running: Arc::new(Mutex::new(false)),
                                    };

                                    // Handle client requests until disconnected or server stops
                                    while *client_running.lock().unwrap() {
                                        // Read the request length
                                        let mut len_bytes = [0u8; 8];
                                        match client_stream.read_exact(&mut len_bytes) {
                                            Ok(_) => {
                                                let request_len =
                                                    u64::from_le_bytes(len_bytes) as usize;
                                                if request_len > 100_000_000 {
                                                    // Sanity check for request size
                                                    println!(
                                                        "Request too large: {} bytes",
                                                        request_len
                                                    );
                                                    break;
                                                }

                                                // Read the request data
                                                let mut request_data = vec![0u8; request_len];
                                                if let Err(e) =
                                                    client_stream.read_exact(&mut request_data)
                                                {
                                                    println!("Error reading from client: {}", e);
                                                    break;
                                                }

                                                // Process the request
                                                match bincode::deserialize::<ClientRequest>(
                                                    &request_data,
                                                ) {
                                                    Ok(request) => {
                                                        // Process using server's forward function
                                                        match server.forward(
                                                            &request.token_embedding,
                                                            request.position,
                                                        ) {
                                                            Ok(logits) => {
                                                                // Create the response
                                                                let response = ServerResponse {
                                                                    logits: logits.clone(),
                                                                    request_id: request.request_id,
                                                                };

                                                                // Serialize and send the response
                                                                if let Ok(serialized) =
                                                                    bincode::serialize(&response)
                                                                {
                                                                    let len =
                                                                        serialized.len() as u64;
                                                                    if let Err(e) = client_stream
                                                                        .write_all(
                                                                            &len.to_le_bytes(),
                                                                        )
                                                                    {
                                                                        println!(
                                                                            "Error sending response length: {}",
                                                                            e
                                                                        );
                                                                        break;
                                                                    }
                                                                    if let Err(e) = client_stream
                                                                        .write_all(&serialized)
                                                                    {
                                                                        println!(
                                                                            "Error sending response data: {}",
                                                                            e
                                                                        );
                                                                        break;
                                                                    }
                                                                    // Ensure data is sent immediately
                                                                    if let Err(e) =
                                                                        client_stream.flush()
                                                                    {
                                                                        println!(
                                                                            "Error flushing data: {}",
                                                                            e
                                                                        );
                                                                        break;
                                                                    }
                                                                } else {
                                                                    println!(
                                                                        "Error serializing response"
                                                                    );
                                                                    break;
                                                                }
                                                            }
                                                            Err(e) => {
                                                                println!(
                                                                    "Error in forward pass: {}",
                                                                    e
                                                                );
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => {
                                                        println!(
                                                            "Error deserializing request: {}",
                                                            e
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                println!("Client disconnected: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("Error cloning client socket: {}", e);
                                }
                            }

                            println!("Client handler thread exiting");
                        });
                    }
                    Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                        // No new connections, sleep a bit
                        thread::sleep(Duration::from_millis(100));
                    }
                    Err(e) => {
                        println!("Error accepting connection: {}", e);
                    }
                }
            }

            println!("Server listener thread exiting");
        });

        Ok(())
    }

    // Stop the server
    pub fn stop(&mut self) {
        *self.running.lock().unwrap() = false;
        self.listener = None;
        self.clients.lock().unwrap().clear();
    }

    // Check if the server is running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    pub fn forward(&mut self, x: &[f32], pos: i32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let p = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let dim = p.dim as usize;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
        let hidden_dim = p.hidden_dim as usize;
        let head_size = dim / p.n_heads as usize;

        s.x.copy_from_slice(x);

        // TODO: Implement the forward pass
        for l in 0..p.n_layers as usize {
            // Attention rmsnorm
            rmsnorm(
                &mut s.xb,
                &s.x,
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
            );

            // Key and value point to the kv cache
            let loff = l * p.seq_len as usize * kv_dim as usize; // kv cache layer offset
            let pos_offset = loff + pos as usize * kv_dim as usize;

            // QKV matmuls for this position
            matmul(
                &mut s.q,
                &s.xb,
                &w.wq[l * dim * dim..(l + 1) * dim * dim],
                dim,
                dim,
            );

            // Write directly into the key and value caches
            matmul(
                &mut s.key_cache[pos_offset..pos_offset + kv_dim as usize],
                &s.xb,
                &w.wk[l * dim * kv_dim as usize..(l + 1) * dim * kv_dim as usize],
                dim,
                kv_dim as usize,
            );
            matmul(
                &mut s.value_cache[pos_offset..pos_offset + kv_dim as usize],
                &s.xb,
                &w.wv[l * dim * kv_dim as usize..(l + 1) * dim * kv_dim as usize],
                dim,
                kv_dim as usize,
            );

            // RoPE relative positional encoding
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0f32 / (10000.0f32).powf(head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim as usize { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only

                for v in 0..rotn {
                    let vec = if v == 0 {
                        &mut s.q
                    } else {
                        &mut s.key_cache[pos_offset..pos_offset + kv_dim as usize]
                    };
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multihead attention
            for h in 0..p.n_heads as usize {
                // Get the query vector for this head
                let q_start = h * head_size;
                let q = &s.q[q_start..q_start + head_size];
                let att = &mut s.att[h * p.seq_len as usize..];
                let xb = &mut s.xb[h * head_size..(h + 1) * head_size];

                // Calculate attention scores
                for t in 0..=pos as usize {
                    // Get the key vector for this head and timestep
                    let k_start = loff + t * kv_dim as usize + (h / kv_mul as usize) * head_size;
                    let k = &s.key_cache[k_start..k_start + head_size];

                    // Calculate attention score as dot product of q and k
                    let score = q
                        .iter()
                        .zip(k.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f32>()
                        / (head_size as f32).sqrt();

                    att[t] = score;
                }

                // Softmax the scores
                let att_slice = &mut att[0..=pos as usize];
                crate::utils::softmax(att_slice);

                // Weighted sum of the values
                xb.fill(0.0);

                for t in 0..=pos as usize {
                    // Get the value vector for this head and timestep
                    let v_start = loff + t * kv_dim as usize + (h / kv_mul as usize) * head_size;
                    let v = &s.value_cache[v_start..v_start + head_size];
                    let a = att[t];

                    // Accumulate weighted value
                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            }

            // Final matmul to get the output of the attention
            matmul(
                &mut s.xb2,
                &s.xb,
                &w.wo[l * dim * dim..(l + 1) * dim * dim],
                dim,
                dim,
            );

            // Residual connection back into x
            for i in 0..dim {
                s.x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            rmsnorm(
                &mut s.xb,
                &s.x,
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
            );

            // Calculate self.w1(x) and self.w3(x)
            matmul(
                &mut s.hb,
                &s.xb,
                &w.w1[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
                dim,
                hidden_dim,
            );
            matmul(
                &mut s.hb2,
                &s.xb,
                &w.w3[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
                dim,
                hidden_dim,
            );

            // SwiGLU non-linearity
            for i in 0..hidden_dim {
                let val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                s.hb[i] = val * (1.0 / (1.0 + (-val).exp())) * s.hb2[i];
            }

            // Final matmul to get the output of the ffn
            matmul(
                &mut s.xb,
                &s.hb,
                &w.w2[l * hidden_dim * dim..(l + 1) * hidden_dim * dim],
                hidden_dim,
                dim,
            );

            // Residual connection
            for i in 0..dim {
                s.x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        let mut x_copy = s.x.clone();
        rmsnorm(&mut x_copy, &s.x, &w.rms_final_weight, dim);
        s.x.copy_from_slice(&x_copy);

        // Classifier into logits
        if let Some(wcls) = &w.wcls {
            matmul(&mut s.logits, &s.x, wcls, dim, p.vocab_size as usize);
        } else {
            // If no classifier weights, use token embedding weights
            matmul(
                &mut s.logits,
                &s.x,
                &w.token_embedding_table,
                dim,
                p.vocab_size as usize,
            );
        }

        Ok(s.logits.clone())
    }

    // Run the server in blocking mode until shutdown
    pub fn run_blocking(&mut self, address: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Start the server
        self.start(address)?;

        println!(
            "Server started and running on {}. Press Ctrl+C to stop.",
            address
        );

        // Create a channel to listen for shutdown signals
        let (shutdown_tx, shutdown_rx) = mpsc::channel();

        // Set up a Ctrl+C handler
        ctrlc::set_handler(move || {
            println!("Received shutdown signal. Gracefully shutting down...");
            let _ = shutdown_tx.send(());
        })?;

        // Block until we receive a shutdown signal
        let _ = shutdown_rx.recv();

        // Stop the server
        self.stop();
        println!("Server shutdown complete.");

        Ok(())
    }
}

// Helper implementation for RunState
impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let n_heads = config.n_heads as usize;
        let seq_len = config.seq_len as usize;
        let n_layers = config.n_layers as usize;
        let vocab_size = config.vocab_size as usize;
        let kv_dim = (config.dim * config.n_kv_heads / config.n_heads) as usize;

        RunState {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            att: vec![0.0; n_heads * seq_len],
            logits: vec![0.0; vocab_size],
            key_cache: vec![0.0; n_layers * seq_len * kv_dim],
            value_cache: vec![0.0; n_layers * seq_len * kv_dim],
        }
    }
}

fn memory_map_tokens(
    config: &Config,
    weight_data: &[u8],
) -> Result<TransformerTokenTable, Box<dyn std::error::Error>> {
    // Convert the byte slice to f32 slice
    let weight_data = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const f32,
            weight_data.len() / mem::size_of::<f32>(),
        )
    };

    // Helper to get or skip weights
    let mut offset = 0;
    let mut get_weights = |size: usize, skip: bool| {
        let slice = &weight_data[offset..offset + size];
        offset += size;
        if skip {
            vec![] // Return empty vec when skipping
        } else {
            slice.to_vec()
        }
    };

    let dim = config.dim as usize;
    let vocab_size = config.vocab_size as usize;

    // Get token embeddings
    let token_embedding_table = get_weights(vocab_size * dim, false);

    // Create the token table structure
    let token_table = TransformerTokenTable {
        token_embedding_table,
    };

    Ok(token_table)
}

fn memory_map_weights(
    config: &Config,
    weight_data: &[u8],
    shared_weights: bool,
) -> Result<TransformerWeights, Box<dyn std::error::Error>> {
    // Convert the byte slice to f32 slice
    let weight_data = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const f32,
            weight_data.len() / mem::size_of::<f32>(),
        )
    };

    // Calculate sizes for each weight tensor
    let dim = config.dim as usize;
    let hidden_dim = config.hidden_dim as usize;
    let n_layers = config.n_layers as usize;
    let vocab_size = config.vocab_size as usize;
    let head_size = dim / config.n_heads as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let seq_len = config.seq_len as usize;

    // Helper to get or skip weights
    let mut offset = 0;
    let mut get_weights = |size: usize, skip: bool| {
        let slice = &weight_data[offset..offset + size];
        offset += size;
        if skip {
            vec![] // Return empty vec when skipping
        } else {
            slice.to_vec()
        }
    };

    // Get token embeddings
    let token_embedding_table = get_weights(vocab_size * dim, false);

    // Get attention weights
    let rms_att_weight = get_weights(n_layers * dim, false);

    // Get query, key, value projection weights
    let wq = get_weights(n_layers * dim * (n_heads * head_size), false);
    let wk = get_weights(n_layers * dim * (n_kv_heads * head_size), false);
    let wv = get_weights(n_layers * dim * (n_kv_heads * head_size), false);
    let wo = get_weights(n_layers * (n_heads * head_size) * dim, false);

    // Get FFN weights
    let rms_ffn_weight = get_weights(n_layers * dim, false);
    let w1 = get_weights(n_layers * dim * hidden_dim, false);
    let w2 = get_weights(n_layers * hidden_dim * dim, false);
    let w3 = get_weights(n_layers * dim * hidden_dim, false);

    // Get final normalization weights
    let rms_final_weight = get_weights(dim, false);

    // Skip RoPE frequency tables
    get_weights(seq_len * head_size / 2, true); // skip freq_cis_real
    get_weights(seq_len * head_size / 2, true); // skip freq_cis_imag

    // Create the weights structure
    let weights = TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        rms_final_weight,
        wcls: if shared_weights {
            None
        } else {
            Some(get_weights(vocab_size * dim, false))
        },
    };

    Ok(weights)
}

// Example function showing how to use the client-server over the internet
#[allow(dead_code)]
pub fn example_internet_client_server_usage() -> Result<(), Box<dyn std::error::Error>> {
    // This is a simplified example - in real usage, the server and client would be on different machines

    // ---------- Server Example -----------
    // Use this in the main function when running in server mode
    #[allow(dead_code)]
    fn run_server() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize server
        let mut server = TransformerServer::new("path/to/your/checkpoint.bin")?;

        // Run in blocking mode - this will keep the process alive until Ctrl+C
        server.run_blocking("127.0.0.1:3000")?;

        Ok(())
    }

    // ---------- Client Example -----------
    // Use this in the main function when running in client mode
    #[allow(dead_code)]
    fn run_client() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize client
        let mut client = TransformerClient::new("path/to/your/checkpoint.bin")?;

        // Connect to server
        client.connect("127.0.0.1:3000")?;
        println!("Client connected to server");

        // Example: Send a token and receive result synchronously
        let token = 100; // Example token ID
        let pos = 0; // Position in sequence

        let result = client.forward(token, pos)?;
        println!("Received result from server with {} elements", result.len());

        // Disconnect client
        client.disconnect()?;

        Ok(())
    }

    // For demonstration purposes, we're just returning Ok
    // In a real application, you would call either run_server() or run_client()
    // based on command line arguments
    Ok(())
}
