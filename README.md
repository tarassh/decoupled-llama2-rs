# Decoupled Llama2-rs

A Rust implementation of Llama2 with a client-server architecture that separates the transformer model into components that communicate over a network.

## Overview

This repository contains an implementation of the Llama2 transformer model with a network-decoupled architecture. The transformer is split into client and server components that can run on different machines and communicate through TCP/IP.

### Features

- **Separated Architecture**: The transformer model is divided into client and server components
- **Network Communication**: Run computation on a server while clients connect remotely
- **Memory Efficiency**: Model weights remain on the server only
- **Multi-client Support**: Multiple clients can connect to a single model server
- **Docker Integration**: Available as a containerized service

## Architecture

This implementation separates the transformer model into two main components:

### TransformerClient

The client handles:
- Loading tokenizers
- Encoding input text
- Sending token embeddings to the server
- Receiving logits from the server
- Sampling the next token
- Managing the generation loop

### TransformerServer

The server manages:
- Loading the model weights
- Storing the KV cache
- Executing the transformer computation
- Processing client requests
- Returning results to clients

### Communication Process

1. Client tokenizes the input prompt
2. For each token position:
   - Client sends token embedding to server
   - Server processes the token through transformer layers
   - Server calculates logits and returns them to client
   - Client samples the next token from the logits
   - Process continues until generation is complete

## Installation

### Requirements

- Rust 1.70 or higher
- A Llama2 model checkpoint

### Getting Test Data

For testing purposes, you can download a small model checkpoint:

```bash
# Download a 42M parameter model for testing
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
```

This provides a small Llama2 model trained on a stories dataset, which is suitable for trying out the application.

### Building from Source

```bash
git clone https://github.com/yourusername/decoupled-llama2-rs.git
cd decoupled-llama2-rs
cargo build --release
```

### Docker

You can also use Docker:

```bash
# Build and run the container
./run_docker.sh
```

## Usage

The application runs in either client or server mode.

### Server Mode

Start a server that loads the model and accepts client connections:

```bash
./target/release/decoupled-llama2-rs -m server -c path/to/model.bin -a 0.0.0.0 -o 8010
```

### Client Mode

Connect to a server and generate text:

```bash
./target/release/decoupled-llama2-rs -m client -a server.ip.address -o 8010 -i "Once upon a time"
```

## Command Line Options

```
OPTIONS:
    -c, --checkpoint <string>     Path to the checkpoint file (required for server mode)
    -t, --temperature <float>     Temperature for sampling [default: 1.0]
    -p, --p_value <float>         P value for nucleus sampling [default: 0.9]
    -s, --seed <int>              Random seed
    -n, --steps <int>             Number of tokens to generate [default: 256]
    -i, --input_prompt <string>   Input prompt
    -z, --tokenizer <string>      Path to tokenizer [default: tokenizer.bin]
    -m, --mode <string>           Mode: client|server [default: client]
    -y, --system_prompt <string>  System prompt for chat mode
    -a, --address <string>        Server address to bind/connect to [default: 127.0.0.1]
    -o, --port <int>              Server port [default: 8080]
```

## Technical Details

### Transformer Architecture

The implementation uses the Llama2 architecture with:
- RMSNorm for layer normalization
- SwiGLU activation for feed-forward layers
- Rotary positional encoding (RoPE)
- Multi-query attention

### Decoupling Approach

The forward pass of the transformer is separated as follows:

1. **Client Side**: 
   - Manages token embeddings
   - Provides user interface

2. **Server Side**:
   - Processes tokens through attention mechanisms
   - Maintains the key-value cache
   - Calculates and returns the logits

### Network Protocol

Client-server communication uses a binary protocol over TCP:
- Length-prefixed messages with bincode serialization
- Request/response pattern with unique request IDs
- Token embeddings sent from client to server
- Logits returned from server to client

## Performance Notes

- Servers need sufficient RAM to store the model
- Multiple clients can use a single server instance
- Network conditions affect generation speed
- The server keeps separate state for each client

## License

[Insert your license here]

## Acknowledgments

- Based on the Llama2 model architecture by Meta AI
- Inspired by [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 