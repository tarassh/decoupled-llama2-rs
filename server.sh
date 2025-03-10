#!/bin/bash

# Navigate to the workspace folder
cd "$(dirname "$0")"

# Build the project
cargo build --release

# Run the server with the specified arguments
./target/release/decoupled-llama2-rs stories42M.bin -m server --address 127.0.0.1 --port 8010