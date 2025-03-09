#!/bin/bash

# Navigate to the workspace folder
cd "$(dirname "$0")"

# Build the project
cargo build --release

# Run the server with the specified arguments
./target/release/decoupled-llama2-rs stories42M.bin -t 0.1 -n 256 -i "One day, Lily met a Shoggoth" -m client --address 127.0.0.1 --port 8010