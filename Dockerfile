FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Download the model *after* caching dependencies
RUN wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin -O /app/stories42M.bin

# Build application
COPY . .
RUN cargo build --release --bin decoupled-llama2-rs && strip target/release/decoupled-llama2-rs

FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=builder /app/target/release/decoupled-llama2-rs /usr/local/bin
COPY --from=builder /app/stories42M.bin /app/stories42M.bin

CMD ["/usr/local/bin/decoupled-llama2-rs", "/app/stories42M.bin", "-m", "server", "--address", "0.0.0.0", "--port", "8080"]