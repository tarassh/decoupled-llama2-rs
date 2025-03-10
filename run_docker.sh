#!/bin/bash
set -e

# Define variables
IMAGE_NAME="decoupled-llama2-rs"
CONTAINER_NAME="llama2-server"
HOST_PORT=8010
CONTAINER_PORT=8080  # As defined in the Dockerfile

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container is already running. Stopping it first..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
elif [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing stopped container..."
    docker rm $CONTAINER_NAME
fi

# Build the Docker image
echo "Building Docker image $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the container
echo "Starting container on port $HOST_PORT..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $HOST_PORT:$CONTAINER_PORT \
  --restart unless-stopped \
  $IMAGE_NAME

echo "Container $CONTAINER_NAME is now running!"
echo "Access the server at http://localhost:$HOST_PORT"

# Show logs
echo "Showing container logs (Ctrl+C to exit logs, container will keep running):"
docker logs -f $CONTAINER_NAME 