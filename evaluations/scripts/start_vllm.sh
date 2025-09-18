#!/bin/bash

# Script to start vLLM server with configuration from models.yaml

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_DIR="$SCRIPT_DIR/../config"
CONFIG_FILE="$CONFIG_DIR/models.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Get model name from argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model-name>"
    echo "Available models: qwen-7b, llama-8b"
    echo ""
    echo "Example: $0 qwen-7b"
    exit 1
fi

MODEL_NAME=$1

# Parse YAML to get model config (using Python for portability)
MODEL_CONFIG=$(python3 -c "
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

if '$MODEL_NAME' not in config.get('models', {}):
    print(f'ERROR: Model {MODEL_NAME} not found in config', file=sys.stderr)
    sys.exit(1)

model = config['models']['$MODEL_NAME']

if model.get('type') != 'open_source':
    print(f'ERROR: {MODEL_NAME} is not an open source model', file=sys.stderr)
    sys.exit(1)

# Extract model path and server URL
model_path = model.get('model_path', '')
server_url = model.get('server_url', 'http://localhost:8000')

# Extract port from URL
import re
port_match = re.search(r':(\d+)', server_url)
port = port_match.group(1) if port_match else '8000'

print(f'{model_path}|{port}')
" 2>&1)

# Check if Python parsing succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}$MODEL_CONFIG${NC}"
    exit 1
fi

# Split the result
IFS='|' read -r MODEL_PATH PORT <<< "$MODEL_CONFIG"

if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: No model_path found for $MODEL_NAME${NC}"
    exit 1
fi

echo -e "${GREEN}Starting vLLM server for $MODEL_NAME${NC}"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use.${NC}"
    echo "Another vLLM server might be running. Kill it first:"
    echo "  lsof -ti:$PORT | xargs kill -9"
    echo ""
    read -p "Do you want to kill the existing process? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        echo "Existing process killed."
        sleep 2
    else
        echo "Exiting without starting new server."
        exit 1
    fi
fi

# Start vLLM server
echo -e "${GREEN}Launching vLLM server...${NC}"
echo "Command: python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port $PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name "$MODEL_PATH"