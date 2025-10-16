#!/bin/bash
# Run all example scripts

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$PROJECT_ROOT/examples"

echo "========================================"
echo "Running NepsisAI Examples"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"

for example in "$EXAMPLES_DIR"/*.py; do
    echo "========================================="
    echo "Running: $(basename "$example")"
    echo "========================================="
    echo ""
    python "$example"
    echo ""
    echo ""
done

echo "========================================="
echo "All examples completed!"
echo "========================================="
