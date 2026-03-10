#!/usr/bin/env bash
#
# Download and export the BGE reranker model to ONNX format.
# Idempotent -- skips export if the model already exists.
#
# The BGE-M3 embedding model (~1.5GB) is NOT handled here; it downloads
# automatically on first use via sentence-transformers.

set -euo pipefail

MODEL_DIR="${HOME}/.cache/dropbox-rag/models/bge-reranker-v2-m3"
MODEL_FILE="${MODEL_DIR}/model.onnx"

echo "=== dropbox-rag: Reranker Model Export ==="
echo ""

if [ -f "${MODEL_FILE}" ]; then
    echo "Reranker ONNX model already exists at:"
    echo "  ${MODEL_FILE}"
    echo ""
    echo "Skipping export. To force re-export, delete the directory:"
    echo "  rm -rf ${MODEL_DIR}"
    exit 0
fi

echo "Reranker ONNX model not found. Exporting from HuggingFace..."
echo ""

# Check if optimum is available; install if not
if ! python3 -c "import optimum" 2>/dev/null; then
    echo "Installing optimum[onnxruntime] (required for ONNX export)..."
    pip install "optimum[onnxruntime]"
    echo ""
fi

echo "Exporting BAAI/bge-reranker-v2-m3 to ONNX format..."
echo "This may take a few minutes on first run."
echo ""

python3 -c "
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

print('Downloading and converting model...')
model = ORTModelForSequenceClassification.from_pretrained(
    'BAAI/bge-reranker-v2-m3', export=True
)

print('Saving ONNX model to ${MODEL_DIR}/')
model.save_pretrained('${MODEL_DIR}')

print('Saving tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
tokenizer.save_pretrained('${MODEL_DIR}')

print('Done.')
"

echo ""
echo "Reranker model exported successfully to:"
echo "  ${MODEL_DIR}/"
echo ""
echo "You can verify the model exists:"
echo "  ls -lh ${MODEL_FILE}"
