#!/usr/bin/env bash
set -euo pipefail

docker pull qdrant/qdrant:v1.17
docker run -d \
  --name dropbox-rag-qdrant \
  -p 6333:6333 \
  -v dropbox-rag-qdrant-data:/qdrant/storage \
  qdrant/qdrant:v1.17

echo "Qdrant running at http://localhost:6333"
