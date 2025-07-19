#!/bin/bash
set -e

echo "🔧 Fixing permissions for /workspace (runtime)..."
chown -R jovyan:users /workspace || true

echo "🚀 Starting Jupyter..."
exec start-notebook.sh --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/workspace
