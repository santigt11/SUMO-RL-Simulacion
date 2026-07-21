#!/bin/bash
# Script para ejecutar el dashboard sin necesidad de activar el venv manualmente
VENV_PYTHON="/home/santiago/Documentos/Tesis/.venv/bin/python3"
APP_DIR="/home/santiago/Documentos/Tesis/dashboard"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python virtual environment not found at $VENV_PYTHON"
    exit 1
fi

cd "$APP_DIR"
exec "$VENV_PYTHON" -m streamlit run app.py --server.headless true
