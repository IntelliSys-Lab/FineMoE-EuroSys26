#!/usr/bin/env bash
set -euo pipefail
exec uv sync --managed-python --python 3.12 --locked
