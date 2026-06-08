#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <local-nyu-depth-v2-dir> <user@host> [remote-root]"
    echo "Example: $0 ~/Downloads/nyu_depth_v2 user@gpu-server ~/METER"
    exit 1
fi

LOCAL_DATA_DIR="$1"
REMOTE_HOST="$2"
REMOTE_ROOT="${3:-~/METER}"
REMOTE_DATA_DIR="${REMOTE_ROOT%/}/data/nyu_depth_v2"

rsync -avh --partial --progress \
    "${LOCAL_DATA_DIR%/}/" \
    "${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
