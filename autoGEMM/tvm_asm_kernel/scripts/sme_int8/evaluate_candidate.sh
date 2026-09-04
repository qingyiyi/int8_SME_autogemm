#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)

exec python3 "${project_root}/python/sme_int8/evaluate_candidate.py" "$@"
