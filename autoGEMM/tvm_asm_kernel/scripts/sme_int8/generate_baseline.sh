#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <int8gemm-kblas-root> <generated-output-dir>" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)
reference_root=$1
output_dir=$2

python3 "${project_root}/python/sme_int8/generate_driver.py" \
  --reference-root "${reference_root}" \
  --output "${output_dir}"

make -n -C "${output_dir}" print-config >/dev/null

echo "Generated baseline bundle in ${output_dir}"
echo "Build remotely with: make -C ${output_dir} REF_ROOT=${reference_root} all check"
