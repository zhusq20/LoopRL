# Source in a task shell; all runtime changes remain in isolated /tmp installs.
LOOPRL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export LOOPRL_NATIVE_PYTHON=/tmp/looprl-sglang-venv/bin/python
export LOOPRL_ORACLE_PYTHON=/tmp/looprl-native-venv/bin/python
export PYTHONPATH="${LOOPRL_REPO}/third_party/slime:/tmp/looprl-native-sources/Megatron-LM-ae76c60e6c35d913867d6dd7bad0f30b128b54e7:/tmp/looprl-native-sources/sglang-e00930c5489053f26d86b179cee0d087f846acbb/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/tmp/looprl-native-compat/package/usr/local/cuda-13.0/compat:/tmp/looprl-native-compiler/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CXX=/tmp/looprl-native-compiler/bin/x86_64-conda-linux-gnu-c++
export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=slime_plugins.sglang_models
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
