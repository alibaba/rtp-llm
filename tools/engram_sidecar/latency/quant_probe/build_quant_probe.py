"""Build original main CUDA kernels and a test binding, without production imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path


DEFAULT_REPO = Path(__file__).resolve().parents[4]

def default_build_directory():
    return os.environ.get("ENGRAM_QUANT_BUILD_DIR",
                          str(Path.home() / ".cache/rtp_llm/engram_quant_main_492ecf45"))


def load_probe(repo, build):
    here = Path(__file__).resolve().parent
    hashes = json.loads((here / 'source_hashes.json').read_text())
    for relative, expected in hashes.items():
        actual = hashlib.sha256((Path(repo) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('source fence failed: ' + relative)
    if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
        raise RuntimeError('explicit TORCH_CUDA_ARCH_LIST required; use CUDA runtime capability')
    os.environ.setdefault('MAX_JOBS', '1')
    from torch.utils.cpp_extension import load
    kernel_dir = Path(repo) / 'rtp_llm/models_py/bindings/cuda/kernels'
    Path(build).mkdir(parents=True, exist_ok=True)
    module = load(
        name='engram_main_quant_probe',
        sources=[str(here / 'quant_shim.cc'),
                 str(kernel_dir / 'per_token_group_quant_8bit.cu'),
                 str(kernel_dir / 'per_token_group_quant_8bit_v2.cu')],
        extra_include_paths=[str(kernel_dir)],
        extra_cflags=['-O2', '-std=c++17', '-DUSING_CUDA=1'],
        extra_cuda_cflags=['-O2', '-std=c++17', '-DUSING_CUDA=1', '-U__CUDA_NO_HALF_OPERATORS__',
                          '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                          '-U__CUDA_NO_HALF2_OPERATORS__'],
        build_directory=str(build), verbose=True,
    )
    return module


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=str(DEFAULT_REPO))
    p.add_argument('--build', default=default_build_directory())
    args = p.parse_args()
    mod = load_probe(args.repo, args.build)
    print(json.dumps({'status': 'BUILD_PASS', 'module': mod.__file__}), flush=True)
