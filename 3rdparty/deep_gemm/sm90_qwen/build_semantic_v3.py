#!/usr/bin/env python3
"""Build CPU host extension + runtime headers under one new isolated import name.
NOT_RUN until used on deployment worker; never install/replace installed deep_gemm.
Run: PYTHONPATH=/isolated/deps python3 build_isolated.py --output /isolated/packages
Dependency bootstrap: python3 -m pip install --target /isolated/deps apache-tvm-ffi==0.1.11 ninja
No CUDA kernels are launched or JIT compiled by this script.
"""
from pathlib import Path
import argparse,sys,shutil,os,json,sysconfig,hashlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'pr323-9e3afe9'; OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
import torch
import tvm_ffi.cpp
cuda=Path(os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda')
# Driver API confirms SM90; nvidia-smi reports 8.9 incorrectly on the worker.
os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST','9.0')
os.environ.setdefault('MAX_JOBS','1')
for var in ['semantic_v3']:
 name='rtp_megagemm_'+var; pkg=OUT/name
 if pkg.exists(): raise RuntimeError(f'Refusing to overwrite {pkg}')
 pkg.mkdir()
 shutil.copy2(SRC/'sgl_deep_gemm/__init__.py',pkg/'__init__.py');shutil.copy2(SRC/'sgl_deep_gemm/cuda_helpers.py',pkg/'cuda_helpers.py')
 for sub in ['utils','testing','legacy','mega']:
  shutil.copytree(SRC/'deep_gemm'/sub,pkg/sub)
 inc=pkg/'include';inc.mkdir()
 shutil.copytree(SRC/'deep_gemm/include/deep_gemm',inc/'deep_gemm')
 for sub in ['cute','cutlass']:
  source=SRC/'third-party/cutlass/include'/sub
  if not source.is_dir(): raise RuntimeError(f'Missing pinned CUTLASS include {source}')
  shutil.copytree(source,inc/sub)
 # csrc only needed at host build time; separate package-specific path handles two variants.
 work=OUT/('_build_'+var);work.mkdir()
 shutil.copytree(SRC/'csrc',work/'csrc')
 if var=='semantic_v3':
  candidate=ROOT/'semantic-v3-src'
  for f in ['csrc/jit_kernels/heuristics/sm90_mega_moe.hpp','csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp','csrc/apis/sm90_mega.hpp']:
   shutil.copy2(candidate/f,work/f)
  for f in ['sm90_fp8_mega_moe.cuh','sm90_mega_moe_pre_dispatch.cuh']:
   shutil.copy2(candidate/'deep_gemm/include/deep_gemm/impls'/f,inc/'deep_gemm/impls'/f)
 # Original Python fallback builder isn't used; _C.so is pre-built here.
 # set include package path to patched headers for both CPU template parsing and JIT.
 ti=Path(torch.__file__).parent
 flags=['-std=c++17','-O3','-fPIC','-Wno-psabi','-Wno-deprecated-declarations',f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
 if int(os.environ.get('DG_JIT_USE_RUNTIME_API','0')):flags.append('-DDG_JIT_USE_RUNTIME_API')
 includes=[str(cuda/'include'),sysconfig.get_path('include'),str(ti/'include'),str(ti/'include/torch/csrc/api/include'),str(inc),str(SRC/'third-party/fmt/include')]
 if (cuda/'include/cccl').exists():includes.append(str(cuda/'include/cccl'))
 links=[f'-L{cuda}/lib64',f'-L{ti}/lib','-lcudart','-lnvrtc','-lcublasLt','-lcublas','-ltorch','-ltorch_cpu','-lc10','-lc10_cuda','-ltorch_cuda']
 # Honor externally configured GCC toolset, CUDA13 library/header paths and LD_LIBRARY_PATH.
 (work/'host-extension').mkdir()
 lib=tvm_ffi.cpp.build(name='_C',cpp_files=[str(work/'csrc/tvm_ffi_api.cpp')],extra_cflags=flags,extra_ldflags=links,extra_include_paths=includes,build_directory=str(work/'host-extension'))
 shutil.copy2(lib,pkg/'_C.so')
 shutil.copy2(SRC/'sgl_deep_gemm/VERSION',pkg/'VERSION')
 manifest={'status':'HOST_BUILD_ONLY_NO_GPU_TEST','variant':var,'semantic_notes':'v2 numerical contract plus MN-major SF Data accounting fix; no I-padding; host rebuilt','package':name,'upstream_revision':'9e3afe91cb145ddfa0b18ae874a11dbb449e16a9','cutlass_revision':'f3fde58372d33e9a5650ba7b80fc48b3b49d40c8','fmt_revision':'553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28','torch':torch.__version__,'so_sha256':hashlib.sha256((pkg/'_C.so').read_bytes()).hexdigest(),'python':sys.version,'compiler':os.environ.get('CXX'),'cuda_home':str(cuda)}
 (pkg/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest),flush=True)
 # Retain host-extension logs by default for reproducibility, no automatic cleanup.
