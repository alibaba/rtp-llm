#!/usr/bin/env python3
"""Stage explicit-promotion-FMA hypothesis in a NEW alias. No imports/build/GPU."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--base-package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parent;base=a.base_package.resolve();dst=a.output.resolve()/'rtp_megagemm_precision_fma_v1'
assert not dst.exists(),f'Refusing overwrite {dst}'
meta=json.loads((base/'build-manifest.json').read_text());assert meta['variant']=='semantic_v6',meta
so_hash=hashlib.sha256((base/'_C.so').read_bytes()).hexdigest();assert so_hash==meta['so_sha256']
oldheader=base/'include/deep_gemm/impls/sm90_fp8_mega_moe.cuh'
assert hashlib.sha256(oldheader.read_bytes()).hexdigest()=='0cc84e1aeba9e5d005c5442a42afab7cd8d52e607253733331234f1bb95a4580'
manifest=json.loads((r/'precision-fma-v1-manifest.json').read_text())
newheader=r/'precision-fma-v1/deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh'
assert hashlib.sha256(newheader.read_bytes()).hexdigest()==manifest['kernel_sha256']
shutil.copytree(base,dst)
shutil.copy2(newheader,dst/'include/deep_gemm/impls/sm90_fp8_mega_moe.cuh')
assert hashlib.sha256((dst/'_C.so').read_bytes()).hexdigest()==so_hash
meta.update(variant='precision_fma_v1',package=dst.name,host_build_reused_from=str(base),host_build_byte_identical=True,so_sha256=so_hash,status='STAGED_NOT_RUN_HYPOTHESIS',precision=manifest,requires_fresh_process=True)
(dst/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
shutil.copy2(r/'precision-fma-v1-manifest.json',dst/'precision-source-manifest.json')
print(json.dumps(meta,indent=2))
