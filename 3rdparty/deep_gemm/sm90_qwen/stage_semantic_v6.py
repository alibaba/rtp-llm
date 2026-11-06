#!/usr/bin/env python3
"""Stage new v6 alias from a verified v3 package, no compile, import or GPU use.
The copied _C.so has exactly the same bytes: host API/ABI did not change.
Only two runtime JIT headers change; generic SM100 scheduler remains intact.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--base-package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parent;base=a.base_package.resolve();dst=a.output.resolve()/'rtp_megagemm_semantic_v6'
assert not dst.exists(),f'Refusing overwrite {dst}'
meta=json.loads((base/'build-manifest.json').read_text());assert meta['variant']=='semantic_v3',meta
so_hash=hashlib.sha256((base/'_C.so').read_bytes()).hexdigest();assert so_hash==meta['so_sha256']
expected={'sm90_fp8_mega_moe.cuh':'a15eeaef37893a62426e54ac74dd4ef50e6fdfdb686ce4955dd9e4184ad3fe90','sm90_mega_moe_pre_dispatch.cuh':'5ed450f8bed37edb3391413bff935fbf63a31195737b32a31c1754203d2d5b8f'}
for name,h in expected.items():assert hashlib.sha256((base/'include/deep_gemm/impls'/name).read_bytes()).hexdigest()==h,name
shutil.copytree(base,dst)
for f in ['impls/sm90_fp8_mega_moe.cuh','scheduler/sm90_mega_moe.cuh']:
 shutil.copy2(r/'semantic-v6-src/deep_gemm/include/deep_gemm'/f,dst/'include/deep_gemm'/f)
newso=hashlib.sha256((dst/'_C.so').read_bytes()).hexdigest();assert newso==so_hash
meta.update(variant='semantic_v6',package=dst.name,host_build_reused_from=str(base),host_build_byte_identical=True,so_sha256=newso,status='STAGED_NO_GPU_TEST',semantic_notes='v3 host ABI unchanged; v6 K128 promotion and baseline PTX SiLU/quant alignment; no padding')
(dst/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
shutil.copy2(r/'semantic-v6-manifest.json',dst/'source-manifest.json')
print(json.dumps(meta,indent=2))
