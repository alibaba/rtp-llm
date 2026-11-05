"""Pinned project-local NCCL selection and CPU-only mapped-library evidence.

Does not import Torch, initialize CUDA, create a communicator, or modify a venv.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import re
import struct
import time

VERSION = '2.29.3'
VERSION_CODE = 22903
SO_BYTES = 230291664
SO_SHA256 = 'b7039d11fe9b00e897308754bbd1cbcb1afc46813e2be5d65d5b85e534ce859c'
WHEEL_SHA256 = 'eab9f5c565ab3326906f1d1b5be5773a174c2a1b47002faed76f9e957392f713'
DEPENDENCY_MANIFEST_SHA256 = 'ccff4928ba53dca5c8e8e3529b15ce899acea0398d84ba0c0be8eccfaa09d984'
ROOT_SUFFIX = 'logs/dependencies/nccl-2.29.3-arm64/extracted'
SCOPE_ENV_KEYS = ('Q397_NCCL_ROOT','Q397_NCCL_SO','Q397_NCCL_SHA256','Q397_NCCL_VERSION',
                  'LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH')


def _require(value, message):
    if not value: raise RuntimeError(message)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def _prepend(value, previous):
    return ':'.join(dict.fromkeys([value]+[part for part in previous.split(':') if part]))


def _preloads(value): return [part for part in re.split(r'[\s:]+',value) if part]


def scope_config(project_root, inherited_env=None):
    """Validate the pinned binary and prepare env for a *new* Python process."""
    inherited=dict(os.environ if inherited_env is None else inherited_env)
    project=Path(project_root).resolve(strict=True)
    expected_root=(project/ROOT_SUFFIX).resolve(strict=True)
    distribution=Path(inherited.get('Q397_NCCL_ROOT') or expected_root).resolve(strict=True)
    _require(distribution==expected_root,'NCCL root is not the pinned project-local distribution')
    library=(distribution/'nvidia/nccl/lib/libnccl.so.2').resolve(strict=True)
    _require(library.is_file() and library.stat().st_size==SO_BYTES,'NCCL library size differs from the official binary')
    with library.open('rb') as stream: header=stream.read(20)
    _require(header[:6]==b'\x7fELF\x02\x01' and struct.unpack('<H',header[18:20])[0]==183,
             'NCCL library must be little-endian ELF64 AArch64')
    _require(_sha(library)==SO_SHA256,'NCCL library SHA differs from the verified official cu13 wheel')
    expected={'Q397_NCCL_ROOT':str(distribution),'Q397_NCCL_SO':str(library),
              'Q397_NCCL_SHA256':SO_SHA256,'Q397_NCCL_VERSION':VERSION}
    for key,value in expected.items():
        _require(not inherited.get(key) or inherited[key]==value,'Inherited pinned NCCL setting conflicts: '+key)
    prior=_preloads(inherited.get('LD_PRELOAD',''))
    for item in prior:
        if Path(item).name.startswith('libnccl.so'):
            _require(Path(item).resolve()==library,'Another NCCL library is explicitly preloaded')
    expected.update(LD_PRELOAD=':'.join(dict.fromkeys([str(library)]+prior)),
                    LD_LIBRARY_PATH=_prepend(str(library.parent),inherited.get('LD_LIBRARY_PATH','')),
                    PYTHONPATH=_prepend(str(distribution),inherited.get('PYTHONPATH','')))
    return dict(distribution_root=str(distribution),library_path=str(library),library_sha256=SO_SHA256,
                library_bytes=SO_BYTES,version=VERSION,version_code=VERSION_CODE,
                wheel_sha256=WHEEL_SHA256,dependency_manifest_sha256=DEPENDENCY_MANIFEST_SHA256,
                environment=expected)


def _start_ticks(pid):
    value=Path('/proc',str(pid),'stat').read_text()
    return int(value[value.rfind(')')+2:].split()[19])


def _mapped_paths(pid):
    libraries=set()
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        parts=line.split(None,5)
        if len(parts)!=6: continue
        path=parts[-1]
        if not Path(path.removesuffix(' (deleted)')).name.startswith('libnccl.so'): continue
        _require(not path.endswith(' (deleted)'),'Mapped NCCL library was deleted/replaced')
        libraries.add(str(Path(path).resolve(strict=True)))
    return sorted(libraries)


def _process_environment(pid):
    if pid==os.getpid(): return dict(os.environ)
    pairs=Path('/proc',str(pid),'environ').read_bytes().split(b'\0')
    return {key.decode():value.decode() for pair in pairs if (parts:=pair.partition(b'='))[1]
            for key,value in [(parts[0],parts[2])] if key.decode() in SCOPE_ENV_KEYS}


def inspect_pid_nccl(pid=None, *, config=None, torch_module=None):
    """Require exactly the pinned mapped file; other PIDs use its binary SHA.

    ncclGetVersion is called only for this process through RTLD_NOLOAD.
    torch.cuda.nccl.version() is a compile-header label in PyTorch 2.11, not
    the runtime version; it is recorded without replacing ncclGetVersion.
    """
    pid=os.getpid() if pid is None else int(pid)
    before=_start_ticks(pid)
    if config is None:
        project=os.environ.get('Q397_PROJECT_ROOT') or os.environ.get('RTP_PROJECT_ROOT')
        _require(project,'Project root is required for pinned NCCL inspection')
        config=scope_config(project)
    _require(config['version_code']==VERSION_CODE and config['library_sha256']==SO_SHA256,
             'NCCL inspection received another binary/version config')
    library=str(Path(config['library_path']).resolve(strict=True))
    paths=_mapped_paths(pid)
    _require(paths==[library],f'Expected exactly the project NCCL library, observed {paths}')
    observed_sha=_sha(library)
    _require(observed_sha==SO_SHA256,'Actual mapped NCCL library SHA differs')
    environment=_process_environment(pid)
    required={'Q397_NCCL_ROOT':config['distribution_root'],'Q397_NCCL_SO':library,
              'Q397_NCCL_SHA256':SO_SHA256,'Q397_NCCL_VERSION':VERSION}
    for key,value in required.items():_require(environment.get(key)==value,'Actual process NCCL scope differs: '+key)
    _require(library in _preloads(environment.get('LD_PRELOAD','')),'Actual process did not inherit explicit NCCL preload')
    _require(config['distribution_root'] in environment.get('PYTHONPATH','').split(':'),
             'Actual process did not inherit the NCCL wheel root for Torch absolute preloading')
    runtime_version=None; torch_version=None
    evidence='official_library_sha256_binding'
    if pid==os.getpid():
        handle=ctypes.CDLL(library,mode=os.RTLD_NOW|os.RTLD_NOLOAD)
        function=handle.ncclGetVersion
        function.argtypes=[ctypes.POINTER(ctypes.c_int)];function.restype=ctypes.c_int
        version=ctypes.c_int()
        _require(function(ctypes.byref(version))==0,'ncclGetVersion failed')
        runtime_version=version.value
        _require(runtime_version==VERSION_CODE,'Actual loaded ncclGetVersion is not 2.29.3')
        evidence='self_ncclGetVersion_and_official_library_sha256'
        if torch_module is not None:
            torch_version=list(torch_module.cuda.nccl.version())
    else:
        _require(torch_module is None,'Do not attribute this process Torch version to another PID')
    _require(_start_ticks(pid)==before,'PID changed/reused during NCCL inspection')
    source=Path(__file__).resolve()
    return dict(schema='q397-nccl-scope-v1',passed=True,pid=pid,proc_start_ticks=before,
                expected_version=VERSION,expected_version_code=VERSION_CODE,library_path=library,
                library_sha256=observed_sha,mapped_nccl_paths=paths,runtime_version_code=runtime_version,
                torch_nccl_version=torch_version,torch_nccl_version_kind='compile_time_header',
                version_evidence=evidence,scope_environment={key:environment.get(key) for key in SCOPE_ENV_KEYS},
                scope_source=dict(path=str(source),bytes=source.stat().st_size,sha256=_sha(source)),
                inspected_unix=time.time())
