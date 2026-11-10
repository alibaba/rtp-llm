"""Reject mixed native NCCL/cuDNN mappings by file identity and pinned digest."""
import hashlib
import json
import os
from pathlib import Path

NATIVE_LIBRARIES = {
    'libcudnn.so.9': ('119059d6ad15f576608a0daa547444538062bf5e6be970f090298b50636e13b6', 6328728),
    'libcudnn_adv.so.9': ('940107ad76ecd853f70c7a48b122cf4369c735fdb57361982d6460ca499a61bf', 132239032),
    'libcudnn_cnn.so.9': ('9022f5c0607cbeff24c7a1418362417e1238c8f42039a14dd8befb9b292d014d', 2501768),
    'libcudnn_engines_precompiled.so.9': ('3b765736f63d724a0de67b7ce915b0d0a780154fe4a2d515ab14040224aa7db4', 299909640),
    'libcudnn_engines_runtime_compiled.so.9': ('7b8f371a7718c29549317efdbdeb8d7c301874f9471762a04830846ac4e83510', 33571912),
    'libcudnn_graph.so.9': ('b601fddd333d84ced4a3d833e75bb412c204e39b21755d79cb890335dce660b8', 10749656),
    'libcudnn_heuristic.so.9': ('368687b9b5c2c01b55c21f74bd930cb05d3654b4e901e998b7bfb33cebac6ae8', 65015024),
    'libcudnn_ops.so.9': ('9765db5f193984c32b76daca262fa5f2e3d77b29082afcabb5467d555f454258', 50342776),
    'libnccl.so.2': ('cc01f863461ea28f3ecc4390716a2d7a5aca9bc00e04eeb6fc7a5c2e666b3dfb', 237975080),
}


def _verify_file(path, name):
    if name not in NATIVE_LIBRARIES:
        raise RuntimeError('unqualified native library: ' + name)
    expected, size = NATIVE_LIBRARIES[name]
    if path.stat().st_size != size:
        raise RuntimeError('native library size mismatch: ' + name)
    with path.open('rb') as stream:
        state = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(chunk)
    if state.hexdigest() != expected:
        raise RuntimeError('native library digest mismatch: ' + name)


def verify_native_search_path(value):
    """Qualify every candidate cuDNN/NCCL library, including later dlopen targets."""
    seen = set()
    for directory in value.split(':'):
        if not directory:
            raise RuntimeError('empty entry in native dynamic library path')
        root = Path(directory)
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if not candidate.name.startswith(('libcudnn', 'libnccl.so')) or not candidate.is_file():
                continue
            path = candidate.resolve(strict=True)
            if path in seen:
                continue
            _verify_file(path, path.name)
            seen.add(path)
    if not seen:
        raise RuntimeError('no native cuDNN/NCCL candidates in library path')


def verify_native_mappings(runtime_version, maps_path='/proc/self/maps', map_files_root='/proc/self/map_files'):
    if runtime_version != 92000:
        raise RuntimeError('native cuDNN runtime is not qualified 9.20.0')
    mapped = {}
    for line in Path(maps_path).read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[-1].startswith('/'):
            continue
        raw = parts[-1]
        name = Path(raw.removesuffix(' (deleted)')).name
        if not name.startswith(('libcudnn', 'libnccl.so')):
            continue
        if raw.endswith(' (deleted)') or name not in NATIVE_LIBRARIES:
            raise RuntimeError('unqualified or deleted native library: ' + name)
        path = Path(raw).resolve(strict=True)
        try:
            mapped_path = Path(os.readlink(Path(map_files_root) / parts[0])).resolve(strict=True)
        except OSError as error:
            raise RuntimeError('cannot resolve mapped native file identity') from error
        if mapped_path != path:
            raise RuntimeError('mapped native file path differs from maps')
        stat = path.stat()
        # Linux overlayfs can report the lower device in /proc/maps while
        # stat reports the overlay device; inode plus full SHA is stable.
        if stat.st_ino != int(parts[4]):
            raise RuntimeError('mapped native library identity differs from its file: '
                + name + f' mapped_inode={parts[4]} stat_inode={stat.st_ino}')
        if name in mapped and mapped[name] != str(path):
            raise RuntimeError('multiple native mappings for ' + name)
        if name not in mapped:
            _verify_file(path, name)
            mapped[name] = str(path)
    if not {'libcudnn.so.9', 'libnccl.so.2'} <= mapped.keys():
        raise RuntimeError('native cuDNN/NCCL mappings missing')
    print('Q36_NATIVE_LIBRARY_GUARD ' + json.dumps({'pid': os.getpid(), 'cudnn_version': runtime_version,
          'mapped': mapped}, sort_keys=True), flush=True)
    return mapped
