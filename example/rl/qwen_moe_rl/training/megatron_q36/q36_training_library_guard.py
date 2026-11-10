"""Verify the qualified Torch overlay and loaded training library identity."""
import importlib.metadata
import hashlib
import functools
import json
import os
from pathlib import Path


def validate_overlay(env):
    overlay = Path(env['Q36_CUDNN_ROOT']) / 'pyoverlay' / 'torch'
    base = Path(env['Q36_BASE_TORCH_DIR'])
    if overlay.is_symlink() or base.is_symlink() or not overlay.is_dir() or not base.is_dir():
        raise RuntimeError('Torch overlay and base package must be actual directories')
    def files(root):
        return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    original, copied = files(base), files(overlay)
    if not original or original.keys() != copied.keys():
        raise RuntimeError('Torch overlay file set differs from the base package')
    for name, source in original.items():
        a, b = source.stat(), copied[name].stat()
        if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
            raise RuntimeError('Torch overlay file identity mismatch: ' + name)
    return len(original)


def verify_runtime(env):
    # Called only from a GPU-assigned worker, never from Ray setup/driver.
    import torch
    import ray
    assigned = ray.get_gpu_ids()
    visible = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(assigned) != 1 or len(visible) != 1 or not visible[0] or torch.cuda.device_count() != 1:
        raise RuntimeError('training library guard requires one Ray-assigned visible GPU')
    if visible[0].isdigit() and float(assigned[0]) != int(visible[0]):
        raise RuntimeError('Ray GPU assignment differs from the visible device')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(properties, 'uuid', None)
    if gpu_uuid is None:
        raise RuntimeError('training device UUID is unavailable')
    overlay = (Path(env['Q36_CUDNN_ROOT']) / 'pyoverlay' / 'torch').resolve(strict=True)
    selected = Path(torch.__file__).resolve(strict=True)
    if selected != overlay / '__init__.py':
        raise RuntimeError('training process did not import Torch from the qualified overlay')
    if torch.__version__ != '2.13.0+cu130':
        raise RuntimeError('unexpected training Torch version')
    if importlib.metadata.version('transformer-engine') != '2.20.2':
        raise RuntimeError('unexpected Transformer Engine version')
    if torch.backends.cudnn.version() != 92801:
        raise RuntimeError('unexpected training cuDNN runtime')
    library_root = (Path(env['CUDNN_HOME']) / 'lib').resolve(strict=True)
    from q36_nonshared_launch_contract import CUDNN_SHA
    mapped, torch_libraries, te_libraries = set(), set(), set()
    torch_lib_root = (overlay / 'lib').resolve(strict=True)
    te_root = Path(env['Q36_BASE_TE_DIR']).resolve(strict=True)
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[-1].startswith('/'):
            continue
        raw = parts[-1]
        name = Path(raw.removesuffix(' (deleted)')).name
        if not (name.startswith(('libcudnn', 'libtorch')) or 'transformer_engine' in name and '.so' in name):
            continue
        if raw.endswith(' (deleted)'):
            raise RuntimeError('deleted training library mapping')
        path = Path(raw).resolve(strict=True)
        try:
            mapped_path = Path(os.readlink(Path('/proc/self/map_files') / parts[0])).resolve(strict=True)
        except OSError as error:
            raise RuntimeError('cannot resolve mapped training file identity') from error
        if mapped_path != path:
            raise RuntimeError('mapped training file path differs from maps')
        stat = path.stat()
        # On overlayfs the maps device can be the lower filesystem while
        # stat reports the overlay device. Compare inode and qualify bytes.
        if stat.st_ino != int(parts[4]):
            raise RuntimeError('mapped training library identity differs from its file')
        if name.startswith('libcudnn'):
            if path.parent != library_root:
                raise RuntimeError('training worker mapped cuDNN outside the qualified root')
            if str(path) not in mapped:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if CUDNN_SHA.get(path.name) != digest:
                    raise RuntimeError('training cuDNN mapped file digest mismatch')
            mapped.add(str(path))
        elif name.startswith('libtorch'):
            if not path.is_relative_to(torch_lib_root):
                raise RuntimeError('training worker mapped Torch outside the qualified overlay')
            corresponding_base = Path(env['Q36_BASE_TORCH_DIR']) / path.relative_to(overlay)
            base_stat = corresponding_base.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size) != (base_stat.st_dev, base_stat.st_ino, base_stat.st_size):
                raise RuntimeError('mapped Torch file is not the qualified base hardlink')
            torch_libraries.add(str(path))
        else:
            if not path.is_relative_to(te_root):
                raise RuntimeError('training worker mapped Transformer Engine outside its qualified root')
            te_libraries.add(str(path))
    if not mapped or not any(Path(p).name.startswith('libcudnn.so') for p in mapped):
        raise RuntimeError('training cuDNN mapping missing')
    if not torch_libraries or not te_libraries:
        raise RuntimeError('training Torch or Transformer Engine library mapping missing')
    print('Q36_TRAINING_LIBRARY_GUARD ' + json.dumps({
        'pid': os.getpid(), 'torch': torch.__version__, 'torch_path': str(selected),
        'transformer_engine': importlib.metadata.version('transformer-engine'),
        'cudnn_version': torch.backends.cudnn.version(), 'mapped_cudnn': sorted(mapped),
        'mapped_torch': sorted(torch_libraries), 'mapped_transformer_engine': sorted(te_libraries),
        'cuda_initialized': torch.cuda.is_initialized(), 'ray_gpu_ids': assigned,
        'visible_device': visible[0], 'device_name': torch.cuda.get_device_name(0), 'gpu_uuid': str(gpu_uuid),
    }, sort_keys=True), flush=True)


def install(module):
    cls = module.MegatronEngine
    original = cls._build_megatron_module
    if getattr(original, '_q36_training_library_guard', False):
        return
    @functools.wraps(original)
    def build(self, *args, **kwargs):
        verify_runtime(os.environ)
        return original(self, *args, **kwargs)
    build._q36_training_library_guard = True
    cls._build_megatron_module = build
