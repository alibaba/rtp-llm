"""AOT compilation for the generate-path post-layers handler.

CUSTOM_PROCESSOR_MODE=compiled: the handler's compiled_module() is
torch.export-ed with a dynamic batch dim and compiled to an AOTInductor
package once at startup; the C++ PostLayersProcessor then runs the package
per step with no Python interpreter on the hot path (no GIL, no GC).

Compilation runs on the serving host so the artifact always matches the
local torch ABI and GPU architecture. The package is cached under a content
hash (torch version, device, weights, module source), so restarting with an
unchanged handler skips the compile.
"""

import hashlib
import inspect
import logging
import os

import torch

# Upper bound of the dynamic batch dim baked into the exported program. A
# step with more context requests than this fails the AOTI run; the engine
# then drops the score for that step and logs (generation is unaffected).
MAX_BATCH = 32768


def _cache_key(module: torch.nn.Module, hidden_size: int, dtype: torch.dtype) -> str:
    h = hashlib.sha256()
    h.update(torch.__version__.encode())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_device_name(0).encode())
    h.update(f"{hidden_size}:{dtype}:{MAX_BATCH}".encode())
    try:
        h.update(inspect.getsource(type(module)).encode())
    except (OSError, TypeError):
        pass
    for name, param in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compile_post_layers_handler(handler, config) -> str:
    module = handler.compiled_module()
    if module is None:
        raise RuntimeError(
            "CUSTOM_PROCESSOR_MODE=compiled requires the handler to implement "
            "compiled_module(); this handler only supports eager"
        )
    args = handler.extend_forward_args()
    if args != ["last_hidden_states"]:
        raise RuntimeError(
            f"compiled mode supports extend_forward_args()==['last_hidden_states'] "
            f"only, got {args}"
        )
    module = module.eval()
    param = next(module.parameters(), None)
    if param is not None:
        dtype, device = param.dtype, param.device
    else:
        from rtp_llm.utils.util import to_torch_dtype

        dtype, device = to_torch_dtype(config.data_type), handler.device

    hidden_size = config.hidden_size
    cache_dir = os.environ.get("CUSTOM_PROCESSOR_AOTI_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "rtp_llm", "post_layers_aoti"
    )
    os.makedirs(cache_dir, exist_ok=True)
    package_path = os.path.join(
        cache_dir, _cache_key(module, hidden_size, dtype) + ".pt2"
    )
    if os.path.exists(package_path):
        logging.info(f"post-layers AOTI cache hit: {package_path}")
        return package_path

    example = torch.zeros(8, hidden_size, dtype=dtype, device=device)
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    exported = torch.export.export(module, (example,), dynamic_shapes=({0: batch},))
    # compile to a sibling temp path, publish atomically: a crash mid-compile
    # must not leave a half-written package at the cache-hit path
    tmp_path = os.path.join(
        os.path.dirname(package_path), f".compiling.{os.getpid()}.pt2"
    )
    torch._inductor.aoti_compile_and_package(exported, package_path=tmp_path)
    os.replace(tmp_path, package_path)
    logging.info(f"post-layers AOTI package compiled: {package_path}")
    return package_path
