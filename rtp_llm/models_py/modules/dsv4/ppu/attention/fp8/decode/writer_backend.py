"""Explicit PPU Decode-only FlashMLA dependency, without global module mutation."""
import functools
import hashlib
import importlib
import importlib.util
import logging
import os
from pathlib import Path
from types import FunctionType

STOCK_SHA = '26037418e0d634cc39eff935fe40afab0e069928dc42f9d4ffaeda3c54efefa9'
WRITER_SHA = 'a8a0e3ec85495170be760d32032da4b75fb837031e3285803419d691776568f5'
INTERFACE_SHA = 'c188dfd92abe8fa7deff738b9b71302e28105f62c95b3847cb666da4d7aeae88'


def checked_path(path, expected):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'FlashMLA dependency SHA mismatch: {path}')
    return path


class DecodeNativeProxy:
    def __init__(self, stock, writer):
        self.stock, self.writer = stock, writer

    def __getattr__(self, name):
        return getattr(self.stock, name)

    def fwd_kvcache_mla(self, q, k_cache, v_cache, head_dim_v, cache_seqlens,
                        block_table, softmax_scale, causal, metadata, splits,
                        is_fp8, indices, sink, lengths, extra_kv, extra_indices,
                        extra_lengths, out):
        if is_fp8 and indices is not None and q.size(-1) == 512 and k_cache.size(-1) == 584:
            return self.writer.sparse_decode_fwd(
                q, k_cache, head_dim_v, indices, sink, lengths, metadata, splits,
                extra_kv, extra_indices, extra_lengths, softmax_scale, out,
            )[:2]
        return self.stock.fwd_kvcache_mla(
            q, k_cache, v_cache, head_dim_v, cache_seqlens, block_table,
            softmax_scale, causal, metadata, splits, is_fp8, indices, sink,
            lengths, extra_kv, extra_indices, extra_lengths, out,
        )


def private_interface(function, stock, writer):
    # Reuse the exact installed Python implementation and scheduler class.
    # Only this function's globals differ; no stock function/module is patched.
    namespace = dict(function.__globals__)
    if namespace.get('flash_mla_cuda') is not stock:
        raise RuntimeError('unexpected stock interface native binding')
    namespace['flash_mla_cuda'] = DecodeNativeProxy(stock, writer)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


@functools.lru_cache(maxsize=1)
def load_writer_backend(path):
    stock = importlib.import_module('flash_mla_cuda')
    interface = importlib.import_module('flash_mla.flash_mla_interface')
    stock_path = checked_path(stock.__file__, STOCK_SHA)
    interface_path = checked_path(interface.__file__, INTERFACE_SHA)
    writer_path = checked_path(path, WRITER_SHA)
    if writer_path == stock_path:
        raise RuntimeError('writer must be a separate dependency')
    spec = importlib.util.spec_from_file_location('_rtp_ppu_writer.flash_mla_cuda', writer_path)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    if writer is stock or Path(writer.__file__).resolve() != writer_path:
        raise RuntimeError('isolated writer native identity mismatch')
    checked_path(writer.__file__, WRITER_SHA)
    if 'arg12:' not in (writer.sparse_decode_fwd.__doc__ or ''):
        raise RuntimeError('writer sparse_decode_fwd ABI mismatch')
    result = private_interface(interface.flash_mla_with_kvcache, stock, writer)
    logging.info('PPU FlashMLA Decode dependency stock=%s sha=%s interface=%s sha=%s writer=%s sha=%s',
                 stock_path, STOCK_SHA, interface_path, INTERFACE_SHA, writer_path, WRITER_SHA)
    return result


def get_writer_backend():
    path = os.environ.get('DSV4_PPU_FLASHMLA_WRITER_PATH', '')
    role = os.environ.get('ROLE_TYPE', '')
    if not path or role == 'PREFILL':
        return None
    if role != 'DECODE':
        raise RuntimeError('explicit writer dependency requires ROLE_TYPE=DECODE')
    return load_writer_backend(path)
