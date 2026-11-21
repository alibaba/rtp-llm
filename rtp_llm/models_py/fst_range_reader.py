"""Bounded rank-local FST reads for CPU staging and online expert updates.

The normal FST parallel loader stages whole files and uses collectives. Neither
is appropriate while inference is running. Use its native CPU range reader
instead: only the selected tensors/slices and one 1 MiB bounce buffer are live.
No process group is constructed or used. GPU postprocessing remains owned by
the temporary expert module, not by FST's background I/O threads.
"""

import logging
import math
from typing import Iterator, Tuple

import torch

from rtp_llm.models_py.read_spec import ReadSpec


def validate_fst_full_tensor_fallback(files, resolver, name_filter=None):
    """Reject segmented plans before a legacy consumer stages/reads tensors."""
    if resolver is None:
        return
    # This guard also runs when FST's private consumer API is incompatible.
    # Inspect only the public checkpoint header, without importing those same
    # private FST interfaces or materializing tensor payloads.
    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as metadata:
            for name in metadata.keys():
                if name_filter is not None and not name_filter(name):
                    continue
                spec = resolver(name, tuple(metadata.get_slice(name).get_shape()))
                if spec.load and spec.segments is not None:
                    raise RuntimeError(
                        "Segmented FST loading requires ReadSpec consumer integration; "
                        "full tensor fallback is forbidden"
                    )


def build_segmented_fst_reader(files):
    """Create a lazy key-to-file range reader for the parallel FST consumer.

    Metadata stores paths, never borrowed file-buffer pointers or open fds.
    Normal FST file staging is retained; only selected consumer tensors avoid
    the full-tensor broadcast. This adds local CPU I/O and H2D transfers.
    """
    index = None

    def read(name, spec):
        nonlocal index
        if spec.segments is None:
            raise ValueError("Segmented FST reader requires a segments ReadSpec")
        if index is None:
            from fastsafetensors.common import SafeTensorsMetadata
            from fastsafetensors.frameworks._torch import TorchOp

            index = {}
            framework = TorchOp()
            for path in files:
                with open(path, "rb", buffering=0) as stream:
                    metadata = SafeTensorsMetadata.from_fd(
                        stream.fileno(), path, framework
                    )
                    for key in metadata.tensors:
                        if key in index:
                            raise RuntimeError(f"Duplicate FST checkpoint key {key!r}")
                        index[key] = path
            logging.getLogger(__name__).info(
                "NewLoader FST segmented consumer: native CPU range reads + H2D; "
                "whole-file FST staging retained, with additional local range I/O; "
                "source buffers may remain live until batch.close (early-release "
                "not preserved); only per-rank consumer full-tensor materialization "
                "is avoided; this is an experimental explicit opt-in path"
            )
        if name not in index:
            raise KeyError(f"FST segmented checkpoint key {name!r} not found")
        iterator = iter_fst_range_tensors(
            [index[name]],
            name_filter=lambda key: key == name,
            read_spec_resolver=lambda key, shape: spec,
        )
        try:
            output_name, value = next(iterator)
            if output_name != spec.output_name:
                raise RuntimeError("FST segmented reader changed the output name")
            return value
        finally:
            # The result owns CPU memory. Closing releases all fd/native state
            # before the enclosing FST file buffer can be closed or reused.
            iterator.close()

    return read


def _selection(shape, indices):
    indices = (slice(None),) * len(shape) if indices is None else indices
    if len(indices) != len(shape):
        raise ValueError("FST range selection rank does not match tensor rank")
    axes, output_shape = [], []
    for size, index in zip(shape, indices):
        if isinstance(index, int):
            index = index + size if index < 0 else index
            if not 0 <= index < size:
                raise IndexError("FST range selection index is out of bounds")
            axes.append(range(index, index + 1))
        elif isinstance(index, slice):
            start, stop, step = index.indices(size)
            if step <= 0:
                raise ValueError("FST range selections require positive strides")
            axis = range(start, stop, step)
            axes.append(axis)
            output_shape.append(len(axis))
        else:
            raise TypeError("FST range selection must contain integers or slices")
    return tuple(axes), tuple(output_shape)


def _element_spans(shape, axes) -> Iterator[Tuple[int, int]]:
    """Yield contiguous (source element offset, count), without a dense plan."""
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]

    def visit(dim, offset):
        if dim == len(shape):
            yield offset, 1
            return
        axis = axes[dim]
        if not axis:
            return
        if axis.step == 1 and all(
            axes[j] == range(shape[j]) for j in range(dim + 1, len(shape))
        ):
            yield offset + axis.start * strides[dim], len(axis) * strides[dim]
        else:
            for index in axis:
                yield from visit(dim + 1, offset + index * strides[dim])

    yield from visit(0, 0)


def native_cpu_range_api():
    """Resolve the actual FST symbols used by the rank-local reader."""
    from fastsafetensors import cpp
    from fastsafetensors.common import SafeTensorsMetadata
    from fastsafetensors.frameworks._torch import TorchOp, dtype_convert

    if not all(
        callable(getattr(cpp, symbol, None))
        for symbol in ("nogds_file_reader", "gds_device_buffer")
    ):
        raise RuntimeError("FST is missing its native CPU range reader APIs")
    return cpp, SafeTensorsMetadata, TorchOp, dtype_convert


def iter_fst_range_tensors(
    files, *, name_filter=None, slice_expander=None, read_spec_resolver=None, catalog=None
):
    """Read via FST C++; CPU tensors own storage beyond the reader lifetime."""
    cpp, SafeTensorsMetadata, TorchOp, dtype_convert = native_cpu_range_api()
    # use_mmap=False permits arbitrary file offsets. CPU mode avoids pinned
    # CUDA allocation and current-device state in the reader's worker threads.
    reader = cpp.nogds_file_reader(False, 1024, 1, False)
    framework = TorchOp()

    def read(fd, metadata, frame, spec):
        shape = tuple(frame.shape)
        if spec.source_shape != shape:
            raise ValueError(f"ReadSpec shape mismatch for {spec.source_name!r}")
        if spec.row_indices is not None:
            row_size = math.prod(shape[1:])
            output_shape = (len(spec.row_indices), *shape[1:])
            # Collapse adjacent vocabulary rows into one native read. Empty
            # rank selections are legal and must not read unrelated rows.
            runs = []
            for row in spec.row_indices:
                if runs and runs[-1][1] == row:
                    runs[-1] = (runs[-1][0], row + 1)
                else:
                    runs.append((row, row + 1))
            spans = ((lo * row_size, (hi - lo) * row_size) for lo, hi in runs)
        elif spec.segments is not None:
            # ReadSpec validates exclusivity and ordered nonoverlapping rows.
            output_shape = (
                sum(end - start for start, end in spec.segments),
                *shape[1:],
            )
            row_size = math.prod(shape[1:])
            spans = (
                (start * row_size, (end - start) * row_size)
                for start, end in spec.segments
            )
        else:
            axes, output_shape = _selection(shape, spec.slices)
            spans = _element_spans(shape, axes)
        tensor = torch.empty(
            output_shape, dtype=dtype_convert[frame.dtype], device="cpu"
        )
        length = tensor.numel() * tensor.element_size()
        if length:
            buffer = cpp.gds_device_buffer(tensor.data_ptr(), length, False)
            destination = 0
            for offset, count in spans:
                count *= tensor.element_size()
                source = metadata.header_length + frame.data_offsets[0]
                source += offset * tensor.element_size()
                request = reader.submit_read(fd, buffer, source, count, destination)
                if request < 0:
                    raise OSError(
                        f"FST range submission failed for {spec.source_name!r}"
                    )
                # Native wait_read returns the destination pointer, NOT byte
                # count. A null pointer (0) means failure in 0.1.19rc5+ali.
                if reader.wait_read(request) <= 0:
                    raise OSError(f"FST range read failed for {spec.source_name!r}")
                destination += count
            if destination != length:
                raise RuntimeError("FST range plan did not fill destination tensor")
        return tensor

    for path in files:
        with open(path, "rb", buffering=0) as stream:
            metadata = SafeTensorsMetadata.from_fd(stream.fileno(), path, framework)
            allowed = catalog.shard_keys(path, metadata.tensors) if catalog is not None else None
            for name, frame in metadata.tensors.items():
                if allowed is not None and name not in allowed:
                    continue
                if name_filter is not None and not name_filter(name):
                    continue
                shape = tuple(frame.shape)
                if slice_expander is not None and slice_expander.handles_safetensor(
                    name
                ):
                    for output_name, index in slice_expander.expand_safetensor(
                        name, shape
                    ):
                        if not shape or not 0 <= index < shape[0]:
                            raise ValueError(f"Invalid stacked expert row for {name!r}")
                        spec = ReadSpec(
                            name,
                            output_name,
                            shape,
                            slices=(index,) + (slice(None),) * (len(shape) - 1),
                        )
                        # Match stacked checkpoint expansion: the expert module
                        # still applies its projection's TP layout to this row.
                        yield output_name, read(stream.fileno(), metadata, frame, spec)
                    continue
                spec = (
                    read_spec_resolver(name, shape)
                    if read_spec_resolver is not None
                    else ReadSpec.full(name, shape)
                )
                if spec.source_name != name:
                    raise ValueError(f"ReadSpec source mismatch for {name!r}")
                if spec.load:
                    tensor = read(stream.fileno(), metadata, frame, spec)
                    yield spec.output_name, spec.wrap_sharded_tensor(tensor)
