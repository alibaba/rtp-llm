"""CPU execution of actual scatter AST + independent page-layout byte oracle.

No Torch/Triton/native-package import and no CUDA initialization. Quantization
arithmetic is source-pinned; numerical BF16/FP4 equality remains a GPU gate.
"""
import ast
import functools
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIRE = ROOT / 'nvfp4_cp_wire.py'
CACHE = ROOT / 'nvfp4_kv_cache.py'


def fn(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class A(np.ndarray):
    def to(self, dtype, **kwargs):
        return self.astype(dtype).view(A)


def arr(x):
    return np.asarray(x).view(A)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class Sparse:
    def __init__(self, size, default=231):
        self.size, self.default, self.writes = size, default, {}

    def read(self, indices):
        return np.array([self.writes.get(int(i), self.default) for i in indices], dtype=np.uint8)


class TL:
    int64 = np.int64
    def __init__(self):
        self.row, self.plane = 0, 0

    def program_id(self, axis):
        return arr(self.row if axis == 0 else self.plane)

    def arange(self, start, end):
        return arr(np.arange(start, end))

    def where(self, *args):
        return arr(np.where(*args))

    def device_assert(self, condition, message):
        assert bool(np.all(condition)), message

    def load(self, pointer, mask=True, other=0):
        indices, mask = np.broadcast_arrays(pointer.offset, mask)
        active = indices[mask]
        size = pointer.data.size
        assert np.all((active >= 0) & (active < size)), 'unmasked read OOB'
        result = np.full(indices.shape, other, dtype=(pointer.data.dtype if isinstance(pointer.data, np.ndarray) else np.uint8))
        result[mask] = pointer.data[active] if isinstance(pointer.data, np.ndarray) else pointer.data.read(active)
        return arr(result)

    def store(self, pointer, values, mask=True):
        indices, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        indices, values = indices[mask], values[mask]
        assert np.all((indices >= 0) & (indices < pointer.data.size)), 'unmasked store OOB'
        if isinstance(pointer.data, np.ndarray):
            pointer.data[indices] = values
        else:
            pointer.data.writes.update(zip(map(int, indices), map(int, values)))


def load_ast(node, ns):
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WIRE), 'exec'), ns)


def kernels():
    tl = TL()
    ns = {'tl': tl}
    load_ast(fn(CACHE, '_scale_128x4_offset'), ns)
    load_ast(fn(WIRE, '_scatter_wire_multirow'), ns)
    return tl, functools.partial(ns['_scatter_wire_multirow'], ROWS_PER_CTA=16)


class WireTests(unittest.TestCase):
    def test_quantizer_exact_expression_source(self):
        current = fn(WIRE, '_pack_wire')
        old = fn(CACHE, '_quantize_main_index_rows_d128_kernel')
        names = {'amax', 'raw_scale', 'stored_scale', 'scale', 'packed_codes'}
        def expressions(node):
            return {n.targets[0].id: ast.dump(n.value) for n in node.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in names}
        self.assertEqual(expressions(current), expressions(old))
        self.assertIn('bitcast=True', ast.unparse(current))
        self.assertIn('torch.bfloat16', ast.unparse(fn(WIRE, 'pack_cp_nvfp4_wire')))

    def test_scatter_has_no_requantization_or_scale_cast(self):
        text = ast.unparse(fn(WIRE, '_scatter_wire_multirow'))
        for forbidden in ('float32', 'float8', '_e2m1_encode', 'tl.max(', 'tl.abs('):
            self.assertNotIn(forbidden, text)
        wrapper = ast.unparse(fn(WIRE, 'scatter_cp_nvfp4_wire'))
        for forbidden in ('.cpu(', '.item(', '.tolist(', 'torch.empty(', 'torch.zeros('):
            self.assertNotIn(forbidden, wrapper)
        self.assertIn('view(torch.uint8)', wrapper)
        self.assertIn('debug=True', wrapper)

    def case(self, sources, slots, persistent_slots, source_rows=17, fi_working_layout=False):
        tl, kernel = kernels()
        sources, slots, persistent_slots = [np.asarray(x, dtype=np.int64) for x in (sources, slots, persistent_slots)]
        wire = ((np.arange(source_rows * 648, dtype=np.int64) * 13 + 7) % 256).astype(np.uint8)
        capacities = (5, 3)
        row_bytes = (32768, 4096, 32768, 4096, 8192, 1024)
        strides = ((32785, 4103, 32785, 4103, 8215, 1033),
                   (32801, 4107, 32801, 4107, 8231, 1041))
        backing = [np.full(capacities[j] * stride + 32, 231, dtype=np.uint8)
                   for j in range(2) for stride in strides[j]]
        expected = [x.copy() for x in backing]
        arguments = [Pointer(wire), Pointer(sources), Pointer(slots), Pointer(persistent_slots)]
        arguments += [Pointer(x, 16) for x in backing]
        options = dict(ROWS=len(slots), SOURCE_ROWS=source_rows,
                       WORK_PAGES=capacities[0], PERSIST_PAGES=capacities[1], FI_WORKING_LAYOUT=fi_working_layout,
                       K_S0=strides[0][0], S_S0=strides[0][1], I_S0=strides[0][4], IS_S0=strides[0][5],
                       PK_S0=strides[1][0], PS_S0=strides[1][1], PI_S0=strides[1][4], PIS_S0=strides[1][5])
        for row in range((len(slots) + kernel.keywords['ROWS_PER_CTA'] - 1) // kernel.keywords['ROWS_PER_CTA']):
            for plane in range(9):
                tl.row, tl.plane = row, plane
                kernel(*arguments, **options)
        # Independent oracle: scatter into HND codes and linear [head,128,8]
        # scales, then transpose the full scale tile into the physical layout.
        for destination, slot_map in enumerate((slots, persistent_slots)):
            cap = capacities[destination]
            codes = [np.full((cap, heads, 128, 64), 231, dtype=np.uint8) for heads in (4, 4, 1)]
            scales = [np.full((cap, heads, 128, 8), 231, dtype=np.uint8) for heads in (4, 4, 1)]
            for source, slot in zip(sources, slot_map):
                if not 0 <= slot < cap * 128:
                    continue
                block, off = divmod(int(slot), 128)
                record = wire[source * 648:(source + 1) * 648].reshape(9, 72)
                for p in range(9):
                    which, head = (0, p) if p < 4 else ((1, p - 4) if p < 8 else (2, 0))
                    codes[which][block, head, off] = record[p, :64]
                    scales[which][block, head, off] = record[p, 64:]
            for which in range(3):
                heads = (4, 4, 1)[which]
                scale_tile = scales[which].reshape(cap, heads, 4, 32, 2, 4).transpose(0, 1, 4, 3, 2, 5)
                if fi_working_layout and destination == 0:
                    if which == 0:
                        scale_tile = scales[which]  # token-major K
                    elif which == 1:
                        scale_tile = scales[which].reshape(cap, heads, 32, 4, 4, 2).transpose(0, 1, 2, 4, 5, 3)
                for plane, values in ((2 * which, codes[which]), (2 * which + 1, scale_tile)):
                    target = expected[destination * 6 + plane]
                    for block in range(cap):
                        begin = 16 + block * strides[destination][plane]
                        target[begin:begin + row_bytes[plane]] = values[block].reshape(-1)
        for actual, oracle in zip(backing, expected):
            np.testing.assert_array_equal(actual, oracle)

    def test_full_bytes_repeated_unsorted_sources_owner_and_page_boundaries(self):
        self.case([9, 1, 9, 0, 16, 4, 8, 7], [0, 127, 128, 511, 512, -1, 640, 33],
                  [-1, 128, -1, 127, 256, 0, -1, 383])

    def test_fi_working_scales_independent_oracle_persistent_unchanged(self):
        self.case([9, 1, 9, 0, 16, 4, 8, 7], [0, 127, 128, 511, 512, -1, 640, 33],
                  [-1, 128, -1, 127, 256, 0, -1, 383], fi_working_layout=True)

    def test_empty(self):
        self.case([], [], [], source_rows=0)

    def test_masked_padding_does_not_read_wire(self):
        self.case([-1, 999999], [-1, 999999], [-1, -1])

    def test_invalid_source_is_contract_failure(self):
        with self.assertRaisesRegex(AssertionError, 'source row out of bounds'):
            self.case([17], [0], [-1])

    def test_int64_source_and_destination_byte_addresses(self):
        tl, kernel = kernels()
        source = 2**31 // 648 + 127
        stride = 2**31 + 32768
        wire = Sparse((source + 1) * 648, default=73)
        destinations = [Sparse(2 * stride) for _ in range(12)]
        tl.row, tl.plane = 0, 8
        kernel(Pointer(wire), Pointer(np.array([source], dtype=np.int64)),
               Pointer(np.array([128], dtype=np.int64)), Pointer(np.array([-1], dtype=np.int64)),
               *[Pointer(x) for x in destinations], ROWS=1, SOURCE_ROWS=source + 1,
               WORK_PAGES=2, PERSIST_PAGES=2,
               **{k: stride for k in ('K_S0', 'S_S0', 'I_S0', 'IS_S0', 'PK_S0', 'PS_S0', 'PI_S0', 'PIS_S0')})
        self.assertEqual(len(destinations[4].writes), 64)
        self.assertEqual(len(destinations[5].writes), 8)
        self.assertTrue(all(i >= 2**31 for i in destinations[4].writes))
        self.assertTrue(all(v == 73 for v in destinations[4].writes.values()))
        self.assertFalse(any(x.writes for x in destinations[6:]))



# Host wrapper tests execute real AST with tensor metadata and a launch recorder.
class MetaTensor:
    next_pointer = 100
    def __init__(self, shape, dtype='u8', stride=None, pointer=None, device='cuda:0', contiguous=True, offset=0):
        self.offset = offset
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda, self.contiguous = len(shape), True, contiguous
        if stride is None:
            stride, product = [], 1
            for size in reversed(shape):
                stride.insert(0, product)
                product *= size
        self.strides = tuple(stride)
        MetaTensor.next_pointer += 1
        self.pointer = MetaTensor.next_pointer if pointer is None else pointer

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return int(np.prod(self.shape))

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.pointer + self.offset

    def untyped_storage(self):
        import types
        return types.SimpleNamespace(data_ptr=lambda: self.pointer)

    def view(self, dtype):
        return MetaTensor(self.shape, dtype, self.strides, self.pointer, self.device, self.contiguous, offset=self.offset)


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kw: self.calls.append((grid, args, kw))


def wrappers():
    import types
    pack, scatter = Launch(), Launch()
    ns = {'torch': types.SimpleNamespace(uint8='u8', bfloat16='bf16', int64='i64', float8_e4m3fn='fp8'),
          'WIRE_BYTES': 648, '_pack_wire': pack, '_scatter_wire_multirow': scatter,
          'triton': types.SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b)}
    load_ast(fn(CACHE, '_validate_disjoint_cp_planes'), ns)
    load_ast(fn(CACHE, '_validate_cp_writer_planes'), ns)
    load_ast(fn(WIRE, 'pack_cp_nvfp4_wire'), ns)
    load_ast(fn(WIRE, 'scatter_cp_nvfp4_wire'), ns)
    return ns, pack, scatter


def planes(pages):
    return tuple(MetaTensor((pages, width), dtype)
                 for width, dtype in zip((32768, 4096, 32768, 4096, 8192, 1024), ('u8', 'fp8') * 3))


class WrapperTests(unittest.TestCase):
    def test_pack_shape_and_zero_rows(self):
        ns, pack, _ = wrappers()
        for rows in (0, 1, 7, 63360):
            ns['pack_cp_nvfp4_wire'](MetaTensor((rows, 1152), 'bf16'), MetaTensor((rows, 648)))
        ns['pack_cp_nvfp4_wire'](MetaTensor((0, 1152), 'bf16', pointer=0), MetaTensor((0, 648), pointer=0))
        self.assertEqual([call[0] for call in pack.calls], [(1, 9), (7, 9), (63360, 9)])
        self.assertTrue(all(c[2]['INPUT_S0'] == 1152 for c in pack.calls))

    def test_pack_rejects_wrong_dtype_layout_device_and_alias(self):
        ns, pack, _ = wrappers()
        cases = [(MetaTensor((2, 1152), 'fp8'), MetaTensor((2, 648))),
                 (MetaTensor((2, 1152), 'bf16'), MetaTensor((2, 656))),
                 (MetaTensor((2, 1152), 'bf16', stride=(1151, 1)), MetaTensor((2, 648))),
                 (MetaTensor((2, 1152), 'bf16'), MetaTensor((2, 648), device='cuda:1')),
                 (MetaTensor((2, 1152), 'bf16', pointer=1), MetaTensor((2, 648), pointer=1))]
        for source, destination in cases:
            with self.assertRaises(ValueError):
                ns['pack_cp_nvfp4_wire'](source, destination)
        self.assertFalse(pack.calls)

    def test_scatter_raw_bytes_strides_and_zero_rows(self):
        ns, _, scatter = wrappers()
        working, persistent = planes(5), planes(3)
        for rows in (0, 3):
            maps = [MetaTensor((rows,), 'i64') for _ in range(3)]
            ns['scatter_cp_nvfp4_wire'](MetaTensor((12, 648)), *maps, working, persistent)
        self.assertEqual(len(scatter.calls), 1)
        grid, args, kwargs = scatter.calls[0]
        self.assertEqual(grid, (1, 9))
        self.assertTrue(all(t.dtype == 'u8' for t in args[4:16]))
        self.assertEqual(kwargs['PS_S0'], 4096)
        self.assertTrue(kwargs['debug'])

    def test_fi_working_layout_is_explicit_and_bool(self):
        ns, _, scatter = wrappers()
        maps = [MetaTensor((3,), 'i64') for _ in range(3)]
        ns['scatter_cp_nvfp4_wire'](MetaTensor((12, 648)), *maps, planes(5), planes(3), fi_working_layout=True)
        self.assertTrue(scatter.calls[0][2]['FI_WORKING_LAYOUT'])
        with self.assertRaises(ValueError):
            ns['scatter_cp_nvfp4_wire'](MetaTensor((12, 648)), *maps, planes(5), planes(3), fi_working_layout=1)

    def test_plane_alias_guard_accepts_interlaced_and_rejects_overlap(self):
        ns, _, _ = wrappers()
        validate = ns['_validate_cp_writer_planes']
        widths = (32768, 4096, 32768, 4096, 8192, 1024)
        offsets = (0, 32768, 36864, 69632, 0, 8192)
        shared = tuple(MetaTensor((5, width), dtype,
                       stride=(73728 if i < 4 else 9216, 1),
                       pointer=1000 if i < 4 else 2000, offset=offset)
                       for i, (width, offset, dtype) in enumerate(zip(widths, offsets, ('u8', 'fp8') * 3)))
        self.assertEqual(validate(shared), 5)
        for index, offset in ((2, 0), (3, 32768), (2, 73728), (5, 8191)):
            invalid = list(shared)
            item = invalid[index]
            invalid[index] = MetaTensor(item.shape, item.dtype, item.strides,
                                        item.pointer, offset=offset)
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                validate(invalid)
        empty = tuple(MetaTensor((0, width), dtype, pointer=0)
                      for width, dtype in zip(widths, ('u8', 'fp8') * 3))
        self.assertEqual(validate(empty), 0)

    def test_plane_alias_guard_against_finite_interval_oracle(self):
        import random
        rng = random.Random(20261010)
        ns, _, _ = wrappers()
        for _ in range(500):
            pitch, pages = rng.randint(1, 200), rng.randint(1, 8)
            starts = (rng.randrange(2000), rng.randrange(2000))
            widths = (rng.randint(1, pitch), rng.randint(1, pitch))
            overlap = any(
                max(starts[0] + i * pitch, starts[1] + j * pitch)
                < min(starts[0] + i * pitch + widths[0], starts[1] + j * pitch + widths[1])
                for i in range(pages) for j in range(pages)
            )
            pair = tuple(MetaTensor((pages, width), stride=(pitch, 1),
                                    pointer=1000, offset=start)
                         for width, start in zip(widths, starts))
            if overlap:
                with self.assertRaises(ValueError):
                    ns['_validate_disjoint_cp_planes'](pair, pages)
            else:
                ns['_validate_disjoint_cp_planes'](pair, pages)

    def test_scatter_rejects_missing_full_map_alias_and_device(self):
        ns, _, scatter = wrappers()
        wire = MetaTensor((12, 648))
        maps = [MetaTensor((3,), 'i64') for _ in range(3)]
        working, persistent = planes(5), planes(3)
        with self.assertRaises(ValueError):
            ns['scatter_cp_nvfp4_wire'](wire, maps[0], maps[1], MetaTensor((1,), 'i64'), working, persistent)
        with self.assertRaises(ValueError):
            ns['scatter_cp_nvfp4_wire'](wire, *maps, working, working)
        working_alias = list(working)
        working_alias[0].pointer = wire.pointer
        with self.assertRaises(ValueError):
            ns['scatter_cp_nvfp4_wire'](wire, *maps, working_alias, persistent)
        self.assertFalse(scatter.calls)


class IntegrationSourceTests(unittest.TestCase):
    def setUp(self):
        self.attention = ROOT.parents[1] / 'modules/hybrid/msa_attention.py'
        self.baseline = ROOT / 'test/nvfp4_cp_wire_reference_contract.py'

    def method(self, path, name):
        return next(n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_flag_default_off_and_native_guard(self):
        text = self.attention.read_text()
        self.assertIn('os.environ.get("RTP_LLM_CP_SUFFIX_NVFP4_WIRE", "0") == "1"', text)
        node = self.method(self.attention, '_cp_suffix_wire_enabled')
        self.assertEqual(ast.unparse(node.body[0]), 'return _CP_SUFFIX_NVFP4_WIRE and self._same_layer_overlap_enabled()')
        self.assertEqual(ast.dump(self.method(self.attention, '_same_layer_overlap_enabled')),
                         ast.dump(self.method(self.baseline, '_same_layer_overlap_enabled')))

    def test_fallback_writer_call_exact_and_ordinary_event_guard(self):
        name = '_write_cp_suffix_to_nvfp4_working_pages'
        old, new = self.method(self.baseline, name), self.method(self.attention, name)
        def writer(n):
            return next(c for c in ast.walk(n) if isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Name) and c.func.id == 'nvfp4_quantize_cp_main_index_rows_to_planes')
        self.assertEqual(ast.dump(writer(old)), ast.dump(writer(new)))
        branch = next(n for n in new.body if isinstance(n, ast.If)
                      and 'self._cp_suffix_wire_enabled()' in ast.unparse(n.test))
        self.assertEqual(ast.unparse(branch.test), 'packed_kv_event is not None and self._cp_suffix_wire_enabled()')
        scatter = next(c for c in ast.walk(branch) if isinstance(c, ast.Call)
                       and isinstance(c.func, ast.Name) and c.func.id == 'scatter_cp_nvfp4_wire')
        self.assertEqual(ast.unparse(scatter.args[3]), 'slot_mapping[:token_count]')
        kwargs = {k.arg: ast.unparse(k.value) for k in scatter.keywords}
        self.assertEqual(kwargs['rows_per_cta'], '16')
        self.assertEqual(kwargs['fi_working_layout'], 'self._flashinfer_nvfp4_prefill')

    def test_gather_one_collective_after_sender_pack_and_producer_fence(self):
        n = self.method(self.attention, '_cp_all_gather_packed_kv')
        calls = sorted((x for x in ast.walk(n) if isinstance(x, ast.Call)), key=lambda x: (x.lineno, x.col_offset))
        names = [ast.unparse(x.func) for x in calls]
        self.assertLess(names.index('pack_cp_nvfp4_wire'), names.index('stream.wait_stream'))
        self.assertLess(names.index('stream.wait_stream'), names.index('all_gather', names.index('stream.wait_stream') + 1))
        self.assertNotIn('restore_stream', ast.unparse(n))


    def test_actual_gather_dispatch_wire_and_default_bf16(self):
        import contextlib
        import sys
        import types
        from unittest.mock import patch
        for enabled in (False, True):
            trace = []
            main = object()
            stream = types.SimpleNamespace(wait_stream=lambda source: trace.append(('producer_wait', source)))
            packed = MetaTensor((7, 1152), 'bf16')
            send = MetaTensor((7, 648 if enabled else 1152), 'u8' if enabled else 'bf16')
            recv = MetaTensor((28, 648 if enabled else 1152), 'u8' if enabled else 'bf16')
            send.copy_ = lambda source: trace.append(('bf16_copy', source))
            owner = (types.SimpleNamespace(side_stream=stream,
                                          suffix_format='nvfp4_wire_v1' if enabled else 'bf16'),
                     types.SimpleNamespace(tensors={'kv_send': send, 'kv_recv': recv}))
            class Ready:
                def record(self, source):
                    trace.append(('ready', source))
            cuda = types.SimpleNamespace(current_stream=lambda device: main,
                                         stream=lambda stream: contextlib.nullcontext(), Event=Ready)
            op = types.ModuleType('rtp_llm.models_py.triton_kernels.common.nvfp4_cp_wire')
            op.pack_cp_nvfp4_wire = lambda source, out: trace.append(('wire_pack', source, out))
            ns = {'torch': types.SimpleNamespace(cuda=cuda), '_CP_PACKED_KV_OVERLAP': True,
                  'Group': types.SimpleNamespace(TP='main', TP_SIDE='side'),
                  'all_gather': lambda source, group, out=None: trace.append(('AG', source, group, out))}
            node = self.method(self.attention, '_cp_all_gather_packed_kv')
            load_ast(node, ns)
            attention = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(tp_size=4),
                                              _same_layer_overlap_enabled=lambda: True,
                                              _cp_suffix_wire_enabled=lambda: enabled,
                                              _same_layer_suffix=owner)
            with patch.dict(sys.modules, {op.__name__: op}):
                output, ready = ns[node.name](attention, packed)
            self.assertIs(output, recv)
            self.assertEqual(trace[0][0], 'wire_pack' if enabled else 'bf16_copy')
            self.assertEqual(trace[1], ('producer_wait', main))
            self.assertEqual(trace[2], ('AG', send, 'side', recv))
            self.assertEqual(trace[3], ('ready', stream))
            self.assertEqual(len(trace), 4)


if __name__ == '__main__':
    unittest.main()
