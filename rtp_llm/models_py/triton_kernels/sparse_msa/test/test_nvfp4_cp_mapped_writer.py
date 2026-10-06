"""CPU source contracts and real production writer GPU regressions."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "common" / "nvfp4_kv_cache.py"
ATTENTION = HERE.parents[2] / "modules" / "hybrid" / "msa_attention.py"


def function(tree, name):
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )


class CPWriterSourceTest(unittest.TestCase):
    def test_disabled_mapping_preserves_frozen_quantization_and_store_body(self):
        current = function(
            ast.parse(SOURCE.read_text()), "_quantize_main_index_rows_d128_kernel"
        )
        reference = function(
            ast.parse((HERE / "nvfp4_cp_writer_reference.py").read_text()),
            "_quantize_main_index_rows_d128_kernel",
        )
        normalized = (
            ast.unparse(current)
            .replace("source_row * K_S0", "row * K_S0")
            .replace("source_row * V_S0", "row * V_S0")
            .replace("source_row * IDX_S0", "row * IDX_S0")
            .replace("mask=input_valid &", "mask=valid_slot &")
        )
        current = ast.parse(normalized).body[0]
        current.body = [
            n
            for n in current.body
            if not (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id in ("source_row", "input_valid")
                    for t in n.targets
                )
            )
            and not (
                isinstance(n, ast.If)
                and isinstance(n.test, ast.Name)
                and n.test.id in ("MAP_SOURCE_ROWS", "WRITE_PERSISTENT")
            )
        ]
        self.assertEqual(
            [ast.dump(n) for n in current.body], [ast.dump(n) for n in reference.body]
        )
        self.assertEqual(
            [ast.literal_eval(n) for n in current.args.defaults[-3:]],
            [False, False, False],
        )

    def test_legacy_wrappers_supply_dummy_pointers_and_do_not_enable_mapping(self):
        tree = ast.parse(SOURCE.read_text())
        for name, slot in (
            ("quantize_main_index_rows", "physical_slots"),
            ("quantize_main_index_rows_to_planes", "slots"),
        ):
            node = function(tree, name)
            call = next(
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "_quantize_main_index_rows_d128_kernel"
            )
            self.assertEqual([ast.unparse(n) for n in call.args[3:6]], [slot] * 3)
            self.assertFalse(
                any(
                    k.arg in ("MAP_SOURCE_ROWS", "MAP_OWNED_ROWS")
                    for k in call.keywords
                )
            )

    def test_cp_keeps_restore_dual_write_clear_order_without_row_gathers(self):
        node = function(
            ast.parse(ATTENTION.read_text()), "_write_cp_suffix_to_nvfp4_working_pages"
        )
        calls = sorted(
            (n for n in ast.walk(node) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        names = [ast.unparse(n.func) for n in calls]
        self.assertFalse(any(name.endswith(".index_select") for name in names))
        writers = [
            i
            for i, n in enumerate(names)
            if n == "nvfp4_quantize_cp_main_index_rows_to_planes"
        ]
        self.assertEqual(len(writers), 1)
        restore = names.index("restore_prefix_planes")
        clear = names.index("clear_packed_working_tail_scales")
        self.assertLess(restore, writers[0])
        self.assertLess(writers[0], clear)
        writer = next(
            n
            for n in calls
            if ast.unparse(n.func) == "nvfp4_quantize_cp_main_index_rows_to_planes"
        )
        self.assertIn("persistent_slots", {k.arg for k in writer.keywords})
        self.assertIn("persistent_planes", {k.arg for k in writer.keywords})

    def test_all_prefill_callers_match_store_signature(self):
        tree = ast.parse(ATTENTION.read_text())
        helper = function(tree, "_write_cp_suffix_to_nvfp4_working_pages")
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == helper.name
        ]
        self.assertEqual(len(calls), 2)  # Ordinary and CP native FP4 prefill.
        for call in calls:
            self.assertEqual(len(call.args), len(helper.args.args) - 1)
            self.assertFalse(call.keywords)

    def test_cp_wrapper_allocates_no_device_metadata_and_reads_maps_on_device(self):
        wrapper = ast.unparse(
            function(
                ast.parse(SOURCE.read_text()), "quantize_cp_main_index_rows_to_planes"
            )
        )
        for forbidden in (
            "index_select(",
            ".cpu(",
            ".item(",
            ".tolist(",
            "torch.empty(",
            "torch.zeros(",
            "torch.tensor(",
        ):
            self.assertNotIn(forbidden, wrapper)
        self.assertIn("1152", wrapper)
        self.assertIn("MAP_SOURCE_ROWS=True", wrapper)
        self.assertIn("MAP_OWNED_ROWS=owned_rows is not None", wrapper)
        kernel = ast.unparse(
            function(
                ast.parse(SOURCE.read_text()), "_quantize_main_index_rows_d128_kernel"
            )
        )
        self.assertIn("owned_rows_ptr + row", kernel)
        self.assertIn("unpad_ptr + logical_row", kernel)
        self.assertGreaterEqual(kernel.count(".to(tl.int64)"), 4)


class CPWriterGPURegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        spec = importlib.util.spec_from_file_location(
            "nvfp4_cp_production_test_module", SOURCE
        )
        cls.op = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.op
        spec.loader.exec_module(cls.op)
        spec = importlib.util.spec_from_file_location(
            "nvfp4_cp_frozen_reference", HERE / "nvfp4_cp_writer_reference.py"
        )
        cls.reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.reference
        spec.loader.exec_module(cls.reference)

    def case(self, rows, rank=0, padded_stride=False, exact=False):
        torch = self.torch
        if exact:
            unpad, owned, ps, ws = [], [], [], []
            for req in range(20):
                for pos in range(7451):
                    segment, within = divmod(pos, 932)
                    cp_rank = segment if segment < 4 else 7 - segment
                    unpad.append(
                        cp_rank * 37280
                        + req * 1864
                        + within
                        + (932 if segment >= 4 else 0)
                    )
                    logical = 66560 + pos
                    ws.append(req * 579 * 128 + logical)
                    if (logical // 128) % 4 == rank:
                        owned.append(req * 7451 + pos)
                        ps.append(
                            (1 + req * 145 + (logical // 128) // 4) * 128
                            + logical % 128
                        )
            self.assertEqual(len(owned), (38400, 38400, 36380, 35840)[rank])
            padded, persistent_pages, working_pages = 149120, 10929, 11580
        else:
            unpad = list(reversed(range(rows)))
            owned = list(range(rank % 4, rows, 4))
            ps, ws = [128 + i for i in range(len(owned))], [
                128 + i for i in range(rows)
            ]
            persistent_pages = max(3, (len(owned) + 127) // 128 + 2)
            working_pages = max(3, (rows + 127) // 128 + 2)
            padded = rows + 5
            for slots, pages in ((ps, persistent_pages), (ws, working_pages)):
                if slots:
                    slots[0] = -1
                if len(slots) > 1:
                    slots[1] = pages * 128
        backing = torch.randn(
            padded, 1168 if padded_stride else 1152, device="cuda", dtype=torch.bfloat16
        )
        packed = backing[:, :1152]
        if rows:
            values = torch.tensor(
                [
                    0.0,
                    -0.0,
                    0.25,
                    0.75,
                    1.25,
                    1.75,
                    2.5,
                    3.5,
                    5.0,
                    6.0,
                    -5.0,
                    -3.5,
                    -2.5,
                    -1.75,
                    0.5,
                    -6.0,
                ],
                device="cuda",
                dtype=torch.bfloat16,
            )
            packed[0, :128].copy_(values.repeat(8))
            packed[0, 128:144].zero_()
            packed[0, 144:160].fill_(1e-30)
            packed[0, 160:176].fill_(1e30)
        maps = [
            torch.tensor(v, dtype=torch.int64, device="cuda")
            for v in (unpad, owned, ps, ws)
        ]
        persistent_slots = torch.full(
            (len(unpad),), -1, device="cuda", dtype=torch.int64
        )
        persistent_slots.scatter_(0, maps[1], maps[2])
        value = torch.full(
            (persistent_pages, 65536), 0xA5, device="cuda", dtype=torch.uint8
        )
        side = torch.full(
            (persistent_pages, 17408), 0x5A, device="cuda", dtype=torch.uint8
        )
        layout = self.op.cache_layout(value, side, 4, 128, 128)
        kp, ks = layout.main_plane(0)
        vp, vs = layout.main_plane(1)
        ip, ix = layout.indexer(128)
        pp = (kp, ks, vp, vs, ip, ix)
        wp = tuple(
            torch.full(
                shape, 0x5A if i % 2 else 0xA5, device="cuda", dtype=torch.uint8
            ).view(torch.float8_e4m3fn if i % 2 else torch.uint8)
            for i, shape in enumerate(
                (
                    (working_pages, 4, 128, 64),
                    (working_pages, 4, 128, 8),
                    (working_pages, 4, 128, 64),
                    (working_pages, 4, 128, 8),
                    (working_pages, 1, 128, 64),
                    (working_pages, 1, 128, 8),
                )
            )
        )
        return packed, backing, maps, pp, wp, persistent_slots, (value, side, *wp)

    def run_chain(self, case, mapped):
        packed, _, (unpad, owned, ps, ws), pp, wp, persistent_slots, _ = case
        if mapped:
            self.op.quantize_cp_main_index_rows_to_planes(
                packed,
                unpad,
                ws,
                *wp,
                persistent_slots=persistent_slots,
                persistent_planes=pp,
            )
        else:
            selected = packed.index_select(0, unpad).contiguous()
            k = selected[:, :512].view(unpad.numel(), 4, 128)
            v = selected[:, 512:1024].view(unpad.numel(), 4, 128)
            idx = selected[:, 1024:].view(unpad.numel(), 1, 128)
            pk, pv, pi = (t.index_select(0, owned) for t in (k, v, idx))
            self.reference.write_reference(pk, pv, pi, ps, pp)
            self.reference.write_reference(k, v, idx, ws, wp)

    def reset(self, case):
        for i, tensor in enumerate(case[-1]):
            tensor.view(self.torch.uint8).fill_(0x5A if i % 2 else 0xA5)

    def compare(self, case, baseline, candidate):
        torch = self.torch
        before = case[1].view(torch.int16).clone()
        self.reset(case)
        baseline()
        expected = [t.view(torch.uint8).clone() for t in case[-1]]
        self.reset(case)
        candidate()
        for actual, old in zip(case[-1], expected):
            self.assertTrue(torch.equal(actual.view(torch.uint8), old))
        self.assertTrue(torch.equal(case[1].view(torch.int16), before))

    def test_production_wrapper_small_exact_and_changed_graph(self):
        torch = self.torch
        for rows in (0, 1, 3, 127, 128, 129, 1934):
            for padded_stride in (False, True):
                case = self.case(rows, padded_stride=padded_stride)
                self.compare(
                    case,
                    lambda: self.run_chain(case, False),
                    lambda: self.run_chain(case, True),
                )
                del case
        for rank in range(4):
            case = self.case(149020, rank=rank, exact=True)
            self.compare(
                case,
                lambda: self.run_chain(case, False),
                lambda: self.run_chain(case, True),
            )
            del case
        for rows in (0, 129):
            case = self.case(rows, padded_stride=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graphs = []
            for mapped in (False, True):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.run_chain(case, mapped)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.run_chain(case, mapped)
                graphs.append(graph)
            for _ in range(3):
                case[0].normal_()
                for tensor in case[2]:
                    tensor.copy_(torch.roll(tensor, 1))
                # Rebuild the full logical map before replay, not in the writer.
                case[5].fill_(-1)
                case[5].scatter_(0, case[2][1], case[2][2])
                self.compare(case, graphs[0].replay, graphs[1].replay)
            del graph, graphs, case

    def test_clear_validation_rejects_other_geometry_dtype_and_map(self):
        case = self.case(3)
        packed, _, (unpad, _, _, ws), pp, wp, persistent_slots, _ = case
        for bad in (packed[:, :1024], packed.float(), packed[:, ::2]):
            with self.assertRaises(ValueError):
                self.op.quantize_cp_main_index_rows_to_planes(bad, unpad, ws, *wp)
        with self.assertRaises(ValueError):
            self.op.quantize_cp_main_index_rows_to_planes(packed, unpad.int(), ws, *wp)
        with self.assertRaises(ValueError):
            self.op.quantize_cp_main_index_rows_to_planes(
                packed, unpad, ws, wp[0], wp[1], wp[2], wp[3], wp[4], wp[5][:, :, :, :4]
            )
        with self.assertRaises(ValueError):
            self.op.quantize_cp_main_index_rows_to_planes(
                packed,
                unpad,
                ws,
                *wp,
                persistent_slots=persistent_slots,
            )
        with self.assertRaises(ValueError):
            self.op.quantize_cp_main_index_rows_to_planes(
                packed,
                unpad,
                ws,
                *wp,
                persistent_slots=persistent_slots[:-1],
                persistent_planes=pp,
            )

    def test_dual_destinations_have_independent_validity(self):
        torch = self.torch
        case = self.case(16)
        _, _, (_, owned, ps, ws), _, _, persistent_slots, _ = case
        # Owned rows0/4/8/12: both, working-only, persistent-only, neither.
        ps.copy_(torch.tensor([128, -1, 130, -1], device="cuda", dtype=torch.int64))
        ws[owned] = torch.tensor([128, 132, -1, -1], device="cuda", dtype=torch.int64)
        persistent_slots.fill_(-1)
        persistent_slots.scatter_(0, owned, ps)
        self.compare(
            case,
            lambda: self.run_chain(case, False),
            lambda: self.run_chain(case, True),
        )

    def test_dual_inactive_rows_do_not_read_sources(self):
        torch = self.torch
        case = self.case(16)
        packed, _, (unpad, owned, ps, ws), pp, wp, persistent_slots, _ = case
        # Active source indices must be valid, as with the old index_select.
        # Inactive rows must not read sources when neither destination is valid.
        ps.copy_(torch.arange(128, 132, device="cuda", dtype=torch.int64))
        ws.copy_(torch.arange(128, 144, device="cuda", dtype=torch.int64))
        ps[:2] = -1
        ws[owned[:2]] = -1
        persistent_slots.scatter_(0, owned, ps)
        unpad[0] = -1
        unpad[4] = packed.shape[0]

        def reference():
            valid = (unpad >= 0) & (unpad < packed.shape[0])
            selected = packed.index_select(
                0, unpad.clamp(0, packed.shape[0] - 1)
            ).contiguous()
            k = selected[:, :512].view(16, 4, 128)
            v = selected[:, 512:1024].view(16, 4, 128)
            idx = selected[:, 1024:].view(16, 1, 128)
            safe_ws = torch.where(valid, ws, -1)
            safe_ps = torch.where(valid.index_select(0, owned), ps, -1)
            self.reference.write_reference(k, v, idx, safe_ws, wp)
            self.reference.write_reference(
                k.index_select(0, owned),
                v.index_select(0, owned),
                idx.index_select(0, owned),
                safe_ps,
                pp,
            )

        self.compare(case, reference, lambda: self.run_chain(case, True))

    def test_legacy_d128_wrappers_match_frozen_writer(self):
        for rows in (0, 3, 129):
            case = self.case(rows, padded_stride=True)
            packed, _, (unpad, owned, ps, ws), pp, wp, _, backing = case
            selected = packed.index_select(0, unpad).contiguous()
            k = selected[:, :512].view(rows, 4, 128)
            v = selected[:, 512:1024].view(rows, 4, 128)
            idx = selected[:, 1024:].view(rows, 1, 128)
            pk, pv, pi = (t.index_select(0, owned) for t in (k, v, idx))
            layout = self.op.cache_layout(backing[0], backing[1], 4, 128, 128)

            def legacy():
                self.op.quantize_main_index_rows(
                    pk, pv, pi, ps, layout, mma_scale_layout=True
                )
                self.op.quantize_main_index_rows_to_planes(k, v, idx, ws, *wp)

            self.compare(case, lambda: self.run_chain(case, False), legacy)

    def test_first_int32_source_address_overflow(self):
        torch = self.torch
        case = self.case(3)
        packed = torch.empty_strided(
            (3, 1152), (2**30 + 128, 1), device="cuda", dtype=torch.bfloat16
        )
        packed.copy_(case[0][:3])
        # Source row2 is valid and actually loaded, not hidden by invalid slots.
        case[2][2][0] = 128
        case[5].scatter_(0, case[2][1], case[2][2])
        case[2][3].copy_(torch.arange(128, 131, device="cuda", dtype=torch.int64))
        case = (packed, packed, *case[2:])
        self.compare(
            case,
            lambda: self.run_chain(case, False),
            lambda: self.run_chain(case, True),
        )


if __name__ == "__main__":
    unittest.main()
