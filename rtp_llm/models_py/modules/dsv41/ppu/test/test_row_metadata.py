"""Row domains, alias semantics and metadata invalidation without payload work."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import (
    RowLayout,
    RowScope,
    RowView,
    SelectorKind,
    is_plain_tensor,
    tensor_metadata_key,
)


def full_view(
    selector, *, scope=None, positions=None, sources=(), layout=RowLayout.FULL
):
    positions = torch.arange(4) if positions is None else positions
    return RowView(
        (positions, torch.zeros_like(positions), selector),
        layout,
        scope or RowScope(object()),
        False,
        sources,
    )


class RowMetadataTest(unittest.TestCase):
    def test_all_true_mask_stays_mask_and_preserves_indexing_semantics(self):
        mask = torch.ones(4, dtype=torch.bool)
        masked, all_rows = full_view(mask), full_view(slice(None))
        self.assertIs(masked.selector.value, mask)
        self.assertIs(masked.selector.kind, SelectorKind.MASK)
        self.assertIs(all_rows.selector.kind, SelectorKind.ALL)
        values = torch.arange(12).reshape(4, 3)
        self.assertFalse(torch._C._is_alias_of(values[masked.selector.value], values))
        self.assertTrue(torch._C._is_alias_of(values[all_rows.selector.value], values))
        self.assertTrue(torch.equal(values[mask], values[all_rows.selector.value]))

    def test_selector_kinds_and_optional_metadata_validation_do_no_value_reads(self):
        for selector, kind in (
            (slice(None), SelectorKind.ALL),
            (slice(1, 3), SelectorKind.RANGE),
            (torch.ones(4, dtype=torch.bool), SelectorKind.MASK),
            (torch.tensor([3, 1]), SelectorKind.INDICES),
        ):
            view = full_view(selector)
            with patch.object(
                torch.Tensor, "item", side_effect=AssertionError("value read")
            ), patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("value read")
            ), patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("device copy")
            ):
                self.assertIs(view.selector.kind, kind)
                self.assertIs(view.validate(), view)
                self.assertTrue(view.selector.belongs_to(view))

    def test_query_abi_keeps_original_tuple_mask_and_cu(self):
        rows = (
            torch.arange(4),
            torch.zeros(4, dtype=torch.int64),
            torch.tensor([0, 4], dtype=torch.int32),
            torch.ones(4, dtype=torch.bool),
        )
        view = RowView(rows, RowLayout.LOCAL_QUERY, RowScope(object()), True)
        self.assertIs(view.as_rows(), rows)
        self.assertIs(view.cu_seqlens, rows[2])
        self.assertIs(view.valid_mask, rows[3])
        self.assertIs(view.validate(), view)
        with self.assertRaises(ValueError):
            view.as_full_rows()
        with self.assertRaises(ValueError):
            full_view(slice(None)).as_rows()

    def test_selector_rejects_changed_metadata_sources_and_new_forward(self):
        source = torch.tensor([0, 4], dtype=torch.int32)
        first = full_view(slice(None), sources=(source,))
        selector = first.selector
        source.add_(1)
        self.assertFalse(selector.belongs_to(first))
        fresh = RowView(
            first.legacy, first.layout, RowScope(object()), False, (source,)
        )
        self.assertFalse(first.selector.belongs_to(fresh))
        self.assertIs(fresh.as_full_rows(), first.legacy)

    def test_layout_context_plan_and_proof_are_distinct_domains(self):
        forward, context, plan, proof = object(), object(), object(), object()
        scope = RowScope(forward, context, plan, proof, (4,), (17,), (4,))
        first = full_view(slice(None), scope=scope)
        same = RowView(first.legacy, first.layout, scope, False)
        self.assertTrue(first.selector.belongs_to(same))
        for other_scope, layout in (
            (RowScope(forward, object(), plan, proof, (4,), (17,), (4,)), first.layout),
            (
                RowScope(forward, context, object(), proof, (4,), (17,), (4,)),
                first.layout,
            ),
            (
                RowScope(forward, context, plan, object(), (4,), (17,), (4,)),
                first.layout,
            ),
            (
                RowScope(forward, context, plan, proof, (4,), (17,), (2, 2)),
                first.layout,
            ),
            (scope, RowLayout.SELECTED),
        ):
            other = RowView(first.legacy, layout, other_scope, False)
            self.assertFalse(first.selector.belongs_to(other))

    def test_inference_storage_reuse_requires_fresh_forward_scope(self):
        with torch.inference_mode():
            source = torch.arange(4)
            first = full_view(slice(None), positions=source, sources=(source,))
            selector = first.selector
            self.assertIsNone(tensor_metadata_key(source)[-1])
            source.add_(10)
            # No version exists; read-only within one forward is still required.
            self.assertTrue(selector.belongs_to(first))
            next_forward = RowView(
                first.legacy, first.layout, RowScope(object()), False, (source,)
            )
            self.assertFalse(selector.belongs_to(next_forward))

    def test_native_tensor_identity_view_geometry_and_version_keys(self):
        value = torch.arange(8)
        self.assertIsNone(tensor_metadata_key(None))
        self.assertEqual(tensor_metadata_key(value)[0], value._cdata)
        alias = value[1:5]
        self.assertTrue(torch._C._is_alias_of(alias, value))
        self.assertNotEqual(tensor_metadata_key(alias), tensor_metadata_key(value))
        before = tensor_metadata_key(value)
        value.add_(1)
        self.assertNotEqual(before, tensor_metadata_key(value))
        with self.assertRaises(AttributeError):
            tensor_metadata_key(object())
        self.assertFalse(is_plain_tensor(None))
        self.assertFalse(is_plain_tensor(torch.tensor([1j]).conj()))

    def test_invalid_structure_is_declined_without_normalizing_selector(self):
        mask = torch.ones(2, dtype=torch.bool)
        view = full_view(mask)
        self.assertIs(view.as_full_rows(), view.legacy)
        self.assertIs(view.selector.value, mask)
        with self.assertRaises(ValueError):
            view.validate()
        invalid = RowView(
            (torch.arange(4), torch.arange(4), slice(None)),
            RowLayout.LOCAL_QUERY,
            RowScope(object()),
            True,
        )
        with self.assertRaises(ValueError):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
