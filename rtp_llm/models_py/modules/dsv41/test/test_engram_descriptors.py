# SPDX-License-Identifier: Apache-2.0
"""Descriptor contract: geometry, identity and capability uniqueness.

These are the negative cases that guard ``validate_documents``/``build_manifest``;
they need neither CUDA nor the checkpoint tools.
"""
import base64
from copy import deepcopy
import json
import unittest
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_descriptors import build_manifest, validate_documents
from rtp_llm.models_py.modules.dsv41.engram_manifest import parse_manifest
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards

GEN = str(UUID(int=1))
HASH = 'sha256:' + 'b' * 64


def descriptors(p=3, dim=64):
    rows = p * 4096 - 7
    plan = plan_shards(rows, dim, p, granularity=4096)
    result = []
    for shard in range(p):
        begin, end = plan.row_range(shard)
        tensors = {}
        for layer in (1, 14):
            for role, columns, size in (('weight', dim, plan.weight_bytes),
                                        ('scale', dim // 32, plan.scale_bytes)):
                aid = str(UUID(int=100 + shard * 100 + layer * 2 + (role == 'scale')))
                tensors[f'layers.{layer}.engram.embed.{role}'] = dict(
                    shape=[rows, columns], row_begin=begin, row_end=end,
                    logical_bytes=(end - begin) * columns, allocation_bytes=size,
                    allocation_granularity=4096, allocation_id=aid,
                    sha256=HASH, handle=base64.b64encode(aid.encode()).decode())
        result.append(dict(schema_version=2, shard=shard, shards=p, generation=GEN,
                           model_revision='revision', hash_abi=HASH, host_id=f'host-{shard}',
                           owner_id=f'host-{shard}', owner_incarnation=str(UUID(int=10 + shard)),
                           tensors=tensors))
    return result


def build(docs):
    return build_manifest(docs, len(docs), GEN, 'revision', HASH)


class DescriptorGeometryTest(unittest.TestCase):
    def test_non_256_columns_and_multiple_p(self):
        for p in (2, 3, 5, 18):
            for dim in (32, 64, 256):
                docs = descriptors(p, dim)
                document = build(list(reversed(docs)))
                manifest = parse_manifest(json.dumps(document).encode())
                for tensor in manifest.tensors:
                    self.assertEqual(len(tensor.shards), p)
                    self.assertEqual(tensor.shape[1],
                                     dim if tensor.name.endswith('weight') else dim // 32)
                    self.assertEqual([s.owner_id for s in tensor.shards],
                                     [f'host-{i}' for i in range(p)])

    def test_out_of_order_documents_still_validate(self):
        docs = descriptors()
        self.assertEqual(len(validate_documents([docs[2], docs[0], docs[1]], 3, GEN)), 3)


class DescriptorRejectionTest(unittest.TestCase):
    def test_invalid_or_mixed_descriptors_fail_before_cuda(self):
        mutations = {
            'revision': lambda d: d[1].update(model_revision='other'),
            'hash': lambda d: d[1].update(hash_abi='sha256:' + 'c' * 64),
            'generation': lambda d: d[1].update(generation=str(UUID(int=3))),
            'count': lambda d: d[1].update(shards=2),
            'index': lambda d: d[1].update(shard=0),
            'boolean': lambda d: d[1].update(shard=True),
            'same_host': lambda d: d[1].update(host_id='host-0', owner_id='host-0'),
            'host_owner': lambda d: d[1].update(owner_id='other'),
            'schema': lambda d: d[1].update(schema_version=1),
            'missing_tensor': lambda d: d[1]['tensors'].pop('layers.1.engram.embed.scale'),
            'allocation': lambda d: d[1]['tensors']['layers.1.engram.embed.weight'].update(
                allocation_id=d[0]['tensors']['layers.1.engram.embed.weight']['allocation_id']),
            'capability': lambda d: d[1]['tensors']['layers.1.engram.embed.weight'].update(
                handle=d[0]['tensors']['layers.1.engram.embed.weight']['handle']),
            'shape': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(shape=[12281, 8]),
            'range': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(row_begin=0),
            'granularity': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(
                allocation_granularity=8192),
            'invalid_base64': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(
                handle='%%%'),
        }
        for name, mutate in mutations.items():
            docs = descriptors()
            mutate(docs)
            with self.subTest(case=name), self.assertRaises(ValueError):
                build(docs)

    def test_swapped_indices_cannot_silently_reorder_handle_rows(self):
        docs = descriptors()
        docs[0]['shard'], docs[1]['shard'] = 1, 0
        with self.assertRaisesRegex(ValueError, 'geometry'):
            build(docs)

    def test_duplicate_documents_are_rejected(self):
        docs = descriptors()
        docs[2] = deepcopy(docs[0])
        with self.assertRaises(ValueError):
            build(docs)

    def test_missing_or_extra_documents_are_rejected(self):
        for docs in (descriptors()[:2], descriptors() + descriptors()[:1]):
            with self.assertRaises(ValueError):
                build(docs)


if __name__ == '__main__':
    unittest.main()