import importlib.util,random,unittest
from pathlib import Path
from source_range_planner import tensor_intervals,page_aligned_union,bounded_batch,subtract_intervals

FILE='model-00001-of-00213.safetensors'

def row(name,canonical,begin,end,local_bytes):
    return dict(name=name,canonical_name=canonical,checkpoint_file=FILE,
                source_begin=begin,source_end=end,local_bytes=local_bytes)

class SourceRangesTest(unittest.TestCase):
    def test_ordinary_axis0(self):
        h={'ordinary':dict(dtype='BF16',shape=[10,4],data_offsets=[0,80])}
        self.assertEqual(tensor_intervals(row('ordinary','ordinary',2,5,24),h,100,180),[(FILE,116,140)])
    def test_expert_crossing_two_original_experts(self):
        h={'layer.experts.gate_up_proj':dict(dtype='BF16',shape=[32,4,3],data_offsets=[0,768])}
        r=row('layer.experts.chunks.1.gate_up_proj','layer.experts.gate_up_proj',3,7,24)
        self.assertEqual(tensor_intervals(r,h,100,868),[(FILE,502,508),(FILE,508,526)])
    def test_vocab_tile_offset(self):
        h={'lm_head.weight':dict(dtype='BF16',shape=[32768,2],data_offsets=[0,131072])}
        r=row('lm_head.tiles.1.weight','lm_head.weight',2,4,8)
        self.assertEqual(tensor_intervals(r,h,32,131104),[(FILE,65576,65584)])
    def test_inward_alignment_and_union_never_fill_gaps(self):
        spans=[(FILE,1,19),(FILE,9,29),(FILE,40,48),(FILE,48,56)]
        self.assertEqual(page_aligned_union(spans,4),[(FILE,4,28),(FILE,40,56)])
        self.assertEqual(bounded_batch(page_aligned_union(spans,4),4,28),[(FILE,4,28),(FILE,40,44)])
    def test_receipt_mismatch_or_unknown_mapping_rejected(self):
        h={'ordinary':dict(dtype='BF16',shape=[10,4],data_offsets=[0,80])}
        with self.assertRaises(ValueError):tensor_intervals(row('ordinary','ordinary',2,5,22),h,100,180)
        with self.assertRaises(ValueError):tensor_intervals(row('invented','ordinary',2,5,24),h,100,180)
        with self.assertRaises(ValueError):tensor_intervals(row('ordinary','ordinary',2,11,72),h,100,180)
    def test_zero_or_unaligned_batch_rejected(self):
        with self.assertRaises(ValueError):bounded_batch([],4096,1)
        with self.assertRaises(ValueError):bounded_batch([(FILE,1,4096)],4096,4096)
    def test_wider_policy_range_excludes_every_exact_page(self):
        self.assertEqual(subtract_intervals([(FILE,0,100)],[(FILE,10,20),(FILE,30,60),(FILE,55,80)]),
                         [(FILE,0,10),(FILE,20,30),(FILE,80,100)])
    def test_expert_ranges_match_actual_loader_contract(self):
        repo=Path(__file__).with_name('qwen38_training_contract.py')
        spec=importlib.util.spec_from_file_location('actual_contract',repo);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=random.Random(811);shape=[512,192,128];header={'e.down_proj':dict(dtype='BF16',shape=shape,data_offsets=[0,512*192*128*2])}
        for _ in range(100):
            chunk=rng.randrange(32);begin=rng.randrange(16*shape[1]);end=rng.randrange(begin,16*shape[1]+1)
            r=row('e.chunks.%d.down_proj'%chunk,'e.down_proj',begin,end,(end-begin)*shape[2]*2)
            got=tensor_intervals(r,header,64,64+header['e.down_proj']['data_offsets'][1])
            fragments=module.flat_source_fragments(shape,chunk*16,(chunk+1)*16,begin,end)
            expected=[(FILE,64+(x['expert']*shape[1]+x['feature_begin'])*shape[2]*2,
                64+(x['expert']*shape[1]+x['feature_end'])*shape[2]*2) for x in fragments]
            self.assertEqual(got,expected)

if __name__=='__main__':unittest.main()
