import unittest
from kimi_k3_logits_compare import compare, metrics

class LogitsCompareTest(unittest.TestCase):
    def test_additive_logit_shift_preserves_distribution(self):
        r=metrics([1.,2.,3.],[11.,12.,13.])
        self.assertTrue(r['top1_equal'])
        self.assertAlmostEqual(r['kl_baseline_candidate'],0.)
        self.assertAlmostEqual(r['rmse'],10.)

    def test_divergent_inputs_and_generated_prefixes_are_rejected(self):
        a=dict(input_ids=[[1]],output_ids=[[2,3]],logits=[[0.,1.]])
        with self.assertRaisesRegex(ValueError,'input token'):
            compare(a,dict(a,input_ids=[[2]]))
        with self.assertRaisesRegex(ValueError,'prefixes diverged'):
            compare(a,dict(a,output_ids=[[4,3]]))
        self.assertTrue(compare(a,dict(a,output_ids=[[2,4]]))[0]['top1_equal'])

    def test_nonfinite_and_missing_logits_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            metrics([0.],[float('nan')])
        a=dict(input_ids=[[1]],output_ids=[[2]],logits=[])
        with self.assertRaisesRegex(ValueError,'missing'):
            compare(a,a)

if __name__=='__main__': unittest.main()
