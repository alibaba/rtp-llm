"""Keep actual untrained-draft accounting distinct from forced fixtures."""
import unittest
from rtp_llm.test.smoke.kimi_k3.mtp_contract import acceptance_contract
from rtp_llm.utils.speculative_acceptance import COUNTERS

class MtpContractTest(unittest.TestCase):
    def aux(self):
        counters={name:0 for name in COUNTERS}
        counters.update(version=1,proposal_rounds=2,proposal_candidates=6,verify_rounds=2,
                        verified_candidates=6,generated_tokens=2,accept_length_0=2)
        return {"speculative_metrics":counters,"speculative_draft_rounds":2}

    def test_zero_acceptance_with_actual_candidates_is_valid(self):
        result=acceptance_contract(self.aux(),untrained=True,required=True)
        self.assertEqual(result["acceptance_rate"],0)
        self.assertEqual(result["mean_accept_length"],0)

    def test_no_verify_candidates_reports_na(self):
        self.assertIsNone(acceptance_contract({},untrained=True)["acceptance_rate"])

    def test_required_execution_cannot_use_absent_or_padding_only_counters(self):
        for aux in ({},self.aux()):
            if aux:aux["speculative_metrics"]["verified_candidates"]=0
            with self.subTest(aux=aux),self.assertRaises(ValueError):
                acceptance_contract(aux,untrained=True,required=True)

    def test_forced_acceptance_never_enters_real_smoke(self):
        aux=self.aux();aux["speculative_metrics"]["excluded_forced_rounds"]=1
        with self.assertRaisesRegex(ValueError,"forced"):
            acceptance_contract(aux,untrained=True,required=True)

if __name__=="__main__":unittest.main()
