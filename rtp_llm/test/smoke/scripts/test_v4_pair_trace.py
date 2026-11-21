import copy
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv4.prefill import v4_fixed_replay as gpu
from rtp_llm.test.smoke import v4_pair_trace as v
sys.path.insert(0, str(Path(__file__).parent))
from compare_v4_pair_trace import compare


class PairTest(unittest.TestCase):
    def test_model_to_leaf_and_compare(self):
        import torch
        ids=gpu.IDS
        hashes=[hashlib.sha256(struct.pack("<"+"i"*n,*ids[:n])).hexdigest() for n in range(18,56)]
        self.assertEqual(hashes[-1],v.SHA)
        text="v4_target_witness_v1 case=17 phase=verify output_token_1based=38 row=2 col=2 width=4 batch=0 accept_len=3 a_id=3315 b_id=2642 raw_a=4 raw_b=3 masked_a=-inf masked_b=3 raw_argmax=3315 masked_argmax=2642 sampled_target=2642 emitted=2642 draft_kind=proposal draft_id=1 prefix_sha256_le_i32="+v.SHA+" committed_sha_chain="+",".join(hashes)
        text += " verify_exec=graph_runner aux_dtype=bf16 head_dtype=bf16 "+" ".join(k+"="+v.SHA for k in ("live_aux40_sha","live_aux41_sha","live_aux42_sha","live_lm_input_sha"))
        text += " live_raw_hc_sha="+",".join([v.SHA]*44)
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{"TEST_TMPDIR":t,"TEST_UNDECLARED_OUTPUTS_DIR":t,"TEST_TARGET":v.TARGET}):
            root=Path(t); marker=root/"v4-live38-replay.on"
            (root/"v4-target-witness-17-verify.txt").write_text(text+"\n")
            net=SimpleNamespace(layers=[None]*43,capture_aux_hidden_layer_ids=(40,41,42))
            def replay(req,timeout):
                self.assertEqual(json.loads(req.data)["prompt"],v.PROMPT)
                for rank in (0,1):
                    pos=torch.arange(rank,55,2).flip(0)
                    loc=torch.tensor(ids).index_select(0,pos)
                    cp=SimpleNamespace(cp_size=2,cp_rank=rank,input_lengths_global=torch.tensor([55]),seq_len_full=55,
                        prefix_length=0,local_is_real=torch.ones_like(pos).bool(),global_positions=pos)
                    self.assertIsNone(gpu.open_trace(net,cp,loc+1,None))
                    obj=gpu.open_trace(net,cp,loc,None)
                    for stage in gpu.STAGES:
                        h=torch.zeros((len(pos),4096) if stage=="final_norm" else (len(pos),4,4096),dtype=torch.bfloat16)
                        obj.observe(stage,h)
                from io import BytesIO
                resp=BytesIO(b"{}");resp.status=200;return resp
            log=root/"model.log";log.write_text("NewLoader load method: requested=auto, selected=fastsafetensors\n")
            manager=SimpleNamespace(log_file_path=log,port=7231)
            with patch("urllib.request.urlopen",replay):
                self.assertTrue(v.run_pair({"prefill":manager,"decode":manager}))
            self.assertFalse(marker.exists())
            new=json.loads((root/"v4_pair_trace.json").read_text());old=copy.deepcopy(new)
            old.update(loader="oldfst",role_loaders={"prefill":"oldfst","decode":"oldfst"})
            self.assertIsNone(compare(old,new)["fixed_prefill_replay"]["first_different_layer"])
            new["layers"][7]["sha256"]="5"*64
            self.assertEqual(compare(old,new)["fixed_prefill_replay"]["first_different_layer"],"layer.6")
            new["sample"]["live_raw_layers"][12]["sha256"]="1"*64
            self.assertEqual(compare(old,new)["first_different_observed_live_stage"],"layer.11.raw_hc")
            for j in range(14,38): new["committed_sha_chain"][j]="a"*64
            new["committed_prefix_sha256"]=new["committed_sha_chain"][-1]
            self.assertFalse(compare(old,new)["comparable_live_sample"])
            self.assertIsNone(compare(old,new)["first_different_observed_live_stage"])
            self.assertEqual(compare(old,new)["first_different_committed_output_1based"],14)
            new["layers"].pop()
            with self.assertRaises(ValueError):compare(old,new)

if __name__=="__main__":
    unittest.main()
