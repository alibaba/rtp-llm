"""Run on one idle GPU to check that the observer preserves real graph results."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.decode import v4_live_layers as obs


class LiveGraphTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_padded_graph_replay_two_calls(self):
        net = SimpleNamespace(layers=[None] * 43, capture_aux_hidden_layer_ids=(40, 41, 42))
        with patch.dict(os.environ, {"TEST_TARGET":"//other:smoke"}):
            self.assertIsNone(obs.allocate(net,"cuda"))
        with patch.dict(os.environ, {"TEST_TARGET":obs.TARGET,"TEST_UNDECLARED_OUTPUTS_DIR":"/unused"}):
            state = obs.allocate(net, "cuda")
        self.assertEqual(state[0].numel()*state[0].element_size(),22*1024**2)
        addr=state[0].data_ptr()
        ids=torch.arange(20,device="cuda",dtype=torch.int32)
        initial=torch.randn(5,4,4,4096,dtype=torch.bfloat16,device="cuda")
        def forward(trace):
            h=initial
            if trace:
                obs.begin(state,ids,4)
                obs.put(state,0,h)
            for i in range(43):
                h=(h+0.125).to(torch.bfloat16)
                if trace: obs.put(state,i+1,h)
            return h
        forward(True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): result=forward(True)
        for value in (81,-29):
            initial.fill_(value)
            ids[9]=value
            graph.replay()
            self.assertTrue(torch.equal(result,forward(False)))
            self.assertIsNone(obs.selected(state,9,4,value+1))
            self.assertIsNone(obs.selected(state,16,4,16))
            self.assertIsNone(obs.selected(state,9,2,value))
            self.assertEqual(state[0].data_ptr(),addr)
            row=obs.selected(state,9,4,value)
            h=initial.flatten(0,1)[9]
            for i in range(44):
                self.assertTrue(torch.equal(row[i],h))
                h=h+0.125


if __name__=="__main__":
    unittest.main()
