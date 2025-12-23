import os
from unittest import TestCase, main, mock

import torch

from rtp_llm.frontend.generation.orchestrator import GenerationOrchestrator
from rtp_llm.ops import PDSepConfig
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs


class SliceStopWordListTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = os.path.join(
            os.getcwd(),
            "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source",
        )
        pd_sep_config = PDSepConfig()
        self.pipeline = GenerationOrchestrator(
            special_tokens=None,
            pd_sep_config=pd_sep_config,
            addresses=["localhost:8080"],  # Default test address
            max_seq_len=1000,
            seq_size_per_block=1,
            tokenizer=None,
            sp_config=None,
        )

    async def mock_generate(self):
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(output_ids=torch.tensor([[[29892]]]), finished=False)
            ]
        )
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(output_ids=torch.tensor([[[825]]]), finished=False)
            ]
        )
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(output_ids=torch.tensor([[[29915]]]), finished=False)
            ]
        )
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(output_ids=torch.tensor([[[29879]]]), finished=False)
            ]
        )
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(output_ids=torch.tensor([[[596]]]), finished=False)
            ]
        )

    async def mock_call(self):
        return self.mock_generate()

    @mock.patch(
        "rtp_llm.async_decoder_engine.backend_rpc_server_visitor.BackendRPCServerVisitor.enqueue"
    )
    async def test_slice(self, mock_enqueue):
        mock_enqueue.return_value = self.mock_call()
        outs = []
        async for out in self.pipeline("hello", stop_words_list=[[29879, 596]]):
            outs.append(out)
        out_str = [response.generate_texts[0] for response in outs]
        self.assertEqual(out_str, [",", ", what", ", what'", ", what'", ", what'"])


if __name__ == "__main__":
    main()
