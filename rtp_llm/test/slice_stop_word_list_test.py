import os
from unittest import TestCase, main, mock

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.models.base_model import GenerateOutput, GenerateOutputs
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader

os.environ["KV_CACHE_MEM_MB"] = "100"
os.environ["RESERVER_RUNTIME_MEM_MB"] = "1"
os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(64 * 1024**2)


class SliceStopWordListTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = os.path.join(
            os.getcwd(),
            "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source",
        )
        engine: BaseEngine = FakeModelLoader(
            "llama", ckpt_path, ckpt_path, max_seq_len=1024
        ).init_engine()
        self.backend_rpc_server_visitor = BackendRPCServerVisitor(engine.config, False)
        self.pipeline = Pipeline(
            engine.config, engine.model.tokenizer, self.backend_rpc_server_visitor
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
