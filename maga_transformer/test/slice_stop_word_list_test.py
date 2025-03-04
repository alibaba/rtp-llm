import os
import torch
from unittest import TestCase, main, mock
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.models.base_model import GenerateOutput, GenerateOutputs
from maga_transformer.test.model_test.test_util.fake_model_loader import  FakeModelLoader

os.environ['KV_CACHE_MEM_MB'] = '100'
os.environ['RESERVER_RUNTIME_MEM_MB'] = '1'
os.environ['DEVICE_RESERVE_MEMORY_BYTES'] = str(64 * 1024 ** 2)

class SliceStopWordListTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        model = FakeModelLoader("llama", 
                                ckpt_path,
                                ckpt_path,
                                WEIGHT_TYPE.FP16,
                                1024).load_model()        
        self.pipeline = Pipeline(model, model.config, model.tokenizer)

    async def mock_generate(self):
        yield GenerateOutputs(generate_outputs=[GenerateOutput(output_ids=torch.tensor([[[29892]]]), finished=False)])
        yield GenerateOutputs(generate_outputs=[GenerateOutput(output_ids=torch.tensor([[[825]]]), finished=False)])
        yield GenerateOutputs(generate_outputs=[GenerateOutput(output_ids=torch.tensor([[[29915]]]), finished=False)])
        yield GenerateOutputs(generate_outputs=[GenerateOutput(output_ids=torch.tensor([[[29879]]]), finished=False)])
        yield GenerateOutputs(generate_outputs=[GenerateOutput(output_ids=torch.tensor([[[596]]]), finished=False)])

    @mock.patch("maga_transformer.async_decoder_engine.backend_rpc_server_visitor.BackendRPCServerVisitor.enqueue")
    def test_slice(self, mock_enqueue):
        mock_enqueue.return_value = self.mock_generate()
        outs = self.pipeline("hello", stop_words_list=[[29879, 596]])
        outs = [out for out in outs]
        out_str = [response.generate_texts[0] for response in outs]
        self.assertEqual(out_str, [',', ', what', ", what'", ", what'", ", what'"])
        
if __name__ == '__main__':
    main()
