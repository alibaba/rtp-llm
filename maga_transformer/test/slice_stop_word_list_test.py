import os
import torch
from unittest import TestCase, main, mock
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.test.model_test.test_util.fake_model_loader import  FakeModelLoader


class SliceStopWordListTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        model = FakeModelLoader("llama", 
                                        ckpt_path,
                                        ckpt_path,
                                        0,
                                        False,
                                        1024).load_model()        
        self.pipeline = Pipeline(model, model.tokenizer)

    async def mock_generate(self):
        yield GenerateOutput(torch.empty([1, 0]), torch.tensor([[[1, 22172, 29892]]]), torch.tensor([False], dtype=torch.bool), None)
        yield GenerateOutput(torch.empty([1, 0]), torch.tensor([[[1, 22172, 29892, 825]]]), torch.tensor([False], dtype=torch.bool), None)
        yield GenerateOutput(torch.empty([1, 0]), torch.tensor([[[1, 22172, 29892, 825, 29915]]]), torch.tensor([False], dtype=torch.bool), None)
        yield GenerateOutput(torch.empty([1, 0]), torch.tensor([[[1, 22172, 29892, 825, 29915, 29879]]]), torch.tensor([False], dtype=torch.bool), None)
        yield GenerateOutput(torch.empty([1, 0]), torch.tensor([[[1, 22172, 29892, 825, 29915, 29879, 596]]]), torch.tensor([True], dtype=torch.bool), None)

    @mock.patch("maga_transformer.models.base_model.BaseModel.generate_stream")
    def test_slice(self, mock_generate_stream):
        mock_generate_stream.return_value = self.mock_generate()
        
        outs = self.pipeline(["hello"], [[]], stop_words_list=[[29879, 596]])
        outs = [out for out in outs]
        out_str = [response.batch_response[0] for response in outs]
        self.assertEqual(out_str, ['', 'what', "what'", "what'", "what'"])
        
if __name__ == '__main__':
    main()
