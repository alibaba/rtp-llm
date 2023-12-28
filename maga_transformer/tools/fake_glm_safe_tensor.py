import torch
from maga_transformer.tools.fake_model_base import *
from safetensors.torch import save_file

if __name__ == '__main__':
    raw_tensor = torch.load("./maga_transformer/test/model_test/fake_test/testdata/chatglm/fake/fake_chatglm_2_2_2_64_512_130528_copy.pt")
    save_file(raw_tensor, "maga_transformer/test/model_test/fake_test/testdata/chatglm/safetensors/fake_chatglm_2_2_2_64_512_130528.safetensors")