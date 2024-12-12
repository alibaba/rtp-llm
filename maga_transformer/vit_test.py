import os
import sys
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))
 
from maga_transformer.model_factory import ModelFactory
from maga_transformer.utils.multimodal_util import (MMUrlType,
                                                    MMPreprocessConfig)
 
model = ModelFactory.create_from_env()
 
url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
res = model.model.mm_part.mm_embedding(url, MMUrlType.IMAGE, configs=MMPreprocessConfig())
print(res)
 
# MODEL_TYPE=fake_qwen2_vl CHECKPOINT_PATH=xxx TOKENIZER_PATH=xxx python maga_transformer/qwen2_vl_vit_benchmark.py