import os
import sys

os.environ["LOAD_PYTHON_MODEL"] = "1"

sys.path.append("/home/wangyin.yx/workspace/FasterTransformer")

import rtp_llm.models
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.distribute.worker_info import g_worker_info, update_master_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.pipeline import Pipeline
from rtp_llm.test.utils.port_util import PortsContext

# import pdb
# pdb.set_trace()

# from librtp_compute_ops.rtp_llm_ops import ParamsBase


start_port = 23345
StaticConfig.server_config.start_port = start_port
update_master_info("127.0.0.1", start_port)
g_worker_info.reload()
StaticConfig.model_config.model_type = "qwen_3"
StaticConfig.model_config.checkpoint_path = "Qwen/Qwen3-0.6B"
os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(3 * 1024 * 1024 * 1024)
model_config = ModelFactory.create_normal_model_config()
model = ModelFactory.creat_standalone_py_model_from_huggingface(
    model_config.ckpt_path, model_config=model_config
)


from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs

attention_inputs = PyAttentionInputs()
inputs = PyModelInputs(
    input_ids=torch.randint(0, 10, (1, 10)), attention_inputs=attention_inputs
)

qwen3_model = Qwen3Model(model.config, model.weight)
qwen3_model.forward(model.input)
