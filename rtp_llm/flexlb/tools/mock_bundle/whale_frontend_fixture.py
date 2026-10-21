"""CPU synthetic Qwen2-VL fixture for an isolated Whale frontend protocol test.

Uses a pinned Python source tree with the container's matching native runtime.
This entry point generates tokenizer/model metadata, never model weights.
"""
import os
import pathlib
import sys
if os.environ.get("RTP_LLM_MOCK_FRONTEND") != "1":
    raise ValueError("test fixture requires RTP_LLM_MOCK_FRONTEND=1")
root = pathlib.Path(os.environ.get("HIPPO_PROC_WORKDIR", "/tmp")) / "mock-vit-main-frontend"
root.mkdir(parents=True, exist_ok=True)
source = pathlib.Path(os.environ["MOCK_FRONTEND_SOURCE_ROOT"])
if not (source / "rtp_llm/server/backend_rpc_server_visitor.py").is_file():
    raise ValueError("MOCK_FRONTEND_SOURCE_ROOT must contain the tested packaged frontend")
for key in ("DSV41_CED", "DSV41_SWA_BOUNDED_REPLAY", "SP_MODEL_TYPE", "PREFILL_CP_SIZE", "PREFILL_CP_KV_CACHE_SHARDED", "SP_CHECKPOINT_PATH", "SP_TOKENIZER_PATH"):
    os.environ.pop(key, None)
os.environ.update(FLEXLB_RUN_DIR=str(root), PYTHONPATH=str(source), TP_SIZE="1", LD_LIBRARY_PATH="/opt/conda310/lib:/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs")
os.chdir(root)
sys.path.insert(0, str(source))
import json
import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

root = Path(os.environ['FLEXLB_RUN_DIR'])
model = root / 'frontend-model'
model.mkdir(exist_ok=True)
special = ['[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|endoftext|>']
vocab = {(special[i] if i < len(special) else f'tok{i}'): i for i in range(4096)}
backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='<|endoftext|>', additional_special_tokens=special[2:7])
tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}{% if message['content'] is string %}{{message['content']}}{% else %}{% for part in message['content'] %}{% if part['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% elif part['type'] == 'text' %}{{part['text']}}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}"
tokenizer.save_pretrained(model)
config = dict(model_type='qwen2_vl', architectures=['Qwen2VLForConditionalGeneration'], vocab_size=4096, num_attention_heads=1, num_key_value_heads=1, hidden_size=128, num_hidden_layers=1, intermediate_size=256, rms_norm_eps=1e-6, rope_theta=1000000, bos_token_id=2, eos_token_id=7, vision_start_token_id=4, vision_end_token_id=5, image_token_id=6, vision_config={'hidden_size':128, 'embed_dim':128, 'patch_size':14, 'spatial_merge_size':2, 'temporal_patch_size':2, 'depth':1, 'num_heads':1})
(model / 'config.json').write_text(json.dumps(config))
os.environ.update(MODEL_TYPE='qwen2_vl', CHECKPOINT_PATH=str(model), TOKENIZER_PATH=str(model), ROLE_TYPE='FRONTEND', START_PORT=os.environ.get('VIT_TEST_FRONTEND_PORT', '20550'), FRONTEND_SERVER_COUNT='1', VIT_SEPARATION='2', MAX_SEQ_LEN='16384', SEQ_SIZE_PER_BLOCK='64', REUSE_CACHE='1', LOG_PATH=str(root/'frontend-logs'), HF_HOME=str(root/'home/hf'), TOKENIZERS_PARALLELISM='false')
from rtp_llm.start_server import main as start_frontend
start_frontend()
