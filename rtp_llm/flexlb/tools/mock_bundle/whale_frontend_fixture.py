"""CPU synthetic Qwen2-VL fixture for an isolated Whale frontend protocol test.

Uses a pinned Python source tree with the container's matching native runtime.
This entry point generates tokenizer/model metadata, never model weights.
"""
import os, pathlib, urllib.request, hashlib, tarfile, subprocess, sys, json
if os.environ.get("RTP_LLM_MOCK_FRONTEND") != "1":
    raise ValueError("test fixture requires RTP_LLM_MOCK_FRONTEND=1")
sha = "76e538acf61653b0a80a6162110a31eed0439d66"
expected = "0ae9a6c936deaf1792028cbaa52a21bab7c9e7992fa064d2fa4f94d2c85cad71"
root = pathlib.Path(os.environ.get("HIPPO_PROC_WORKDIR", "/tmp")) / ("codex-vit-" + sha[:10])
root.mkdir(parents=True, exist_ok=True)
archive = root / "source.tar.gz"
if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    with urllib.request.urlopen("https://github.com/alibaba/rtp-llm/archive/" + sha + ".tar.gz", timeout=90) as response:
        archive.write_bytes(response.read())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, "frontend source archive mismatch"
with tarfile.open(archive) as tar:
    for member in tar:
        parts = pathlib.PurePosixPath(member.name).parts
        if len(parts) < 3 or parts[1] != "rtp_llm" or ".." in parts or not member.isfile():
            continue
        if not member.name.endswith((".py", ".json", ".proto", ".jinja", ".j2")):
            continue
        target = root.joinpath(*parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tar.extractfile(member).read())
libs = root / "rtp_llm/libs"
libs.mkdir(exist_ok=True)
link = libs / "libth_transformer_config.so"
if not link.exists():
    link.symlink_to("/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs/libth_transformer_config.so")
subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "-I", ".", "--python_out=.", "--grpc_python_out=.",
                "rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto", "rtp_llm/cpp/model_rpc/proto/flexlb_schedule_service.proto"], cwd=root, check=True)
for key in ("DSV41_CED", "DSV41_SWA_BOUNDED_REPLAY", "SP_MODEL_TYPE", "PREFILL_CP_SIZE", "PREFILL_CP_KV_CACHE_SHARDED", "SP_CHECKPOINT_PATH", "SP_TOKENIZER_PATH"):
    os.environ.pop(key, None)
os.environ.update(FLEXLB_RUN_DIR=str(root), PYTHONPATH=str(root), TP_SIZE="1", LD_LIBRARY_PATH="/opt/conda310/lib:/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs")
os.chdir(root)
sys.path.insert(0, str(root))
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
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.config.log_config import setup_logging
from rtp_llm.utils.concurrency_controller import ConcurrencyController, set_global_controller
setup_logging()
cfg = setup_args([])
set_global_controller(ConcurrencyController(128))
FrontendApp(cfg, separated_frontend=True).start()
