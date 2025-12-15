import sys
from pathlib import Path

# if you haven't install the wheel, you should add your wheel path to sys.path to find the rtp_llm package
rtp_opensouce_path = Path(__file__).resolve().parent.parent
sys.path.append(str(rtp_opensouce_path))
print(sys.path)
from rtp_llm.models_py.standalone.auto_model import AutoModel

my_model = AutoModel.from_pretrained("/home/lvjiang.lj/models/qwen3-0.6B")
messages = [{"role": "user", "content": "你好，请问你是谁？"}]
input_text = my_model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
input_ids = my_model.tokenizer.encode(input_text)
output_ids = my_model.generate(input_ids, max_new_tokens=200)
output_text = my_model.tokenizer.decode(output_ids)
print(output_text)
