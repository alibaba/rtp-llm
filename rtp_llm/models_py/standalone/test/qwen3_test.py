import sys
from pathlib import Path

# only need to import the rtp_simple_model
from rtp_llm.models_py.standalone.rtp_simple_model import RtpSimplePyModel

# if you want to run the script directly and haven't installed the rtp-llm wheel, you need to add the dev directory to the python path
# rtp_opensouce_path = Path(__file__).resolve().parent.parent.parent.parent.parent
# print(f"rtp_opensouce_path: {rtp_opensouce_path}")
# sys.path.append(str(rtp_opensouce_path))


# create simplemodel, support load model from huggingface
my_model = RtpSimplePyModel(
    model_type="qwen_3",
    model_path_or_name="Qwen/Qwen3-0.6B",
    act_type="FP16",
)

# generate text and token_ids
messages = [{"role": "user", "content": "你好，请用较长篇幅介绍自己"}]
input_text = my_model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
input_ids = my_model.tokenizer.encode(input_text)

# generate token_ids and text
output_ids = my_model.generate(input_ids, max_new_tokens=1000)
output_text = my_model.tokenizer.decode(output_ids)
print(f"\nanswer of Q1:\n\n {output_text}")


# generate text and token_ids
messages = [{"role": "user", "content": "3.9和3.11哪个大"}]
input_text = my_model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
input_ids = my_model.tokenizer.encode(input_text)

# generate token_ids and text
output_ids = my_model.generate(input_ids, max_new_tokens=1000)
output_text = my_model.tokenizer.decode(output_ids)
print(f"\nanswer of Q2:\n\n {output_text}")
