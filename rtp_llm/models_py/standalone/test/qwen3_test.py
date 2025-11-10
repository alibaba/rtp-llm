import logging
import sys
from unittest import TestCase, main

from rtp_llm.models_py.standalone.rtp_simple_model import RtpSimplePyModel

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Qwen3SimplePyModelTest(TestCase):

    def test_qwen3_simple_py_model(self):
        qwen3_simple_py_model = self._create_qwen3_simple_py_model()
        logging.info(f"qwen3_simple_py_model created")

        msg1 = [{"role": "user", "content": "你好，请用较长篇幅介绍自己"}]
        output_text1 = self._run_message(qwen3_simple_py_model, msg1, max_new_tokens=20)
        logging.info(f"output_text1: {output_text1}")
        assert (
            output_text1
            == "你好！我是你的虚拟助手，一个专注于帮助你解决问题和提供支持的AI助手。我"
        )

        msg2 = [{"role": "user", "content": "3.9和3.11哪个大"}]
        output_text2 = self._run_message(qwen3_simple_py_model, msg2, max_new_tokens=10)
        logging.info(f"output_text2: {output_text2}")
        assert output_text2 == "3.9 和 3.11 中"

    def _run_message(
        self,
        model: RtpSimplePyModel,
        message: list[dict[str, str]],
        max_new_tokens: int = 1000,
    ) -> str:
        input_text = model.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        output_text = self._run_raw_text(model, input_text, max_new_tokens)
        return output_text

    def _run_raw_text(
        self, model: RtpSimplePyModel, text: str, max_new_tokens: int
    ) -> str:
        input_ids = model.tokenizer.encode(text)
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        output_text = model.tokenizer.decode(output_ids)
        return output_text

    def _create_qwen3_simple_py_model(self) -> RtpSimplePyModel:
        qwen3_simple_py_model = RtpSimplePyModel(
            model_type="qwen_3",
            model_path_or_name="Qwen/Qwen3-0.6B",
            act_type="FP16",
        )
        return qwen3_simple_py_model


if __name__ == "__main__":
    main()
