import copy
import os
from typing import Any, List, Optional
from unittest import TestCase, main

from transformers import AutoTokenizer

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, GenerateConfig
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.structure.request_extractor import (
    RequestExtractor,
    request_id_field_name,
)


class GenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(os.getcwd(), "rtp_llm/test")

    def _create_generate_config(self):
        return {
            "stop_words_str": ["hello", "what's your name"],
            "stop_words_list": [[8848]],
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.8,
            "max_new_tokens": 100,
        }

    def _create_generate_config_for_select_tokens_id(self):
        return {"select_tokens_id": [0, 3]}

    def _create_kwargs(self):
        return {
            "stop_words_str": ["hi"],
            "stop_words_list": [[1551]],
            "top_k": 2,
            "top_p": 0.5,
            "max_new_tokens": 20,
        }

    def test_simple(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

        generate_config = Pipeline.create_generate_config(
            generate_config={},
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
            **self._create_generate_config(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

    def test_kwargs_overwrite(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
            **self._create_kwargs(),
        )
        self.assertEqual(generate_config.stop_words_list, [[1551]])
        self.assertEqual(generate_config.stop_words_str, ["hi"])
        self.assertEqual(generate_config.top_k, 2)
        self.assertEqual(generate_config.top_p, 0.5)
        self.assertEqual(generate_config.max_new_tokens, 20)

    def test_stop_words_merge(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912]])
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_stop_words_merge_with_toeknizer(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=GenerateEnvConfig(),
        )
        self.assertEqual(
            generate_config.stop_words_list,
            [[8848], [1233, 19912], [14990], [12555, 594, 697, 829], [14398]],
        )
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_select_tokens_id(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config_for_select_tokens_id(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        self.assertEqual(generate_config.select_tokens_id, [0, 3])
        self.assertEqual(generate_config.select_tokens_str, [])

        with self.assertRaisesRegex(Exception, "should be less than vocab_size"):
            generate_config = Pipeline.create_generate_config(
                generate_config=self._create_generate_config_for_select_tokens_id(),
                vocab_size=2,
                special_tokens=special_tokens,
                tokenizer=None,
                generate_env_config=GenerateEnvConfig(),
            )

    def test_batch_shared_config_no_accumulation(self):
        # Regression: batch 请求里 request_extractor 用 [config] * N 让 N 条 query 共享
        # 同一个 GenerateConfig 对象。Pipeline.create_generate_config 对「对象入参」走
        # 复用分支(config = generate_config),于是 convert_select_tokens / add_special_tokens
        # 会在同一个对象上被调用 N 次。修复前:select_tokens_id 累积成 N 份(logits 被重复
        # N 次)、special stop words 被追加 N 遍。
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )

        shared = GenerateConfig(
            select_tokens_str=["1", "2", "3", "4"],
            stop_words_str=["hello"],
            stop_words_list=[[8848]],
        )

        first_select_len = None
        for _ in range(3):  # 3 条 query 共享同一个对象
            cfg = Pipeline.create_generate_config(
                generate_config=shared,  # 传对象 → 命中复用分支
                vocab_size=200000,
                special_tokens=special_tokens,
                tokenizer=tokenizer,
                generate_env_config=GenerateEnvConfig(),
            )
            self.assertIs(cfg, shared)  # 确认确实复用同一对象
            if first_select_len is None:
                first_select_len = len(shared.select_tokens_id)
            # 每次调用后长度都应等于第一次,不随 batch 累积
            self.assertEqual(len(shared.select_tokens_id), first_select_len)

        # 不只是「不累积」,值也不能丢:词表是仓内固定测试数据,直接钉字面量
        self.assertEqual(shared.select_tokens_id, [16, 17, 18, 19])

        # special stop words 只合入一次,且用户已传入的 stop words 原样保留一份
        self.assertEqual(shared.stop_words_str, ["hello", "gg"])
        self.assertEqual(shared.stop_words_str.count("hello"), 1)
        self.assertEqual(shared.stop_words_list.count([8848]), 1)
        self.assertEqual(shared.stop_words_list.count([1233, 19912]), 1)

    def test_select_tokens_str_id_union_dedup(self):
        # 同时提供 select_tokens_str 与 select_tokens_id 时取去重并集:显式 id 保留、
        # str 派生 token 全部并入、整体无重复;重复调用(batch 共享)保持幂等。
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        str_ids = []
        for token_str in ["1", "2"]:
            str_ids += tokenizer.encode(token_str)

        config = GenerateConfig(select_tokens_str=["1", "2"], select_tokens_id=[99999])
        for _ in range(2):  # 幂等:第二次调用不应改变结果
            Pipeline.create_generate_config(
                generate_config=config,
                vocab_size=200000,
                special_tokens=SpecialTokens(),
                tokenizer=tokenizer,
                generate_env_config=GenerateEnvConfig(),
            )

        # 显式 id 保留,str 未被丢弃,且无重复
        self.assertIn(99999, config.select_tokens_id)
        for token_id in str_ids:
            self.assertIn(token_id, config.select_tokens_id)
        self.assertEqual(
            len(config.select_tokens_id), len(set(config.select_tokens_id))
        )

    def test_prepare_chain_idempotent(self):
        # 契约守卫:create_generate_config 链上所有 prepare 方法必须幂等。
        # 共享对象重复调用后,被 mutate 的字段应与首次调用后完全一致;未来新增
        # 的非幂等 prepare 方法会在此暴露。
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102

        shared = GenerateConfig(
            select_tokens_str=["1", "2", "3", "4"],
            stop_words_str=["hello"],
            stop_words_list=[[8848]],
        )

        def snapshot(c):
            # 全量 dump 而不是列举字段:prepare 链未来 mutate 任何新字段都会被这里
            # 捕获,不需要同步维护一张白名单。
            return copy.deepcopy(c.model_dump())

        first = None
        for _ in range(3):
            Pipeline.create_generate_config(
                generate_config=shared,
                vocab_size=200000,
                special_tokens=special_tokens,
                tokenizer=tokenizer,
                generate_env_config=generate_env_config,
            )
            snap = snapshot(shared)
            if first is None:
                first = snap
            self.assertEqual(snap, first)

    def test_batch_adapter_name_shallow_copy_no_accumulation(self):
        # 累积缺陷有两条触发路径,这条覆盖第二条:_get_adapter 在带 adapter_name 时
        # 对每条 query 做 copy.copy(request_extractor.py),浅拷贝让各副本与原对象
        # 共享 select_tokens_id / stop_words_* 的 list 引用,重复 append 依然互相
        # 可见。(第一条是不带 adapter_name 时的 [config] * N,见
        # test_batch_shared_config_no_accumulation。)
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )

        request, _ = RequestExtractor(GenerateConfig()).extract_request(
            {
                "prompt_batch": ["q1", "q2", "q3"],
                "generate_config": {
                    "select_tokens_str": ["1", "2", "3", "4"],
                    "adapter_name": ["lora_a", "lora_b", "lora_c"],
                },
                request_id_field_name: 1,
            }
        )
        configs = request.generate_configs
        self.assertEqual(len(configs), 3)
        self.assertEqual(
            [c.adapter_name for c in configs], ["lora_a", "lora_b", "lora_c"]
        )
        # 各副本是不同对象。注:当前 _get_adapter 用 copy.copy,副本之间的 list 字段
        # 仍共享同一引用,这正是原地 append 会串扰的原因;这里不对别名本身做断言——
        # 将来若在 _get_adapter 层深拷贝消除别名,下面的值断言依旧成立。
        self.assertIsNot(configs[0], configs[1])

        # oracle 用词表固定的字面量,不镜像生产侧的去重算法
        expected_ids = [16, 17, 18, 19]
        self.assertEqual(
            [tokenizer.encode(s)[0] for s in ["1", "2", "3", "4"]], expected_ids
        )

        for config in configs:
            Pipeline.create_generate_config(
                generate_config=config,
                vocab_size=200000,
                special_tokens=special_tokens,
                tokenizer=tokenizer,
                generate_env_config=GenerateEnvConfig(),
            )

        for config in configs:
            self.assertEqual(config.select_tokens_id, expected_ids)
            self.assertEqual(config.stop_words_str.count("gg"), 1)
            self.assertEqual(config.stop_words_list.count([1233, 19912]), 1)

    def test_same(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]

        a = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        b = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))

    def test_add_thinking_params(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        special_tokens = SpecialTokens()
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 109})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 109)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [102])

    def test_add_thinking_params_with_think_token(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>"
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = BaseTokenizer(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649])

    def test_add_thinking_params_with_think_token_2(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = BaseTokenizer(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649, 271])


class OpenaiGenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata"
        )
        self.tokenizer = QWenTokenizer(
            os.path.join(self.test_data_path, "qwen_7b/tokenizer/qwen.tiktoken"),
            *args,
            **kwargs,
        )

    def _generate_config_with_stop_word(
        self,
        model_stop_word_str: Optional[List[str]] = None,
        model_stop_word_list: Optional[List[str]] = None,
        env_stop_word_str: Optional[str] = None,
        env_stop_word_list: Optional[str] = None,
        req_stop: Optional[List[str]] = None,
        req_config_stop_word_str: Optional[List[str]] = None,
        req_config_stop_word_list: Optional[List[List[int]]] = None,
        response_format: Optional[dict] = None,
        json_format: Optional[bool] = None,
    ):
        special_tokens = SpecialTokens()
        if model_stop_word_str is not None:
            special_tokens.stop_words_str_list = model_stop_word_str
        if model_stop_word_list is not None:
            special_tokens.stop_words_id_list = model_stop_word_list

        generate_env_config = GenerateEnvConfig()
        if env_stop_word_str is not None:
            generate_env_config.stop_words_str = env_stop_word_str
        if env_stop_word_list is not None:
            generate_env_config.stop_words_list = env_stop_word_list

        # Create ModelConfig object
        model_config = ModelConfig()
        model_config.generate_env_config = generate_env_config
        model_config.render_config = RenderConfig()
        model_config.special_tokens = special_tokens
        model_config.max_seq_len = 1024
        model_config.template_type = None
        model_config.model_name = ""
        model_config.ckpt_path = ""

        openai_endpoint = OpenaiEndpoint(
            model_config=model_config,
            misc_config=PyMiscellaneousConfig(),
            vit_config=VitConfig(),
            tokenizer=self.tokenizer,
            backend_rpc_server_visitor=None,
        )

        request = ChatCompletionRequest(messages=[])
        request.response_format = response_format
        request.json_format = json_format
        if req_stop is not None:
            request.stop = req_stop
        if req_config_stop_word_str is not None:
            if request.extra_configs is None:
                request.extra_configs = GenerateConfig()
            request.extra_configs.stop_words_str = req_config_stop_word_str
        if req_config_stop_word_list is not None:
            if request.extra_configs is None:
                request.extra_configs = GenerateConfig()
            request.extra_configs.stop_words_list = req_config_stop_word_list

        return openai_endpoint._extract_generation_config(request)

    def test_response_format_is_rejected_before_generation(self):
        with self.assertRaises(FtRuntimeException) as raised:
            self._generate_config_with_stop_word(
                response_format={"type": "json_object"}
            )
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )
        self.assertIn("response_format", raised.exception.message)

    def test_json_format_is_rejected_before_generation(self):
        with self.assertRaises(FtRuntimeException) as raised:
            self._generate_config_with_stop_word(json_format=True)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )
        self.assertIn("json_format", raised.exception.message)

    def assert_config_stop_word(
        self,
        expect_stop_word_str: Optional[List[str]] = None,
        expect_stop_word_list: Optional[List[List[str]]] = None,
        **kwargs,
    ):
        config = self._generate_config_with_stop_word(**kwargs)
        if expect_stop_word_str is not None:
            self.assertEqual(
                sorted(config.stop_words_str), sorted(expect_stop_word_str)
            )
        if expect_stop_word_list is not None:
            self.assertEqual(
                sorted(config.stop_words_list), sorted(expect_stop_word_list)
            )

    def test_stop_word_config(self):
        self.assert_config_stop_word(
            expect_stop_word_str=["<|im_end|>", "<|endoftext|>"],
            expect_stop_word_list=[[151643], [151645]],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "model stop word",
                "another model stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2528, 2936, 3409],
                [41963, 1614, 2936, 3409],
            ],
            model_stop_word_str=["model stop word", "another model stop word"],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "model stop list",
                "another model stop list",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],
            ],
            model_stop_word_list=[[2528, 2936, 1140], [41963, 1614, 2936, 1140]],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "env stop word",
                "another env stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [3160, 2936, 3409],
                [41963, 6105, 2936, 3409],
            ],
            env_stop_word_str='["env stop word", "another env stop word"]',
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "env stop list",
                "another env stop list",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [3160, 2936, 1140],
                [41963, 6105, 2936, 1140],
            ],
            env_stop_word_list="[[3160, 2936, 1140], [41963, 6105, 2936, 1140]]",
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "req stop word",
                "another req stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2936, 3409],
                [41963, 4232, 2936, 3409],
            ],
            req_stop=["req stop word", "another req stop word"],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "req config stop word",
                "another config req stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2193, 2936, 3409],
                [41963, 2193, 4232, 2936, 3409],
            ],
            req_config_stop_word_str=[
                "req config stop word",
                "another config req stop word",
            ],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=["<|im_end|>", "<|endoftext|>"],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
            ],
            req_config_stop_word_list=[
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
            ],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",  # default stop word
                "model stop word",
                "another model stop word",  # model_stop_word_str
                "model stop list",
                "another model stop list",  # model_stop_word_list
                "env stop word",
                "another env stop word",  # env_stop_word_str
                "env stop list",
                "another env stop list",  # env_stop_word_list
                "req stop word",
                "another req stop word",  # req_stop
                "req config stop word",
                "another config req stop word",  # req_config_stop_word_str
                "dup stop word",
                "dup stop list",  # duplicate stop word
            ],
            expect_stop_word_list=[
                [151643],
                [151645],  # default stop word list
                [2528, 2936, 3409],
                [41963, 1614, 2936, 3409],  # model_stop_word_str
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],  # model_stop_word_list
                [3160, 2936, 3409],
                [41963, 6105, 2936, 3409],  # env_stop_word_str
                [3160, 2936, 1140],
                [41963, 6105, 2936, 1140],  # env_stop_word_list
                [2958, 2936, 3409],
                [41963, 4232, 2936, 3409],  # req_stop
                [2958, 2193, 2936, 3409],
                [41963, 2193, 4232, 2936, 3409],  # req_config_stop_word_str
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],  # req_config_stop_word_list
                [21912, 2936, 1140],
                [21912, 2936, 3409],  # duplicate stop word
            ],
            model_stop_word_str=[
                "model stop word",
                "another model stop word",
                "dup stop word",
            ],
            model_stop_word_list=[
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],
                [21912, 2936, 1140],
            ],
            env_stop_word_str='["env stop word", "another env stop word", "dup stop word"]',
            env_stop_word_list="[[3160, 2936, 1140], [41963, 6105, 2936, 1140], [21912, 2936, 1140]]",
            req_stop=["req stop word", "another req stop word", "dup stop word"],
            req_config_stop_word_str=[
                "req config stop word",
                "another config req stop word",
                "dup stop word",
            ],
            req_config_stop_word_list=[
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
                [21912, 2936, 1140],
            ],
        )


if __name__ == "__main__":
    main()
