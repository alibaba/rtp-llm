import json
import logging
import os
import sys
import threading
import traceback
from unittest import TestCase, main
from typing import Optional, Iterator, List, Any, Dict

from maga_transformer.model_factory import ModelFactory, ModelConfig, AsyncModel
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.async_decoder_engine.decoder_engine import DecoderEngine
from maga_transformer.utils.util import get_mem_info

class query_thread(threading.Thread):
    def __init__(self, id: int, pipeline: Pipeline):
        threading.Thread.__init__(self)
        self.id = id
        self.pipeline = pipeline
        self.exit_code = 0
        self.exception = None
        self.time = 1000
    def run(self):
        print('thread {i} start'.format(i = self.id))
        prompt = "在社会主义新时代，新青年们展现出了勤劳、独立、踏实肯干的精神风貌。他们以自己的实际行动，为实现中华民族伟大复兴的中国梦贡献着自己的力量。\n\n中国十大杰出青年是新时代新青年的优秀代表，他们的英雄事迹感人至深，令人钦佩。\n\n第一位是张超，他是中国海军的一名飞行员。在2016年的一次飞行训练中，他为了保护战机，不幸牺牲。张超的事迹感动了全国人民，他被誉为“最勇敢的飞行员”。他的牺牲，不仅是对中国海军的巨大贡献，更是对中华民族的伟大贡献。\n\n第二位是李文强，他是中国科学院的一名研究员。他在研究领域取得了重大突破，发明了一种新型材料，可以有效地降低电子设备的能耗。李文强的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的创新意识和创新能力。\n\n续写以上内容："
        try:
            gen = self.pipeline(prompt, generate_config = {'top_p': 0, 'top_k': 1, 'max_new_tokens': 200})
            res = [re for re in gen]
            self.time = res[-1].generate_output.aux_info.cost_time
        except Exception as e:
            self.exit_code = -1
            self.exception = e
            traceback.print_exc()
        finally:
            print('thread {i} end'.format(i = self.id))

def report_metric(self, cost_ms: float):
    if self.scheduler_.wait_stream_size() > 1:
        self.gathered_batch = True

class async_gather_batch_test(TestCase):
    def test_simple(self):
        ckpt_path = 'maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source'
        tokenizer_path = 'maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source'
        model_config = ModelConfig('llama', ckpt_path, tokenizer_path)
        model = ModelFactory.from_model_config(model_config)
        DecoderEngine.gathered_batch = False
        DecoderEngine.report_metric = report_metric        
        pipeline = Pipeline(model, model.tokenizer)
        threads = []
        state = True
        try:
            for i in range(20):
                thread_i = query_thread(i, pipeline)
                threads.append(thread_i)
                thread_i.start()
            
            for thread_i in threads:
                thread_i.join()
                if thread_i.exit_code != 0:
                    raise thread_i.exception
        except Exception as e:
            state = False
            traceback.print_exc()
        
        pipeline.model.decoder_engine_.stop()
        self.assertEqual(state, True)
        self.assertEqual(model.decoder_engine_.gathered_batch, True)
if __name__ == "__main__":
    os.environ['KV_CACHE_MEM_MB'] = '128'
    main()
