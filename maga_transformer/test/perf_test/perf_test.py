import json
import logging
import argparse
import os
import sys
import threading
import asyncio
import time
import torch
from typing import Optional, Tuple, Dict, Any, List
from filelock import FileLock, Timeout
from odps import ODPS

from maga_transformer.model_factory import *
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.model_factory import AsyncModel

origin_prompt = "在社会主义新时代，新青年们展现出了勤劳、独立、踏实肯干的精神风貌。他们以自己的实际行动，为实现中华民族伟大复兴的中国梦贡献着自己的力量。\n\n中国十大杰出青年是新时代新青年的优秀代表，他们的英雄事迹感人至深，令人钦佩。\n\n第一位是张超，他是中国海军的一名飞行员。在2016年的一次飞行训练中，他为了保护战机，不幸牺牲。张超的事迹感动了全国人民，他被誉为“最勇敢的飞行员”。他的牺牲，不仅是对中国海军的巨大贡献，更是对中华民族的伟大贡献。\n\n第二位是李文强，他是中国科学院的一名研究员。他在研究领域取得了重大突破，发明了一种新型材料，可以有效地降低电子设备的能耗。李文强的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的创新意识和创新能力。\n\n第三位是刘洋，她是中国第一位女航天员。在2012年，她成功完成了神舟九号飞船的飞行任务，成为中国航天史上的重要人物。刘洋的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的责任感和使命感。\n\n第四位是王俊凯，他是中国著名歌手和演员。他在音乐和影视领域取得了巨大的成就，成为了中国年轻人的偶像。王俊凯的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的自我价值感和自我实现意识。\n\n第五位是陈薇，她是中国疾病预防控制中心的一名研究员。在2020年新冠疫情爆发期间，她带领团队研发出了中国首个新冠疫苗，为全球抗疫做出了巨大贡献。陈薇的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的责任感和使命感。\n\n第六位是李佳琦，他是中国著名网络主播。他在直播领域取得了巨大的成就，成为了中国年轻人的偶像。李佳琦的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的自我价值感和自我实现意识。\n\n第七位是王小川，他是中国著名互联网企业家。他创立的搜狗公司成为了中国最大的搜索引擎之一，为中国的互联网发展做出了巨大贡献。王小川的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的创新意识和创新能力。\n\n第八位是张瑞敏，他是中国著名企业家。他创立的海尔集团成为了中国最大的家电企业之一，为中国的经济发展做出了巨大贡献。张瑞敏的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的责任感和使命感。\n\n第九位是马云，他是中国著名互联网企业家。他创立的阿里巴巴集团成为了中国最大的电子商务企业之一，为中国的经济发展做出了巨大贡献。马云的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的创新意识和创新能力。\n\n第十位是马化腾，他是中国著名互联网企业家。他创立的腾讯公司成为了中国最大的社交网络企业之一，为中国的互联网发展做出了巨大贡献。马化腾的事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的自我价值感和自我实现意识。\n\n这些新青年的英雄事迹表明，新时代的青年们不仅有着高超的技能，更有着强烈的责任感和使命感，他们用自己的实际行动，为实现中华民族伟大复兴的中国梦贡献着自己的力量。\n\n续写以上内容："

def write_odps(records):
    if 'ODPS_PROJECT' not in os.environ:
        logging.warning("no odps config")
        for record in records:
            logging.info(f'{record}')
        return
    partition_name = os.environ['PARTITION_NAME']
    odps = ODPS(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        os.environ['ODPS_PROJECT'],
        endpoint='http://service-corp.odps.aliyun-inc.com/api',
    )

    table = odps.get_table(os.environ['ODPS_TABLE'])
    if table.exist_partition(partition_name):
        partition = table.get_partition(partition_name)
    else:
        partition = table.create_partition(partition_name)
    with table.open_writer(partition=partition_name) as writer:
        writer.write(records)

def test_pipeline_time(pipeline, **kwargs):
    context_time = 0.0
    acc_time = 0.0
    token_count = 0
    exception = None

    async def generator():
        nonlocal context_time
        nonlocal acc_time
        nonlocal exception
        nonlocal token_count
        try:
            begin_time = time.time()
            async for x in pipeline.pipeline_async(**kwargs):
                end_time = time.time()
                cost_time = end_time - begin_time
                if token_count == 0:
                    context_time = cost_time
                else:
                    acc_time += cost_time
                begin_time = end_time
                token_count += 1
        except Exception as e:
            logging.error(f'except: {e}')
            exception = e

    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generator())

    backgroud_thread = threading.Thread(target=start_loop)
    backgroud_thread.start()
    backgroud_thread.join()

    if token_count != 0:
        avg_time = acc_time / token_count
    else:
        avg_time = 0
    return exception is None, context_time, avg_time

def get_prompt(tokenizer, prompt, seqlen):
    while len(tokenizer.encode(prompt)) < seqlen:
        prompt += prompt
    for dec_step in [1024, 256, 64, 16, 2, 1]:
        while len(tokenizer.encode(prompt[:-dec_step])) >= seqlen:
            prompt = prompt[:-dec_step]
    return prompt

def run(model_args: Dict[str, Any], test_args: Dict[str, Any]):
    state = True
    model_args['async_mode'] = 1

    model = ModelFactory.from_model_type(ModelConfig(**model_args))
    if isinstance(model, AsyncModel):
        tokenizer = model.model.tokenizer
    else:
        tokenizer = model.tokenizer
    pipeline = Pipeline(model, tokenizer)

    test_batchsize = test_args['test_batchsize']
    test_input_len = test_args['test_input_len']

    # test
    context_decoder_time = []
    decoder_time = []

    prompts = {x:get_prompt(tokenizer, origin_prompt, x)
               for x in set(test_input_len)}
    for bs, seqlen in zip(test_batchsize, test_input_len):
        # find closest prompt input > seqlen
        prompt = prompts[seqlen]
        actual_seqlen = len(tokenizer.encode(prompt))
        test_pipeline_time(
            pipeline=pipeline,
            prompts=[prompt]*bs, images = [[]] * bs, top_k=1, top_p=0, temperature=1,
            generate_config={'max_new_tokens': 10, 'min_new_tokens':10})

        torch.cuda.nvtx.range_push(f"batchsize={bs}, seqlen={actual_seqlen}")
        cur_state, context_time, generate_time = test_pipeline_time(
            pipeline=pipeline,
            prompts=[prompt]*bs, images = [[]] * bs, top_k=1, top_p=0, temperature=1,
            generate_config={'max_new_tokens': 10, 'min_new_tokens':10})
        state = state and cur_state
        torch.cuda.nvtx.range_pop()
        print(f"--- model_type={model_args['model_type']} batchsize={bs}, seqlen={actual_seqlen} ---", file=sys.stderr)
        print(f"context time: {context_time*1000:.3f} ms, generate time: {generate_time*1000:.3f} ms", file=sys.stderr)
        context_decoder_time.append(context_time)
        decoder_time.append(generate_time)


    if isinstance(pipeline.model, AsyncModel):
        pipeline.model.decoder_engine_.stop()

    del pipeline
    torch.cuda.empty_cache()

    return state, context_decoder_time, decoder_time

class testcase:
    def __init__(self, model_type, ckpt_path, tokenizer_path, prec, test_batch_size, test_input_len):
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.tokenizer_path = tokenizer_path
        self.test_batch_size = test_batch_size
        self.test_input_len = test_input_len
        self.prec = prec

    def get_args(self):
        return {
            'model_args': {
                'model_type': self.model_type, 'ckpt_path': self.ckpt_path, 'tokenizer_path': self.tokenizer_path,
                'weight_type':  WEIGHT_TYPE.INT8 if self.prec == 'int8' else WEIGHT_TYPE.FP16,
                'max_seq_len': int(os.environ['MAX_SEQ_LEN'])
            },
            'test_args': {
                'test_batchsize': self.test_batch_size, 'test_input_len': self.test_input_len
            }
        }

if __name__ == '__main__':
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='perf runner')

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_size', type=float, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--batch_size', type=str, required=True)
    parser.add_argument('--input_len', type=str, required=True)
    parser.add_argument('--prec', type=str, required=True)
    args, _ = parser.parse_known_args()

    t_case = testcase(args.model_type, args.ckpt_path, args.tokenizer_path, args.prec,
        [int(x) for x in args.batch_size.split(',')],
        [int(x) for x in args.input_len.split(',')])

    lock_path = '/tmp/maga_transformer/perf_test/gpu_status_lock'
    lock = FileLock(lock_path)

    device_name = torch.cuda.get_device_name(0)

    while True:
        try:
            lock.acquire()
        except:
            logging.info("lock file failed")
            time.sleep(1)
            continue
        state, context_decoder_time, decoder_time = run(**t_case.get_args())
        assert state
        records = []
        for i in range(len(t_case.test_batch_size)):
            records.append([
                args.model_type,
                args.model_size,
                args.prec,
                device_name,
                'maga_transformer',
                str(os.environ.get('CIS_ENV_COMMIT_ID')),
                t_case.test_batch_size[i],
                t_case.test_input_len[i],
                context_decoder_time[i] * 1000,
                decoder_time[i] * 1000,
            ])
        write_odps(records)
        break
