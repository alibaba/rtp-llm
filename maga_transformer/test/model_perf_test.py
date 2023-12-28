from collections import defaultdict
import os, time, sys, traceback
sys.path.append('/home/zw193905/FasterTransformer')
sys.path.append('/home/zw193905/FasterTransformer/bazel-FasterTransformer/external/com_taobao_aios')
sys.path.append('/home/zw193905/FasterTransformer/bazel-FasterTransformer/external/com_taobao_aios/aios/kmonitor/python_client')

from maga_transformer.model_factory import ModelFactory, ModelConfig
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.config.generate_config import GenerateConfig
 
import torch
import csv

model_list = {
    # "codex-7b": {
    #     "size": 7,
    #     "model_type": "codex",
    #     "ckpt_path": "/home/zw193905/models_data/codex/data/codex_data.pt",
    #     "tokenizer_path": "/home/zw193905/models_data/codex/tokenizer"
    # },
    # "llama-7b-int8": {
    #     "model_type": "llama",
    #     "size": 7,
    #     "tokenizer_path": "/home/zw193905/models_data/llama-ckpt/tokenizer.model",
    #     "ckpt_path": "/home/zw193905/models_data/llama-ckpt/7B",
    #     "int8_mode": 1        
    # },
    # "llama-7b": {
    #     "model_type": "llama",
    #     "size": 7,        
    #     "tokenizer_path": "/home/zw193905/models_data/llama-ckpt/tokenizer.model",
    #     "ckpt_path": "/home/zw193905/models_data/llama-ckpt/7B"
    # },    
    # "llama-13b-int8": {
    #     "model_type": "llama",
    #     "size": 13,        
    #     "tokenizer_path": "/home/zw193905/models_data/ugllm13B_model/tokenizer.model",
    #     "ckpt_path": "/home/zw193905/models_data/ugllm13B_model",
    #     "int8_mode": 1        
    # },
    # "llama-13b": {
    #     "model_type": "llama",
    #     "size": 13,        
    #     "tokenizer_path": "/home/zw193905/ugllm13B_model/tokenizer.model",
    #     "ckpt_path": "/home/zw193905/ugllm13B_model/tokenizer.model"
    # },    
    # "chatglm-6b": {
    #     "model_type": "chat_glm",
    #     "size": 6,        
    #     "tokenizer_path": "/home/zw193905/models_data/chatglm",
    #     "ckpt_path": "/home/zw193905/models_data/chatglm"
    # },
    #"starcoder-16b-int8": {
    #    "size": 16,        
    #    "model_type": "gpt_bigcode",
    #    "tokenizer_path": "/home/zw193905/models_data/starcoder",
    #    "ckpt_path": "/home/zw193905/models_data/starcoder",
    #    "int8_mode": 1
    # },
    "chatglm2": {
        "size": 13,        
        "model_type": "chatglm2",
        "tokenizer_path": "/mnt/nas1/hf/chatglm2-6b",
        "ckpt_path": "/mnt/nas1/hf/chatglm2-6b",
        "int8_mode": 0
    },
    # "starcoder-16b": {
    #     "size": 16,
    #     "model_type": "gpt_bigcode",
    #     "tokenizer_path": "/home/zw193905/chatglm",
    #     "ckpt_path": "/home/zw193905/chatglm"
    # }    
}



# batch_sizes = [1,2,3,4,5,6,7,8] + [i for i in range(16, 129, 8)]
batch_sizes = [1]

seq_step = 128

test_query = "test "

def gen_tokens(model, length, generate_config):
    size = 0
    query = test_query
    if isinstance(model, AsyncModel):
        tokenizer = model.model.tokenizer
    else:
        tokenizer = model.tokenizer   
 
    pipeline = Pipeline(model, tokenizer)
    while size < length:
        input_tensor, input_lengths = pipeline.encode_tokens([query], generate_config)
        size = input_lengths[0]
        query += test_query
    return input_tensor, input_lengths
    
def perf_context_decoder(model, warmup):
    perf_res = []
    max_seq_len = model.config.max_seq_len
    max_seq_len = min(max_seq_len, 128)
    for batch_size in batch_sizes:
        for step in [128]:
            print("begin context decoder perf test, batch_size = %s, step = %s" % (batch_size, step))
            try:
                generate_config = GenerateConfig()
                generate_config.max_new_tokens = 1
                generate_config.random_seed = None

                input_tensor, input_lengths = gen_tokens(model, step, generate_config)
                be = time.perf_counter() * 1000
                stream = model.generate_stream(inputs=torch.tensor(input_tensor * batch_size),
                                                input_lengths=torch.tensor(input_lengths * batch_size),
                                                generate_config=generate_config)
                
                [t for t in stream]
                en = time.perf_counter() * 1000
                if not warmup:
                    perf_res.append((batch_size, step, en - be))               
                    print("context test_out:", batch_size, step, en - be)
            except Exception as e:
                print('error:', batch_size, step, str(e), traceback.format_exc())
                break
            
            print("done context decoder perf test, batch_size = %s, step = %s" % (batch_size, step))
            sys.stdout.flush()
            
    return perf_res

def perf_decoder(model, warmup):
    perf_res = []
    max_seq_len = model.config.max_seq_len
    print('max_seq_len:', max_seq_len)
    max_seq_len = min(max_seq_len, 128)    
    for batch_size in batch_sizes:
        while model.decoder_engine.query_manager.has_query():
            model.decoder_engine.async_process()
        count = 0
        print("begin decoder perf test, batch_size = %s" % (batch_size))
        try:
            generate_config = GenerateConfig()
            generate_config.max_new_tokens = max_seq_len
            generate_config.topk = 1
            generate_config.random_seed = None
            
            input_tensor, input_lengths = gen_tokens(model, 1, generate_config)
            
            # input_tensor, input_lengths = model.encode_tokens(test_input, generate_config)
            generate_stream = model.generate_stream(inputs=torch.tensor(input_tensor * batch_size),
                                                    input_lengths=torch.tensor(input_lengths * batch_size),
                                                    generate_config=generate_config)
             
            be = time.perf_counter() * 1000
            for _ in generate_stream:
                count += 1       
                en = time.perf_counter() * 1000
                if count % seq_step == 0:
                    if not warmup: 
                        perf_res.append((batch_size, count, en - be))
                        print("decoder test_out:", batch_size, count, en - be)                    
                be = time.perf_counter() * 1000
            if not warmup: 
                print('count:', count)
            # assert(count + 10 > max_seq_len)
        except Exception as e:
            print('error:', batch_size, count, str(e), traceback.format_exc())
            continue
        
        print("done decoder perf test, batch_size = %s" % (batch_size))
        sys.stdout.flush()
    
    return perf_res

def perf_model(model):
    # warm up
    
    print("start warm up")
    for i in range(0, 2):
    	perf_context_decoder(model, True)
    	perf_decoder(model, True)

    print("warm up done")

    context_perf_res = perf_context_decoder(model, False)
    decoder_perf_res = perf_decoder(model, False)
    
    print("real perf test done")
    
    return context_perf_res, decoder_perf_res

def record(name, data):
    rec = defaultdict(lambda: defaultdict(lambda: 0))
    seq_lens = set()
    for (b,s,t) in data:
        rec[b][s] = t
        seq_lens.add(s)
    seq_lens = sorted(list(seq_lens))        
    row_datas = [['batch_size'] + seq_lens]

    # row_datas = [[-1] * len(seq_lens)] * len(rec)
    for b in sorted(list(rec.keys())):
        row_data = [b] + [-1] * len(seq_lens)        
        for s, t in rec[b].items():
            row_data[seq_lens.index(s) + 1] = t
        row_datas.append(row_data)
    with open("perf_res/" + name + ".csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(row_datas)

def perf_and_record_model(name, model_config):
    print('perf model:', name)
    
    free_gpu_mem_size = torch.cuda.get_device_properties("cuda:0").total_memory / 1024 / 1024 / 1024        
    size = model_config['size']
    int8_mode = model_config.get('int8_mode', 0)
    if size * (2 - int8_mode) >= free_gpu_mem_size:
        return
    
    try:
        '''
        config = ModelFactory.from_model_type(model_type=model_config['model_type'],
                                            ckpt_path=model_config['ckpt_path'],
                                            tokenizer_path=model_config['tokenizer_path'],
                                            async_mode=True,
                                            gen_config=True,
                                            int8_mode=int8_mode)
        #max_seq_len = config.max_seq_len
        #head_num = config.head_num
        #head_num_kv = config.head_num_kv
        #if head_num_kv <= 0:
        #    head_num_kv = head_num
        # os.system(f'./gpt_gemm {16} {1} {max_seq_len/2} {head_num} {head_num_kv} {config.size_per_head} {config.inter_size} {config.vocab_size} {1} {1} {0}')
        '''

        model = ModelFactory.from_model_type(ModelConfig(model_type=model_config['model_type'],
                                            ckpt_path=model_config['ckpt_path'],
                                            tokenizer_path=model_config['tokenizer_path'],
                                            async_mode=True,
                                            int8_mode=int8_mode))
        if isinstance(model, AsyncModel): 
            model.stop()
        return
        
        res1, res2 = perf_model(model)
    except Exception as e:
        print('run error:', name, str(e), traceback.format_exc())
        return
    print('res1:', res1)
    print('res2:', res2)
    record(name + '-context', res1)
    record(name + '-decoder', res2)
    
    sys.stdout.flush()

from multiprocessing import Process
import time
def main():    
    for name, model_config in model_list.items():
        # for k,v in model_config.items():
        #     os.environ[k] = v
        be = time.perf_counter()
        p = Process(target=perf_and_record_model, args=(name, model_config))
        p.start()
        p.join()
        en = time.perf_counter()        
        print('perf model:', name, en - be)
    
    sys.stdout.flush()
    
if __name__ == '__main__':
    main()
