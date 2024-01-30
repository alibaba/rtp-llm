import logging
import torch
import os
import time
import threading
import traceback
import asyncio
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union, Any, Dict, AsyncGenerator
from maga_transformer.utils.util import get_mem_info, AtomicCounter
from maga_transformer.async_decoder_engine.batch_query import QueryStats
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.async_decoder_engine.scheduler import Scheduler
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.metrics import GaugeMetrics, kmonitor
from maga_transformer.utils.time_util import Timer
from maga_transformer.structure.raw_query import RawQuery
from maga_transformer.async_decoder_engine.normal_model_executor import ExecutorBase

class QueryResponse:
    def __init__(self, batch_size, beam_width, eos_token_id, max_seq_len):
        self.hidden_states: List[torch.Tensor] = []
        self.output_tokens = eos_token_id * torch.ones(
            (batch_size, beam_width, max_seq_len), dtype=torch.int32)
        self.finished: List[bool] = [False] * batch_size * beam_width
        self.aux_info: List[Dict[Any, Any]] = [{} for _ in range(batch_size)]
        self.loss: List[Union[None, torch.Tensor]] = [None] * batch_size
        self.logits: List[torch.Tensor] = []
        
class DecoderEngine:
    def __init__(self, executor: ExecutorBase, scheduler: Scheduler, config: GptInitModelParameters) -> None:        
        self.executor_ = executor
        self.scheduler_ = scheduler
        self.config_ = config        

        self.need_stop_ = False
        self.wait_decode_counter_ = AtomicCounter()
        self.thread = threading.Thread(target=self.run_engine, daemon=True)
        self.thread.start()

        logging.info(f'last mem info:{get_mem_info().used} {get_mem_info().free}')

    def stop(self):
        logging.info("decoder engine begin stop")
        self.need_stop_ = True
        self.thread.join()
        logging.info("decoder engine stop done")

    def process_query_response(self, queries: List[QueryStats], current_response) -> GenerateOutput:
        beam_width = queries[0].generate_config.num_beams
        
        current_response.hidden_states = []
        current_response.logits = []
        current_time = time.time()
        
        for i, query in enumerate(queries):
            if query.error_info:
                raise Exception(query.error_info)
            current_response.hidden_states.append(query.hidden_states)
            if query.generate_config.return_logits:
                current_response.logits.append(query.logits)
            if current_response.finished[i * beam_width]:
                continue
            token_ids = query.sliced_output_token_ids
            finish = query.finish
            current_response.output_tokens[i, :, :token_ids[0].numel()] = token_ids
            current_response.loss[i] = query.loss
            current_response.aux_info[i] = {
                "cost_time": current_time - query.begin_time,
                "iter_count": query.iter_count,
                "input_len": query.context_length,
                "output_len": query.seq_length - query.context_length,
                "cum_log_probs": query.cum_log_probs.tolist(),
            }
            if finish:
                if query.generate_config.return_input_ids:
                    current_response.aux_info[i].update({"input_ids": query.input_token_ids.tolist()})
                current_response.finished[i * beam_width: (i + 1) * beam_width] = [True] * beam_width

    async def decoder(self, raw_query: RawQuery) -> AsyncGenerator[GenerateOutput, None]:
        current_response = QueryResponse(raw_query.batch_size, raw_query.generate_config.num_beams,
                                         self.config_.special_tokens.eos_token_id, self.config_.max_seq_len)
        begin_time = time.time()
        queries: List[QueryStats] = []
        
        try:
            queries = self.scheduler_.enqueue(raw_query, self.executor_.base_model_ops.gpt_op.weight.lora_resource)
            counter = self.wait_decode_counter_.get()
            while not all(current_response.finished):
                if raw_query.generate_config.timeout_ms > 0 and (time.time() - begin_time) * 1000 > raw_query.generate_config.timeout_ms:
                    raise Exception(f"{(time.time() - begin_time) * 1000} ms timeout")
                
                while True:
                    new_counter = self.wait_decode_counter_.get()
                    if new_counter != counter:
                        counter = new_counter
                        break
                    await asyncio.sleep(0.001)
                    
                self.process_query_response(queries, current_response)
                yield GenerateOutput(current_response.hidden_states, current_response.output_tokens,
                       torch.BoolTensor(current_response.finished), current_response.aux_info,
                       current_response.loss, current_response.logits)
        finally:
            self.scheduler_.set_stop(queries)

    def report_metric(self, cost_ms: float):
        kmonitor.report(GaugeMetrics.ASYNC_BATCH_SIZE_METRIC,
                        self.scheduler_.running_batch_size())
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_QUERY_SIZE_METRIC,
                        self.scheduler_.wait_query_size())
        kmonitor.report(GaugeMetrics.ASYNC_ITERATE_LANTENCY, cost_ms)
        kmonitor.report(GaugeMetrics.KV_CACHE_MEM_USED_RATIO_METRIC, self.scheduler_.cache_manager_.block_used_ratio())

    # 这个后台任务一直在跑，应该用线程实现，用 Python 自己的线程切换机制。
    # 如果用协程的话对外返回的 decode 协程会因为 run_engine 协程一直运行被饿死。
    @torch.inference_mode()
    def run_engine(self):
        while True:
            if self.need_stop_:
                logging.info("need stop flag is true, exit run_engine")
                return
            if not self.scheduler_.has_query() and g_parallel_info.tp_rank == 0:
                time.sleep(0.001)
                continue
            
            batch_query = None
            try:
                with Timer() as t:
                    be = time.perf_counter()
                    
                    torch.cuda.nvtx.range_push('pre_input')
                    batch_query = self.scheduler_.schedule()
                    if batch_query.total_batch_size == 0 and g_parallel_info.tp_rank == 0:
                        self.wait_decode_counter_.increment()
                        torch.cuda.nvtx.range_pop()
                        continue
                    batch_query.generate_model_input()
                    batch_query.tp_sync()
                    torch.cuda.nvtx.range_pop()
                    
                    self.executor_.process(batch_query)
                    
                    torch.cuda.nvtx.range_push('update')
                    if g_parallel_info.tp_rank == 0:
                        self.scheduler_.prepare_for_next_step()
                    torch.cuda.nvtx.range_pop()
                    
                self.report_metric(t.cost_ms())
                
                # torch.cuda.synchronize();
                # en = time.perf_counter()
                # print('async token time:', (en - be) * 1000)
            except Exception as e:
                if batch_query:
                    self.scheduler_.update_all_errors(str(e))
                logging.error(
                    f'process run error: {e}, Traceback: {traceback.format_exc()}'
                )
                if (g_parallel_info.tp_size) > 1 or ("CUDA" in str(e)):
                    kmonitor.report(GaugeMetrics.ERROR_EXIT_METRIC, 1)
                    kmonitor.flush()
                    time.sleep(0.1)
                    # NOTE: nccl could hang when any error. GPU may hang under CUDA error.
                    os._exit(-1)
            self.wait_decode_counter_.increment()
