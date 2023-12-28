import logging
import torch
import os
import time
import threading
import traceback
import asyncio
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union, Any, Dict
from maga_transformer.utils.util import get_mem_info, AtomicCounter
from maga_transformer.async_decoder_engine.batch_query import QueryStats
from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.query_manager import QueryManager
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.metrics import GaugeMetrics, kmonitor
from maga_transformer.utils.time_util import Timer
from maga_transformer.async_decoder_engine.base_model_executor import ExecutorBase

class DecoderEngine:
    def __init__(self, executor: ExecutorBase, query_manager: QueryManager, config: GptInitModelParameters) -> None:        
        self.executor_ = executor
        self.query_manager_ = query_manager
        self.config_ = config        

        self.stop_flag_ = False
        self.has_quit_process_flag_ = False
        self.p = threading.Thread(target=self.async_process)
        self.p.start()
        self.wait_decode_counter_ = AtomicCounter()

        logging.info(f'last mem info:{get_mem_info().used} {get_mem_info().free}')

    def stop(self):
        logging.info("decoder engine begin stop")
        self.stop_flag_ = True
        self.p.join()
        logging.info("decoder engine stop done")

    async def decoder(
        self, 
        input_token_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor],
        images: List[List[str]],
        generate_config: GenerateConfig
    ) -> Iterator[Tuple[
            List[Union[None, torch.Tensor]], torch.Tensor, torch.Tensor, List[Dict[Any, Any]], torch.Tensor, 
            Optional[Union[torch.Tensor, List[torch.Tensor]]]
        ]]:
        begin_time = time.time()
        queries: List[QueryStats] = []
        batch_size: int = input_token_ids.shape[0]
        beam_width = generate_config.num_beams
        finished: List[bool] = [False] * batch_size * beam_width
        aux_info: List[Dict[Any, Any]] = [{} for _ in range(batch_size)]
        loss: List[Union[None, torch.Tensor]] = [None] * batch_size
        logits: List[Optional[torch.Tensor]] = [None] * batch_size
        output_tokens = self.config_.special_tokens.eos_token_id * torch.ones(
            (batch_size, beam_width, self.config_.max_seq_len),
            dtype=torch.int32
        )
        try:
            queries = self.query_manager_.put_requests_to_queue(input_token_ids, input_lengths, images, generate_config)
            counter = self.wait_decode_counter_.get()
            iter_count = 0
            while not all(finished):
                if generate_config.timeout_ms > 0 and (time.time() - begin_time) * 1000 > generate_config.timeout_ms:
                    raise Exception(f"{(time.time() - begin_time) * 1000} ms timeout")
                while True:
                    new_counter = self.wait_decode_counter_.get()
                    if new_counter != counter:
                        counter = new_counter
                        break
                    await asyncio.sleep(0.001)
                iter_count += 1
                current_time = time.time()
                hidden_states = []
                logits = []
                for i, query in enumerate(queries):
                    if query.error_info:
                        raise Exception(query.error_info)
                    hidden_states.append(query.hidden_states)
                    if query.generate_config.return_logits:
                        logits.append(query.logits)
                    if finished[i * beam_width]:
                        continue
                    token_ids = query.sliced_output_token_ids
                    finish = query.finish
                    output_tokens[i, :, :token_ids[0].numel()] = token_ids
                    loss[i] = query.loss
                    aux_info[i] = {
                        "cost_time": current_time - begin_time,
                        "iter_count": iter_count,
                        "input_len": query.context_length,
                        "output_len": query.seq_length - query.context_length,
                        "cum_log_probs": query.cum_log_probs.tolist(),
                    }
                    if finish:
                        # release query first, let other query use
                        # self.query_manager_.free([query])
                        finished[i * beam_width: (i + 1) * beam_width] = [True] * beam_width
                        continue
                yield (hidden_states, output_tokens,
                       torch.BoolTensor(finished), aux_info, loss, logits)
        finally:
            self.query_manager_.set_stop(queries)

    def report_metric(self, cost_ms: float):
        kmonitor.report(GaugeMetrics.ASYNC_BATCH_SIZE_METRIC,
                        self.query_manager_.running_batch_size())
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_QUERY_SIZE_METRIC,
                        self.query_manager_.wait_query_size())
        kmonitor.report(GaugeMetrics.ASYNC_ITERATE_LANTENCY, cost_ms)
        kmonitor.report(GaugeMetrics.KV_CACHE_MEM_USED_RATIO_METRIC, self.query_manager_.block_used_ratio())

    # 这个后台任务一直在跑，应该用线程实现，用 Python 自己的线程切换机制。
    # 如果用协程的话对外返回的 decode 协程会因为 async_process 协程一直运行被饿死。
    @torch.inference_mode()
    def async_process(self):
        while True:
            if self.stop_flag_:
                logging.info("stop flag is true, exit async_process")
                self.has_quit_process_flag_ = True
                return
            if not self.query_manager_.has_query() and g_parallel_info.tp_rank == 0:
                time.sleep(0.001)
                continue
            batch_query = None
            try:
                with Timer() as t:
                    be = time.perf_counter()
                    torch.cuda.nvtx.range_push('pre_input')
                    batch_query = self.query_manager_.get_batch_request()
                    batch_query.generate()
                    batch_query.tp_sync()
                    torch.cuda.nvtx.range_pop()
                    self.executor_.process(batch_query)
                    torch.cuda.nvtx.range_push('update')
                    if g_parallel_info.tp_rank == 0:
                        self.query_manager_.update_batch_query()
                self.report_metric(t.cost_ms())
                torch.cuda.nvtx.range_pop()
                # torch.cuda.synchronize();
                # en = time.perf_counter()
                # print('async token time:', (en - be) * 1000)
            except Exception as e:
                if batch_query:
                    self.query_manager_.update_all_errors(str(e))
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
