import logging
import torch
import os
import time
import threading
import traceback
import asyncio
from typing import AsyncGenerator
from maga_transformer.utils.util import get_mem_info, AtomicCounter
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.base_model import GenerateInput, GenerateOutput
from maga_transformer.async_decoder_engine.scheduler import Scheduler
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.metrics import GaugeMetrics, kmonitor
from maga_transformer.utils.time_util import Timer
from maga_transformer.async_decoder_engine.normal_model_executor import ExecutorBase
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.utils.model_weight import LoraResourceHolder

class DecoderEngine:
    def __init__(self, executor: ExecutorBase, scheduler: Scheduler, config: GptInitModelParameters) -> None:
        self.executor_ = executor
        self.scheduler_ = scheduler
        self.config_ = config
        self.wait_decode_counter_ = AtomicCounter()
        logging.info(f'last mem info:{get_mem_info().used} {get_mem_info().free}')

    def start(self):
        self.need_stop_ = False
        self.thread = threading.Thread(target=self.run_engine, daemon=True)
        self.thread.start()

    def stop(self):
        logging.info("decoder engine begin stop")
        self.need_stop_ = True
        self.thread.join()
        logging.info("decoder engine stop done")

    def decode(self, input: GenerateInput) -> AsyncGenerator[GenerateOutput, None]:
        stream = self.create_stream(input)        
        # 保证性能测试时能凑批到一起，都用一个起始 counter
        init_counter = self.wait_decode_counter_.get()
        return self._generator_loop_wrap(stream, init_counter)

    async def _generator_loop_wrap(self, stream: GenerateStream, init_counter: int):
        try:
            async for output in self._generate_loop(stream, init_counter):
                yield output
        except BaseException as e:
            # Note that BaseException is used here to catch GeneratorExit and ordinary types of Exception.
            error_msg = f"request_id = {stream._stream_id}, exception type = {type(e)}, exception str {str(e)}"
            logging.info(error_msg)
            # Note: can't release resources here
            stream.set_stop(error_msg)
            raise e

    async def _generate_loop(self, stream: GenerateStream, init_counter: int):
        counter = init_counter
        while True:
            while True:
                new_counter = self.wait_decode_counter_.get()
                if stream.stopped:
                    raise Exception(stream.stop_reason)
                if new_counter != counter:
                    counter = new_counter
                    break
                await asyncio.sleep(0.001)
                
            output = stream.output
            yield output
            if output.finished:
                break

    def report_metric(self, cost_ms: float):
        kmonitor.report(GaugeMetrics.ASYNC_BATCH_SIZE_METRIC,
                        self.scheduler_.running_batch_size())
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_QUERY_SIZE_METRIC,
                        self.scheduler_.wait_stream_size())
        kmonitor.report(GaugeMetrics.ASYNC_ITERATE_LANTENCY, cost_ms)
    
    # public for ptuning
    def create_stream(self, input: GenerateInput) -> GenerateStream:
        if input.prompt_length <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                                     f"model tokens can not be empty, request length is {input.prompt_length}")
        max_new_tokens = min(self.config_.max_seq_len - input.prompt_length, input.generate_config.max_new_tokens)
        if max_new_tokens <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                                     f"model max tokens is {self.config_.max_seq_len}, request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}")
        stream = GenerateStream(input=input,
                                max_seq_len=self.config_.max_seq_len)
        if input.generate_config.adapter_name is not None:
            lora_resource_holder = LoraResourceHolder(self.executor_.base_model_ops.gpt_op.weight.lora_resource,
                                     input.generate_config.adapter_name)
            input.lora_id = lora_resource_holder.lora_id
            stream.add_resource_dtor(lambda: lora_resource_holder.release())
        self.scheduler_.enqueue(stream)
        return stream

    # 这个后台任务一直在跑，应该用线程实现，用 Python 自己的线程切换机制。
    # 如果用协程的话对外返回的 decode 协程会因为 run_engine 协程一直运行被饿死。
    @torch.inference_mode()
    def step(self):
        batch_query = None
        try:
            with Timer() as t:
                be = time.perf_counter()
                torch.cuda.nvtx.range_push('pre_input')
                batch_query = self.scheduler_.schedule()
                if batch_query.total_batch_size == 0 and g_parallel_info.tp_rank == 0:
                    torch.cuda.nvtx.range_pop()
                    time.sleep(0.001)
                    return
                batch_query.generate_model_input()
                batch_query.tp_sync()
                torch.cuda.nvtx.range_pop()

                self.executor_.process(batch_query)

                torch.cuda.nvtx.range_push('update')
                if g_parallel_info.tp_rank == 0:                    
                    self.scheduler_.prepare_next_step()
                torch.cuda.nvtx.range_pop()

            self.report_metric(t.cost_ms())

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

    def run_engine(self):
        while not self.need_stop_:
            self.step()
        logging.info("need stop flag is true, exit run_engine")
