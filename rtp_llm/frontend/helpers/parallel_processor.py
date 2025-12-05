"""
Parallel Processor

处理并行批处理逻辑，管理多个异步生成器的并行执行
"""

import asyncio
import logging
import traceback
from typing import Any, AsyncGenerator, Dict, List, Set, Union

from .pipeline_response import BatchPipelineResponse, PipelineResponse


class ParallelProcessor:
    """并行处理器，负责管理多个异步生成器的并行执行"""

    async def parallel_batch_async_generators(
        self,
        incremental: bool,
        generators: List[AsyncGenerator[Dict[str, Any], None]],
        batch_infer: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """并行处理多个异步生成器"""
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs: Set[int] = set()
        batch_state: List[Any] = [None] * len(iterators)

        while True:
            # Create parallel tasks
            tasks = []
            for idx, itr in enumerate(iterators):
                if idx not in done_idxs:  # Create tasks only for unfinished iterators
                    tasks.append((idx, itr.__anext__()))

            # Use asyncio.gather() to get results
            if tasks:
                results = await asyncio.gather(
                    *(task[1] for task in tasks), return_exceptions=True
                )
                for idx, result in zip((task[0] for task in tasks), results):
                    if isinstance(result, Exception):
                        # Handle exception cases, such as StopAsyncIteration
                        if isinstance(result, StopAsyncIteration):
                            done_idxs.add(idx)
                            if batch_state[idx] is None:
                                batch_state[idx] = PipelineResponse()
                            if incremental:
                                batch_state[idx] = PipelineResponse()
                        else:
                            error_msg = f'ErrorMsg: {str(result)} \n Traceback: {"".join(traceback.format_tb(result.__traceback__))}'
                            logging.warning(error_msg)
                            raise result
                    else:
                        batch_state[idx] = result

            # Check if all iterators are done
            if len(done_idxs) == len(iterators):
                break

            # Process batch data
            batch = batch_state
            if batch_infer:
                yield BatchPipelineResponse(response_batch=batch)
            else:
                yield batch[0]
