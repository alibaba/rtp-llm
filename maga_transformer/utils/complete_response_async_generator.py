from typing import AsyncGenerator, List, Callable, Any

class CompleteResponseAsyncGenerator:
    def __init__(self, generator: AsyncGenerator, collect_complete_response_func: Callable):
        self._generator = generator
        self._collect_complete_response_func = collect_complete_response_func
        self._all_responses = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            response = await self._generator.__anext__()
            self._all_responses.append(response)
            return response
        except StopAsyncIteration:
            raise

    async def aclose(self):
        return await self._generator.aclose()

    async def gen_complete_response_once(self) -> Any:
        return await self._collect_complete_response_func(CompleteResponseAsyncGenerator.generate_from_list(self._all_responses))

    @staticmethod
    async def generate_from_list(response_list) -> AsyncGenerator:
        for response in response_list:
            yield response

    @staticmethod
    async def get_last_value(all_responses: AsyncGenerator):
        response = None
        try:
            async for response in all_responses:
                pass
        except StopAsyncIteration:
            pass
        return response