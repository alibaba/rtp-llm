import torch
from unittest import TestCase, main
from typing import Any, List, Optional, Union, Dict, Tuple
from maga_transformer.async_decoder_engine.generate_stream import GenerateInput, GenerateStream
from maga_transformer.config.generate_config import GenerateConfig 
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.async_decoder_engine.stream_cache_manager import StreamCacheManager
from maga_transformer.config.cache_config import CacheConfigGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from unittest import mock

class MockMemInfo:
    free: int  = 10 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
class ReuseCacheLengthTest(TestCase):
    @staticmethod
    def _init_stream_cache_manager(head_num_kv: int = 2,
                                   size_per_head: int = 4,
                                   seq_size_per_block: int = 2,
                                   layer_num: int = 16,
                                   data_type: str = 'fp16',
                                   reserve_runtime_mem_mb: int = 0,
                                   img_start_id: Optional[int] = None,
                                   img_end_id: Optional[int] = None):
        config = GptInitModelParameters(head_num=head_num_kv,
                                        head_num_kv=head_num_kv,
                                        size_per_head=size_per_head,
                                        layer_num=layer_num,
                                        max_seq_len=0,
                                        seq_size_per_block=seq_size_per_block,
                                        data_type=data_type,
                                        reserve_runtime_mem_mb=reserve_runtime_mem_mb,
                                        vocab_size=0)
        config.reuse_cache = True
        if img_start_id and img_end_id:
            config.vit_related_params.vit_special_token_ids['image_start_id'] = img_start_id
            config.vit_related_params.vit_special_token_ids['image_end_id'] = img_end_id
        cache_config = CacheConfigGenerator.create_config(config)
        cache_manager = CacheManager(cache_config, None)
        return StreamCacheManager(config, cache_manager, 1)

    @staticmethod
    def _init_generate_stream(token_ids: List[int]):
        generate_input = GenerateInput(token_ids = torch.tensor(token_ids), generate_config=GenerateConfig())
        return GenerateStream(generate_input)

    def test_simple(self):
        stream_cache_manager = self._init_stream_cache_manager()
        input_token = [1, 2, 3, 4, 5]
        stream = self._init_generate_stream(input_token)
        stream_cache_manager.init_kvcache(stream)
        stream_cache_manager.free_block_cache(stream)
        self.assertEqual(stream.reuse_length, 0)

        stream2 = self._init_generate_stream([1, 2, 3, 4])
        stream_cache_manager.init_kvcache(stream2)
        self.assertEqual(stream2.reuse_length, 2)

    def test_multimodal_base(self):
        stream_cache_manager = self._init_stream_cache_manager(img_start_id = 1, img_end_id = 2)
        
        input_token = [1, 3, 2, 4, 5, 6]
        stream = self._init_generate_stream(input_token)
        stream_cache_manager.init_kvcache(stream)
        self.assertEqual(stream.reuse_length, 0)
        self.assertEqual(stream.block_indice, [[1, 2, 3]])
        stream_cache_manager.free_block_cache(stream)

        # unclosed image tags
        try:
            stream2 = self._init_generate_stream([1, 3, 4])
            stream_cache_manager.init_kvcache(stream2)
        except Exception as e:   
            self.assertEqual(str(e), 'unclosed image tag pair in [1, 3, 4]')
            stream_cache_manager.cache_manager_.free([stream2.block_indice])

        # reuse length not cover whole image -> shrink to left tag
        stream3 = self._init_generate_stream([1, 3, 4, 2])
        stream_cache_manager.init_kvcache(stream3)
        stream_cache_manager.free_block_cache(stream3)
        self.assertEqual(stream3.reuse_length, 0)

        # after align, reuse length not cover whole image
        stream4 = self._init_generate_stream([1, 3, 2, 1, 2])
        stream_cache_manager.init_kvcache(stream4)
        stream_cache_manager.free_block_cache(stream4)
        self.assertEqual(stream4.reuse_length, 0)

        # reuse cache
        stream5 = self._init_generate_stream([1, 3, 2, 4, 6])
        stream_cache_manager.init_kvcache(stream5)
        self.assertEqual(stream5.block_indice, [[1, 2, 9]])
        stream_cache_manager.free_block_cache(stream5)
        self.assertEqual(stream5.reuse_length, 4)

        # check if free cache can be malloc
        stream6 = self._init_generate_stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        stream_cache_manager.init_kvcache(stream6)
        self.assertEqual(len(stream6.block_indice[0]), 9)
        stream_cache_manager.free_block_cache(stream6)

if __name__ == '__main__':
    main()
