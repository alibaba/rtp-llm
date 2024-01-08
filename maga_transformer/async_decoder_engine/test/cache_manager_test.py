from unittest import TestCase, main
from maga_transformer.async_decoder_engine.cache_manager import CacheManager, CacheConfigGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from unittest import mock

class MockMemInfo:
    free: int  = 16 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.async_decoder_engine.cache_manager.get_mem_info', MockMemInfo)
class CacheManagerTest(TestCase):
    @staticmethod
    def _init_config(head_num_kv: int = 2,
                     size_per_head: int = 4,
                     seq_size_per_block: int = 8,
                     layer_num: int = 16,
                     data_type: str = 'fp16',
                     reserve_runtime_mem_mb: int = 0):
        config = GptInitModelParameters(head_num=head_num_kv,
                                        head_num_kv=head_num_kv,
                                        size_per_head=size_per_head,
                                        layer_num=layer_num,
                                        max_seq_len=0,
                                        seq_size_per_block=seq_size_per_block,
                                        data_type=data_type,
                                        reserve_runtime_mem_mb=reserve_runtime_mem_mb,
                                        vocab_size=0)
        cache_config = CacheConfigGenerator.create_config(config)
        return cache_config

    def test_simple(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        self.assertEqual(len(cache_manager.free_blocks_index), 3)
        index1 = cache_manager.malloc(1)
        index2 = cache_manager.malloc(2)
        with self.assertRaisesRegex(Exception, "failed to malloc 1 blocks, only 0 blocks left"):
            cache_manager.malloc(1)
        cache_manager.free([index1])
        cache_manager.free([index2])

    def test_allocate_with_free_cache(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        self.assertEqual(len(cache_manager.free_blocks_index), 3)
        index1, _ = cache_manager.malloc_with_cache(3, token_ids=[1,2,3])
        cache_manager.free_with_cache([index1], token_ids=[1,2,3])
        index2 = cache_manager.malloc(2)
        self.assertEqual(len(cache_manager.free_blocks_index), 1)

    def test_allocate_with_reuse(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        self.assertEqual(len(cache_manager.free_blocks_index), 3)
        index1 = cache_manager.malloc(2)
        cache_manager.free_with_cache([index1], token_ids=[1000,1002], chat_id="1234")
        index2, reuse_num = cache_manager.malloc_with_cache(2, [1000,1002], chat_id="1234")
        self.assertEqual(len(cache_manager.free_blocks_index), 1)
        self.assertEqual(index2, [1,2])
        self.assertEqual(reuse_num, 1)
        cache_manager.free_with_cache([index2], [1000, 1002], chat_id="1234")
        index3, reuse_num = cache_manager.malloc_with_cache(3, [1000, 1002, 1003], chat_id="1234")
        self.assertEqual(index3, [1,2,3])
        self.assertEqual(reuse_num, 1)

    def test_lack_mem(self):
        with self.assertRaises(AssertionError):
            config = self._init_config(reserve_runtime_mem_mb=1)
            cache_manager = CacheManager(config, None)

if __name__ == '__main__':
    main()
