from unittest import TestCase, main
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.cache_config import CacheConfigGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from unittest import mock

class MockMemInfo:
    free: int  = 2 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
class CacheManagerTest(TestCase):
    @staticmethod
    def _init_config(head_num_kv: int = 2,
                     size_per_head: int = 4,
                     seq_size_per_block: int = 1,
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

        self.assertEqual(cache_manager.free_block_nums, 3)
        index1 = cache_manager.malloc(1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        
        index2 = cache_manager.malloc(2)
        with self.assertRaisesRegex(Exception, "failed to malloc 1 blocks, only 0 blocks left"):
            cache_manager.malloc(1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        
        cache_manager.free([index1])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 0)
        
        cache_manager.free([index2])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        
        self.assertEqual(cache_manager.free_block_nums, 3)

    def test_allocate_with_free_cache(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 3)
        
        index1, _ = cache_manager.malloc_with_cache(3, token_ids=[1,2,3])
        self.assertEqual(index1, [1,2,3])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        
        cache_manager.free_with_cache([index1], token_ids=[1,2,3])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        
        index2 = cache_manager.malloc(2)
        self.assertEqual(index2, [1,2])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        
        self.assertEqual(cache_manager.free_block_nums, 1)

    def test_allocate_with_reuse(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        # malloc cache item 1
        self.assertEqual(cache_manager.free_block_nums, 3)
        index1 = cache_manager.malloc(2)
        self.assertEqual(index1, [1,2])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        self.assertEqual(cache_manager.cache_item_num, 0)
        
        # insert cache item 1
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        self.assertEqual(cache_manager.cache_item_num, 1)
        
        # malloc cache item 1
        index2, reuse_num = cache_manager.malloc_with_cache(2, [1000,1002])
        self.assertEqual(cache_manager.free_block_nums, 1)
        self.assertEqual(index2, [1,3])
        self.assertEqual(reuse_num, 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        
        # insert cache item 1
        cache_manager.free_with_cache([index2], token_ids=[1000, 1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        
        # malloc cache item 2
        index3, reuse_num = cache_manager.malloc_with_cache(3, [1000, 1002, 1003])
        self.assertEqual(index3, [1,2,3])
        self.assertEqual(reuse_num, 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        self.assertEqual(cache_manager.cache_item_num, 1)
        
        # insert cache item 2
        cache_manager.free_with_cache([index3], token_ids=[1000, 1002, 1003])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        self.assertEqual(cache_manager.cache_item_num, 2)

    def test_match_max_len(self):
        config = self._init_config()
        config = config._replace(block_nums=100)
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 99)
        
        # malloc cache item 1
        index1, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002])
        self.assertEqual(index1, [1,2])
        self.assertEqual(reuse_num, 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        
        # insert cache item 1
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        
        # malloc cache item 2
        index2, reuse_num = cache_manager.malloc_with_cache(3, token_ids=[1000, 1002, 1003])
        self.assertEqual(index2, [1,3,4])
        self.assertEqual(reuse_num, 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        
        # insert cache item 2
        cache_manager.free_with_cache([index2], [1000, 1002, 1003])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(4), 0)
        
        # malloc cache item 3
        index3, reuse_num = cache_manager.malloc_with_cache(4, [1000, 1002, 1003, 1004])
        self.assertEqual(index3, [1,3,5,6])
        self.assertEqual(reuse_num, 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 3)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(4), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(5), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(6), 1)
        
        # insert cache item 3
        cache_manager.free_with_cache([index3], [1000, 1002, 1003, 1004])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 3)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(4), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(5), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(6), 0)
        
        # trigger match max len cache item
        index4, reuse_num = cache_manager.malloc_with_cache(4, [1000, 1002, 1003, 1004])
        self.assertEqual(index4, [1,3,5,7])
        self.assertEqual(reuse_num, 3)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 4)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 3)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(4), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(5), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(6), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(7), 1)

    # pop no resident cache item
    def test_pop(self):
        config = self._init_config()
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 3)
        
        # malloc cache item 1
        index1 = cache_manager.malloc(2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        
        # insert cache item 1
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        
        # trigger reuse cache, pop cache item 1, malloc from free failed
        with self.assertRaisesRegex(Exception, "failed to malloc 3 blocks, only 2 blocks left"):
            index2, reuse_num = cache_manager.malloc_with_cache(4, [1000,1002,1003,1004])
        self.assertEqual(cache_manager.free_block_nums, 3)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
    
        # trigger malloc block from free failed
        with self.assertRaisesRegex(Exception, "failed to malloc 4 blocks, only 3 blocks left"):
            index2, reuse_num = cache_manager.malloc_with_cache(4, [1000,1002,1003,1004])
    
        # insert cache item 2
        index1, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[100, 1002])
        self.assertEqual(index1, [3,1])
        self.assertEqual(reuse_num, 0)
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertEqual(cache_manager.free_block_nums, 2)
        
        # trigger pop cache item 2 from cache, malloc success
        index2, reuse_num = cache_manager.malloc_with_cache(3, [2000,2002,2003])
        self.assertEqual(index2, [2,3,1])
        self.assertEqual(reuse_num, 0)
        self.assertEqual(cache_manager.free_block_nums, 0)
    
    def test_pop_two_cache(self):
        config = self._init_config()
        config = config._replace(block_nums=7)
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 6)
        
        # insert cache item 1
        index1, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002])
        self.assertEqual(index1, [1,2])
        self.assertEqual(reuse_num, 0)
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertEqual(cache_manager.free_block_nums, 5)
        
        # insert cache item 2
        index1, reuse_num = cache_manager.malloc_with_cache(3, token_ids=[2000,2002,2003])
        self.assertEqual(index1, [3,4,5])
        self.assertEqual(reuse_num, 0)
        cache_manager.free_with_cache([index1], token_ids=[2000,2002,2003])
        self.assertEqual(cache_manager.free_block_nums, 3)
        self.assertTrue(cache_manager.block_cache.has_key([2000,2002]))

        # malloc cache item 3 lead to pop cache item 2
        index1, reuse_num = cache_manager.malloc_with_cache(5, token_ids=[1000,1002,1003,1004,1005])
        self.assertEqual(index1, [1,6,4,2,3])
        self.assertEqual(reuse_num, 1)
        # cache item 1 is in cache
        self.assertTrue(cache_manager.block_cache.has_key([1000]))
        # cache item 2 is not in cache
        self.assertFalse(cache_manager.block_cache.has_key([2000,2002]))
        
    def test_pop_with_resident(self):
        config = self._init_config()
        config = config._replace(block_nums=6)
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 5)
        
        # insert resident cache item
        index1 = cache_manager.malloc(2)
        self.assertEqual(index1, [1,2])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        cache_manager.insert_resident_cache(index1, token_ids=[1000, 1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        
        # insert cache item 2
        index1, reuse_num = cache_manager.malloc_with_cache(3, token_ids=[2000,2002,2003])
        self.assertEqual(index1, [3,4,5])
        self.assertEqual(reuse_num, 0)
        cache_manager.free_with_cache([index1], token_ids=[2000,2002,2003])
        self.assertEqual(cache_manager.free_block_nums, 2)
        self.assertTrue(cache_manager.block_cache.has_key([2000,2002]))
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        
        # malloc cache item 3 lead to pop cache item 2
        with self.assertRaisesRegex(Exception, "failed to malloc 3 blocks, only 2 blocks left"):
            index1, reuse_num = cache_manager.malloc_with_cache(5, token_ids=[2000,2002,2003,2004,2005])
        # cache item 1 is in cache
        self.assertTrue(cache_manager.block_cache.has_key([1000]))
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        # cache item 2 is not in cache
        self.assertFalse(cache_manager.block_cache.has_key([2000,2002]))
        
        
    def test_resident(self):
        config = self._init_config()
        config = config._replace(block_nums=100)
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 99)
        
        # malloc for resident block
        index1 = cache_manager.malloc(2)
        self.assertEqual(index1, [1,2])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        # insert resident cache item
        cache_manager.insert_resident_cache(index1, token_ids=[1000, 1002])
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        
        # put not pop resident cache item
        index2, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002])
        self.assertEqual(index2, [1,3])
        self.assertEqual(reuse_num, 1)
        cache_manager.free_with_cache([index2], token_ids=[1000, 1002])
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        
        # match resident cache item
        index2, reuse_num = cache_manager.malloc_with_cache(3, token_ids=[1000, 1002, 1003])
        self.assertEqual(index2, [1,4,5])
        self.assertEqual(reuse_num, 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 0)

        # put not pop resident cache item
        cache_manager.free_with_cache([index2], token_ids=[1000, 1002, 1003])
        self.assertTrue(cache_manager.block_cache.is_resident([1000]))
        
        # not match
        index3, reuse_num = cache_manager.malloc_with_cache(3, token_ids=[2000, 2002, 2003])
        self.assertEqual(index3, [6,7,8])
        self.assertEqual(reuse_num, 0)

    def test_seq_size_per_block(self):
        config = self._init_config(seq_size_per_block=2)
        config = config._replace(block_nums=100)
        cache_manager = CacheManager(config, None)

        self.assertEqual(cache_manager.free_block_nums, 99)
        
        # malloc cache item 1
        index1, reuse_num = cache_manager.malloc_with_cache(1, token_ids=[1000, 1002])
        self.assertEqual(index1, [1])
        self.assertEqual(reuse_num, 0)
        # insert cache item 1
        cache_manager.free_with_cache([index1], token_ids=[1000,1002])
        self.assertFalse(cache_manager.block_cache.has_key([1000,1002]))
        
        # malloc cache item 2
        index2, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002, 1003])
        self.assertEqual(index2, [2,3])
        self.assertEqual(reuse_num, 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 1)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 1)
        cache_manager.free_with_cache([index2], token_ids=[1000,1002,1003])
        self.assertTrue(cache_manager.block_cache.has_key([1000,1002]))
        
        index3, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002, 1003, 1004])
        self.assertEqual(index3, [2,4])
        self.assertEqual(reuse_num, 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(1), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(2), 2)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(3), 0)
        self.assertEqual(cache_manager.block_ref_counter.get_ref_counter(4), 1)
        
        cache_manager.free_with_cache([index3], token_ids=[1000,1002,1003,1004,1005])
        index4, reuse_num = cache_manager.malloc_with_cache(2, token_ids=[1000, 1002, 1003, 1004, 1005])
        self.assertEqual(index4, [2,4])
        self.assertEqual(reuse_num, 4)
            
    def test_lack_mem(self):
        with self.assertRaises(AssertionError):
            config = self._init_config(reserve_runtime_mem_mb=1)
            cache_manager = CacheManager(config, None)

if __name__ == '__main__':
    main()
