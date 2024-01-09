from unittest import TestCase, main
from maga_transformer.async_decoder_engine.query_manager import QueryManager
from maga_transformer.async_decoder_engine.ptuning import PrefixParams, PrefixType
from maga_transformer.async_decoder_engine.cache_manager import CacheConfigGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from unittest import mock
import torch
import logging

class MockMemInfo:
    free: int  = 32 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.async_decoder_engine.cache_manager.get_mem_info', MockMemInfo)
class QueryManagerTest(TestCase):
    @staticmethod
    def _init_config(head_num_kv: int = 2,
                     size_per_head: int = 8,
                     seq_size_per_block: int = 8,
                     layer_num: int = 8,
                     data_type: str = 'fp16',
                     reserve_runtime_mem_mb: int = 0):
        config = GptInitModelParameters(head_num=head_num_kv,
                                        head_num_kv=head_num_kv,
                                        size_per_head=size_per_head,
                                        layer_num=layer_num,
                                        max_seq_len=128,
                                        seq_size_per_block=seq_size_per_block,
                                        data_type=data_type,
                                        reserve_runtime_mem_mb=reserve_runtime_mem_mb,
                                        vocab_size=0)
        cache_config = CacheConfigGenerator.create_config(config)
        return config, cache_config
    
    def _get_batch_query(self, query_manager: QueryManager):    
        batch_query = query_manager.get_batch_request()
        batch_query.generate()
        return batch_query

    def test_simple(self):
        config, cache_config = self._init_config()
        query_manager = QueryManager(config, cache_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)
        self.assertFalse(query_manager.has_query())
        inputs = torch.IntTensor([[1,2,3,4,5,6,7,8], [4,5,0,0,0,0,0,0]])
        context_lengths = torch.IntTensor([8,2])
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        images = [[]] * 2

        queries = query_manager.put_requests_to_queue(
            inputs, None, context_lengths, images, generate_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 5)
        self.assertEqual(query_manager.running_batch_size(), 0)
        self.assertEqual(query_manager.wait_query_size(), 2)        
        batch_query = self._get_batch_query(query_manager)
        self.assertEqual(query_manager.running_batch_size(), 2)
        self.assertEqual(query_manager.wait_query_size(), 0)
        finished = torch.BoolTensor([False, True])
        new_tokens = [[1,2,3,4,5,6,7,8,4], [4,5,0,0,0,0,0,0,6]]
        hidden_states = torch.zeros((2, 32), dtype=torch.float32)
        logits = torch.zeros((2, 32), dtype=torch.float32)
        query_manager.batch_query_.finished = finished
        query_manager.batch_query_.hidden_states = hidden_states
        query_manager.batch_query_.logits = logits
        query_manager.batch_query_.update_length = [1, 1]
        query_manager.batch_query_.updated_token_ids = torch.IntTensor(new_tokens)
        query_manager.batch_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        query_manager.update_batch_query()
        self.assertEqual(queries[0].finish, False)
        self.assertEqual(queries[0].block_indice, [[1,3]])
        self.assertEqual(queries[0].error_info, '')
        self.assertEqual(queries[0].seq_length, 9)
        self.assertEqual(queries[0].output_token_ids.numpy().tolist(), [[1,2,3,4,5,6,7,8,4]])
        self.assertEqual(queries[1].finish, True)
        self.assertEqual(queries[1].block_indice, [])
        self.assertEqual(queries[1].error_info, '')
        self.assertEqual(queries[1].seq_length, 3)
        self.assertEqual(queries[1].output_token_ids.numpy().tolist(), [[4,5,6]])

        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 5)
        batch_query = self._get_batch_query(query_manager)
        self.assertEqual(query_manager.running_batch_size(), 1)
        self.assertEqual(query_manager.wait_query_size(), 0)
        [query_manager._release_query_resource(x) for x in queries]
        self.assertEqual(queries[0].block_indice, [])
        self.assertEqual(queries[1].block_indice, [])
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)

    def test_put_lack_mem(self):
        config, cache_config = self._init_config()
        query_manager = QueryManager(config, cache_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)
        self.assertFalse(query_manager.has_query())
        inputs = torch.IntTensor([list(range(64))] * 3)
        context_lengths = torch.IntTensor([33, 8, 9])
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        images = [[]] * 3

        with self.assertRaisesRegex(Exception, "failed to malloc 2 blocks, only 1 blocks left"):
            queries = query_manager.put_requests_to_queue(
                inputs, None, context_lengths, images,generate_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)

    def test_extend_lack_mem(self):
        config, cache_config = self._init_config()
        query_manager = QueryManager(config, cache_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)
        self.assertFalse(query_manager.has_query())
        inputs = torch.IntTensor([list(range(64))] * 3)
        context_lengths = torch.IntTensor([32, 4, 16])
        images = [[]] * 3
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        queries = query_manager.put_requests_to_queue(
            inputs, None, context_lengths, images, generate_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 0)
        batch_query = self._get_batch_query(query_manager)
        finished = torch.BoolTensor([False, False, False])
        new_tokens = [[1]*32 + [4], [1]*32 + [6], [1]*32 + [4]]
        hidden_states = torch.zeros((3, 32), dtype=torch.float32)
        logits = torch.zeros((3, 32), dtype=torch.float32)
        query_manager.batch_query_.finished = finished
        query_manager.batch_query_.hidden_states = hidden_states
        query_manager.batch_query_.logits = logits
        query_manager.batch_query_.update_length = [1, 1, 1]
        query_manager.batch_query_.updated_token_ids = torch.IntTensor(new_tokens)
        query_manager.batch_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        query_manager.update_batch_query()
        self.assertEqual(queries[0].error_info, 'LACK_MEM')
        self.assertEqual(queries[2].error_info, '')
        self.assertEqual(queries[0].finish, True)
        self.assertEqual(queries[1].finish, False)
        self.assertEqual(queries[2].finish, False)
        self.assertEqual(len(queries[1].block_indice), 1)
        self.assertEqual(len(query_manager.batch_query_.queries), 2)

    @mock.patch.dict('os.environ', {'USE_BLOCK_CACHE': '1'})
    def test_reuse(self):
        config, cache_config = self._init_config()
        query_manager = QueryManager(config, cache_config)
        self.assertTrue(query_manager.use_cache_)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 7)
        self.assertFalse(query_manager.has_query())
        inputs = torch.IntTensor([list(range(64))] * 1)
        context_lengths = torch.IntTensor([16])
        images = [[]]
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False, chat_id='aaaa')
        queries = query_manager.put_requests_to_queue(inputs, None, context_lengths, images, generate_config)
        self.assertEqual(queries[0].block_indice, [[1, 2]])
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 5)
        batch_query = self._get_batch_query(query_manager)
        finished = torch.BoolTensor([True])
        new_tokens = [list(range(64)) + [6]]
        hidden_states = torch.zeros((1, 32), dtype=torch.float32)
        logits = torch.zeros((1, 32), dtype=torch.float32)
        query_manager.batch_query_.finished = finished
        query_manager.batch_query_.hidden_states = hidden_states
        query_manager.batch_query_.logits = logits
        query_manager.batch_query_.update_length = [1]
        query_manager.batch_query_.updated_token_ids = torch.IntTensor(new_tokens)
        query_manager.batch_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        query_manager.update_batch_query()        
        self.assertEqual(queries[0].block_indice, [])
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 5)
        self.assertEqual(len(query_manager.batch_query_.queries), 0)

        context_lengths = torch.IntTensor([32])
        queries = query_manager.put_requests_to_queue(inputs, None, context_lengths, images, generate_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 3)
        self.assertEqual(queries[0].block_indice, [[1, 2, 3, 4]])
        self.assertEqual(queries[0].reuse_length, 16)
        batch_query = self._get_batch_query(query_manager)
        query_manager.batch_query_.finished = finished
        query_manager.batch_query_.hidden_states = hidden_states
        query_manager.batch_query_.logits = logits
        query_manager.batch_query_.update_length = [1]
        query_manager.batch_query_.updated_token_ids = torch.IntTensor(new_tokens)
        query_manager.batch_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        query_manager.update_batch_query()
        self.assertEqual(len(query_manager.batch_query_.queries), 0)

        context_lengths = torch.IntTensor([24])
        queries = query_manager.put_requests_to_queue(inputs, None, context_lengths, images, generate_config)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 4)
        self.assertEqual(queries[0].block_indice, [[1, 2, 3]])
        self.assertEqual(queries[0].reuse_length, 23)
        batch_query = self._get_batch_query(query_manager)
        prefix_lengths, count_length, max_prefix_length = query_manager.get_prefix_args(batch_query)
        self.assertEqual(prefix_lengths.numpy().tolist(), [23])
        self.assertEqual(count_length.numpy().tolist(), [True])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

        finished = torch.BoolTensor([False])
        query_manager.batch_query_.finished = finished
        query_manager.batch_query_.hidden_states = hidden_states
        query_manager.batch_query_.logits = logits
        query_manager.batch_query_.update_length = [1]
        query_manager.batch_query_.updated_token_ids = torch.IntTensor(new_tokens)
        query_manager.batch_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        query_manager.update_batch_query()
        self.assertEqual(queries[0].block_indice, [[1, 2, 3, 5]])

    @mock.patch.dict('os.environ', {'USE_BLOCK_CACHE': '1'})
    def test_ptuning(self):
        config, cache_config = self._init_config()
        prefix_seq_len = 9
        prefix_prompt = torch.zeros((config.layer_num * 2, config.head_num_kv, prefix_seq_len, config.size_per_head), dtype=torch.float16, device="cuda:0")
        prefix_param = PrefixParams(prefix_prompt, PrefixType.PTuningV2, None)
        query_manager = QueryManager(config, cache_config, prefix_param)
        self.assertEqual(len(query_manager.cache_manager_.free_blocks_index), 5)
        self.assertFalse(query_manager.use_cache_)
        self.assertEqual(query_manager.ptuning_.prefix_block_indice, [1])
        self.assertEqual(query_manager.ptuning_.prefix_additional_block, 2)

        self.assertFalse(query_manager.has_query())
        inputs = torch.IntTensor([list(range(64))] * 2)
        context_lengths = torch.IntTensor([8, 9])
        images = [[]] * 2
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        queries = query_manager.put_requests_to_queue(
                inputs, None, context_lengths, images, generate_config)
        batch_query = self._get_batch_query(query_manager)
        prefix_lengths, count_length, max_prefix_length = query_manager.get_prefix_args(batch_query)

        self.assertEqual(prefix_lengths.numpy().tolist(), [9, 9])
        self.assertEqual(count_length.numpy().tolist(), [False])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

if __name__ == '__main__':
    main()
