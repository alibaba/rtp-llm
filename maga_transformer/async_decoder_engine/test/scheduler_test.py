from unittest import mock
import torch
import logging
from unittest import TestCase, main
from maga_transformer.async_decoder_engine.scheduler import Scheduler
from maga_transformer.async_decoder_engine.ptuning import PrefixParams, PrefixType
from maga_transformer.async_decoder_engine.cache_manager import CacheConfigGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.model_weight import LoraResource
from maga_transformer.structure.raw_query import RawQuery


class MockMemInfo:
    free: int  = 32 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
class SchedulerTest(TestCase):
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
    
    def _get_batch_query(self, scheduler: Scheduler):    
        batch_query = scheduler.get_batch_request()
        batch_query.generate()
        return batch_query

    def test_simple(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.has_query())
        inputs = torch.IntTensor([[1,2,3,4,5,6,7,8], [4,5,0,0,0,0,0,0]])
        context_lengths = torch.IntTensor([8,2])
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        images = [[]] * 2

        queries = scheduler.enqueue(
            RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertEqual(scheduler.running_batch_size(), 2)
        self.assertEqual(scheduler.wait_query_size(), 0)
        self.assertEqual(scheduler.running_batch_size(), 2)
        self.assertEqual(scheduler.wait_query_size(), 0)
        finished = torch.BoolTensor([False, True])
        new_tokens = [[1,2,3,4,5,6,7,8,4], [4,5,0,0,0,0,0,0,6]]
        hidden_states = torch.zeros((2, 32), dtype=torch.float32)
        logits = torch.zeros((2, 32), dtype=torch.float32)
        scheduler.running_query_.finished = finished
        scheduler.running_query_.hidden_states = hidden_states
        scheduler.running_query_.logits = logits
        scheduler.running_query_.update_length = [1, 1]
        scheduler.running_query_.updated_token_ids = torch.IntTensor(new_tokens)
        scheduler.running_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        scheduler.update_batch_query()
        self.assertEqual(queries[0].finish, False)
        self.assertEqual(queries[0].block_indice, [[1,3]])
        self.assertEqual(queries[0].error_info, '')
        self.assertEqual(queries[0].seq_length, 9)
        self.assertEqual(queries[0].output_token_ids.numpy().tolist(), [[1,2,3,4,5,6,7,8,4]])
        self.assertEqual(queries[1].finish, True)
        self.assertEqual(queries[1].block_indice, [[]])
        self.assertEqual(queries[1].error_info, '')
        self.assertEqual(queries[1].seq_length, 3)
        self.assertEqual(queries[1].output_token_ids.numpy().tolist(), [[4,5,6]])

        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(scheduler.running_batch_size(), 1)
        self.assertEqual(scheduler.wait_query_size(), 0)
        [scheduler._release_query_resource(x) for x in queries]
        self.assertEqual(queries[0].block_indice, [[]])
        self.assertEqual(queries[1].block_indice, [[]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)

    def test_put_lack_mem(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.has_query())
        inputs = torch.IntTensor([list(range(64))])
        context_lengths = torch.IntTensor([57])
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        images = [[]]

        queries = scheduler.put_requests_to_queue(
            RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        scheduler.get_batch_request()
        self.assertEqual(queries[0].error_info, "failed to malloc 8 blocks, only 7 blocks left")
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)

    def test_extend_lack_mem(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.has_query())
        inputs = torch.IntTensor([list(range(64))] * 3)
        context_lengths = torch.IntTensor([32, 4, 16])
        images = [[]] * 3
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        queries = scheduler.enqueue(
            RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 0)
        finished = torch.BoolTensor([False, False, False])
        new_tokens = [[1]*32 + [4], [1]*32 + [6], [1]*32 + [4]]
        hidden_states = torch.zeros((3, 32), dtype=torch.float32)
        logits = torch.zeros((3, 32), dtype=torch.float32)
        scheduler.running_query_.finished = finished
        scheduler.running_query_.hidden_states = hidden_states
        scheduler.running_query_.logits = logits
        scheduler.running_query_.update_length = [1, 1, 1]
        scheduler.running_query_.updated_token_ids = torch.IntTensor(new_tokens)
        scheduler.running_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        scheduler.update_batch_query()
        self.assertEqual(queries[0].error_info, 'LACK_MEM')
        self.assertEqual(queries[2].error_info, '')
        self.assertEqual(queries[0].finish, True)
        self.assertEqual(queries[1].finish, False)
        self.assertEqual(queries[2].finish, False)
        self.assertEqual(len(queries[1].block_indice), 1)
        self.assertEqual(len(scheduler.running_query_.queries), 2)

    @mock.patch.dict('os.environ', {'REUSE_CACHE': '1'})
    def test_reuse(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertTrue(scheduler.reuse_cache_)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.has_query())
        inputs = torch.IntTensor([list(range(64))] * 1)
        context_lengths = torch.IntTensor([16])
        images = [[]]
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False, chat_id='aaaa')
        queries = scheduler.enqueue(RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(queries[0].block_indice, [[1, 2]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        finished = torch.BoolTensor([True])
        new_tokens = [list(range(64)) + [6]]
        hidden_states = torch.zeros((1, 32), dtype=torch.float32)
        logits = torch.zeros((1, 32), dtype=torch.float32)
        scheduler.running_query_.finished = finished
        scheduler.running_query_.hidden_states = hidden_states
        scheduler.running_query_.logits = logits
        scheduler.running_query_.update_length = [1]
        scheduler.running_query_.updated_token_ids = torch.IntTensor(new_tokens)
        scheduler.running_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        scheduler.update_batch_query()        
        self.assertEqual(queries[0].block_indice, [[]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertEqual(len(scheduler.running_query_.queries), 0)

        context_lengths = torch.IntTensor([32])
        queries = scheduler.enqueue(RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 3)
        self.assertEqual(queries[0].block_indice, [[1, 2, 3, 4]])
        self.assertEqual(queries[0].reuse_length, 16)
        scheduler.running_query_.finished = finished
        scheduler.running_query_.hidden_states = hidden_states
        scheduler.running_query_.logits = logits
        scheduler.running_query_.update_length = [1]
        scheduler.running_query_.updated_token_ids = torch.IntTensor(new_tokens)
        scheduler.running_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        scheduler.update_batch_query()
        self.assertEqual(len(scheduler.running_query_.queries), 0)

        context_lengths = torch.IntTensor([24])
        queries = scheduler.enqueue(RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 4)
        self.assertEqual(queries[0].block_indice, [[1, 2, 3]])
        self.assertEqual(queries[0].reuse_length, 23)
        prefix_lengths, count_length, max_prefix_length = scheduler.get_prefix_args(batch_query)
        self.assertEqual(prefix_lengths.numpy().tolist(), [23])
        self.assertEqual(count_length.numpy().tolist(), [True])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

        finished = torch.BoolTensor([False])
        scheduler.running_query_.finished = finished
        scheduler.running_query_.hidden_states = hidden_states
        scheduler.running_query_.logits = logits
        scheduler.running_query_.update_length = [1]
        scheduler.running_query_.updated_token_ids = torch.IntTensor(new_tokens)
        scheduler.running_query_.cum_log_probs = torch.Tensor([0.0, 0.0, 0.0])
        scheduler.update_batch_query()
        self.assertEqual(queries[0].block_indice, [[1, 2, 3, 5]])

    @mock.patch.dict('os.environ', {'REUSE_CACHE': '1'})
    def test_ptuning(self):
        config, cache_config = self._init_config()
        prefix_seq_len = 9
        prefix_prompt = torch.zeros((config.layer_num * 2, config.head_num_kv, prefix_seq_len, config.size_per_head), dtype=torch.float16, device="cuda:0")
        prefix_param = PrefixParams(prefix_prompt, PrefixType.PTuningV2, None)
        scheduler = Scheduler(config, cache_config, prefix_param)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertFalse(scheduler.reuse_cache_)
        self.assertEqual(scheduler.ptuning_.prefix_block_indice, [1])
        self.assertEqual(scheduler.ptuning_.prefix_additional_block, 2)

        self.assertFalse(scheduler.has_query())
        inputs = torch.IntTensor([list(range(64))] * 2)
        context_lengths = torch.IntTensor([8, 9])
        images = [[]] * 2
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        queries = scheduler.enqueue(
                RawQuery(inputs, context_lengths, images, generate_config, None), LoraResource(dict(), None, None, None))
        batch_query = self._get_batch_query(scheduler)
        prefix_lengths, count_length, max_prefix_length = scheduler.get_prefix_args(batch_query)

        self.assertEqual(prefix_lengths.numpy().tolist(), [9, 9])
        self.assertEqual(count_length.numpy().tolist(), [False])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

if __name__ == '__main__':
    main()
