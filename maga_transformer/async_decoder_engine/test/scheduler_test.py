from unittest import mock
import torch
import logging
from unittest import TestCase, main
from maga_transformer.async_decoder_engine.scheduler import Scheduler
from maga_transformer.async_decoder_engine.batch_query import ModelOutput
from maga_transformer.async_decoder_engine.ptuning import PrefixParams, PrefixType
from maga_transformer.async_decoder_engine.cache_manager import CacheConfigGenerator
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.models.base_model import GenerateInput
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.model_weight import LoraResource


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
        streams =  [x for x in scheduler._waiting_streams]
        batch_query = scheduler.schedule()
        for s in streams:
            logging.info(s.stop_reason)
        logging.info(f'{len(scheduler._waiting_streams)}, {len(batch_query.streams)}')
        batch_query.generate_model_input()
        return batch_query

    def test_simple(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.have_streams())
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        stream1 = GenerateStream(GenerateInput(
            token_ids=torch.tensor([1,2,3,4,5,6,7,8]),
            generate_config=generate_config))
        scheduler.enqueue(stream1)
        stream2 = GenerateStream(GenerateInput(
            token_ids=torch.tensor([4,5]),
            generate_config=generate_config))
        scheduler.enqueue(stream2)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertEqual(scheduler.running_batch_size(), 2)
        self.assertEqual(scheduler.wait_stream_size(), 0)
        self.assertEqual(scheduler.running_batch_size(), 2)
        self.assertEqual(scheduler.wait_stream_size(), 0)
        finished = torch.BoolTensor([False, True])
        update_length = [1, 1]
        update_token_ids = torch.tensor([[0] * 8 + [4], [0] * 8 + [6]])
        scheduler.batch_query.model_output = ModelOutput(
            finished=finished, update_length=update_length, update_token_ids=update_token_ids)

        scheduler.prepare_next_step()
        self.assertEqual(stream1.finished, False)
        self.assertEqual(stream1.stop_reason, '')
        self.assertEqual(stream1.seq_length, 9)
        self.assertEqual(stream1.block_indice, [[1,3]])
        self.assertEqual(stream1.output.output_ids.numpy().tolist(), [[4]])
        self.assertEqual(stream2.finished, True)
        self.assertEqual(stream2.stop_reason, '')
        self.assertEqual(stream2.seq_length, 3)
        self.assertEqual(stream2.block_indice, [[]])
        self.assertEqual(stream2.output.output_ids.numpy().tolist(), [[6]])

        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(scheduler.running_batch_size(), 1)
        self.assertEqual(scheduler.wait_stream_size(), 0)
        [s.release_resource() for s in [stream1, stream2]]
        self.assertEqual(stream1.block_indice, [[]])
        self.assertEqual(stream2.block_indice, [[]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)

    def test_put_lack_mem(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.have_streams())
        inputs = torch.IntTensor([list(range(64))])
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        images = [[]]

        stream = GenerateStream(GenerateInput(token_ids=inputs, generate_config=generate_config))
        scheduler.enqueue(stream)
        scheduler.schedule()
        self.assertEqual(stream.stop_reason, "failed to malloc 8 blocks, only 7 blocks left")
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)

    def test_extend_lack_mem(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.have_streams())
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        streams = [
            GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(32))), generate_config=generate_config)),
            GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(4))), generate_config=generate_config)),
            GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(16))), generate_config=generate_config)),
        ]
        for stream in streams:
            scheduler.enqueue(stream)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 0)
        finished = torch.BoolTensor([False, False, False])
        update_length = [1, 1, 1]
        update_token_ids = torch.IntTensor([[1]*32 + [4], [1]*32 + [6], [1]*32 + [4]])

        scheduler.batch_query.model_output = ModelOutput(
            finished=finished, update_length=update_length, update_token_ids=update_token_ids)

        scheduler.prepare_next_step()
        self.assertEqual(streams[0].stop_reason, 'LACK_MEM')
        self.assertEqual(streams[1].stop_reason, '')
        self.assertEqual(streams[2].stop_reason, '')
        self.assertEqual(streams[0].stopped, True)
        self.assertEqual(streams[1].stopped, False)
        self.assertEqual(streams[2].stopped, False)
        self.assertEqual(len(streams[1].block_indice), 1)
        self.assertEqual(len(scheduler.batch_query.streams), 3)

    @mock.patch.dict('os.environ', {'REUSE_CACHE': '1'})
    def test_reuse(self):
        config, cache_config = self._init_config()
        scheduler = Scheduler(config, cache_config)
        self.assertTrue(scheduler._stream_cache_manager.reuse_cache_)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 7)
        self.assertFalse(scheduler.have_streams())
        generate_config: GenerateConfig = GenerateConfig(using_hf_sampling=False, chat_id='aaaa')
        stream = GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(16))), generate_config=generate_config))
        scheduler.enqueue(stream)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(stream.block_indice, [[1, 2]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        finished = torch.BoolTensor([True])
        update_length = [1]
        update_token_ids = torch.IntTensor([list(range(64)) + [6]])

        scheduler.batch_query.model_output = ModelOutput(
            finished=finished, update_length=update_length, update_token_ids=update_token_ids)

        scheduler.prepare_next_step()
        stream.release_resource()
        self.assertEqual(stream.block_indice, [[]])
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertEqual(len(scheduler.batch_query.streams), 0)

        stream = GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(32))), generate_config=generate_config))
        scheduler.enqueue(stream)

        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 3)
        self.assertEqual(stream.block_indice, [[1, 2, 3, 4]])
        self.assertEqual(stream.reuse_length, 16)
        update_length = [1]

        scheduler.batch_query.model_output = ModelOutput(
            finished=finished, update_length=update_length, update_token_ids=update_token_ids)

        scheduler.prepare_next_step()
        stream.release_resource()
        self.assertEqual(len(scheduler.batch_query.streams), 0)
        stream = GenerateStream(GenerateInput(token_ids=torch.tensor(list(range(24))), generate_config=generate_config))
        scheduler.enqueue(stream)
        batch_query = self._get_batch_query(scheduler)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 4)
        self.assertEqual(stream.block_indice, [[1, 2, 3]])
        self.assertEqual(stream.reuse_length, 23)
        prefix_lengths, count_length, max_prefix_length = scheduler._stream_cache_manager.get_prefix_args(batch_query)
        self.assertEqual(prefix_lengths.numpy().tolist(), [23])
        self.assertEqual(count_length.numpy().tolist(), [True])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

        finished = torch.BoolTensor([False])
        update_length = [1]

        scheduler.batch_query.model_output = ModelOutput(
            finished=finished, update_length=update_length, update_token_ids=update_token_ids)

        scheduler.prepare_next_step()
        self.assertEqual(stream.block_indice, [[1, 2, 3, 5]])

    @mock.patch.dict('os.environ', {'REUSE_CACHE': '1'})
    def test_ptuning(self):
        config, cache_config = self._init_config()
        prefix_seq_len = 9
        prefix_prompt = torch.zeros((config.layer_num * 2, config.head_num_kv, prefix_seq_len, config.size_per_head), dtype=torch.float16, device="cuda:0")
        prefix_param = PrefixParams(prefix_prompt, PrefixType.PTuningV2, None)
        scheduler = Scheduler(config, cache_config, prefix_param)
        self.assertEqual(len(scheduler.cache_manager_.free_blocks_index), 5)
        self.assertFalse(scheduler._stream_cache_manager.reuse_cache_)
        self.assertEqual(scheduler._stream_cache_manager.ptuning_.prefix_block_indice, [1])
        self.assertEqual(scheduler._stream_cache_manager.ptuning_.prefix_additional_block, 2)

        self.assertFalse(scheduler.have_streams())
        inputs = torch.tensor(list(range(8)))
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        stream = GenerateStream(GenerateInput(token_ids=inputs, generate_config=generate_config))
        scheduler.enqueue(stream)
        batch_query = self._get_batch_query(scheduler)
        prefix_lengths, count_length, max_prefix_length = scheduler._stream_cache_manager.get_prefix_args(batch_query)

        self.assertEqual(prefix_lengths.numpy().tolist(), [9])
        self.assertEqual(count_length.numpy().tolist(), [False])
        self.assertEqual(max_prefix_length.numpy().tolist(), [0])

if __name__ == '__main__':
    main()
