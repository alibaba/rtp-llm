"""Exercise complete-world policy admission without claiming native execution."""

from types import SimpleNamespace
import unittest
import uuid

from q38_native32_control import Native32Policy, SourceLease
from q38_native32_topology import Native32Topology, expert_map


NAMES = frozenset(f'model.weight.{i}' for i in range(1590))
INCARNATION = 'q38-test-native32'


def topology():
    return Native32Topology(tuple(f'10.0.0.{i+1}' for i in range(8)),
                            tuple(tuple(str(uuid.UUID(int=4*i+j+1)) for j in range(4)) for i in range(8)),
                            39500)


class Control:
    def __init__(self):
        self.phase = 'paused'
        self.version = INCARNATION+':0'
        self.update_id = ''
        self.step = 0
        self.applied = 0
        self.names = []
        self.commits = 0
        self.fail_push = False

    async def status(self):
        return [SimpleNamespace(world_rank=r, phase=self.phase, model_version=self.version,
                                global_step=self.step, update_id=self.update_id,
                                quiescent=self.phase=='paused', received_weights=len(self.names),
                                applied_weights=self.applied, kv_cache_ready=True) for r in range(32)]

    async def pause(self):
        self.phase = 'paused'
        return await self.status()

    async def begin_update(self, *, global_step, max_buffer_bytes):
        self.pending = global_step
        self.update_id = f'{INCARNATION}:{global_step+1}'
        self.phase = 'updating'
        self.names = []

    async def push_weight(self, name, descriptors):
        if self.fail_push:
            raise TimeoutError('injected native ACK timeout')
        self.names.append(name)

    async def commit_update(self):
        self.commits += 1
        self.version, self.update_id = self.update_id, ''
        self.step = self.pending
        self.applied = 2
        self.phase = 'paused'
        return await self.status()

    async def resume(self):
        self.phase = 'running'
        return await self.status()


class Token:
    def __init__(self, control):
        self.control = control

    async def generate(self, request_id, prompt_ids, sampling, *, expected_version):
        if expected_version != self.control.version:
            raise RuntimeError('request is not bound to the committed policy')
        return SimpleNamespace(model_version=self.control.version, global_step=self.control.step,
                               log_probs=[-1.0], token_ids=[request_id])


class Native32Test(unittest.IsolatedAsyncioTestCase):
    def test_one_identity_expert_map_and_eight_distinct_owners(self):
        value = topology()
        self.assertEqual(len(value.control_addresses), 32)
        self.assertEqual(value.generation_addresses, [f'10.0.0.{i}:39501' for i in (1,3,5,7)])
        self.assertEqual(value.node_environment(7, INCARNATION, local_experts=True)['RTP_WORLD_RANK'], '28')
        self.assertEqual(expert_map(), [list(range(512)) for _ in range(92)])
        with self.assertRaises(ValueError):
            Native32Topology(value.hosts, value.gpu_uuids[:-1]+value.gpu_uuids[:1], value.start_port)

    def controller(self):
        control = Control()
        return Native32Policy(topology(), INCARNATION, NAMES, control, [Token(control) for _ in range(4)]), control

    async def test_two_complete_steps_gate_versioned_rollout(self):
        controller, control = self.controller()
        sampling = SimpleNamespace(max_new_tokens=1, logprobs=True, logprobs_mode='raw_logprobs')
        releases = []
        with self.assertRaisesRegex(RuntimeError, 'admission is closed'):
            await controller.generate([1], sampling, request_id=1)
        for step in (0, 1):
            async def leases():
                for name in sorted(NAMES):
                    async def ack():
                        self.assertEqual(control.names[-1], name)
                        releases.append((step,name))
                    yield SourceLease(name, (b'ipc',)*32, ('a'*64,)*32, ack)
            record = await controller.refit(step, leases())
            self.assertEqual(record['received_sources'], 1590)
            self.assertEqual(record['model_version'], f'{INCARNATION}:{step+1}')
            result = await controller.generate([1], sampling, request_id=2)
            self.assertEqual(result.global_step, step)
        self.assertEqual(len(releases), 3180)

    async def test_ack_timeout_keeps_the_exact_source_lease(self):
        controller, control = self.controller()
        control.fail_push = True
        async def fail_release():
            self.fail('uncertain ACK cannot release its source')
        lease = SourceLease(next(iter(NAMES)), (b'ipc',)*32, ('a'*64,)*32, fail_release)
        async def stream():
            yield lease
        with self.assertRaises(TimeoutError):
            await controller.refit(0, stream())
        self.assertIs(controller.held_lease, lease)
        self.assertTrue(controller.poisoned)
        self.assertEqual(control.commits, 0)

    async def test_incomplete_stream_never_commits(self):
        controller, control = self.controller()
        async def stream():
            for _ in ():
                yield _
        with self.assertRaisesRegex(RuntimeError, 'incomplete native policy'):
            await controller.refit(0, stream())
        self.assertEqual(control.commits, 0)
        self.assertTrue(controller.poisoned)


if __name__ == '__main__':
    unittest.main()
