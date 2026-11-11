"""Serve verified RTP tokens to rLLM and serialize on-policy refits."""

import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import struct

from q38_driver_recipe import MAX_PROMPT, MAX_RESPONSE, PROMPTS, ROLLOUTS, STEPS


async def _ray_refit(worker, method, *args):
    return await worker.q38_refit.remote(method, *args)


def _write_json(path, payload):
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class NativeCoordinator:
    """Own one native incarnation, its admission gate, and the live refit leases."""

    def __init__(self, topology, incarnation, metadata, shape_manifest, journal_dir):
        from q38_native32_control import Native32Policy
        from rtp_request_journal import NativeRpcJournal

        self.evidence = Path(journal_dir)
        self.evidence.mkdir(exist_ok=False)
        eos = json.loads((Path(metadata) / "generation_config.json").read_text())["eos_token_id"]
        eos = [eos] if type(eos) is int else eos
        if not eos or any(type(x) is not int or not 0 <= x < 248320 for x in eos):
            raise ValueError("the official generation configuration lacks valid EOS tokens")
        self.incarnation = incarnation
        self.step = -1
        self.version = None
        self.accepting = False
        self.poisoned = False
        self.active = 0
        self.next_request_id = 1
        self.generations = Counter()
        self.receipts = []
        self.stream = None
        self.changed = []
        self.condition = asyncio.Condition()
        self.lifecycle = asyncio.Lock()
        journals = [NativeRpcJournal(self.evidence, rank, self._journal_state) for rank in range(4)]
        self.policy = Native32Policy.from_runtime(
            topology, incarnation, metadata, shape_manifest, eos, request_journals=journals,
        )

    def _journal_state(self):
        return {"native_incarnation": self.incarnation, "policy_step": self.step}

    async def ownership(self):
        import ray

        context = ray.get_runtime_context()
        return {"ip": ray.util.get_node_ip_address(), "node_id": context.get_node_id(),
                "actor_id": context.get_actor_id(), "pid": os.getpid(),
                "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                "native_incarnation": self.incarnation}

    async def bind(self, trainers, relays, source_shapes, rollout_uuids):
        from q38_refit_stream import SplitRefitStream

        if self.stream is not None:
            raise RuntimeError("the native refit stream is already bound")
        await self.policy.bootstrap_status()
        self.stream = SplitRefitStream(
            trainers, relays, self.incarnation, source_shapes, rollout_uuids, call=_ray_refit,
        )
        return len(self.policy.sources)

    async def sleep(self):
        async with self.condition:
            self.accepting = False
            await asyncio.wait_for(self.condition.wait_for(lambda: self.active == 0), timeout=900)

    async def refit(self, step):
        async with self.lifecycle:
            if self.poisoned or self.stream is None or type(step) is not int or step != self.step + 1 or step > STEPS:
                raise RuntimeError("split refit received an invalid step or held transaction")
            try:
                await self.sleep()
                if step > 0 and self.generations[step - 1] < PROMPTS * ROLLOUTS:
                    raise RuntimeError("preceding on-policy generation evidence is incomplete")
                receipt = await self.policy.refit(step, self.stream.leases(step, self.policy.control))
                digests = receipt["source_digests"]
                expected_version = f"{self.incarnation}:{step + 1}"
                if (receipt["model_version"] != expected_version or receipt["global_step"] != step
                        or receipt["native_ranks"] != list(range(32)) or receipt["logical_policy_instances"] != 1
                        or receipt["received_sources"] != 1590 or set(digests) != self.policy.sources):
                    raise RuntimeError("the all-rank native refit receipt does not match the committed policy")
                changed = len(digests) if step == 0 else sum(
                    digests[name] != self.receipts[-1]["source_digests"][name] for name in digests
                )
                if changed == 0:
                    raise RuntimeError("Adam produced no BF16 source changes in the new native policy")
                _write_json(self.evidence / f"refit-step{step}.json", dict(receipt, changed_sources=changed))
                self.receipts.append(receipt)
                self.changed.append(changed)
                async with self.condition:
                    self.step = step
                    self.version = expected_version
                    self.accepting = step < STEPS
                return {"global_step": step, "native_version": expected_version, "changed_sources": changed}
            except BaseException:
                self.poisoned = True
                raise

    async def generate(self, prompt_ids, params):
        async with self.condition:
            if self.poisoned or not self.accepting:
                raise RuntimeError("rollout is closed until a fully committed policy is admitted")
            step, version = self.step, self.version
            request_id = self.next_request_id
            self.next_request_id += 1
            self.active += 1
        try:
            result = await self.policy.generate(prompt_ids, params, request_id=request_id)
            if (result.global_step != step or result.model_version != version
                    or not result.token_ids or result.log_probs is None
                    or len(result.log_probs) != len(result.token_ids)
                    or result.logprobs_mode != "raw_logprobs"
                    or any(not math.isfinite(score) or score > 0 for score in result.log_probs)):
                raise RuntimeError("native generation did not carry finite FP32 scores for the admitted policy")
            bits = [struct.pack("<f", value).hex() for value in result.log_probs]
            self.generations[step] += 1
            return {"result": result, "raw_fp32_le": bits, "native_request_id": request_id}
        except BaseException:
            self.poisoned = True
            raise
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()

    async def finish(self):
        async with self.lifecycle:
            await self.sleep()
            if (self.poisoned or self.step != STEPS or len(self.receipts) != STEPS + 1
                    or set(self.generations) != set(range(STEPS))):
                raise RuntimeError("split policy did not complete two fresh on-policy updates")
            await self.stream.close()
            await asyncio.gather(*(client.close() for client in self.policy.clients))
            await self.policy.control.close()
            summary = {
                "native_incarnation": self.incarnation,
                "versions": [r["model_version"] for r in self.receipts],
                "changed_sources": self.changed,
                "native_requests_by_policy_step": dict(self.generations),
            }
            _write_json(self.evidence / "NATIVE-TRAINING-COMPLETE.json", summary)
            return summary


class NativeTokenClient:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None,
                       video_data=None, audio_data=None, **kwargs):
        from rtp_llm.RL.token_rpc_client import TokenSamplingParams
        from verl.workers.rollout.replica import TokenOutput

        if image_data or video_data or audio_data:
            raise ValueError("the Qwen3.8 policy only supports the official text workload")
        if not prompt_ids or len(prompt_ids) > MAX_PROMPT + MAX_RESPONSE:
            raise ValueError("invalid text rollout prompt budget")
        params = dict(sampling_params)
        maximum = params.get("max_tokens")
        if type(maximum) is not int or not 1 <= maximum <= MAX_RESPONSE:
            raise ValueError("the rLLM gateway must submit an explicit bounded token budget")
        if (params.get("temperature", 1.0) != 1.0 or params.get("top_p", 1.0) != 1.0
                or params.get("top_k", -1) not in (-1, 0) or params.get("min_tokens", 0)
                or params.get("ignore_eos", False) or params.get("stop")
                or params.get("presence_penalty", 0) or params.get("frequency_penalty", 0)
                or params.get("repetition_penalty", 1) != 1 or params.get("logit_bias")):
            raise ValueError("sampling processors would change the on-policy FP32 score semantics")
        sampling = TokenSamplingParams(
            max_new_tokens=maximum, temperature=1.0, top_p=1.0, top_k=0,
            seed=params.get("seed"), stop_token_ids=tuple(params.get("stop_token_ids") or ()),
            logprobs=True, logprobs_mode="raw_logprobs",
        )
        sampling.validate()
        reply = await self.coordinator.generate.remote(list(prompt_ids), sampling)
        result = reply["result"]
        return TokenOutput(
            token_ids=result.token_ids, log_probs=result.log_probs,
            stop_reason="completed", num_preempted=0,
            extra_fields={
                "global_step": result.global_step,
                "global_steps": result.global_step, "min_global_steps": result.global_step,
                "max_global_steps": result.global_step, "model_version": result.model_version,
                "native_request_id": reply["native_request_id"], "native_finish_reason": result.finish_reason,
                "raw_fp32_le": reply["raw_fp32_le"],
            },
        )


class NativeServerManager:
    def __init__(self, coordinator):
        self.client = NativeTokenClient(coordinator)

    def get_client(self):
        return self.client

    def get_addresses(self):
        return []
