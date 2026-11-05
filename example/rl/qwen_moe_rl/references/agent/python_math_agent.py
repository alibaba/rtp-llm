"""Original Qwen3-4B math AgentFlow with a bounded Python/SymPy tool."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from openai import AsyncOpenAI
import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory


TOOLS = [{"type": "function", "function": {
    "name": "python", "description": (
        "Run bounded mathematical Python with math and SymPy. Imports of math or sympy, "
        "variables, loops, lists and mathematical functions are supported. Use print() "
        "to display results. No network, files, subprocesses, or other modules. Each call "
        "starts a fresh process; repeat required definitions. Prefer symbols/Integer/Rational "
        "to preserve exactness."
    ), "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}}]

SYSTEM_PROMPT = """Solve the competition mathematics problem carefully. You may call the
python tool for exact arithmetic, symbolic algebra, enumeration, or checking a conjecture.
The tool has math and SymPy, and prints its results; each call is independent.
Only use tools when helpful. After checking your reasoning, give the final answer in
\\boxed{ANSWER}. Do not put a full solution or multiple candidate answers inside the box.
"""


def execute_python(code: str, timeout_s: float = 12.0) -> dict:
    """Run a fresh child with no inherited credentials or network capability."""
    worker = Path(__file__).with_name("sandbox_worker.py").resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rllm-python-tool-") as directory:
        clean_env = {
            "HOME": directory, "TMPDIR": directory, "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
        }
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=directory, env=clean_env, close_fds=True, start_new_session=True,
            )
            process.stdin.write(json.dumps({"code": code}).encode())
            process.stdin.close()
            streams = selectors.DefaultSelector()
            streams.register(process.stdout, selectors.EVENT_READ, "stdout")
            streams.register(process.stderr, selectors.EVENT_READ, "stderr")
            output = {"stdout": bytearray(), "stderr": bytearray()}
            deadline = time.monotonic() + timeout_s
            while streams.get_map():
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, timeout_s)
                for key, _ in streams.select(timeout=min(.2, max(0, deadline - time.monotonic()))):
                    chunk = os.read(key.fileobj.fileno(), 16384)
                    if not chunk:
                        streams.unregister(key.fileobj)
                        continue
                    output[key.data].extend(chunk)
                    if sum(map(len, output.values())) > 65536:
                        raise RuntimeError("Python tool exceeded the 64 KiB subprocess-output limit")
            process.wait(timeout=max(.1, deadline - time.monotonic()))
            if process.returncode != 0:
                stderr = output["stderr"].decode(errors="replace")
                return {"ok": False, "stdout": "", "error": f"worker exited {process.returncode}: {stderr[-1500:]}", "elapsed_s": time.monotonic() - started}
            return json.loads(output["stdout"])
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": "Python tool wall-time limit exceeded", "elapsed_s": time.monotonic() - started}
        except Exception as exc:
            return {"ok": False, "stdout": "", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.monotonic() - started}
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                for stream in (process.stdout, process.stderr, process.stdin):
                    if stream is not None:
                        stream.close()


def extract_boxed(text: str) -> str:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return ""


@rllm.rollout(name="python-math")
async def python_math_agent(task: Task, config: AgentConfig) -> Episode:
    request_timeout = float(os.environ.get('RLLM_PYTHON_TOOL_HTTP_TIMEOUT', '180'))
    http_max_retries = int(os.environ.get('RLLM_PYTHON_TOOL_HTTP_MAX_RETRIES', '1'))
    if not 0 < request_timeout < float('inf') or http_max_retries < 0:
        raise ValueError('Python Tool HTTP timeout must be positive/finite and HTTP retries nonnegative')
    max_turns = int(os.environ.get("RLLM_PYTHON_TOOL_MAX_TURNS", "4"))
    call_tokens = int(os.environ.get("RLLM_PYTHON_TOOL_CALL_TOKENS", "2048"))
    total_tokens = int(os.environ.get("RLLM_PYTHON_TOOL_TOTAL_TOKENS", "8192"))
    thinking = os.environ.get("RLLM_PYTHON_TOOL_ENABLE_THINKING", "0") == "1"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.instruction}]
    steps, tool_records = [], []
    final_content, used_tokens = "", 0
    async with AsyncOpenAI(base_url=config.base_url, api_key="EMPTY", max_retries=http_max_retries) as client:
        for turn in range(max_turns):
            remaining = total_tokens - used_tokens
            if remaining < 64:
                break
            response = await client.chat.completions.create(
                model=config.model, messages=messages, tools=TOOLS,
                tool_choice="auto", max_tokens=min(call_tokens, remaining),
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                timeout=request_timeout,
            )
            if not response.choices:
                raise RuntimeError("Model returned no choices")
            message = response.choices[0].message
            content = message.content or ""
            used_tokens += response.usage.completion_tokens if response.usage else 0
            assistant = {"role": "assistant", "content": content}
            if message.tool_calls:
                assistant["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
            messages.append(assistant)
            steps.append(Step(chat_completions=[dict(m) for m in messages], model_response=content, action=content))
            if not message.tool_calls:
                final_content = content
                break
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    code = str(arguments.get("code", ""))
                    if call.function.name != "python":
                        result = {"ok": False, "error": "Unknown tool", "stdout": ""}
                    else:
                        result = await asyncio.to_thread(execute_python, code)
                except Exception as exc:
                    code = ""
                    result = {"ok": False, "error": f"Invalid tool arguments: {exc}", "stdout": ""}
                tool_records.append({"turn": turn, "tool_call_id": call.id, "code": code, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    answer = extract_boxed(final_content)
    return Episode(
        trajectories=[Trajectory(name="python-math", steps=steps)],
        artifacts={"answer": answer, "final_content": final_content, "python_tools": tool_records, "thinking_enabled": thinking},
        metrics={"python_tool_calls": float(len(tool_records)), "python_tool_successes": float(sum(bool(r["result"].get("ok")) for r in tool_records)), "agent_completion_tokens": float(used_tokens)},
    )


class PythonMathEvaluator:
    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy

        answer = str(episode.artifacts.get("answer", ""))
        truth = str(task.metadata.get("ground_truth", task.metadata.get("answer", "")))
        correct = bool(answer and truth and (grade_answer_mathd(answer, truth) or grade_answer_sympy(answer, truth)))
        trace_dir = os.environ.get("RLLM_PYTHON_TOOL_TRACE_DIR")
        if trace_dir:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # The upstream EpisodeLogger omits artifacts and token arrays. Save
            # an independent complete audit after gateway enrichment, before
            # the trainer transforms/pads the trajectories.
            record = {
                "episode_id": episode.id, "task_id": task.id, "task": task.metadata,
                "reward": float(correct), "is_correct": correct,
                "artifacts": episode.artifacts, "metrics": episode.metrics,
                "trajectories": [{"name": trajectory.name, "steps": [step.to_dict() for step in trajectory.steps]} for trajectory in episode.trajectories],
            }
            safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in episode.id)
            (directory / f"enriched_{safe_id}.json").write_text(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        # Tool usage is audited separately. The reward is only final-answer correctness.
        return EvalOutput(reward=float(correct), is_correct=correct, signals=[Signal(name="accuracy", value=float(correct))])


python_math_evaluator = PythonMathEvaluator()
