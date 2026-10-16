"""Validated four-video fixtures and input accounting for the manual E1PD4 test."""

import concurrent.futures
import copy
import hashlib
import json
import re
from collections import Counter
from itertools import groupby
from pathlib import Path

ENCODER_VIDEO_TOTAL_MIN_PIXELS = 2500000
ENCODER_VIDEO_TOTAL_MAX_PIXELS = 36864000


def load_workload(directory, *, omit_max_tokens=False, enable_thinking=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    entries = manifest["requests"]
    if [r["id"] for r in entries] != [1, 2, 3, 4]:
        raise ValueError("fixture must contain request IDs 1 through 4 in order")
    workload = []
    for entry in entries:
        for key, digest_key in (
            ("template", "request_sha256"),
            ("video", "video_sha256"),
        ):
            relative = Path(entry[key])
            path = directory / relative
            # Bazel runfiles legitimately symlink each asset outside this directory.
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("fixture path escapes its directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry[digest_key]:
                raise ValueError("fixture checksum mismatch: " + entry[key])
        body = json.loads((directory / entry["template"]).read_text())
        if body.get("stream") is not False or body.get("max_tokens") != 8192:
            raise ValueError("expected original nonstream request with max_tokens=8192")
        body = copy.deepcopy(body)
        videos = [
            part
            for msg in body["messages"]
            for part in msg.get("content", [])
            if isinstance(part, dict) and part.get("type") == "video_url"
        ]
        if len(videos) != 1 or videos[0]["video_url"]["url"] != entry["video_url"]:
            raise ValueError("expected the original single video URL")
        videos[0]["video_url"]["url"] = str((directory / entry["video"]).resolve())
        if omit_max_tokens:
            del body["max_tokens"]
        if enable_thinking is not None:
            body["enable_thinking"] = bool(enable_thinking)
            body["chat_template_kwargs"] = dict(
                body.get("chat_template_kwargs", {}),
                enable_thinking=bool(enable_thinking),
            )
        workload.append(dict(id=entry["id"], payload=body))
    return workload, manifest


def output_token_budget(body, prompt_tokens, capacity, default_max_new_tokens):
    """Output upper bound; the engine may also reserve sequence slots."""
    requested = body.get("max_tokens")
    if prompt_tokens >= capacity or (
        requested is not None and prompt_tokens + requested > capacity
    ):
        raise ValueError("complete request exceeds configured capacity")
    return min(
        default_max_new_tokens if requested is None else requested,
        capacity - prompt_tokens,
    )


def expected_inputs(model, workload, capacity):
    """Use the service's renderer and sampling policy, without model weights."""
    import av
    import torch
    from transformers import AutoProcessor

    from rtp_llm.config.generate_config import GenerateConfig
    from rtp_llm.config.py_config_modules import GenerateEnvConfig
    from rtp_llm.multimodal.qwen3_vl_video import (
        resolve_video_size,
        sample_frame_indices,
        video_resize_shape,
        video_timestamp_tokens,
    )
    from rtp_llm.openai.api_datatype import ChatCompletionRequest
    from rtp_llm.openai.renderers.custom_renderer import RendererParams
    from rtp_llm.openai.renderers.qwen35_renderer import Qwen35Renderer

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    tokenizer, vp = processor.tokenizer, processor.video_processor
    renderer = Qwen35Renderer(
        tokenizer,
        RendererParams(
            model_type="qwen35_moe",
            max_seq_len=capacity,
            eos_token_id=tokenizer.eos_token_id,
            stop_word_ids_list=[],
            ckpt_path=model,
        ),
        GenerateEnvConfig(),
        ckpt_path=model,
    )
    expected = {}
    for entry in workload:
        body = entry["payload"]
        rendered = renderer.render_chat(ChatCompletionRequest(**body))
        if len(rendered.multimodal_inputs) != 1:
            raise ValueError("expected one video and preprocessing configuration")
        config = rendered.multimodal_inputs[0].mm_preprocess_config
        with av.open(rendered.multimodal_inputs[0].url) as container:
            stream = container.streams.video[0]
            total, fps = stream.frames, float(stream.average_rate)
            height, width = stream.height, stream.width
            if not total:
                total = sum(1 for _ in container.decode(video=0))
        indices = sample_frame_indices(total, fps, config)
        size = resolve_video_size(
            vp, ENCODER_VIDEO_TOTAL_MIN_PIXELS, ENCODER_VIDEO_TOTAL_MAX_PIXELS
        )
        h, w = video_resize_shape(config, len(indices), height, width, vp, size)
        t = (len(indices) + vp.temporal_patch_size - 1) // vp.temporal_patch_size
        grid = torch.tensor([[t, h // vp.patch_size, w // vp.patch_size]])
        timestamps = video_timestamp_tokens(grid, indices, fps, processor)
        visual = int(grid.prod()) // vp.merge_size**2
        expanded = visual + sum(len(x) + 2 for x in timestamps)
        ids = rendered.input_ids
        if ids.count(tokenizer.convert_tokens_to_ids("<|video_pad|>")) != 1:
            raise ValueError("expected exactly one video placeholder")
        prompt = len(ids) - 1 + expanded
        output_budget = output_token_budget(
            body, prompt, capacity, GenerateConfig().max_new_tokens
        )
        expected[str(entry["id"])] = dict(
            fixture_id=entry["id"],
            thinking_enabled=body.get("enable_thinking", True),
            prompt_tokens=prompt,
            requested_max_tokens=body.get("max_tokens"),
            backend_default_max_new_tokens=GenerateConfig().max_new_tokens,
            output_token_upper_bound=output_budget,
            video_tokens=expanded,
            visual_tokens=visual,
            frames=len(indices),
            grid=grid.tolist(),
            resized_height=h,
            resized_width=w,
            template_tokens=len(ids),
            timestamp_tokens=sum(map(len, timestamps)),
            method="current Qwen35Renderer and shared video sampling/grid expansion",
        )
    return expected


def preflight(manager, workload, expected, out, request, save):
    from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

    def send(entry, rank):
        url = f"http://127.0.0.1:{manager.port + rank * MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"
        row = request(
            url, entry["payload"], rank, expected["by_fixture"][str(entry["id"])]
        )
        row.update(fixture_id=entry["id"], client_rank=rank)
        return row

    serial = []
    for rank, entry in enumerate(workload):
        serial.append(send(entry, rank))
        save(out / "mixed-serial-preflight.json", serial)
        if not serial[-1]["ok"]:
            raise RuntimeError(f"video {entry['id']} serial preflight failed")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda pair: send(pair[1], pair[0]), enumerate(workload)))
    save(out / "mixed-concurrent-preflight.json", rows)
    if not all(r["ok"] for r in rows):
        raise RuntimeError("mixed four-concurrent preflight failed")
    return serial + rows


def summarize(rows, warmup_s, window_s, windows, summary):
    return {
        str(i): summary(
            [r for r in rows if r.get("fixture_id") == i], warmup_s, window_s, windows
        )
        for i in range(1, 5)
    }


def validate_content(content, fixture_id):
    """Check the four current prompt contracts without limiting output length."""
    errors = []
    final = content.split("</think>", 1)[-1].strip()
    if fixture_id in (1, 4):
        headings = (
            "Core Summary",
            "Scene",
            "Subjects Description",
            "Dynamic Narrative",
            "Style, Color",
        )
        if any(not re.search(r"(?im)^#+\s*" + re.escape(h), final) for h in headings):
            errors.append("missing requested video description sections")
    elif fixture_id == 2:
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", final, flags=re.IGNORECASE)
        try:
            body = json.loads(payload)
            keys = ("scenes", "visual_events", "on_screen_text")
            valid = isinstance(body, dict) and set(body) == set(keys)
            if valid:
                for key in keys:
                    fields = {"start", "end", "description"}
                    if key == "on_screen_text":
                        fields = {"start", "end", "text", "position"}
                    items = body[key]
                    if not isinstance(items, list):
                        valid = False
                        break
                    for item in items:
                        if (
                            not isinstance(item, dict)
                            or set(item) != fields
                            or any(
                                item[name] is not None
                                and not isinstance(item[name], str)
                                for name in ("start", "end")
                            )
                            or any(
                                not isinstance(item[name], str)
                                for name in fields - {"start", "end"}
                            )
                        ):
                            valid = False
                            break
        except (ValueError, TypeError):
            valid = False
        if not valid:
            errors.append("invalid requested visual-event JSON")
    elif fixture_id == 3:
        headings = (
            "Core Summary",
            "Visible Subjects",
            "Action and Interaction Timeline",
            "On-screen Text",
            "Visibility Limitations",
        )
        if tuple(re.findall(r"(?m)^#\s+(.+?)\s*$", final)) != headings:
            errors.append("missing requested visual subject sections")
    # A final answer may repeat a draft from reasoning. Count each section
    # independently so that normal finalization does not cross a loop threshold.
    for section in content.split("</think>", 1):
        lines = Counter(
            line.strip() for line in section.splitlines() if len(line.strip()) > 30
        )
        if max(lines.values(), default=0) >= 10:
            errors.append("degenerate generation: long line repeated at least 10 times")
        nonempty_lines = [line.strip() for line in section.splitlines() if line.strip()]
        blocks = Counter(
            "\n".join(nonempty_lines[start : start + 4])
            for start in range(max(0, len(nonempty_lines) - 3))
            if len("\n".join(nonempty_lines[start : start + 4])) >= 64
        )
        if max(blocks.values(), default=0) >= 4:
            errors.append(
                "degenerate generation: multi-line block repeated at least 4 times"
            )
        words = re.findall(r"\w+", section.lower())
        if any(sum(1 for _ in group) >= 32 for _, group in groupby(words)):
            errors.append(
                "degenerate generation: word repeated consecutively at least 32 times"
            )
    return list(dict.fromkeys(errors))


def select_request(workload, index, seed):
    """A stable uniform choice for each logical request across client processes."""
    if len(workload) != 4 or index < 0:
        raise ValueError("expected four videos and a nonnegative logical request index")
    key = f"{seed}:{index}".encode("ascii")
    position = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 4
    return workload[position]
