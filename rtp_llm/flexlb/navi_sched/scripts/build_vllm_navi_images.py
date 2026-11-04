#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ALIBABA_REPO = "alibaba/alibaba"
VLLM_REPO = "alibaba/vllm"
WHEEL_PIPELINE = 287644
VLLM_PIPELINE = 27142
WHEEL_IMAGE = "hub.docker.alibaba-inc.com/isearch/navi_sched_vllm_wheel"
CUDA_IMAGE = "hub.docker.alibaba-inc.com/isearch/vllm_server_cuda"
ROCM_IMAGE = "hub.docker.alibaba-inc.com/isearch/vllm_server_rocm"
MASTER_IMAGE = "hub.docker.alibaba-inc.com/isearch/vllm_navi_sched_master"
SUCCESS_STATUSES = {"COMPLETE", "SUCCESS"}


def run(command, cwd=None, print_output=True):
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if print_output and result.stdout:
        print(result.stdout.rstrip(), flush=True)
    return result.stdout


def git_value(repo, *args):
    return run(["git", *args], cwd=repo).strip()


def verify_branch(repo, branch):
    local_branch = git_value(repo, "branch", "--show-current")
    if local_branch != branch:
        raise RuntimeError(f"{repo}: expected branch {branch}, got {local_branch}")
    head = git_value(repo, "rev-parse", "HEAD")
    remote_line = git_value(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    remote_head = remote_line.split()[0] if remote_line else ""
    if head != remote_head:
        raise RuntimeError(
            f"{repo}: local HEAD {head} does not match origin/{branch} {remote_head}"
        )
    return head


def trigger_pipeline(repo, pipeline, branch, parameters):
    command = [
        "a1",
        "ci",
        "pipeline",
        "run",
        str(pipeline),
        "--repo",
        repo,
        "--branch",
        branch,
    ]
    for key, value in parameters.items():
        command.extend(["--param", f"{key}={value}"])
    command.extend(["-f", "json"])
    output = run(command)
    match = re.search(r"Run ID:\s*(\d+)", output)
    if not match:
        raise RuntimeError(f"cannot parse run ID from pipeline {pipeline} output")
    return int(match.group(1))


def get_run(repo, run_id):
    output = run(
        ["a1", "ci", "run", "get", str(run_id), "--repo", repo, "-f", "json"],
        print_output=False,
    )
    return json.loads(output)


def bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def verify_wheel_run(run_data, revision, branch):
    params = run_data.get("params") or {}
    checks = {
        "repo": run_data.get("repo") == ALIBABA_REPO,
        "pipeline": run_data.get("pipelineId") == WHEEL_PIPELINE,
        "branch": run_data.get("branch") == branch,
        "revision": (run_data.get("revision") or run_data.get("commit"))
        == revision,
        "build_arm": bool_value(params.get("build_arm")),
        "build_vllm_wheels": bool_value(params.get("build_vllm_wheels")),
        "build_rtpllm_images": not bool_value(params.get("build_rtpllm_images")),
        "rebuild_rtp_llm": not bool_value(params.get("rebuild_rtp_llm")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"wheel run validation failed: {', '.join(failed)}")


def verify_vllm_run(run_data, revision, wheel_revision, wheel_version, branch):
    params = run_data.get("params") or {}
    checks = {
        "repo": run_data.get("repo") == VLLM_REPO,
        "pipeline": run_data.get("pipelineId") == VLLM_PIPELINE,
        "branch": run_data.get("branch") == branch,
        "revision": (run_data.get("revision") or run_data.get("commit"))
        == revision,
        "wheel_revision": params.get("navi_sched_wheel_revision")
        == wheel_revision,
        "wheel_version": params.get("navi_sched_wheel_version") == wheel_version,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"vLLM run validation failed: {', '.join(failed)}")


def wait_for_success(repo, run_id, poll_seconds):
    while True:
        run_data = get_run(repo, run_id)
        print(
            f"run {run_id}: status={run_data.get('status')} "
            f"finished={run_data.get('isFinished')} "
            f"failure={run_data.get('isFailure')}",
            flush=True,
        )
        if run_data.get("isFinished"):
            if (
                run_data.get("isFailure")
                or run_data.get("status") not in SUCCESS_STATUSES
            ):
                raise RuntimeError(f"run {run_id} did not complete successfully")
            return run_data
        time.sleep(poll_seconds)


def verify_wheel_images(revision):
    digests = {}
    for arch in ("x86_64", "arm64"):
        image = f"{WHEEL_IMAGE}:{revision}_{arch}"
        digest = run(["crane", "digest", image]).strip().splitlines()[-1]
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"invalid digest for {image}: {digest}")
        digests[arch] = digest
    return digests


def find_image(repository, pattern):
    output = run(["crane", "ls", repository], print_output=False)
    tags = sorted(tag for tag in output.splitlines() if re.fullmatch(pattern, tag))
    if not tags:
        raise RuntimeError(f"cannot find image in {repository} matching {pattern}")
    return f"{repository}:{tags[-1]}"


def verify_image(image, expected_platforms):
    digest = run(["crane", "digest", image], print_output=False).strip()
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"invalid digest for {image}: {digest}")
    manifest = json.loads(run(["crane", "manifest", image], print_output=False))
    platforms = {
        (item.get("platform") or {}).get("os", "")
        + "/"
        + (item.get("platform") or {}).get("architecture", "")
        for item in manifest.get("manifests", [])
    }
    if not platforms:
        config = json.loads(run(["crane", "config", image], print_output=False))
        platforms = {f"{config.get('os', '')}/{config.get('architecture', '')}"}
    if platforms != set(expected_platforms):
        raise RuntimeError(f"unexpected platforms for {image}: {platforms}")
    return {"image": image, "digest": digest, "platforms": sorted(platforms)}


def verify_vllm_images(vllm_revision, wheel_revision):
    prefix = (
        rf"v[0-9A-Za-z.+-]+_vllm_{re.escape(vllm_revision[:10])}"
        rf"_navi_{re.escape(wheel_revision[:10])}_"
    )
    images = {
        "cuda": find_image(
            CUDA_IMAGE, prefix + r"\d{14}_multi_arch_accelerated"
        ),
        "rocm": find_image(ROCM_IMAGE, prefix + r"\d{14}_accelerated"),
        "master": find_image(
            MASTER_IMAGE, prefix + r"\d{14}_master_multi_arch_accelerated"
        ),
    }
    return {
        "cuda": verify_image(images["cuda"], {"linux/amd64", "linux/arm64"}),
        "rocm": verify_image(images["rocm"], {"linux/amd64"}),
        "master": verify_image(
            images["master"], {"linux/amd64", "linux/arm64"}
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build NaviSched wheels, validate them, then build vLLM images."
    )
    parser.add_argument("--alibaba-dir", default="/home/zq103303/alibaba")
    parser.add_argument("--vllm-dir", default="/home/zq103303/vllm")
    parser.add_argument("--branch", default="navi_sched")
    parser.add_argument("--wheel-version", default="0.1.0")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument(
        "--wheel-run-id",
        type=int,
        help="Resume and validate an existing wheel pipeline run.",
    )
    parser.add_argument(
        "--vllm-run-id",
        type=int,
        help="Resume and validate an existing vLLM pipeline run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alibaba_dir = str(Path(args.alibaba_dir).resolve())
    vllm_dir = str(Path(args.vllm_dir).resolve())
    wheel_revision = verify_branch(alibaba_dir, args.branch)
    vllm_revision = verify_branch(vllm_dir, args.branch)

    wheel_params = {
        "build_arm": "true",
        "build_rtpllm_images": "false",
        "build_vllm_wheels": "true",
        "rebuild_rtp_llm": "false",
    }
    vllm_params = {
        "navi_sched_wheel_revision": wheel_revision,
        "navi_sched_wheel_version": args.wheel_version,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "wheel": {
                        "repo": ALIBABA_REPO,
                        "pipeline": WHEEL_PIPELINE,
                        "revision": wheel_revision,
                        "params": wheel_params,
                    },
                    "vllm": {
                        "repo": VLLM_REPO,
                        "pipeline": VLLM_PIPELINE,
                        "revision": vllm_revision,
                        "params": vllm_params,
                    },
                },
                indent=2,
            )
        )
        return

    wheel_run_id = args.wheel_run_id or trigger_pipeline(
        ALIBABA_REPO,
        WHEEL_PIPELINE,
        args.branch,
        wheel_params,
    )
    wheel_run = get_run(ALIBABA_REPO, wheel_run_id)
    verify_wheel_run(wheel_run, wheel_revision, args.branch)
    wait_for_success(ALIBABA_REPO, wheel_run_id, args.poll_seconds)
    digests = verify_wheel_images(wheel_revision)

    vllm_run_id = args.vllm_run_id or trigger_pipeline(
        VLLM_REPO,
        VLLM_PIPELINE,
        args.branch,
        vllm_params,
    )
    vllm_run = get_run(VLLM_REPO, vllm_run_id)
    verify_vllm_run(
        vllm_run,
        vllm_revision,
        wheel_revision,
        args.wheel_version,
        args.branch,
    )
    wait_for_success(VLLM_REPO, vllm_run_id, args.poll_seconds)
    images = verify_vllm_images(vllm_revision, wheel_revision)
    print(
        json.dumps(
            {
                "wheel_run_id": wheel_run_id,
                "wheel_revision": wheel_revision,
                "wheel_digests": digests,
                "vllm_run_id": vllm_run_id,
                "vllm_revision": vllm_revision,
                "vllm_images": images,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
