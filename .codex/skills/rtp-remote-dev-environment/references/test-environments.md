# RTP-LLM Test Environments

Last observed: 2026-07-31 from the local Mac.

Treat this as discovered state. Re-probe connectivity, GPU inventory, container state, and image identity before scheduling work.

## Connection Routing

- Connect to `11.163.39.110` through `11.163.39.115` with native SSH. Do not route these fixed hosts through WebTerminal. The local `~/.ssh/config` has a direct-host rule before the WebTerminal include with `ProxyCommand none` for these six addresses.
- Connect to dynamic or deployed endpoints, including the long `*.default-*` names below, through `webterminal-cli`.
- If WebTerminal authentication has expired, refresh it through the local `webterminal-cli` project before retrying the endpoint. Do not reinterpret an expired WebTerminal session as a target-host outage.

## Host-Level Environments

The user refers to these hosts as B300 environments. At the last probe, `nvidia-smi` reported eight `NVIDIA L20D` GPUs with driver `580.105.08` on every reachable host. Keep the user-facing platform label and the detected GPU identity distinct until the naming is confirmed.

| Address | SSH | Architecture | Docker/container observation |
|---|---|---|---|
| `11.163.39.110` | Reachable with native SSH; local `id_rsa` public key enrolled and batch-mode login verified | `x86_64` | `zhanghuidong.zhd_GPU` created and verified with the common CUDA 13 image, eight GPUs, host network, `/data3` mounted, and Docker init enabled; host Docker socket remained world-writable and should not be used as the desired configuration |
| `11.163.39.111` | Reachable | `x86_64` | Docker socket permission denied; user was not in `docker` group |
| `11.163.39.112` | Reachable with native SSH | `x86_64` | `zhanghuidong.zhd_GPU` running with the common CUDA 13 image; socket was world-writable and should not be used as the desired configuration |
| `11.163.39.113` | Reachable | `x86_64` | Docker socket permission denied; user was not in `docker` group |
| `11.163.39.114` | Reachable | `x86_64` | `zhanghuidong.zhd_GPU` running with the common CUDA 13 image |
| `11.163.39.115` | Reachable; SSH alias `b300` | `x86_64` | Build hub candidate; Docker socket permission denied; user was not in `docker` group |

The common observed container name is `zhanghuidong.zhd_GPU`.

The common observed image is:

```text
hub.docker.alibaba-inc.com/isearch/rtp_llm_base_gpu_cuda13:2026_05_01_14_37_e91b801
```

When Docker access fails on `/var/run/docker.sock`, use the `remote-docker-access` skill. Do not store or transmit the sudo password in this skill, the repository, command arguments, or logs.

## Validated `.110` Container Creation

The `.115` source files copied to `.110` are:

```text
~/work/alibaba/docker/ddev/ddev
~/work/alibaba/docker/dev/version
```

The `ddev` script SHA-256 at validation time is:

```text
afc67368821896f2e874c504125e6ecca6cc0cd64290c9bfa1a84cb503f87c24
```

Use the image repository and tag as separate arguments because this `ddev` implementation concatenates them internally. Use `--gpu` to reproduce the complete GPU device and driver-volume mapping on `.115`. Mount `/data3` because the `.110` user home resolves there. Use `--init` to reap Bazel and test child processes:

```bash
python2 "$HOME/work/alibaba/docker/ddev/ddev" create "${USER}_GPU" \
  --gpu \
  --docker_args="--init -v /data3:/data3" \
  --image=hub.docker.alibaba-inc.com/isearch/rtp_llm_base_gpu_cuda13 \
  --tag=2026_05_01_14_37_e91b801
```

Validated `.110` artifact identity:

```text
Container: zhanghuidong.zhd_GPU
Image ID: sha256:37f1ea4cfce1462a65a9e97d01b0c7d72a7739c38c2ac3a5c8dbb089d85f967e
Repository digest: sha256:a70597717ec8ff182b9ef4ae2e0b8376595f325ae19cf661810ccb85708afd6a
```

The image does not contain a Docker CLI. The mounted Docker socket therefore does not provide nested `docker` commands unless the image is deliberately extended later.

## Validated `.110` Python Dependencies

Source lock file on `.115`:

```text
/data1/zhanghuidong.zhd/workspace/RTP-LLM/github-opensource-k3-cuda-graph/deps/requirements_lock_torch_gpu_cuda13.txt
```

Persistent copy visible inside the `.110` container:

```text
/data3/zhanghuidong.zhd/workspace/RTP-LLM/github-opensource-k3-cuda-graph/deps/requirements_lock_torch_gpu_cuda13.txt
```

Validated lock SHA-256:

```text
7ff027ce1aa9bc3fe04e55af1337f846e9418cee0a029d186a99b13245d02db0
```

Install into the image-provided Python 3.10 environment as root inside this disposable development container:

```bash
docker exec zhanghuidong.zhd_GPU \
  /opt/conda310/bin/python -m pip install \
  --disable-pip-version-check \
  -r /data3/zhanghuidong.zhd/workspace/RTP-LLM/github-opensource-k3-cuda-graph/deps/requirements_lock_torch_gpu_cuda13.txt
```

The validated result includes Torch `2.11.0+cu130`, Torchvision `0.26.0+cu130`, FlashInfer `0.6.9`, DeepGEMM `2.5.0`, and RTP Kernel `0.1.0`. A CUDA allocation smoke test saw eight `NVIDIA L20D` devices and completed successfully.

Known validation notes:

- `pip check` reports `decord 0.6.0 is not supported on this platform`, although `import decord` succeeds. Treat this as a lock/package-metadata issue until the lock is regenerated or the package is replaced.
- Importing `rtp_kernel` warns that optional `sparse_attention_fp8` is unavailable because `flash_attn_interface` is absent. Core `rtp_kernel` import succeeds; test that optional path separately before relying on it.

## Prebuilt Container Endpoints

These endpoints are already inside deployed container environments. The absence of a nested `docker` command is expected and is not a connectivity failure.

| Endpoint | SSH | Architecture | Observed environment |
|---|---|---|---|
| `asi-adc-c2group-online-asi-zjk-gs-ksyun-t05-tre-01-inst-50799.default-9eec3496-a-4982` | Reachable | `x86_64` | Docker marker present; eight `NVIDIA L20D`; driver `580.105.08`; Python `3.12.7` |
| `asi-adc-c2group-online-asi-zjk-gs-ksyun-t05-tre-01-inst-50808.default-ed58969a-a-309f` | Reachable | `x86_64` | Docker marker present; eight `NVIDIA L20D`; driver `580.105.08`; Python `3.12.7` |

## Probe Rules

1. Probe with read-only commands before assigning a test.
2. Distinguish network reachability, SSH authentication, Docker permission, container existence, and application readiness.
3. Use native SSH for `.110-.115`; use `webterminal-cli` for the deployed endpoints and refresh its authentication when required.
4. Never fix Docker access with `chmod 666` or retain a world-writable Docker socket.
5. Do not assume the two prebuilt endpoints have the same Python, PyTorch, CUDA, libraries, or model mounts as the common host container merely because their GPU and driver match.
