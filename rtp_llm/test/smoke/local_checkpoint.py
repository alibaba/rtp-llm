import json
import os
import shlex
import subprocess
from pathlib import Path


def resolve_local_checkpoint(model_name, explicit_path=None):
    root = os.environ.get("SMOKE_LOCAL_CHECKPOINT_ROOT")
    if not explicit_path and not root:
        raise ValueError(
            "Set SMOKE_LOCAL_CHECKPOINT_ROOT to a local directory containing "
            f"{model_name}, or supply an explicit checkpoint path; NAS fallback is disabled"
        )
    checkpoint = Path(explicit_path) if explicit_path else Path(root) / model_name
    if not checkpoint.is_absolute():
        raise ValueError(f"Checkpoint path must be absolute: {checkpoint}")
    checkpoint = checkpoint.resolve(strict=True)
    filesystem = subprocess.check_output(
        ["stat", "-f", "--format=%T", "--", str(checkpoint)], text=True
    ).strip()
    if filesystem in {"nfs", "nfs4", "cifs", "smb2"}:
        raise ValueError(
            f"Checkpoint must be on local disk, not {filesystem}: {checkpoint}"
        )
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError(f"Checkpoint has no weight shards: {index_path}")
    required = shards | {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        index_path.name,
    }
    for filename in required:
        path = (checkpoint / filename).resolve(strict=True)
        if (
            not path.is_relative_to(checkpoint)
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ValueError(f"Missing, empty or external checkpoint file: {path}")
    return str(checkpoint)


def relocate_model_paths(smoke_args, env_args, original, checkpoint):
    def replace_token(token):
        if token == original:
            return checkpoint
        key, separator, value = token.partition("=")
        return (
            key + separator + checkpoint if separator and value == original else token
        )

    def replace_args(arguments):
        return shlex.join([replace_token(token) for token in shlex.split(arguments)])

    if isinstance(smoke_args, dict):
        smoke_args = {role: replace_args(args) for role, args in smoke_args.items()}
    else:
        smoke_args = replace_args(smoke_args)
    if isinstance(env_args, dict):
        env_args = {
            role: [replace_token(value) for value in values]
            for role, values in env_args.items()
        }
    else:
        env_args = [replace_token(value) for value in env_args]
    return smoke_args, env_args
