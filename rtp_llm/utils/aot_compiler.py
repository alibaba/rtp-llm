import fcntl
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import safetensors
import torch
import torch.nn as nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.weight_type import WEIGHT_TYPE


def get_file_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    if not os.path.exists(file_path):
        return ""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_config_md5(config_dict: Dict[str, Any]) -> str:
    """Calculate MD5 hash of a config dictionary."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def import_class_from_path(module_path_str: str, ckpt_path: str):
    """
    Import a class dynamically from a module path string.
    Example: "item_embedding.ItemEmbedding"
    """
    if "." not in module_path_str:
        raise ValueError(
            f"Invalid module path: {module_path_str}. Expected 'module.Class'"
        )

    module_name, class_name = module_path_str.rsplit(".", 1)

    # Add ckpt_path and current dir to sys.path to ensure we can find the module
    if ckpt_path not in sys.path:
        sys.path.append(ckpt_path)
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    try:
        # Try loading via standard import first
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        logging.warning(
            f"Standard import failed for {module_name}: {e}. Trying absolute path..."
        )
        # Fallback: try loading from ckpt_path/module_name.py
        try:
            from rtp_llm.utils.import_util import load_module

            module_file_path = os.path.join(ckpt_path, module_name + ".py")
            if os.path.exists(module_file_path):
                module = load_module(module_file_path)
                return getattr(module, class_name)
        except Exception as e2:
            logging.error(
                f"Failed to load {class_name} from {module_name} even with absolute path: {e2}"
            )

        raise e


def load_embedding_weight(
    ckpt_path: str, config: GptInitModelParameters
) -> torch.Tensor:
    """
    Intelligently load embedding weights.
    Scans ckpt_path for main model embeddings using CkptDatabase.
    """
    logging.info(
        f"Auto-compile: Scanning {ckpt_path} for main model embedding table using CkptDatabase..."
    )

    try:
        # Initialize database (this handles index.json parsing or glob scanning automatically)
        db = CkptDatabase(ckpt_path)
    except Exception as e:
        raise RuntimeError(f"Auto-compile: Failed to initialize CkptDatabase: {e}")

    # Common keys for embedding tables
    candidate_keys = [
        "model.embeddings.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "transformer.word_embeddings.weight",
        "word_embeddings.weight",
    ]

    # Resolve target data type from config
    target_dtype = None
    if config.data_type:
        try:
            target_dtype = WEIGHT_TYPE.from_str(config.data_type).to_torch_dtype()
            logging.info(
                f"Auto-compile: Target dtype resolved from config: {target_dtype}"
            )
        except Exception as e:
            logging.warning(
                f"Auto-compile: Failed to resolve dtype from config.data_type='{config.data_type}': {e}. Falling back to default."
            )

    if target_dtype is None:
        # Fallback default if not specified or failed
        target_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        logging.info(f"Auto-compile: Using default target dtype: {target_dtype}")

    # Get all available tensor names for fast lookup
    all_tensor_names = set(db.get_pretrain_tensor_names())

    for k in candidate_keys:
        if k in all_tensor_names:
            logging.info(f"Auto-compile: Found embedding key '{k}'")
            # load_tensor returns a list of tensors (in case of sharding)
            tensors = db.load_tensor(k, data_type=target_dtype)

            if tensors:
                return tensors[0]

    raise RuntimeError(
        "Auto-compile: Could not find any suitable embedding table in checkpoint."
    )


def generate_dummy_inputs(
    config: GptInitModelParameters, custom_config: Dict[str, Any], device="cpu"
):
    """
    Generate dummy inputs for tracing based on config.
    """
    feat_num = custom_config.get("feat_num", 10)  # Default from example
    token_len = custom_config.get("trace_token_len", 14)

    batch_size = 2

    # Input shape: [Batch, FeatNum, TokenLen]
    # Use embedding_table.shape[0] to ensure we don't generate out-of-bounds indices
    vocab_limit = config.vocab_size
    input_ids = torch.randint(
        0, vocab_limit, (batch_size, feat_num, token_len), dtype=torch.int32
    ).to(device)
    attention_mask = torch.ones(
        (batch_size, feat_num, token_len), dtype=torch.int32
    ).to(device)

    return input_ids, attention_mask


def try_auto_compile(ckpt_path: str, config: GptInitModelParameters):
    """
    Main entry point for auto-compilation.
    """
    # 1. Check if AOT is enabled
    if config.vit_separation != 3:
        return

    custom_config = config.custom_modal
    if not custom_config:
        # It's possible custom_modal is not in GptInitModelParameters yet if not updated from config
        # But ModelFactory calls us after create_frontend_config which calls update_config_with_custom_modal
        logging.info("Auto-compile: No custom_modal config found, skipping.")
        return

    # 2. Identify output path
    # Default to a 'custom_modal' directory inside HIPPO_APP_INST_ROOT or ckpt_path
    root_path = os.path.join(os.getcwd(), "custom_modal_artifacts")
    output_dir = os.path.join(root_path, "custom_modal")
    os.makedirs(output_dir, exist_ok=True)
    so_path = os.path.join(output_dir, "custom_modal.so")
    metadata_path = os.path.join(output_dir, "metadata.json")

    # 3. Resolve model class and source file
    embedding_module_str = custom_config.get("embedding_module_path")
    if not embedding_module_str:
        logging.error(
            "Auto-compile: embedding_module_path not found in custom_modal config."
        )
        return

    module_name, _ = embedding_module_str.rsplit(".", 1)
    # the python file is in the same dir or cwd
    source_file = os.path.join(ckpt_path, f"{module_name}.py")

    # 4. Force Compile (Skip Fingerprint Check)
    # We used to calculate MD5 of source and config here to skip compilation.
    # Now we force compilation every time to ensure weights are up-to-date.
    
    # 5. Multi-process coordination (Rank 0 compiles, others wait)
    # We use a file lock.
    # IMPORTANT: In a multi-GPU setting on the same node, we want only one process to compile.
    # Assuming standard distributed launch where they share the filesystem.
    lock_file_path = os.path.join(output_dir, ".compile.lock")
    lock_file = open(lock_file_path, "w")

    try:
        # Acquire lock (blocking)
        logging.info("Auto-compile: Checking lock...")
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        # Force compile, no double check.
        logging.info(f"Auto-compile: Compiling {source_file} to {so_path}...")

        # 6. Instantiate Model
        ModelClass = import_class_from_path(embedding_module_str, ckpt_path)
        model_instance = ModelClass(config=config)

        # 7. Load Embedding Weights (Intelligent Loading)
        embedding_table = load_embedding_weight(ckpt_path, config)

        # Move embedding table to device before passing to model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_table = embedding_table.to(device)

        # Wrap tensor in ModelWeights to match runtime interface
        model_weights = ModelWeights(
            num_layers=0, device=device, dtype=embedding_table.dtype
        )
        model_weights.set_global_weight("embedding", embedding_table)

        # 8. Load Model Weights
        # The CustomModalEmbedding.load_weight method takes embedding_table and loads other weights itself
        if hasattr(model_instance, "load_weight"):
            model_instance.load_weight(model_weights)
        else:
            logging.warning(
                "Auto-compile: Model class does not have 'load_weight' method. Weights might be missing!"
            )

        # 9. Generate Dummy Inputs
        model_instance.to(device)
        model_instance.eval()
        input_ids, attention_mask = generate_dummy_inputs(config, custom_config, device)

        # 10. Define Dynamic Shapes
        batch_dim = torch.export.Dim("batch_dim")
        token_dim = torch.export.Dim("token_len")
        dynamic_shapes = {
            "input_ids": {0: batch_dim, 2: token_dim},
            "attention_mask": {0: batch_dim, 2: token_dim},
        }

        # 11. Run Compile
        options = {
            "aot_inductor.output_path": so_path,
            "max_autotune": True,
        }

        logging.info(
            "Auto-compile: Starting torch._export.aot_compile (this may take a while)..."
        )
        start_time = time.time()

        # Using aot_compile
        torch._export.aot_compile(
            model_instance,
            args=(input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes,
            options=options,
        )

        elapsed = time.time() - start_time
        logging.info(f"Auto-compile: Compilation finished in {elapsed:.2f}s.")

    except Exception as e:
        logging.error(f"Auto-compile failed: {e}")
        if os.path.exists(so_path):
            os.remove(so_path)
        raise e
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
