import torch
import torch.nn as nn
import logging
import os
import json
import hashlib
import importlib
import sys
import fcntl
import time
import safetensors
from typing import Any, Dict, List, Optional

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_parallel_info

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
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def import_class_from_path(module_path_str: str, ckpt_path: str):
    """
    Import a class dynamically from a module path string.
    Example: "item_embedding.ItemEmbedding"
    """
    if "." not in module_path_str:
        raise ValueError(f"Invalid module path: {module_path_str}. Expected 'module.Class'")
        
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
        logging.warning(f"Standard import failed for {module_name}: {e}. Trying absolute path...")
        # Fallback: try loading from ckpt_path/module_name.py
        try:
            from rtp_llm.utils.import_util import load_module
            module_file_path = os.path.join(ckpt_path, module_name + ".py")
            if os.path.exists(module_file_path):
                module = load_module(module_file_path)
                return getattr(module, class_name)
        except Exception as e2:
            logging.error(f"Failed to load {class_name} from {module_name} even with absolute path: {e2}")
            
        raise e

def load_embedding_weight(ckpt_path: str, custom_config: Dict[str, Any]) -> torch.Tensor:
    """
    Intelligently load embedding weights.
    Prioritizes config-specified weight_path, falls back to scanning ckpt_path for main model embeddings.
    """
    # 1. Check config first
    weight_path = custom_config.get("weight_path")
    if weight_path:
        if not os.path.isabs(weight_path):
            weight_path = os.path.join(ckpt_path, weight_path)
        if os.path.exists(weight_path):
            logging.info(f"Auto-compile: Loading embedding table from configured path: {weight_path}")
            return torch.load(weight_path, map_location='cpu')
        else:
            logging.warning(f"Auto-compile: Configured weight_path {weight_path} not found.")

    # 2. Fallback: Scan checkpoint for main model embeddings
    logging.info(f"Auto-compile: Scanning {ckpt_path} for main model embedding table...")
    
    # Common keys for embedding tables in various architectures
    # 'model.embed_tokens.weight' -> Llama, Qwen, Mistral
    # 'transformer.wte.weight' -> GPT-2, GPT-Neo, Qwen-Legacy
    # 'word_embeddings.weight' -> ChatGLM
    candidate_keys = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "transformer.word_embeddings.weight",
        "word_embeddings.weight"
    ]

    files = os.listdir(ckpt_path)
    # Prefer safetensors
    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    bin_files = [f for f in files if f.endswith(".bin") or f.endswith(".pt")]
    
    # Sort to usually find the first shard (where embeddings usually live)
    safetensors_files.sort()
    bin_files.sort()

    # Strategy: Try safetensors first (supports slice loading)
    for f in safetensors_files:
        full_path = os.path.join(ckpt_path, f)
        try:
            with safetensors.safe_open(full_path, framework="pt", device="cpu") as st:
                keys = st.keys()
                for k in candidate_keys:
                    if k in keys:
                        logging.info(f"Auto-compile: Found embedding key '{k}' in {f}")
                        return st.get_tensor(k)
        except Exception as e:
            logging.warning(f"Auto-compile: Failed to read {f}: {e}")

    # Fallback to bin files (pytorch load) - potentially slow as it loads full dict
    for f in bin_files:
        full_path = os.path.join(ckpt_path, f)
        try:
            # map_location='cpu' is important
            state_dict = torch.load(full_path, map_location="cpu")
            for k in candidate_keys:
                if k in state_dict:
                    logging.info(f"Auto-compile: Found embedding key '{k}' in {f}")
                    return state_dict[k]
            del state_dict # Free memory
        except Exception as e:
            logging.warning(f"Auto-compile: Failed to read {f}: {e}")
            
    raise RuntimeError("Auto-compile: Could not find any suitable embedding table in checkpoint.")

def generate_dummy_inputs(config: GptInitModelParameters, custom_config: Dict[str, Any], device='cpu'):
    """
    Generate dummy inputs for tracing based on config.
    """
    feat_num = custom_config.get("feat_num", 10) # Default from example
    token_len = custom_config.get("trace_token_len", 14) 
    
    batch_size = 1
    
    # Input shape: [Batch, FeatNum, TokenLen]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, feat_num, token_len), dtype=torch.int32).to(device)
    attention_mask = torch.ones((batch_size, feat_num, token_len), dtype=torch.int32).to(device)
    
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
    root_path = os.environ.get("HIPPO_APP_INST_ROOT", ckpt_path)
    output_dir = os.path.join(root_path, "custom_modal")
    os.makedirs(output_dir, exist_ok=True)
    so_path = os.path.join(output_dir, "custom_modal.so")
    metadata_path = os.path.join(output_dir, "metadata.json")

    # 3. Resolve model class and source file
    embedding_module_str = custom_config.get("embedding_module_path")
    if not embedding_module_str:
        logging.error("Auto-compile: embedding_module_path not found in custom_modal config.")
        return
    
    module_name, class_name = embedding_module_str.rsplit(".", 1)
    # Assume the python file is in the same dir or cwd
    source_file = f"{module_name}.py"
    if not os.path.exists(source_file):
         # Try joining with ckpt_path
         source_file = os.path.join(ckpt_path, f"{module_name}.py")
    
    # 4. Calculate Fingerprint
    current_source_md5 = get_file_md5(source_file)
    current_config_md5 = get_config_md5(custom_config)
    
    need_compile = True
    if os.path.exists(so_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                saved_meta = json.load(f)
            if (saved_meta.get('source_md5') == current_source_md5 and 
                saved_meta.get('config_md5') == current_config_md5):
                need_compile = False
                logging.info(f"Auto-compile: Artifacts up-to-date for {source_file}. Skipping compilation.")
        except Exception as e:
            logging.warning(f"Auto-compile: Failed to read metadata, forcing compile. Error: {e}")

    if not need_compile:
        return

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
        
        # Double check after acquiring lock (in case another process finished it)
        if os.path.exists(metadata_path):
             with open(metadata_path, 'r') as f:
                saved_meta = json.load(f)
             if (saved_meta.get('source_md5') == current_source_md5 and 
                saved_meta.get('config_md5') == current_config_md5):
                logging.info("Auto-compile: Artifacts updated by another process. Skipping.")
                return

        logging.info(f"Auto-compile: Compiling {source_file} to {so_path}...")
        
        # 6. Instantiate Model
        ModelClass = import_class_from_path(module_name, class_name)
        model_instance = ModelClass(config=config)
        
        # 7. Load Embedding Weights (Intelligent Loading)
        embedding_table = load_embedding_weight(ckpt_path, custom_config)
        
        # 8. Load Model Weights
        # The CustomModalEmbedding.load_weight method takes embedding_table and loads other weights itself
        if hasattr(model_instance, "load_weight"):
            model_instance.load_weight(embedding_table)
        else:
            logging.warning("Auto-compile: Model class does not have 'load_weight' method. Weights might be missing!")

        # 9. Generate Dummy Inputs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        logging.info("Auto-compile: Starting torch._export.aot_compile (this may take a while)...")
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

        # 12. Write Metadata
        meta_content = {
            "source_md5": current_source_md5,
            "config_md5": current_config_md5,
            "timestamp": time.time()
        }
        with open(metadata_path, 'w') as f:
            json.dump(meta_content, f)

    except Exception as e:
        logging.error(f"Auto-compile failed: {e}")
        if os.path.exists(so_path):
            os.remove(so_path)
        raise e
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()