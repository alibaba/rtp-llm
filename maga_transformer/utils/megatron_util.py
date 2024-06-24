from typing import Any, Dict, List, Set, Union, Optional, NamedTuple
from pathlib import PosixPath, Path
import os
import logging
import re

class MegatronUtil:
    @staticmethod
    def is_megatron_ckpt(ckpt_path: Path) -> bool:
        subdirs = list(ckpt_path.rglob(r"mp_rank_??"))
        if len(subdirs) > 0:
            return True
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_???"))
        if len(subdirs) > 0:
            return True
        
        # Megatron 只输出global_stepxxx/mp_rank_xxx_model_states.pt,但是deepspeed还会输出*.bin
        patterns = ["*.pth", "*.bin", "*.pt"]
        ckpt_files = [file for pattern in patterns for file in ckpt_path.glob(pattern)]
        if len(ckpt_files) > 0:
            return False
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_model_states.pt"))
        for subdir in subdirs:
            training_config_path = subdir.parent / 'training_config.ini'
            if not training_config_path.exists():
                return True

        return False
    
    @staticmethod
    def get_megatron_info(ckpt_path: Path):
        subdirs = list(ckpt_path.rglob(r"mp_rank_??"))
        if len(subdirs) > 0:
            tp = len(subdirs)
            pp = 1
            return subdirs[0].parents[0], len(subdirs), 1
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_model_states.pt"))
        if len(subdirs) > 0:
            return subdirs[0].parents[0], len(subdirs), 1
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_???"))
        assert len(subdirs) > 0, "magatron path must like: [mp_rank_?? | mp_rank_??_model_states.pt | mp_rank_??_???]"
        mp: int = len(subdirs)
        tp_ranks = set()
        pp_ranks = set()
        for subdir in map(str, subdirs):
            tp_rank = re.findall(r"mp_rank_(\d\d)_000$", subdir)
            pp_rank = re.findall(r"mp_rank_00_(\d\d\d)$", subdir)
            if len(tp_rank) > 0:
                tp_ranks.add(tp_rank[0])
            if len(pp_rank) > 0:
                pp_ranks.add(pp_rank[0])
        tp = len(tp_ranks)
        pp = len(pp_ranks)
        assert tp * pp == mp , f"tp:{tp} * pp{pp} != mp:{mp}"
        return subdirs[0].parents[0], tp, pp
    
    @staticmethod
    def detect_ckpt_file(dir: str, pp_rank: int, tp_rank: int, pp_size: int, tp_size:int) -> Path:
        if pp_size == 1:
            suffix = f"mp_rank_{tp_rank:02}"
            if (dir / suffix).exists():
                path = dir / suffix
            else:
                assert((dir / f"mp_rank_{tp_rank:02}_model_states.pt").exists())
                path = dir
        else:
            suffix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
            path = dir / suffix

        if not path.exists():
            raise FileNotFoundError(f"{path.absolute()} not exists!")
        if (path / "model_rng.pt").exists():
            suffix = f"model_rng.pt"
        elif (path / "model_optim_rng.pt").exists():
            suffix = f"model_optim_rng.pt"
        elif (path /  f"mp_rank_{tp_rank:02}_model_states.pt").exists():
            suffix =  f"mp_rank_{tp_rank:02}_model_states.pt"
        path = path / suffix
        return path