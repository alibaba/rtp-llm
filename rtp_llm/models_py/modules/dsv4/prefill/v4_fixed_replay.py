"""Opt-in CP row fingerprints for the ordinary V4 ARM test's fixed replay only."""
import hashlib
import json
import os
from pathlib import Path

TARGET = "//rtp_llm/test/smoke:smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100"
IDS = (0,3476,477,260,11502,22896,16,128803,17916,713,102323,16652,303,14479,16734,13199,128804,128822,24313,37399,16562,6127,10177,50219,16562,16905,10177,88634,16562,856,2614,37,10177,34859,13168,16562,1059,7,10177,28730,16562,2398,1821,223,21,2029,10177,16051,20035,16562,939,23,15,3663,15)
STAGES = ("embed", *(f"layer.{i}" for i in range(43)), "final_norm")


def open_trace(v4, cp, input_ids, prepare_hidden):
    if (os.getenv("TEST_TARGET") != TARGET or prepare_hidden is not None or
        len(v4.layers) != 43 or cp is None):
        return None
    import torch
    if input_ids.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    tmp = Path(os.getenv("TEST_TMPDIR", ""))
    root = Path(os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", ""))
    if not (tmp.is_absolute() and root.is_absolute() and (tmp/"v4-live38-replay.on").is_file()):
        return None
    if tuple(v4.capture_aux_hidden_layer_ids)!=(40,41,42):
        return None
    lengths = cp.input_lengths_global
    if (int(cp.cp_size)!=2 or int(cp.seq_len_full)!=55 or int(cp.prefix_length)!=0 or
        (lengths is not None and lengths.reshape(-1).tolist()!=[55])):
        return None
    rows = cp.local_is_real.reshape(-1).bool().nonzero(as_tuple=True)[0]
    pos = cp.global_positions.reshape(-1).index_select(0,rows).long().cpu().tolist()
    tokens = input_ids.reshape(-1).index_select(0,rows).cpu().tolist()
    if len(pos) != len(set(pos)) or any(not 0 <= p < len(IDS) or IDS[p] != v for p,v in zip(pos,tokens)):
        return None
    return Trace(root, int(cp.cp_rank), rows, pos)


class Trace:
    def __init__(self, root, rank, rows, pos):
        self.root, self.rank, self.rows, self.pos = root, rank, rows, pos
        self.stages = []

    def observe(self, stage, hidden):
        import torch
        if stage != STAGES[len(self.stages)] or hidden.dtype != torch.bfloat16:
            raise ValueError("V4 fixed replay stage or dtype mismatch")
        shape = list(hidden.shape[1:])
        if shape != ([4096] if stage=="final_norm" else [4,4096]):
            raise ValueError("V4 fixed replay hidden geometry mismatch")
        # No hidden values, weights or tokens leave the worker, only one SHA per
        # real row. CP padding is discarded before hashing.
        bytes_ = hidden.detach().index_select(0,self.rows).contiguous().view(torch.uint8).cpu().numpy()
        self.stages.append({"stage":stage,"sha":[hashlib.sha256(row.tobytes()).hexdigest() for row in bytes_]})
        if stage == "final_norm":
            name = self.root / f"v4-replay-rank{self.rank}.json"
            with name.open("x") as f:
                json.dump({"rank":self.rank,"positions":self.pos,"stages":self.stages}, f)
