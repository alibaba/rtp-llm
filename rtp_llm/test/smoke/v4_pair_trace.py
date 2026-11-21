"""One additional authorized replay on an already-running V4 ARM smoke service."""
import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path

TARGET = "//rtp_llm/test/smoke:smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100"
SCHEMA = "rtp.v4.live38.v2"
FIXTURE = "v4_0731_xgrammar_17"
SHA = "cba0af7f8765ce15615d4bb2c34e043080fbbc8816c41623db1abaa9578c3e47"
PROMPT_SHA = "9bfcb2dd441fbf20c44897a746a66b3add4d5bc2b42d965f1845e318e1505e1c"
STAGES = ["embed", *(f"layer.{i}" for i in range(43)), "final_norm"]
RAW_LIVE = ["embed.raw_hc", *(f"layer.{i}.raw_hc" for i in range(43))]
PROMPT = '<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>查询下北京的天气，json格式返回<｜Assistant｜></think>{"city":"北京","weather":"晴","temperature":"15°C","humidity":"30%","wind":"北风 3级","update_time":"2025-04-'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def validsha(s):
    return isinstance(s, str) and re.fullmatch(r"[0-9a-f]{64}",s) is not None


def loader(path):
    seen = set()
    with Path(path).open(errors="replace") as f:
        for line in f:
            if "finally choose load method: fastsafetensors" in line: seen.add("oldfst")
            if re.search(r"NewLoader load method: requested=(?:auto|fastsafetensors), selected=fastsafetensors\b",line): seen.add("newfst")
            if "finally choose load method: scratch" in line or "selected=scratch" in line: seen.add("scratch")
    require(seen in ({"oldfst"},{"newfst"}),"ambiguous_or_missing_loader")
    return seen.pop()


def sample(root):
    lines = (root/"v4-target-witness-17-verify.txt").read_text().splitlines()
    require(len(lines)==1 and lines[0].startswith("v4_target_witness_v1 "),"missing_or_duplicate_sample")
    values = dict(s.split("=",1) for s in lines[0].split()[1:])
    require(values["phase"]=="verify" and int(values["case"])==17 and int(values["output_token_1based"])==38,"wrong_sample")
    chain = values["committed_sha_chain"].split(",")
    require(len(chain)==38 and all(validsha(s) for s in chain) and chain[0]==PROMPT_SHA and chain[-1]==values["prefix_sha256_le_i32"],"invalid_chain")
    keys = "row col width batch accept_len a_id b_id raw_argmax masked_argmax sampled_target emitted draft_id".split()
    record = {k:int(values[k]) for k in keys}
    record["draft_kind"]=values["draft_kind"]
    record["verify_exec"]=values["verify_exec"]
    require(record["verify_exec"] in ("eager","graph_runner"),"missing_verify_execution_mode")
    record["live_layers"]=[{"stage":k,"sha256":values[key],"dtype":values[dtype]} for k,key,dtype in (
        ("layer.40.mean_hc","live_aux40_sha","aux_dtype"),("layer.41.mean_hc","live_aux41_sha","aux_dtype"),
        ("layer.42.mean_hc","live_aux42_sha","aux_dtype"),("lm_head_input","live_lm_input_sha","head_dtype"))]
    require(all(validsha(z["sha256"]) and z["dtype"] in ("bf16","fp32") for z in record["live_layers"]),"incomplete_live_layers")
    raw = values["live_raw_hc_sha"].split(",")
    require(len(raw)==len(RAW_LIVE) and all(validsha(h) for h in raw),"missing_live_raw_layers")
    record["live_raw_layers"]=[{"stage":k,"sha256":h,"dtype":"bf16"} for k,h in zip(RAW_LIVE,raw)]
    for k in ("raw_a","raw_b","masked_a","masked_b"):
        x = float(values[k])
        require(math.isfinite(x) or (k.startswith("masked") and x==-math.inf),"invalid_score")
        record[k]=x if math.isfinite(x) else "-inf"
    require(record["a_id"]==3315 and record["b_id"]==2642 and
            record["row"]==record["batch"]*record["width"]+record["col"] and
            0<=record["col"]<record["accept_len"]<=record["width"],"invalid_verify_row")
    require((record["draft_kind"]=="bonus" and record["col"]==record["width"]-1 and record["draft_id"]==-1) or
            (record["draft_kind"]=="proposal" and record["col"]<record["width"]-1 and record["draft_id"]>=0),"invalid_draft")
    return chain, record


def merge_layers(root):
    ranks = [json.loads((root/f"v4-replay-rank{i}.json").read_text()) for i in (0,1)]
    for i,r in enumerate(ranks):
        require(r["rank"]==i and [v["stage"] for v in r["stages"]]==STAGES,"rank_or_stage_mismatch")
    positions = ranks[0]["positions"]+ranks[1]["positions"]
    require(sorted(positions)==list(range(55)),"missing_or_overlapping_cp_positions")
    result=[]
    for j,stage in enumerate(STAGES):
        rows={}
        for r in ranks:
            pos, hashes = r["positions"], r["stages"][j]["sha"]
            require(len(hashes)==len(pos) and all(validsha(s) for s in hashes),"invalid_replay_hash")
            rows.update(zip(pos,hashes))
        result.append({"stage":stage,"sha256":hashlib.sha256(b"".join(bytes.fromhex(rows[p]) for p in range(55))).hexdigest(),
                       "last_sha256":rows[54]})
    return result


def run_pair(managers):
    root, tmp = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]), Path(os.environ["TEST_TMPDIR"])
    require(root.is_absolute() and tmp.is_absolute() and os.environ.get("TEST_TARGET")==TARGET,"unauthorized_target")
    marker = tmp/"v4-live38-replay.on"
    leaf={"schema":SCHEMA,"fixture":FIXTURE,"complete":False,"generated_index":38,
          "layer_mode":"fixed_prefill_replay","layer_prefix_sha256":SHA}
    created=False
    errors=[]
    try:
        chain, record=sample(root)
        leaf.update(committed_prefix_sha256=chain[-1],committed_sha_chain=chain,sample=record)
    except Exception as exc:
        errors.append("sample:"+type(exc).__name__)
    try:
        roles={role:loader(managers[role].log_file_path) for role in ("prefill","decode")}
        require(roles["prefill"]==roles["decode"],"mixed_role_loaders")
        leaf.update(loader=roles["prefill"],role_loaders=roles)
    except Exception as exc:
        errors.append("loader:"+type(exc).__name__)
    try:
        with marker.open("x"):
            created=True
        payload={"prompt":PROMPT,"generate_config":{"max_new_tokens":1,"top_k":1,"is_streaming":False,
                 "can_use_pd_separation":False,"reuse_cache":False}}
        req=urllib.request.Request(f"http://127.0.0.1:{int(managers['prefill'].port)}/",data=json.dumps(payload).encode(),
                                   headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=600) as res:
            require(res.status==200,"replay_not_200")
            res.read(4096)
        leaf["layers"]=merge_layers(root)
    except Exception as exc:
        # Test/service exceptions may include HTTP bodies, filesystem paths or
        # raw logs. Persist only a fixed class and structural failure code.
        errors.append("replay:"+type(exc).__name__)
    finally:
        if created: marker.unlink()
        leaf.update(complete=not errors,errors=errors)
        out=root/"v4_pair_trace.json"
        with out.open("x") as f: json.dump(leaf,f,allow_nan=False)
    return leaf["complete"]
