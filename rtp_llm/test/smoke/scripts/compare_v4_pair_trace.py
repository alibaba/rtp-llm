"""Compare sanitized V4 Old/New JSON leaves; a real difference is exit 0."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_pair_trace import FIXTURE, PROMPT_SHA, RAW_LIVE, SCHEMA, SHA, STAGES, require, validsha


def validate(x,loader):
    require(x["schema"]==SCHEMA and x["fixture"]==FIXTURE and x["complete"] is True and not x["errors"],"incomplete")
    require(x["loader"]==loader and x["role_loaders"]=={"prefill":loader,"decode":loader},"loader_mismatch")
    require(x["generated_index"]==38 and x["layer_mode"]=="fixed_prefill_replay" and x["layer_prefix_sha256"]==SHA,"wrong_probe")
    c=x["committed_sha_chain"]
    require(len(c)==38 and all(validsha(v) for v in c) and c[0]==PROMPT_SHA and c[-1]==x["committed_prefix_sha256"],"invalid_chain")
    require([v["stage"] for v in x["layers"]]==STAGES and all(validsha(v["sha256"]) and validsha(v["last_sha256"]) for v in x["layers"]),"incomplete_layers")
    s=x["sample"]
    require([z["stage"] for z in s["live_raw_layers"]]==RAW_LIVE and
            all(z["dtype"]=="bf16" and validsha(z["sha256"]) for z in s["live_raw_layers"]),"missing_live_raw_layers")
    require(s["verify_exec"] in ("eager","graph_runner") and
        [z["stage"] for z in s["live_layers"]]==["layer.40.mean_hc","layer.41.mean_hc","layer.42.mean_hc","lm_head_input"] and
        all(validsha(z["sha256"]) and z["dtype"] in ("bf16","fp32") for z in s["live_layers"]),"missing_live_layers")
    require(s["a_id"]==3315 and s["b_id"]==2642 and 0<=s["col"]<s["accept_len"]<=s["width"] and s["row"]==s["batch"]*s["width"]+s["col"],"invalid_row")
    require(all(type(s[k]) is int and s[k]>=0 for k in ("row","batch","raw_argmax","masked_argmax","sampled_target","emitted")),"invalid_token")
    require((s["draft_kind"]=="bonus" and s["draft_id"]==-1 and s["col"]==s["width"]-1) or (s["draft_kind"]=="proposal" and s["draft_id"]>=0 and s["col"]<s["width"]-1),"invalid_draft")
    for k in ("raw_a","raw_b","masked_a","masked_b"):
        require((type(s[k]) in (float,int) and math.isfinite(s[k])) or (k.startswith("masked_") and s[k]=="-inf"),"invalid_score")


def compare(old,new):
    validate(old,"oldfst"); validate(new,"newfst")
    first=next((i for i,(a,b) in enumerate(zip(old["committed_sha_chain"],new["committed_sha_chain"])) if a!=b),None)
    replay=next((a["stage"] for a,b in zip(old["layers"],new["layers"]) if a["sha256"]!=b["sha256"]),None)
    last=next((a["stage"] for a,b in zip(old["layers"],new["layers"]) if a["last_sha256"]!=b["last_sha256"]),None)
    live_order=lambda s: s["live_raw_layers"]+s["live_layers"]
    first_live=next((a["stage"] for a,b in zip(live_order(old["sample"]),live_order(new["sample"])) if a!=b),None)
    return {"valid":True,"comparable_live_sample":first is None,"first_different_committed_output_1based":first,
            "first_different_observed_live_stage":first_live if first is None else None,
            "live_layers_0_to_39":"observed_raw_hc_verified_against_native_aux",
            "live38":{"old":old["sample"],"new":new["sample"]},
            "fixed_prefill_replay":{"first_different_layer":replay,"first_different_last_position_layer":last,
            "prefix_sha256":SHA,"same_as_both_live_histories":old["committed_prefix_sha256"]==new["committed_prefix_sha256"]==SHA},
            "note":"Fixed prefill replay is not the live decode or proof of causality."}


if __name__=="__main__":
    try:
        require(len(sys.argv)==3,"usage")
        report=compare(*(json.loads(Path(p).read_text()) for p in sys.argv[1:]))
    except (ValueError,KeyError,TypeError,IndexError,OSError) as e:
        print(json.dumps({"valid":False,"error":type(e).__name__}))
        sys.exit(2)
    print(json.dumps(report,allow_nan=False))
