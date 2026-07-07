#!/usr/bin/env python3
"""Load a single model once (old OR new loader, decided by USE_NEW_LOADER env)
and let the DUMP_WEIGHTS hook write the fingerprint json. Run INSIDE the GPU
container; invoked by run_dump.sh. Not meant to be run on the Mac side.

IMPORTANT: this driver takes NO command-line args. rtp_llm's config system runs
its own global argparse over sys.argv during model/config init and would abort on
unknown flags. So everything comes through env vars (same convention as the
server_test --test_env=... flow):

  MODEL_TYPE        required, e.g. qwen_3_moe
  CHECKPOINT_PATH   required, ckpt dir
  TOKENIZER_PATH    optional, defaults to CHECKPOINT_PATH
  ACT_TYPE          optional, defaults to bf16
  QUANTIZATION      optional, defaults to "" (none)
  TP_SIZE           optional, defaults to 1 (single-process; tp>1 needs torchrun)

  USE_NEW_LOADER=1  -> NewModelLoader (rtp_llm/models_py)
  USE_NEW_LOADER=0  -> legacy ModelLoader (rtp_llm/model_loader)
  LOAD_METHOD       auto|scratch|fastsafetensors  (honored by FakeModelLoader)
  DUMP_WEIGHTS=/dir -> both loaders dump rank{tp_rank}.json there
"""
import logging
import os

logging.basicConfig(level=logging.INFO)


def main():
    model_type = os.environ.get("MODEL_TYPE")
    ckpt = os.environ.get("CHECKPOINT_PATH")
    if not model_type or not ckpt:
        raise SystemExit(
            "MODEL_TYPE and CHECKPOINT_PATH env vars are required "
            f"(got MODEL_TYPE={model_type!r}, CHECKPOINT_PATH={ckpt!r})"
        )
    tokenizer = os.environ.get("TOKENIZER_PATH") or ckpt
    act_type = os.environ.get("ACT_TYPE", "bf16")
    quant = os.environ.get("QUANTIZATION", "")
    tp = int(os.environ.get("TP_SIZE", "1"))

    if tp != 1:
        # tp>1 requires a distributed launch (torchrun) so each rank sets
        # ParallelismConfig.tp_rank; FakeModelLoader uses defaults (tp=1).
        # Validate single-rank weight correctness first, then extend.
        logging.warning(
            "TP_SIZE=%d but FakeModelLoader runs single-process (tp=1). "
            "TP-slicing validation needs a torchrun harness; see PORTING_GUIDE.md.",
            tp,
        )

    from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader

    use_new = os.environ.get("USE_NEW_LOADER", "0") == "1"
    logging.info(
        "Loading model_type=%s via %s loader (DUMP_WEIGHTS=%s)",
        model_type,
        "NEW" if use_new else "OLD",
        os.environ.get("DUMP_WEIGHTS"),
    )

    loader = FakeModelLoader(
        model_type=model_type,
        tokenizer_path=tokenizer,
        ckpt_path=ckpt,
        act_type=act_type,
        quantization=quant,
        load_py_model=use_new,
    )
    model = loader.init_model()
    logging.info("Loaded OK. py_model=%s", type(getattr(model, "py_model", None)))


if __name__ == "__main__":
    main()
