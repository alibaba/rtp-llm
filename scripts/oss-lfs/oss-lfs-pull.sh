#!/bin/bash
# Download all OSS-hosted large files referenced by tests.
#
# Single URL pattern, single hash algo, single function shape:
#
#     download <local-target-path> <sha256>
#
# URL is always <OSS_BASE>/lfs/<sha256> — content-addressed; the same hash
# both locates the object on OSS and validates the bytes after download.
#
# To add a new file:
#   1. sha=$(sha256sum FILE | cut -d' ' -f1)
#   2. ossutil cp FILE oss://rtp-opensource/lfs/$sha
#      (--access-key-id / --access-key-secret / --region cn-hangzhou required)
#   3. add `download "<path-in-repo>" "$sha" & pids+=($!)` to the manifest
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OSS_BASE="https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    local target="$1" expected="$2"
    if [ "${#expected}" -ne 64 ]; then
        echo "[oss-lfs] FAIL invalid sha256 (len=${#expected}) for $target" >&2
        return 1
    fi

    mkdir -p "$(dirname "$target")"
    # Symlinks (legacy layout where targets pointed into rtp_llm/test-data/)
    # must be replaced with a real file so bazel runfiles materialize correctly;
    # never short-circuit on a symlink even if its target hash matches.
    if [ -L "$target" ]; then
        rm -f "$target"
    elif [ -f "$target" ] && [ "$(sha256_of "$target")" = "$expected" ]; then
        return 0
    fi

    echo "[oss-lfs] downloading $target"
    # Retry transient network failures — CI runners and developer machines
    # behind flaky networks routinely see one-shot curl failures resolve on
    # the next attempt. Keep this short (3 tries, 1s/2s backoff) so a truly
    # dead OSS object still fails fast rather than hanging the whole pull.
    local attempt
    for attempt in 1 2 3; do
        if curl -sS -f "$OSS_BASE/lfs/$expected" -o "$target.tmp"; then
            break
        fi
        if [ "$attempt" = 3 ]; then
            echo "[oss-lfs] FAIL download $target (sha=$expected) after 3 attempts" >&2
            rm -f "$target.tmp"
            return 1
        fi
        echo "[oss-lfs] retry $attempt/3 for $target" >&2
        sleep "$attempt"
    done
    local actual; actual=$(sha256_of "$target.tmp")
    if [ "$actual" != "$expected" ]; then
        echo "[oss-lfs] FAIL sha256 $target expected=$expected got=$actual" >&2
        rm -f "$target.tmp"
        return 1
    fi
    mv "$target.tmp" "$target"
    git update-index --assume-unchanged "$target" 2>/dev/null || true
}

# ---- Manifest: 45 entries ------------------------------------------------
# Each line: download <local-path> <sha256> & pids+=($!)
pids=()

# testdata fixtures — written directly to the source-tree path so bazel
# package globs (e.g. //rtp_llm/test/tokenizer_test:testdata) pick up the real
# file. Pre-restructure these were git-LFS pointers; the restructure replaced
# them with symlinks into rtp_llm/test-data/, which broke runfiles
# materialization. We download into the source path and let the symlink-
# replacement branch in download() handle the legacy symlinks on first run.
download "rtp_llm/test/model_test/fake_test/testdata/kimi_k2/tokenizer/chat_template.jinja" "8a4cd421d3a014c1700ce2adddc16c0ab8c3bdff2f13622c7d93d1fcd6418c6a" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/kimi_k2/tokenizer/tiktoken.model" "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source/pytorch_model.bin" "30e2f243cf6380041d865181f71f9d1d270162b1b712c06fed66cd5aee50d332" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source/tokenizer.model" "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/qwen_0.5b/layer_0.pt" "7e212d49f6390a5d1a87dfd992f05774341422c7533d89fb997f3c08384e9111" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/qwen_0.5b/layer_1.pt" "71556ef1c1113924305f4aa69a74463028479943fb115172d7451e4803f7f95c" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/qwen_0.5b/layer_2.pt" "11965410f2bee2811c3ac2aa736822819ca085b86e434c53d6e5e3006d8b5427" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/qwen_0.5b/layer_3.pt" "4bd80b259307e5098aeec4b8ad674c549e333e46d621dfa5e230c46884bba5fc" & pids+=($!)
download "rtp_llm/test/model_test/fake_test/testdata/qwen_0.5b/layer_4.pt" "2e86f6c89e13b40c3bb37868d4502fdb15589b02cddfc56ec553892748589c46" & pids+=($!)
download "rtp_llm/test/tokenizer_test/testdata/chatglm3_tokenizer/tokenizer.model" "e7dc4c393423b76e4373e5157ddc34803a0189ba96b21ddbb40269d31468a6f2" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/bin_testdata/pytorch_model.bin" "30e2f243cf6380041d865181f71f9d1d270162b1b712c06fed66cd5aee50d332" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/lora_testdata/adapter_model.bin" "644fbc1d45c5985325ed4d628c184a3bd493d65183888604a70ff5ca3b37161a" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/lora_testdata_safetensor/adapter_model.safetensors" "06845a0c6b47b0a7b419af543ed54a0fc508cbc916df0526fd2013f3839dccc3" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/pt_testdata/test.pt" "8662568434af656ed9970cf41d57c55ddaf7f0248742df90aa83214718ddcd3b" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/safetensor_testdata/test.safetensors" "ab0f233a43d26026aed7f12a17ab6b15764dfd3bd7002124b67b82e9c714a65b" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/mixture_testdata/pytorch_model.bin" "30e2f243cf6380041d865181f71f9d1d270162b1b712c06fed66cd5aee50d332" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/mixture_testdata/test.pt" "8662568434af656ed9970cf41d57c55ddaf7f0248742df90aa83214718ddcd3b" & pids+=($!)
download "rtp_llm/utils/test/testdata/ckpt_database_testdata/mixture_testdata/test.safetensors" "ab0f233a43d26026aed7f12a17ab6b15764dfd3bd7002124b67b82e9c714a65b" & pids+=($!)

# smoke/data files (rtp_llm/test/smoke/data/...) — pointer files in repo, overwritten in place
download "rtp_llm/test/smoke/data/model/bert/colbert_roberta_epect_1.pt" "cafb2a5f9a47b7dcfb98c0f073c9fab21f0a44be48e665f1439fb3e5d64fabb3" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/colbert_roberta_expect_0.pt" "3e9de8c24548d49e6cf63594550326597373a1cf32f5034ac524fff1a271f058" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/colbert_roberta_expect_1.pt" "92861837c099bbec05376c1c2add1435ed4060fd1bf59ca363b0edb22fb2ce7f" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/expect.pt" "be5c2cc3ce5f81be52616771ee7122f37566d41f3499f137c34c2bfabdda8ec6" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/expect_tp2.pt" "d0388ab4d415f8410cfd08dd875e49f393b0298170fb0e75038623f5d43a5b06" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/megatron_bert_expect_1.pt" "2c47607b37d246a89e0be4af3b4e1e389bb1ca870721468a377683749b591be3" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/megatron_bert_expect_2.pt" "f71769c4b5fc7999811c86fcc43dfc64f148e4f51b123024fcf4a265ded69625" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/roberta_expect.pt" "fe410aa5a99ecdd4cc8ad3cd8149bf0df5023460218b50cb7f41292c074ffb08" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_0.pt" "9d1d56992badbb32631cda00a4778e384c03fd6a51ab65da9097f68ff8800f73" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_1.pt" "8fe7d940180cd0e993f04a11d292d0fb9322fa3dbd926a2d26e06673248153a9" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_2.pt" "e05412706cba0fa8b0e147fa663541e6668aeb6796e2fdb226fca2b3e25c43b4" & pids+=($!)
download "rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_3.pt" "e0bb7cf792941c6b4e98d6a0706d8da24200d1acc0f2e6da6fe3d7af9af09d10" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/expect_3090.pt" "d1a0c913d1280c06f991247ba27a7a75c124c70e42fe114453ab9e8f9c68e091" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/expect_int8_3090.pt" "c87e6c72cff97398a4e5c6926350edc1daee35c515de7cb62fc726a2beafc593" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/expect_int8_ppu.pt" "463aa21076c452d1cd2b1cb14b7e75b00fefeff1dd8543bd7b33a250b025be9c" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/expect_ppu.pt" "f6f53507911b2175fdcbfc909c7d7aeea7fec136b877d26a7e24edfca8031953" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/logits_3090.pt" "bb771f1d370f91ab0b41a2317573045ff7d359a19f00cf5ae1a0c96acb5309bc" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/logits_int8_3090.pt" "6add21cb7aa1f9b142c3526d7c19beeb889c871317a8effa1e0b4dd34aa8c329" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/logits_int8_ppu.pt" "ac067643fbf674ea63f1c957b7fd4a2809951cff5221bd79e6ffa5ab98107948" & pids+=($!)
download "rtp_llm/test/smoke/data/model/deepseek_v2/logits_ppu.pt" "ce93670d6b9395bf0f2a556abd8f8fe293fad0b2096dd146e7dfc5d7d6bf5390" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen25/expect_embedding.pt" "5010b0c9e225b0195ee67f128e5994bfd930ce2ae65b6cf98e5e87350d7d04a3" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen25/expect.pt" "5b979bd25cd417c76d5ce532e03468c589ddfa8c18cf4a427aa7706c21c665b6" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen25/expect_select_tokens_logits.pt" "21d784c2066cb38855980c6c2fae0d9a63dbdb80bd3c42f9c88df3ae35dbac16" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen25/input_embedding_fp32.pt" "a7409b7b0c205deea1c49671a8738380cad67c6d3a2a6e7a13983faa2e24c417" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen25/input_embedding.pt" "2170688000c1b9d77d65e524aa410b9759da59039d7cba0dcaa93355e0d83ee6" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen2/gte-embedding.pt" "8aa4d7534c7bb0e3944c442bdd57c5b30404cf780bf8cb7c2e82dea44e47d6b0" & pids+=($!)
download "rtp_llm/test/smoke/data/model/qwen3/qwen3_embed_expect.pt" "fa39ecd9bf08b58f1bdd20e425e667c53b4c2ab1f3a5346c8e3ae94524a1004f" & pids+=($!)

# ---- Wait + propagate failure -------------------------------------------
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

if [ "$failed" -gt 0 ]; then
    echo "[oss-lfs] $failed/${#pids[@]} file(s) FAILED" >&2
    exit 1
fi
echo "[oss-lfs] all ${#pids[@]} downloads completed"
