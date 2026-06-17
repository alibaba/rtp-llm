from rtp_llm.omni.engine.stage_connector import StageOutput

# Codec special tokens (pad=8292, bos=8293, eos=8294, etc.) occupy IDs >= this value.
# Must be filtered before passing to the vocoder.  Matches the C++ production path
# in omni_engine.py which uses `codec_tokens[0] < CODEC_SPECIAL_TOKEN_MIN`.
CODEC_SPECIAL_TOKEN_MIN = 8292


def thinker2talker(source_output: StageOutput) -> StageOutput:
    return StageOutput(
        embeddings=source_output.embeddings,
        metadata={
            "source_token_ids": source_output.token_ids,
            "source_text": source_output.metadata.get("text", ""),
        },
    )


def talker2code2wav(source_output: StageOutput) -> StageOutput:
    token_ids = source_output.token_ids or []
    filtered = [t for t in token_ids if t < CODEC_SPECIAL_TOKEN_MIN]
    return StageOutput(
        token_ids=filtered,
        metadata={"codec_token_count": len(filtered)},
    )
