from rtp_llm.omni.engine.stage_connector import StageOutput

CODEC_END_TOKEN = 8294


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
    filtered = [t for t in token_ids if t != CODEC_END_TOKEN]
    return StageOutput(
        token_ids=filtered,
        metadata={"codec_token_count": len(filtered)},
    )
