package org.flexlb.it.configuration;

import org.flexlb.enums.BlockHashStrategyType;

/**
 * Engine-facing cache-key contract used by an integration-test context.
 *
 * <p>VLLM and SGLang derive keys from {@code input_ids} and query KVCM. RTP-LLM deliberately
 * models the production caller-provided {@code block_cache_keys} contract, so it disables KVCM
 * and uses the local worker cache-status index instead.
 */
public enum EngineMode {
    VLLM(BlockHashStrategyType.VLLM, true),
    SGLANG(BlockHashStrategyType.SGLANG, true),
    RTP_LLM(BlockHashStrategyType.VLLM, false);

    private final BlockHashStrategyType blockHashStrategy;
    private final boolean usesKvcm;

    EngineMode(BlockHashStrategyType blockHashStrategy, boolean usesKvcm) {
        this.blockHashStrategy = blockHashStrategy;
        this.usesKvcm = usesKvcm;
    }

    /** Returns the configured strategy for modes that derive block keys in FlexLB. */
    public BlockHashStrategyType blockHashStrategy() {
        return blockHashStrategy;
    }

    /** Returns whether this mode configures the scripted KVCM boundary. */
    public boolean usesKvcm() {
        return usesKvcm;
    }
}
