package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.BlockHashStrategyType;

/**
 * Configuration-source-owned block hash algorithm parameters.
 *
 * <p>This model is deserialized as {@link FlexlbConfig#blockHashConfig} from FlexLB configuration
 * sources. {@link #type} selects the algorithm when the strategy bean is created, while
 * {@link #hashSeed} is the vLLM hash seed and can be applied to an active vLLM strategy by a
 * {@link ConfigService} update.
 *
 * <p>This is intentionally distinct from {@code org.flexlb.cache.domain.WorkerBlockHashConfig},
 * whose {@code blockSize} and {@code lookaheadTokens} are reported by alive engine workers. Do
 * not put those worker-status-derived values in this configuration model.
 */
@Getter
@Setter
public class BlockHashConfig {

    /**
     * Preferred block hash algorithm. When absent, FlexLB falls back to the deprecated
     * {@link FlexlbConfig#blockHashStrategy}. Changing this value requires a restart because the
     * strategy bean is selected during construction.
     */
    private BlockHashStrategyType type;

    /**
     * vLLM {@code sha256_cbor} seed. This configurable value defaults to {@code "0"} and is
     * updated at runtime for an active vLLM strategy through the {@link ConfigService} listener.
     * It has no effect when the resolved strategy is SGLang.
     */
    private String hashSeed = "0";
}
