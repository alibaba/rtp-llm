package org.flexlb.cache.hash;

import org.flexlb.cache.domain.WorkerBlockHashConfig;

/**
 * Supplies worker-status-derived block shape parameters required to calculate request cache keys.
 *
 * <p>The returned {@link WorkerBlockHashConfig} contains only {@code blockSize} and
 * {@code lookaheadTokens}; it is independent of the configurable algorithm type and hash seed in
 * {@link org.flexlb.config.BlockHashConfig}.
 */
public interface BlockHashConfigResolver {

    WorkerBlockHashConfig resolve();
}
