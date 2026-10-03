package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashConfig;

/**
 * Supplies the block hash configuration required to calculate request cache keys.
 */
public interface BlockHashConfigResolver {

    BlockHashConfig resolve();
}
