package org.flexlb.cache.domain;

import java.util.Collections;
import java.util.List;

/**
 * Block hashes calculated with the Local Standby block size.
 */
public record LocalStandbyHashResult(List<Long> blockCacheKeys, long blockSize) {

    public static LocalStandbyHashResult empty() {
        return new LocalStandbyHashResult(Collections.emptyList(), 0);
    }
}
