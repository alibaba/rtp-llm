package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import java.util.List;

/** Local-residency cache matching; remote KV payload loading is not enabled. */
@Getter
@Setter
public final class KvcmCacheMatchingConfig implements CacheMatchingConfig {
    private long requestTimeoutMs = 500;
    private long leaderRefreshIntervalMs = 10000;
    private int globalKvsHostCount;
    private boolean enableP2p;
    private List<String> medium = List.of("hbm");
}
