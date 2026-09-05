package org.flexlb.config;

import java.util.Optional;

/**
 * Local Standby settings that can be applied without rebuilding cache-match components.
 */
public record LocalStandbyRuntimeSettings(
        long maximumEntries,
        double capacityMultiplier,
        long ttlMs,
        long minimumTtlMs,
        double ttlReductionStartRatio) {

    /**
     * Extracts the Local Standby fields that can change at runtime.
     *
     * <p>Configuration snapshots are validated by {@link ConfigService} before their settings are
     * applied.
     */
    public static LocalStandbyRuntimeSettings from(LocalStandbyConfig config) {
        LocalStandbyConfig localStandby = config == null ? new LocalStandbyConfig() : config;
        long maximumEntries = localStandby.getMaximumEntries();
        double capacityMultiplier = localStandby.getCapacityMultiplier();
        long ttlMs = localStandby.getTtlMs();
        long minimumTtlMs = localStandby.getMinimumTtlMs();
        double ttlReductionStartRatio = localStandby.getTtlReductionStartRatio();
        return new LocalStandbyRuntimeSettings(
                maximumEntries, capacityMultiplier, ttlMs, minimumTtlMs, ttlReductionStartRatio);
    }

    /**
     * Extracts Local Standby runtime settings from a complete FlexLB configuration snapshot.
     *
     * <p>The extraction intentionally does not require KVCM to be active; runtime consumers only
     * need the nested Local Standby settings when they are present.
     */
    public static Optional<LocalStandbyRuntimeSettings> fromFlexlbConfig(FlexlbConfig config) {
        return findLocalStandbyConfig(config)
                .map(LocalStandbyRuntimeSettings::from);
    }

    /**
     * Finds the Local Standby configuration in a complete FlexLB configuration snapshot.
     *
     * <p>A LOCAL_SYNC snapshot has no Local Standby configuration and returns an empty value.
     */
    public static Optional<LocalStandbyConfig> findLocalStandbyConfig(FlexlbConfig config) {
        return Optional.ofNullable(config)
                .map(FlexlbConfig::getCacheMatching)
                .filter(KvcmCacheMatchingConfig.class::isInstance)
                .map(KvcmCacheMatchingConfig.class::cast)
                .map(KvcmCacheMatchingConfig::getLocalStandby);
    }
}
