package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.FlexlbConfig;

import java.util.HashMap;
import java.util.Map;

/**
 * Read-only cluster view captured before building an admission plan.
 *
 * <p>Versions captured here (prefill queue versions) are validated at commit
 * time. Decode admission versions are captured later, right after the
 * scheduler's own reservation, so that only interference between plan build
 * and commit is detected.
 *
 * <p>O(1) snapshot redesign: the normal placement path consumes only
 * aggregate scalars, while the full per-entry decode lists are needed solely
 * by the low-frequency eviction/failure-classification paths. With the
 * {@code autoTpmSnapshotMode} gray switch set to {@code summary}, capture
 * takes only {@link DecodeEndpointSummary} aggregates (atomic getters, no
 * admission lock, no per-entry deep copy) and {@link #decodes()} lazily
 * upgrades to full snapshots on demand. At the default ({@code full}) the
 * legacy eager full capture is kept bit-for-bit.
 *
 * @param prefills        prefill endpoint snapshots keyed by "ip:httpPort"
 * @param decodeSummaries lightweight decode aggregate summaries keyed by
 *                        "ip:httpPort" (captured in both modes)
 * @param eagerDecodes    legacy eager full decode snapshots keyed by
 *                        "ip:httpPort"; {@code null} in summary mode, where
 *                        {@link #decodes()} lazily builds them instead
 */
public record ClusterSnapshot(
        Map<String, PrefillEndpointSnapshot> prefills,
        Map<String, DecodeEndpointSummary> decodeSummaries,
        Map<String, DecodeEndpointSnapshot> eagerDecodes) {

    public static ClusterSnapshot capture(EndpointRegistry registry, FlexlbConfig config) {
        return config.isAutoTpmSnapshotSummaryMode()
                ? captureSummary(registry, config)
                : captureFull(registry, config);
    }

    /** Legacy path (switch OFF): eager full decode snapshots at capture time. */
    static ClusterSnapshot captureFull(EndpointRegistry registry, FlexlbConfig config) {
        return new ClusterSnapshot(capturePrefills(registry, config),
                captureSummaries(registry, config), captureDecodes(registry, config));
    }

    /** Summary path (switch ON): aggregate scalars only, no per-entry copies. */
    static ClusterSnapshot captureSummary(EndpointRegistry registry, FlexlbConfig config) {
        return new ClusterSnapshot(capturePrefills(registry, config),
                captureSummaries(registry, config), null);
    }

    private static Map<String, PrefillEndpointSnapshot> capturePrefills(EndpointRegistry registry,
                                                                        FlexlbConfig config) {
        Map<String, PrefillEndpointSnapshot> prefills = new HashMap<>();
        registry.getPrefillEndpoints().forEach((key, ep) ->
                prefills.put(key, PrefillEndpointSnapshot.capture(ep, config.getFlexlbBatchQueueMaxSize())));
        return Map.copyOf(prefills);
    }

    private static Map<String, DecodeEndpointSummary> captureSummaries(EndpointRegistry registry,
                                                                       FlexlbConfig config) {
        Map<String, DecodeEndpointSummary> summaries = new HashMap<>();
        registry.getDecodeEndpoints().forEach((key, ep) ->
                summaries.put(key, DecodeEndpointSummary.capture(ep, config.getDecodeConcurrencyLimit())));
        return Map.copyOf(summaries);
    }

    /**
     * Fresh full decode snapshots straight from the live endpoints. Used by
     * the eviction/failure-classification paths when the schedule loop runs
     * on a TTL-cached snapshot: a cached view may predate the newest
     * reservations, and planning victims from it would make a just-reserved
     * lower-priority request invisible. Immutable for the same sharing reason
     * as the snapshot maps.
     */
    public static Map<String, DecodeEndpointSnapshot> captureDecodes(EndpointRegistry registry,
                                                                     FlexlbConfig config) {
        Map<String, DecodeEndpointSnapshot> decodes = new HashMap<>();
        registry.getDecodeEndpoints().forEach((key, ep) ->
                decodes.put(key, DecodeEndpointSnapshot.capture(ep, config.getDecodeConcurrencyLimit())));
        // Immutable maps: the snapshot may be shared across requests by the
        // short-TTL cache (flexlbClusterSnapshotCacheTtlMs), so it must not
        // be mutable anywhere downstream.
        return Map.copyOf(decodes);
    }

    /**
     * Full decode snapshots keyed by "ip:httpPort". In full mode this returns
     * the eagerly captured map (same instance on every call — the legacy
     * decision path is unchanged). In summary mode this lazily upgrades every
     * summary via {@link DecodeEndpointSummary#toFullSnapshot()}, building a
     * fresh map per call: each call acquires the admission lock per endpoint,
     * so callers on the eviction/classification paths should call it once and
     * reuse the result.
     */
    public Map<String, DecodeEndpointSnapshot> decodes() {
        if (eagerDecodes != null) {
            return eagerDecodes;
        }
        Map<String, DecodeEndpointSnapshot> full = new HashMap<>(decodeSummaries.size());
        decodeSummaries.forEach((key, summary) -> full.put(key, summary.toFullSnapshot()));
        return Map.copyOf(full);
    }
}
