package org.flexlb.it.fixture.engine;

import org.flexlb.dao.route.RoleType;

import java.util.EnumMap;
import java.util.Map;

/**
 * Role-aware fake-worker declaration for one integration-test JVM.
 *
 * <p>The topology is intentionally role/count based rather than exposing fixed worker slots.
 * {@link ScriptedWorkerCluster} binds one topology for its JVM, matching Failsafe's isolated fork
 * lifecycle.
 *
 * @param workerCounts positive worker counts keyed by engine role
 */
public record WorkerTopology(Map<RoleType, Integer> workerCounts) {

    public WorkerTopology {
        EnumMap<RoleType, Integer> normalized = new EnumMap<>(RoleType.class);
        if (workerCounts != null) {
            workerCounts.forEach((role, count) -> {
                if (role == null || count == null || count <= 0) {
                    throw new IllegalArgumentException("Each worker topology count must be positive and role-specific");
                }
                normalized.put(role, count);
            });
        }
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("Worker topology must contain at least one role");
        }
        workerCounts = Map.copyOf(normalized);
    }

    /** Creates a topology that contains exactly one role. */
    public static WorkerTopology of(RoleType roleType, int workerCount) {
        return new WorkerTopology(Map.of(roleType, workerCount));
    }

    /** Returns the configured count for {@code roleType}, or zero when the role is absent. */
    public int workerCount(RoleType roleType) {
        return workerCounts.getOrDefault(roleType, 0);
    }
}
