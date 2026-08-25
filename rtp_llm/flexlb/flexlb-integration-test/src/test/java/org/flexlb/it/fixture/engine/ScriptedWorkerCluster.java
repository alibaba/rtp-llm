package org.flexlb.it.fixture.engine;

import org.flexlb.dao.route.RoleType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Owns the role-aware worker collection for one Failsafe JVM.
 *
 * <p>A cluster cannot change topology after startup. This protects the production static worker
 * status maps from a test accidentally combining incompatible endpoint declarations.
 */
final class ScriptedWorkerCluster implements AutoCloseable {

    private final Map<RoleType, List<ScriptedWorker>> workersByRole = new EnumMap<>(RoleType.class);
    private WorkerTopology topology;

    synchronized void start(WorkerTopology requestedTopology) throws IOException {
        if (topology != null && !topology.equals(requestedTopology)) {
            throw new IllegalStateException(
                    "A scripted worker cluster is bound to one topology per test JVM; requested="
                            + requestedTopology + ", active=" + topology);
        }
        if (topology == null) {
            topology = requestedTopology;
            for (Map.Entry<RoleType, Integer> entry : topology.workerCounts().entrySet()) {
                List<ScriptedWorker> workers = new ArrayList<>(entry.getValue());
                for (int index = 0; index < entry.getValue(); index++) {
                    workers.add(new ScriptedWorker(IntegrationTestFixtures.WORKER_IP, entry.getKey()));
                }
                workersByRole.put(entry.getKey(), workers);
            }
        }
        for (ScriptedWorker worker : allWorkers()) {
            worker.start();
        }
    }

    ScriptedWorker worker(RoleType roleType, int index) {
        List<ScriptedWorker> workers = workersByRole.get(roleType);
        if (workers == null || index < 0 || index >= workers.size()) {
            throw new IllegalArgumentException("No scripted worker for role=" + roleType + ", index=" + index);
        }
        return workers.get(index);
    }

    Collection<ScriptedWorker> allWorkers() {
        return workersByRole.values().stream().flatMap(List::stream).toList();
    }

    WorkerTopology topology() {
        if (topology == null) {
            throw new IllegalStateException("Scripted worker cluster has not started");
        }
        return topology;
    }

    @Override
    public synchronized void close() throws InterruptedException {
        for (ScriptedWorker worker : allWorkers()) {
            worker.close();
        }
        workersByRole.clear();
        topology = null;
    }
}
