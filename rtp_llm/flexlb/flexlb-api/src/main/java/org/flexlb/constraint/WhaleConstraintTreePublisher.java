package org.flexlb.constraint;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;
import org.flexlb.constraint.ConstraintTreeModels.WorkerPublication;
import org.flexlb.constraint.ConstraintTreeModels.WorkerUpdateResponse;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Slf4j
@Service
public class WhaleConstraintTreePublisher implements ConstraintTreePublisher {

    static final String UPDATE_PATH = "/update_constraint_tree";
    static final String STATUS_PATH = "/constraint_tree_status";

    private final WorkerAddressService workerAddressService;
    private final GeneralHttpNettyService httpService;
    private final ExecutorService publishExecutor;
    private final Duration publishTimeout;

    @Autowired
    public WhaleConstraintTreePublisher(WorkerAddressService workerAddressService,
                                        GeneralHttpNettyService httpService) {
        this(workerAddressService,
                httpService,
                configuredConcurrency(),
                Duration.ofSeconds(configuredTimeoutSeconds()));
    }

    WhaleConstraintTreePublisher(WorkerAddressService workerAddressService,
                                 GeneralHttpNettyService httpService,
                                 int concurrency,
                                 Duration publishTimeout) {
        this.workerAddressService = workerAddressService;
        this.httpService = httpService;
        this.publishExecutor = Executors.newFixedThreadPool(
                Math.max(1, concurrency), new NamedThreadFactory("constraint-tree-publisher"));
        this.publishTimeout = publishTimeout;
    }

    @Override
    public PublicationResult publish(SerializedArtifact artifact) {
        Map<String, URI> targets = discoverTargets(artifact.metadata().model());
        if (targets.isEmpty()) {
            log.warn("no Whale inference workers discovered for constraint tree model={}, version={}",
                    artifact.metadata().model(), artifact.version());
            return new PublicationResult(0, 0, List.of());
        }

        List<Callable<WorkerPublication>> tasks = targets.entrySet().stream()
                .<Callable<WorkerPublication>>map(entry ->
                        () -> publishOne(entry.getKey(), entry.getValue(), artifact))
                .toList();
        List<WorkerPublication> results = new ArrayList<>(tasks.size());
        try {
            for (Future<WorkerPublication> future : publishExecutor.invokeAll(tasks)) {
                try {
                    results.add(future.get());
                } catch (Exception e) {
                    results.add(new WorkerPublication("unknown", false, 0, rootMessage(e)));
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new PublicationResult(targets.size(), 0,
                    List.of(new WorkerPublication("all", false, 0, "publication interrupted")));
        }

        int published = (int) results.stream().filter(WorkerPublication::success).count();
        log.info("constraint tree publication completed model={}, version={}, workers={}/{}, bytes={}",
                artifact.metadata().model(), artifact.version(), published, targets.size(), artifact.payload().length);
        return new PublicationResult(targets.size(), published, List.copyOf(results));
    }

    private Map<String, URI> discoverTargets(String model) {
        Map<String, URI> targets = new LinkedHashMap<>();
        addTargets(targets, workerAddressService.getEngineWorkerList(model, RoleType.DECODE));
        addTargets(targets, workerAddressService.getEngineWorkerList(model, RoleType.PDFUSION));
        return targets;
    }

    private void addTargets(Map<String, URI> targets, List<WorkerHost> hosts) {
        for (WorkerHost host : hosts) {
            if (host == null || host.getIp() == null || host.getIp().isBlank() || host.getHttpServerPort() <= 0) {
                continue;
            }
            String address = host.getIp() + ":" + host.getHttpServerPort();
            targets.putIfAbsent(address, URI.create("http://" + address));
        }
    }

    private WorkerPublication publishOne(String worker, URI uri, SerializedArtifact artifact) {
        try {
            WorkerPublication current = checkCurrent(worker, uri, artifact);
            if (current != null) {
                return current;
            }
            WorkerUpdateResponse response = httpService
                    .requestRawJson(artifact.payload(), uri, UPDATE_PATH, WorkerUpdateResponse.class)
                    .timeout(publishTimeout)
                    .block();
            if (response == null) {
                return new WorkerPublication(worker, false, 0, "worker returned an empty response");
            }
            String status = response.status() == null ? "" : response.status().toLowerCase(Locale.ROOT);
            boolean accepted = status.equals("accepted")
                    || status.equals("already_accepted")
                    || status.equals("updated")
                    || status.equals("already_current")
                    || status.equals("ready");
            return new WorkerPublication(worker, accepted, response.version(), response.message());
        } catch (Exception e) {
            log.warn("constraint tree publication failed worker={}, version={}: {}",
                    worker, artifact.version(), rootMessage(e));
            return new WorkerPublication(worker, false, 0, rootMessage(e));
        }
    }

    private WorkerPublication checkCurrent(String worker, URI uri, SerializedArtifact artifact) {
        try {
            WorkerUpdateResponse response = httpService
                    .get(uri, STATUS_PATH, WorkerUpdateResponse.class)
                    .timeout(publishTimeout)
                    .block();
            if (response == null) {
                return null;
            }
            if (response.version() > artifact.version()) {
                return new WorkerPublication(worker, false, response.version(),
                        "worker already has a newer tree version");
            }
            if (response.version() == artifact.version()) {
                return new WorkerPublication(worker, true, response.version(), "already current");
            }
            String status = response.status() == null ? "" : response.status().toLowerCase(Locale.ROOT);
            if (response.requestedVersion() == artifact.version()
                    && (status.equals("queued") || status.equals("loading"))) {
                return new WorkerPublication(worker, true, response.version(), "already loading");
            }
        } catch (Exception ignored) {
            // A status probe is only an optimization. The update POST remains authoritative.
        }
        return null;
    }

    private static int configuredConcurrency() {
        return configuredPositiveInt("CONSTRAINT_TREE_PUBLISH_CONCURRENCY", 2);
    }

    private static int configuredTimeoutSeconds() {
        return configuredPositiveInt("CONSTRAINT_TREE_PUBLISH_TIMEOUT_SECONDS", 120);
    }

    private static int configuredPositiveInt(String name, int defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            return Math.max(1, Integer.parseInt(value));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be an integer", e);
        }
    }

    private static String rootMessage(Throwable throwable) {
        Throwable current = throwable;
        while (current.getCause() != null) {
            current = current.getCause();
        }
        return current.getMessage() == null ? current.getClass().getSimpleName() : current.getMessage();
    }

    @PreDestroy
    public void destroy() {
        publishExecutor.shutdownNow();
    }
}
