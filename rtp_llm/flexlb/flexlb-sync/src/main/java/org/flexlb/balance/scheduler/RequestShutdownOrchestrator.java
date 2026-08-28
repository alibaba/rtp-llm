package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Objects;

/** Owns the one ordered shutdown protocol for request and endpoint lifecycles. */
@Component
final class RequestShutdownOrchestrator {

    private final RequestLifecycleCoordinator lifecycle;
    private final EndpointRegistry registry;
    private final Runnable closePlacement;

    RequestShutdownOrchestrator(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry) {
        this(lifecycle, registry, () -> { });
    }

    @Autowired
    RequestShutdownOrchestrator(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry,
            RequestScheduler scheduler) {
        this(lifecycle, registry, scheduler::closePlacement);
    }

    private RequestShutdownOrchestrator(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry,
            Runnable closePlacement) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.registry = Objects.requireNonNull(registry, "registry");
        this.closePlacement = Objects.requireNonNull(
                closePlacement, "closePlacement");
    }

    @PreDestroy
    void shutdown() {
        Throwable failure = null;
        boolean ownsShutdown;
        try {
            closePlacement.run();
        } catch (Throwable closeFailure) {
            failure = closeFailure;
        }
        try {
            ownsShutdown = lifecycle.closeAdmissionAndAwaitMutations();
        } catch (Throwable closeFailure) {
            // The production gate is total. If an implementation violates that
            // contract, retain the failure and still attempt every close leaf.
            failure = append(failure, closeFailure);
            ownsShutdown = true;
        }
        if (!ownsShutdown) {
            return;
        }

        try {
            try {
                lifecycle.closeOutstandingAndTerminalize();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
            try {
                lifecycle.closeExpiration();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
            try {
                registry.close();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
        } finally {
            try {
                lifecycle.closePublisher();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
        }
        rethrow(failure);
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            try {
                first.addSuppressed(next);
            } catch (Throwable ignored) {
                // Preserve the fixed primary failure and continue shutdown.
            }
        }
        return first;
    }

    private static void rethrow(Throwable failure) {
        if (failure == null) {
            return;
        }
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException("Request shutdown failed", failure);
    }
}
