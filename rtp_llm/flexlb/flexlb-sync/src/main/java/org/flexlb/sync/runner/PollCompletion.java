package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.function.BiConsumer;

/** Completes asynchronous polls and releases their exact leases, including executor rejection. */
final class PollCompletion {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private PollCompletion() { }

    static <T> void registerResultCallback(WorkerStatus.PollLease lease, Executor executor, String kind, String address,
                           CompletableFuture<T> response, BiConsumer<T, Throwable> callback) {
        response.handleAsync((value, failure) -> {
            try {
                callback.accept(value, unwrap(failure));
            } catch (Throwable callbackFailure) {
                logger.error("{} callback failed for {}", kind, address, callbackFailure);
            }
            return null;
        }, executor).whenComplete((ignored, callbackFailure) -> {
            lease.close();
            if (callbackFailure != null) {
                logger.error("{} callback was not scheduled for {}", kind, address, unwrap(callbackFailure));
            }
        });
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }
}
