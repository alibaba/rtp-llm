package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.EndpointTrace.Stage;
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
        response.whenComplete((value, failure) -> {
            lease.trace().record(Stage.RPC_COMPLETED, unwrap(failure));
            lease.trace().record(Stage.CALLBACK_READY);
        }).handleAsync((value, failure) -> {
            try {
                lease.trace().record(Stage.CALLBACK_STARTED);
                callback.accept(value, unwrap(failure));
            } catch (Throwable callbackFailure) {
                lease.trace().record(Stage.CALLBACK_FAILED, callbackFailure);
                logger.error("{} callback failed for {}", kind, address, callbackFailure);
            }
            finally { lease.trace().record(Stage.CALLBACK_ENDED); }
            return null;
        }, executor).whenComplete((ignored, callbackFailure) -> {
            lease.close();
            if (callbackFailure != null) {
                lease.trace().record(Stage.CALLBACK_SUBMISSION_FAILED, unwrap(callbackFailure));
                logger.error("{} callback was not scheduled for {}", kind, address, unwrap(callbackFailure));
            }
        });
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }
}
