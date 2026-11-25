package org.flexlb.engine.grpc.client;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import io.grpc.ManagedChannel;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.timing.grpc.PullRequestTimingsRequest;
import org.flexlb.timing.grpc.PullRequestTimingsResponse;
import org.flexlb.timing.grpc.RequestTimingServiceGrpc;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Pulls bounded timing summaries from DashServing's inference gRPC port.
 */
@Component
public class RequestTimingClient {
    private final GrpcChannelFactory channelFactory;
    private final Map<GrpcTarget, ManagedChannel> channels = new HashMap<>();

    public RequestTimingClient(GrpcChannelFactory channelFactory) {
        this.channelFactory = channelFactory;
    }

    public CompletableFuture<PullRequestTimingsResponse> pull(GrpcTarget target,
                                                              PullRequestTimingsRequest request,
                                                              long timeoutMs) {
        ManagedChannel channel;
        synchronized (channels) {
            channel = channels.computeIfAbsent(target, channelFactory::create);
        }
        var future = RequestTimingServiceGrpc.newFutureStub(channel)
                .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS).pullRequestTimings(request);
        CompletableFuture<PullRequestTimingsResponse> result = new CompletableFuture<>();
        Futures.addCallback(future, new FutureCallback<>() {
            @Override
            public void onSuccess(PullRequestTimingsResponse response) {
                result.complete(response);
            }

            @Override
            public void onFailure(Throwable failure) {
                result.completeExceptionally(failure);
            }
        }, Runnable::run);
        result.whenComplete((response, failure) -> {
            if (result.isCancelled()) {
                future.cancel(true);
            }
        });
        return result;
    }

    public void remove(GrpcTarget target) {
        ManagedChannel channel;
        synchronized (channels) {
            channel = channels.remove(target);
        }
        if (channel != null) {
            channel.shutdownNow();
        }
    }

    @PreDestroy
    public void close() {
        synchronized (channels) {
            channels.values().forEach(ManagedChannel::shutdownNow);
            channels.clear();
        }
    }
}
