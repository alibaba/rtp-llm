package org.flexlb.engine.grpc;

import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.netty.channel.EventLoopGroup;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;

import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Test-only access to deterministic Engine channel readiness. */
public final class EngineGrpcClientTestAccess {

    private EngineGrpcClientTestAccess() {
    }

    public static EngineGrpcClient create(
            CustomNameResolver nameResolver,
            ThreadPoolExecutor executor,
            EventLoopGroup eventLoopGroup,
            GrpcReporter reporter,
            int connectTimeoutMillis) {
        return new EngineGrpcClient(
                nameResolver,
                executor,
                eventLoopGroup,
                reporter,
                connectTimeoutMillis);
    }

    public static void awaitBatchEnqueueReady(
            EngineGrpcClient client,
            String ip,
            int port,
            long timeout,
            TimeUnit unit) throws InterruptedException, TimeoutException {
        Objects.requireNonNull(client, "client");
        Objects.requireNonNull(ip, "ip");
        Objects.requireNonNull(unit, "unit");
        String channelKey = AbstractGrpcClient.createKey(
                ip, port, AbstractGrpcClient.ServiceType.BATCH_ENQUEUE);
        var invoker = client.channelPool.get(channelKey);
        if (invoker == null) {
            ManagedChannel candidate = client.createChannel(channelKey);
            invoker = client.putInvokerIfAbsent(channelKey, candidate);
        }

        ManagedChannel channel = invoker.getChannel();
        long deadlineNanos = System.nanoTime() + unit.toNanos(timeout);
        ConnectivityState state = channel.getState(true);
        while (state != ConnectivityState.READY
                && System.nanoTime() < deadlineNanos) {
            if (state == ConnectivityState.SHUTDOWN) {
                throw new IllegalStateException(
                        "BATCH_ENQUEUE channel shut down before READY: "
                                + channelKey);
            }
            if (state == ConnectivityState.TRANSIENT_FAILURE) {
                channel.resetConnectBackoff();
            }
            CountDownLatch changed = new CountDownLatch(1);
            channel.notifyWhenStateChanged(state, changed::countDown);
            long remainingNanos = deadlineNanos - System.nanoTime();
            if (remainingNanos <= 0L) {
                break;
            }
            changed.await(
                    Math.min(remainingNanos, TimeUnit.MILLISECONDS.toNanos(250)),
                    TimeUnit.NANOSECONDS);
            state = channel.getState(true);
        }
        if (state != ConnectivityState.READY) {
            throw new TimeoutException(
                    "BATCH_ENQUEUE channel did not become READY: "
                            + channelKey + " state=" + state);
        }
    }
}
