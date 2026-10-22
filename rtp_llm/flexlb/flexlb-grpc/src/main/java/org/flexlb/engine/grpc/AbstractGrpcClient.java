package org.flexlb.engine.grpc;

import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import lombok.Getter;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * @author zjw
 * description:
 * date: 2025/4/23
 */
public abstract class AbstractGrpcClient implements CustomNameResolver.Listener {

    /**
     * Maintain different channel for different service type.
     *
     * @see ServiceType
     */
    protected final Map<String /*ip:port:serviceType*/, Invoker> channelPool =
            new ConcurrentHashMap<>();

    protected final GrpcReporter grpcReporter;

    protected AbstractGrpcClient(GrpcReporter grpcReporter) {
        this.grpcReporter = grpcReporter;
    }

    /**
     * Handle service address update event
     * When service discovery detects worker list changes, synchronously update gRPC channel pool and cache view
     *
     * @param ipPortList Latest worker address list in format ip:httpPort
     */
    @Override
    public void onAddressUpdate(List<String /*ip:port*/> ipPortList) {
        if (ipPortList == null) {
            Logger.error("received null ipPort list");
            return;
        }

        // Update gRPC channel pool
        updateGrpcChannelPool(ipPortList);
    }

    /**
     * Update gRPC channel pool based on latest ipPortList
     * Create new channels for newly online workers, remove channels for offline workers
     *
     * @param ipPortList Latest worker address list in format ip:httpPort
     */
    private void updateGrpcChannelPool(List<String> ipPortList) {
        Set<String /*ip:port:serviceType*/> currentKeys = new HashSet<>(channelPool.keySet());
        List<String /*ip:port:serviceType*/> addedKeys = new ArrayList<>();

        // Identify new and retained workers, mark channels to be removed
        for (String ipPort : ipPortList) {
            String[] parts = ipPort.split(":");
            String ip = parts[0];
            int httpPort = Integer.parseInt(parts[1]);
            int grpcPort = CommonUtils.toGrpcPort(httpPort);

            for (ServiceType serviceType : ServiceType.values()) {
                String serviceKey = createKey(ip, grpcPort, serviceType);
                boolean contained = currentKeys.remove(serviceKey);
                if (!contained) {
                    addedKeys.add(serviceKey);
                }
            }
        }

        if (addedKeys.isEmpty() && currentKeys.isEmpty()) {
            return;
        }

        // Create channels for newly online workers
        int addedChannelCount = 0;
        for (String newKey : addedKeys) {
            try {
                ManagedChannel managedChannel = createChannel(newKey);
                Invoker invoker = putInvokerIfAbsent(newKey, managedChannel);
                if (invoker.getChannel() == managedChannel) {
                    addedChannelCount++;
                }
            } catch (Exception e) {
                Logger.error("create channel for ipPort {} failed", newKey, e);
            }
        }

        // Close and remove channels for offline workers
        int removedChannelCount = 0;
        for (String key : currentKeys) {
            Invoker invoker = channelPool.remove(key);
            if (invoker != null) {
                removedChannelCount++;
                try {
                    invoker.shutdown(CloseReason.DISCOVERY_REMOVED);
                } catch (Exception e) {
                    Logger.error(
                            "shutdown channel for ipPort {} failed", invoker.getChannelKey(), e);
                }
            }
        }

        Logger.info(
                "gRPC channel pool changed: hosts={}, addedChannels={}, removedChannels={}, totalChannels={}",
                ipPortList.size(),
                addedChannelCount,
                removedChannelCount,
                channelPool.size());
    }

    @Scheduled(fixedRate = 2000)
    public void reportChannelPoolSize() {
        grpcReporter.reportChannelPoolSize(channelPool.size());
    }

    protected Invoker getInvoker(String channelKey) {
        Invoker invoker = channelPool.get(channelKey);
        if (invoker == null) {
            Logger.warn("ip:{} grpc channel not found, channelPool:{}", channelKey, channelPool);
        }
        return invoker;
    }

    protected Invoker putInvokerIfAbsent(String channelKey, ManagedChannel managedChannel) {
        Invoker newInvoker = new Invoker(channelKey, managedChannel);
        Invoker existing = channelPool.putIfAbsent(channelKey, newInvoker);
        if (existing != null) {
            newInvoker.shutdown(CloseReason.DUPLICATE_DISCARDED);
            return existing;
        }
        return newInvoker;
    }

    protected Invoker replaceInvoker(
            String channelKey, Invoker expectedInvoker, ManagedChannel managedChannel) {
        Invoker newInvoker = new Invoker(channelKey, managedChannel);
        if (expectedInvoker == null) {
            Invoker existing = channelPool.putIfAbsent(channelKey, newInvoker);
            if (existing != null) {
                newInvoker.shutdown(CloseReason.DUPLICATE_DISCARDED);
                return existing;
            }
            return newInvoker;
        }

        boolean replaced = channelPool.replace(channelKey, expectedInvoker, newInvoker);
        if (!replaced) {
            newInvoker.shutdown(CloseReason.DUPLICATE_DISCARDED);
            return channelPool.get(channelKey);
        }

        expectedInvoker.shutdown(CloseReason.REPLACED);
        return newInvoker;
    }

    @PreDestroy
    public void shutdownChannelPool() {
        Set<String> channelKeys = new HashSet<>(channelPool.keySet());
        for (String channelKey : channelKeys) {
            Invoker invoker = channelPool.remove(channelKey);
            if (invoker != null) {
                try {
                    invoker.shutdown(CloseReason.CLIENT_SHUTDOWN);
                } catch (Exception e) {
                    Logger.error(
                            "shutdown channel for ipPort {} failed in pre-destroy", channelKey, e);
                }
            }
        }
        Logger.warn("grpc channel pool shutdown finished");
    }

    protected abstract ManagedChannel createChannel(String hostKey);

    protected static String createKey(String ip, int port, ServiceType serviceType) {
        return ip + ":" + port + ":" + serviceType.getSuffix();
    }

    protected static String[] parseServiceKey(String serviceKey) {
        String[] parts = serviceKey.split(":");
        if (parts.length == 3) {
            return new String[] {parts[0], parts[1], parts[2]};
        }

        throw new IllegalArgumentException("Invalid service key format: " + serviceKey);
    }

    public enum CloseReason {
        DISCOVERY_REMOVED,
        REPLACED,
        CLIENT_SHUTDOWN,
        DUPLICATE_DISCARDED
    }

    /** A local close request; it does not establish the cause of a preceding transport failure. */
    public record CloseObservation(CloseReason reason, long nanoTime, boolean forced) {}

    /**
     * Wrapper class for FutureStub (async gRPC calls)
     */
    public static class GrpcFutureStubWrapper {
        private final RpcServiceGrpc.RpcServiceFutureStub rpcServiceFutureStub;
        private final MultimodalRpcServiceGrpc.MultimodalRpcServiceFutureStub multimodalFutureStub;

        public GrpcFutureStubWrapper(
                RpcServiceGrpc.RpcServiceFutureStub rpcServiceFutureStub,
                MultimodalRpcServiceGrpc.MultimodalRpcServiceFutureStub multimodalFutureStub) {
            this.rpcServiceFutureStub = rpcServiceFutureStub;
            this.multimodalFutureStub = multimodalFutureStub;
        }

        public RpcServiceGrpc.RpcServiceFutureStub getRpcServiceFutureStub() {
            return rpcServiceFutureStub;
        }

        public MultimodalRpcServiceGrpc.MultimodalRpcServiceFutureStub getMultimodalFutureStub() {
            return multimodalFutureStub;
        }

        public GrpcFutureStubWrapper withDeadlineAfter(long timeout, TimeUnit unit) {
            return withDeadline(Deadline.after(timeout, unit));
        }

        public GrpcFutureStubWrapper withDeadline(Deadline deadline) {
            return new GrpcFutureStubWrapper(
                    rpcServiceFutureStub.withDeadline(deadline),
                    multimodalFutureStub.withDeadline(deadline));
        }

        public GrpcFutureStubWrapper withTrace(GrpcCallTrace trace) {
            return new GrpcFutureStubWrapper(
                    rpcServiceFutureStub.withOption(GrpcCallTrace.CALL_TRACE, trace),
                    multimodalFutureStub.withOption(GrpcCallTrace.CALL_TRACE, trace));
        }
    }

    @Getter
    public class Invoker {

        private final String channelKey;
        private final ManagedChannel channel;
        private final long createTime;
        private volatile long lastUsedTime;
        private volatile long expireTime;
        private final AtomicReference<CloseObservation> closeObservation = new AtomicReference<>();

        public Invoker(String channelKey, ManagedChannel channel) {
            this.channelKey = channelKey;
            this.channel = channel;
            long currentTime = System.nanoTime() / 1000;
            this.createTime = currentTime;
            this.lastUsedTime = currentTime;
            this.expireTime = 0;
        }

        public void updateLastUsedTime() {
            this.lastUsedTime = System.nanoTime() / 1000;
        }

        public void markExpired() {
            this.expireTime = System.nanoTime() / 1000;
        }

        public long getConnectionDuration() {
            return expireTime > 0 ? expireTime - createTime : System.nanoTime() / 1000 - createTime;
        }

        public CloseObservation getCloseObservation() {
            return closeObservation.get();
        }

        public void shutdown(CloseReason reason) {
            if (channel == null) {
                return;
            }
            try {
                closeObservation.compareAndSet(
                        null, new CloseObservation(reason, System.nanoTime(), false));
            } catch (RuntimeException | Error ignored) {
                // Recording diagnostics must not prevent closing a channel under memory pressure.
            }
            logCloseSafely(reason, false, "requested", null);
            try {
                channel.shutdown();
                if (reason == CloseReason.REPLACED || reason == CloseReason.DUPLICATE_DISCARDED) {
                    // These paths can run from a saturated gRPC callback on its Netty eventloop.
                    // Waiting here stalls the same loop needed to finish channel termination.
                    if (!channel.isTerminated()) {
                        forceShutdown(reason);
                    }
                    return;
                }
                if (!channel.awaitTermination(1000, TimeUnit.MILLISECONDS)) {
                    logCloseSafely(reason, true, "graceful_timeout", null);
                    forceShutdown(reason);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logCloseSafely(reason, true, "interrupted", e);
                forceShutdown(reason);
            } catch (Exception e) {
                logCloseSafely(reason, true, "shutdown_failed", e);
                forceShutdown(reason);
            }
        }

        private void forceShutdown(CloseReason reason) {
            try {
                // A graceful request does not cancel active calls. Record the distinct forced
                // action.
                closeObservation.updateAndGet(
                        previous ->
                                previous != null && previous.forced()
                                        ? previous
                                        : new CloseObservation(reason, System.nanoTime(), true));
            } catch (RuntimeException | Error ignored) {
                // Diagnostics must not prevent releasing the channel.
            }
            channel.shutdownNow();
        }

        private void logCloseSafely(
                CloseReason reason, boolean force, String trigger, Throwable failure) {
            try {
                String format =
                        "event=engine_grpc_channel_close reason={} channelKey={} createTimeUs={} force={} trigger={}";
                if (force) {
                    Logger.warn(format, reason, channelKey, createTime, true, trigger, failure);
                } else {
                    Logger.info(format, reason, channelKey, createTime, false, trigger);
                }
            } catch (RuntimeException | Error ignored) {
                // Logging failures cannot prevent shutdown or its forceful fallback.
            }
        }
    }

    /**
     * gRPC service type, in order to mark different gRPC connection in channelPool
     *
     * @see AbstractGrpcClient#channelPool
     * @see EngineGrpcClient
     */
    public enum ServiceType {
        WORKER_STATUS("worker", "GetWorkerStatus"),
        CACHE_STATUS("cache", "GetCacheStatus"),
        MULTIMODAL_WORKER_STATUS("multimodal_worker", "GetWorkerStatus"),
        MULTIMODAL_CACHE_STATUS("multimodal_cache", "GetCacheStatus"),
        BATCH_ENQUEUE("batch_enqueue", "EnqueueBatch"),
        // AutoTPM Cancel: single-shot engine cancel, reuses
        // the pooled channel of the target engine worker.
        ENGINE_CANCEL("engine_cancel", "Cancel");

        @Getter private final String suffix;
        @Getter private final String operationName;

        ServiceType(String suffix, String operationName) {
            this.suffix = suffix;
            this.operationName = operationName;
        }
    }
}
