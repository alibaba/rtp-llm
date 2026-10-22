package org.flexlb.engine.grpc;

import com.google.common.util.concurrent.ForwardingListenableFuture;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.Context;
import io.grpc.ForwardingClientCall;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

@Timeout(30)
class EngineGrpcClientCancellationAuditTest {
    @Test
    void cancellingReturnedFutureCancelsInitialLiveRpc() throws Exception {
        try (Fixture f = new Fixture(false, false, false)) {
            var result = f.call();
            assertTrue(
                    f.firstReceived.await(5, TimeUnit.SECONDS),
                    "server did not receive request; result=" + f.resultDiagnostic());
            assertTrue(result.cancel(true));
            assertTrue(result.isCancelled());
            assertTrue(
                    f.cancelled.await(1, TimeUnit.SECONDS),
                    "wrapper cancellation must reach the live RPC before its 10s deadline");
            assertEquals(1, f.attempts.get());
        }
    }

    @Test
    void retryKeepsTheOriginalDeadlineInsteadOfRenewingItsBudget() throws Exception {
        try (Fixture f = new Fixture(true, false, false, 600)) {
            var result = f.call(1000);
            assertTrue(f.secondReceived.await(5, TimeUnit.SECONDS));
            assertTrue(
                    f.retryDeadlineMs.get() <= 500,
                    "retry must consume the original remaining budget: " + f.retryDeadlineMs.get());
            ExecutionException error =
                    assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(
                    Status.Code.DEADLINE_EXCEEDED,
                    Status.fromThrowable(error.getCause()).getCode());
            assertEquals(2, f.attempts.get());
        }
    }

    @Test
    void earlierCallerContextDeadlineRemainsAuthoritativeAcrossRetry() throws Exception {
        ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();
        try (Fixture f = new Fixture(true, false, false, 400);
                Context.CancellableContext caller =
                        Context.ROOT.withDeadlineAfter(700, TimeUnit.MILLISECONDS, timer)) {
            var result = caller.call(f::call);
            assertTrue(f.secondReceived.await(5, TimeUnit.SECONDS));
            assertTrue(
                    f.retryDeadlineMs.get() <= 500,
                    "retry must respect the caller context deadline");
            ExecutionException error =
                    assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
            assertEquals(
                    Status.Code.DEADLINE_EXCEEDED,
                    Status.fromThrowable(error.getCause()).getCode());
        } finally {
            timer.shutdownNow();
        }
    }

    @Test
    void wrapperTimeoutAlsoCancelsLiveRpc() throws Exception {
        try (Fixture f = new Fixture(false, false, false)) {
            var result = f.call();
            assertTrue(
                    f.firstReceived.await(5, TimeUnit.SECONDS),
                    "server did not receive request; result=" + f.resultDiagnostic());
            result.orTimeout(30, TimeUnit.MILLISECONDS);
            assertThrows(ExecutionException.class, () -> result.get(1, TimeUnit.SECONDS));
            assertTrue(f.cancelled.await(1, TimeUnit.SECONDS));
            assertEquals(1, f.attempts.get());
        }
    }

    @Test
    void cancellingReturnedFutureCancelsRetryLiveRpc() throws Exception {
        try (Fixture f = new Fixture(true, false, false)) {
            var result = f.call();
            assertTrue(f.secondReceived.await(5, TimeUnit.SECONDS));
            assertTrue(result.cancel(true));
            assertTrue(f.cancelled.await(1, TimeUnit.SECONDS));
            assertEquals(2, f.attempts.get());
        }
    }

    @Test
    void cancellationDuringRetryChannelCreationPreventsAnotherRpc() throws Exception {
        try (Fixture f = new Fixture(true, true, false)) {
            var result = f.call();
            assertTrue(f.client.retryCreating.await(5, TimeUnit.SECONDS));
            assertTrue(result.cancel(true));
            f.client.allowRetryCreation.countDown();
            // Place a barrier behind the first failure callback that initiated retry creation.
            f.executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            assertEquals(1, f.attempts.get());
            assertFalse(f.secondReceived.await(100, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    void cancellationWhileRetryStartHasNotReturnedItsFutureCancelsTheAttemptContext()
            throws Exception {
        try (Fixture f = new Fixture(true, false, false)) {
            f.client.gateRetryStart = true;
            var result = f.call();
            assertTrue(f.client.retryStarting.await(5, TimeUnit.SECONDS));
            assertTrue(result.cancel(true));
            f.client.allowRetryStart.countDown();
            f.executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            assertTrue(
                    f.client.contextCancelledAtStart.get(),
                    "retry must see cancelled owner even before future binding");
            assertFalse(f.secondReceived.await(100, TimeUnit.MILLISECONDS));
            assertEquals(1, f.attempts.get());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void callbackRegistrationFailureCancelsTheAlreadyStartedRealRpc() throws Exception {
        try (Fixture f = new Fixture(false, false, false)) {
            Function<
                            AbstractGrpcClient.GrpcFutureStubWrapper,
                            ListenableFuture<EngineRpcService.WorkerStatusPB>>
                    start =
                            stub -> {
                                var live =
                                        stub.getRpcServiceFutureStub()
                                                .getWorkerStatus(
                                                        EngineRpcService.StatusVersionPB
                                                                .getDefaultInstance());
                                return new ForwardingListenableFuture
                                        .SimpleForwardingListenableFuture<>(live) {
                                    @Override
                                    public void addListener(Runnable listener, Executor executor) {
                                        await(f.firstReceived); // prove a real RPC is alive
                                        // before injecting failure
                                        throw new RejectedExecutionException(
                                                "injected callback-registration failure");
                                    }
                                };
                            };
            var execute =
                    EngineGrpcClient.class.getDeclaredMethod(
                            "executeGrpcCallAsync",
                            String.class,
                            int.class,
                            Function.class,
                            long.class,
                            AbstractGrpcClient.ServiceType.class);
            execute.setAccessible(true);
            var result =
                    (CompletableFuture<EngineRpcService.WorkerStatusPB>)
                            execute.invoke(
                                    f.client,
                                    "127.0.0.1",
                                    f.server.getPort(),
                                    start,
                                    10000L,
                                    AbstractGrpcClient.ServiceType.WORKER_STATUS);
            ExecutionException error =
                    assertThrows(ExecutionException.class, () -> result.get(1, TimeUnit.SECONDS));
            assertInstanceOf(RejectedExecutionException.class, error.getCause());
            assertTrue(
                    f.cancelled.await(1, TimeUnit.SECONDS),
                    "registration failure must not leave the live RPC until deadline");
        }
    }

    @Test
    void cancelledCallDoesNotRetryAfterLaterEof() throws Exception {
        try (Fixture f = new Fixture(false, false, false)) {
            var result = f.call();
            assertTrue(
                    f.firstReceived.await(5, TimeUnit.SECONDS),
                    "server did not receive request; result=" + f.resultDiagnostic());
            assertTrue(result.cancel(true));
            assertTrue(f.cancelled.await(1, TimeUnit.SECONDS));
            f.connection.get().shutdownOutput();
            f.executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            assertEquals(1, f.client.created.get());
        }
    }

    @Test
    void saturatedCallbackCanReplaceChannelWithoutBlockingItsOwnEventloop() throws Exception {
        try (Fixture f = new Fixture(false, false, true)) {
            var result = f.call();
            assertTrue(
                    f.firstReceived.await(5, TimeUnit.SECONDS),
                    "server did not receive request; result=" + f.resultDiagnostic());
            CountDownLatch blocked = new CountDownLatch(1);
            f.executor.execute(
                    () -> {
                        blocked.countDown();
                        await(f.release);
                    });
            assertTrue(blocked.await(5, TimeUnit.SECONDS));
            f.executor.execute(() -> {});
            f.connection.get().shutdownOutput();
            assertTrue(f.client.replacing.await(5, TimeUnit.SECONDS));
            // The failure callback runs inline on this same loop when its bounded executor is full.
            f.clientLoop.next().submit(() -> {}).get(400, TimeUnit.MILLISECONDS);
            assertEquals(response(), result.get(5, TimeUnit.SECONDS));
            assertTrue(
                    f.client.replacementThread.get().contains("nioEventLoopGroup"),
                    f.client.replacementThread.get());
            assertTrue(
                    f.client.replacementMillis.get() < 400,
                    "replacement must not wait for its own loop to terminate the channel");
            assertTrue(f.inline.get() > 0);
        }
    }

    private static EngineRpcService.WorkerStatusPB response() {
        return EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build();
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AssertionError(error);
        }
    }

    private static final class TestClient extends EngineGrpcClient {
        final AtomicInteger created = new AtomicInteger();
        final CountDownLatch retryCreating = new CountDownLatch(1);
        final CountDownLatch allowRetryCreation = new CountDownLatch(1);
        final CountDownLatch replacing = new CountDownLatch(1);
        final AtomicReference<String> replacementThread = new AtomicReference<>();
        final AtomicLong replacementMillis = new AtomicLong();
        final boolean blockRetryCreation;
        volatile boolean gateRetryStart;
        final CountDownLatch retryStarting = new CountDownLatch(1),
                allowRetryStart = new CountDownLatch(1);
        final AtomicBoolean contextCancelledAtStart = new AtomicBoolean();

        TestClient(
                ThreadPoolExecutor executor, NioEventLoopGroup loop, boolean blockRetryCreation) {
            super(
                    mock(CustomNameResolver.class),
                    executor,
                    loop,
                    mock(GrpcReporter.class),
                    2000,
                    5000);
            this.blockRetryCreation = blockRetryCreation;
        }

        @Override
        protected ManagedChannel createChannel(String key) {
            if (created.incrementAndGet() == 2 && blockRetryCreation) {
                retryCreating.countDown();
                await(allowRetryCreation);
            }
            ManagedChannel real = super.createChannel(key);
            if (created.get() != 2 || !gateRetryStart) {
                return real;
            }
            return new ManagedChannel() {
                public ManagedChannel shutdown() {
                    real.shutdown();
                    return this;
                }

                public boolean isShutdown() {
                    return real.isShutdown();
                }

                public boolean isTerminated() {
                    return real.isTerminated();
                }

                public ManagedChannel shutdownNow() {
                    real.shutdownNow();
                    return this;
                }

                public boolean awaitTermination(long timeout, TimeUnit unit)
                        throws InterruptedException {
                    return real.awaitTermination(timeout, unit);
                }

                public String authority() {
                    return real.authority();
                }

                public <ReqT, RespT> ClientCall<ReqT, RespT> newCall(
                        MethodDescriptor<ReqT, RespT> method, CallOptions options) {
                    return new ForwardingClientCall.SimpleForwardingClientCall<>(
                            real.newCall(method, options)) {
                        @Override
                        public void start(Listener<RespT> listener, Metadata headers) {
                            retryStarting.countDown();
                            await(allowRetryStart);
                            contextCancelledAtStart.set(Context.current().isCancelled());
                            super.start(listener, headers);
                        }
                    };
                }
            };
        }

        @Override
        protected Invoker replaceInvoker(String key, Invoker stale, ManagedChannel replacement) {
            replacementThread.set(Thread.currentThread().getName());
            replacing.countDown();
            long start = System.nanoTime();
            try {
                return super.replaceInvoker(key, stale, replacement);
            } finally {
                replacementMillis.set(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        final AtomicInteger attempts = new AtomicInteger(), inline = new AtomicInteger();
        final AtomicLong retryDeadlineMs = new AtomicLong();
        final AtomicReference<SocketChannel> connection = new AtomicReference<>();
        final CountDownLatch firstReceived = new CountDownLatch(1),
                secondReceived = new CountDownLatch(1),
                cancelled = new CountDownLatch(1),
                release = new CountDownLatch(1);
        final NioEventLoopGroup serverLoop = new NioEventLoopGroup(1),
                clientLoop = new NioEventLoopGroup(1);
        final ThreadPoolExecutor executor =
                new ThreadPoolExecutor(
                        1,
                        1,
                        0,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(1),
                        (task, ignored) -> {
                            inline.incrementAndGet();
                            task.run();
                        });
        final TestClient client;
        final Server server;

        Fixture(boolean closeFirst, boolean blockRetry, boolean succeedRetry) throws Exception {
            this(closeFirst, blockRetry, succeedRetry, 0);
        }

        Fixture(boolean closeFirst, boolean blockRetry, boolean succeedRetry, long firstEofDelayMs)
                throws Exception {
            server =
                    NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                            .bossEventLoopGroup(serverLoop)
                            .workerEventLoopGroup(serverLoop)
                            .channelFactory(
                                    () ->
                                            new NioServerSocketChannel() {
                                                @Override
                                                protected int doReadMessages(List<Object> messages)
                                                        throws Exception {
                                                    int read = super.doReadMessages(messages);
                                                    if (read > 0) {
                                                        connection.set(
                                                                (SocketChannel)
                                                                        messages.get(
                                                                                messages.size()
                                                                                        - 1));
                                                    }
                                                    return read;
                                                }
                                            })
                            .directExecutor()
                            .addService(
                                    new RpcServiceGrpc.RpcServiceImplBase() {
                                        @Override
                                        public void getWorkerStatus(
                                                EngineRpcService.StatusVersionPB request,
                                                StreamObserver<EngineRpcService.WorkerStatusPB>
                                                        observer) {
                                            int attempt = attempts.incrementAndGet();
                                            Context.current()
                                                    .addListener(
                                                            ignored -> {
                                                                if (attempt > 1 || !closeFirst) {
                                                                    cancelled.countDown();
                                                                }
                                                            },
                                                            Runnable::run);
                                            if (attempt == 1) {
                                                firstReceived.countDown();
                                                if (closeFirst) {
                                                    SocketChannel first = connection.get();
                                                    serverLoop
                                                            .next()
                                                            .schedule(
                                                                    () -> {
                                                                        first.shutdownOutput();
                                                                    },
                                                                    firstEofDelayMs,
                                                                    TimeUnit.MILLISECONDS);
                                                }
                                            } else {
                                                retryDeadlineMs.set(
                                                        Context.current()
                                                                .getDeadline()
                                                                .timeRemaining(
                                                                        TimeUnit.MILLISECONDS));
                                                secondReceived.countDown();
                                                if (succeedRetry) {
                                                    observer.onNext(response());
                                                    observer.onCompleted();
                                                }
                                            }
                                        }
                                    })
                            .build()
                            .start();
            client = new TestClient(executor, clientLoop, blockRetry);
        }

        CompletableFuture<EngineRpcService.WorkerStatusPB> latest;

        String resultDiagnostic() {
            return latest == null ? "not started" : latest.toString();
        }

        CompletableFuture<EngineRpcService.WorkerStatusPB> call() {
            return call(10000);
        }

        CompletableFuture<EngineRpcService.WorkerStatusPB> call(long timeoutMs) {
            latest =
                    client.getWorkerStatusAsync(
                            "127.0.0.1",
                            server.getPort(),
                            EngineRpcService.StatusVersionPB.getDefaultInstance(),
                            timeoutMs);
            return latest;
        }

        @Override
        public void close() throws Exception {
            release.countDown();
            client.allowRetryCreation.countDown();
            client.allowRetryStart.countDown();
            client.shutdownChannelPool();
            executor.shutdownNow();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            clientLoop.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
            serverLoop.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
        }
    }
}
