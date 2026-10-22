package org.flexlb.httpserver;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import io.grpc.*;
import io.grpc.netty.*;
import io.grpc.stub.*;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import java.lang.reflect.Field;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.flexlb.config.*;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.*;
import org.springframework.mock.env.MockEnvironment;

@Timeout(20)
class GrpcReceivingGuardWiringTest {
  @Test
  void scheduleQueueSaturationRejectsNewBusinessButCancelControlWorks() throws Exception {
    var pool =
        new ThreadPoolExecutor(
            1,
            1,
            0,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1),
            new FlexlbGrpcServer.CountingCallerRunsHandler());
    var control =
        new ThreadPoolExecutor(
            1,
            1,
            0,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1),
            new FlexlbGrpcServer.CountingCallerRunsHandler());
    var release = new CountDownLatch(1);
    var blocked = new CountDownLatch(1);
    var entered = new AtomicInteger();
    var guard = new GrpcReceivingGuard(8, 1, 1000, pool, control);
    var alloc = new PooledByteBufAllocator(true);
    var call = GrpcReceivingGuardRepro.METHOD;
    var cancel = call.toBuilder().setFullMethodName("receiving.Test/Cancel").build();
    var service =
        ServerServiceDefinition.builder("receiving.Test")
            .addMethod(
                call,
                ServerCalls.asyncUnaryCall(
                    (byte[] b, StreamObserver<byte[]> o) -> {
                      entered.incrementAndGet();
                      o.onNext(new byte[1]);
                      o.onCompleted();
                    }))
            .addMethod(
                cancel,
                ServerCalls.asyncUnaryCall(
                    (byte[] b, StreamObserver<byte[]> o) -> {
                      o.onNext(new byte[1]);
                      o.onCompleted();
                    }))
            .build();
    var server =
        NettyServerBuilder.forAddress(new java.net.InetSocketAddress("127.0.0.1", 0))
            .directExecutor()
            .callExecutor(guard)
            .addStreamTracerFactory(guard)
            .intercept(new GrpcOverloadInterceptor(pool, control))
            .intercept(guard)
            .addService(service)
            .withChildOption(ChannelOption.ALLOCATOR, alloc)
            .build()
            .start();
    var channel =
        NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
    try {
      GrpcReceivingGuardRepro.awaitReady(channel);
      assertTrue(GrpcReceivingGuardMatrix.full(channel, call).get(3, TimeUnit.SECONDS).isOk());
      entered.set(0);
      pool.execute(
          () -> {
            blocked.countDown();
            try {
              release.await();
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
          });
      assertTrue(blocked.await(3, TimeUnit.SECONDS));
      pool.execute(() -> {});
      long started = System.nanoTime();
      assertEquals(
          Status.Code.RESOURCE_EXHAUSTED,
          GrpcReceivingGuardMatrix.full(channel, call).get(500, TimeUnit.MILLISECONDS).getCode());
      assertEquals(0, entered.get());
      assertTrue(
          GrpcReceivingGuardMatrix.full(channel, cancel).get(500, TimeUnit.MILLISECONDS).isOk());
      GrpcReceivingGuardRepro.until(
          () ->
              guard.receiving() == 0
                  && guard.controlReceiving() == 0
                  && alloc.pinnedDirectMemory() == 0,
          "saturated close cleanup with TCP open");
      System.out.println(
          "GUARD_OVERLOAD_CONTROL elapsedMs="
              + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started)
              + " entered="
              + entered.get()
              + " pinned="
              + alloc.pinnedDirectMemory());
    } finally {
      release.countDown();
      channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      guard.close();
      pool.shutdown();
      control.shutdown();
      assertTrue(pool.awaitTermination(3, TimeUnit.SECONDS));
      assertTrue(control.awaitTermination(3, TimeUnit.SECONDS));
    }
  }

  @Test
  void flexMonitorReceivesGuardAndBothExecutorLaneMetrics() throws Exception {
    ConfigService config = mock(ConfigService.class);
    when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
    var server =
        new FlexlbGrpcServer(
            null,
            config,
            new MockEnvironment(),
            null,
            null,
            mock(GrpcServerTimingInterceptor.class),
            null);
    var monitor = mock(FlexMonitor.class);
    var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>(1));
    var control = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>(1));
    var guard = new GrpcReceivingGuard(1, 1000, pool);
    try {
      set(server, "flexMonitor", monitor);
      set(server, "receivingGuard", guard);
      set(server, "grpcExecutor", pool);
      set(server, "grpcControlExecutor", control);
      var registration = FlexlbGrpcServer.class.getDeclaredMethod("registerReceivingMetrics");
      registration.setAccessible(true);
      registration.invoke(server);
      server.reportReceivingMetrics();
      verify(monitor, times(11)).register(anyString(), eq(FlexMetricType.GAUGE));
      verify(monitor).report("app.grpc.server.receiving.active", 0d);
      verify(monitor).report("app.grpc.server.receiving.control.active", 0d);
      verify(monitor).report("app.grpc.server.receiving.rejected.total", 0d);
      verify(monitor).report("app.grpc.server.receiving.timed.out.total", 0d);
      verify(monitor).report("app.grpc.server.control.queue.size", 0d);
      verify(monitor).report("app.grpc.server.receiving.budget.used.bytes", 0d);
      verify(monitor).report("app.grpc.server.receiving.control.budget.used.bytes", 0d);
    } finally {
      guard.close();
      pool.shutdown();
      control.shutdown();
    }
  }

  static void set(Object instance, String name, Object value) throws Exception {
    Field f = instance.getClass().getDeclaredField(name);
    f.setAccessible(true);
    f.set(instance, value);
  }
}
