package org.flexlb.httpserver;

import static org.junit.jupiter.api.Assertions.*;

import io.grpc.*;
import io.grpc.netty.*;
import io.grpc.stub.*;
import io.netty.buffer.*;
import io.netty.channel.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.*;

@Timeout(30)
class GrpcReceivingWireBudgetTest {
  @Test
  void partialBodiesAcrossConnectionsCloseAtWireBudgetWithoutEarlyReuse() throws Exception {
    var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
    var release = new CountDownLatch(1);
    var control = Executors.newSingleThreadExecutor();
    var guard =
        new GrpcReceivingGuard(2, 1, 5000, pool, control, 64L << 10, 128L << 10);
    var alloc = new PooledByteBufAllocator(true);
    var cancelMethod =
        GrpcReceivingGuardRepro.METHOD.toBuilder()
            .setFullMethodName("receiving.Test/Cancel")
            .build();
    var service =
        ServerServiceDefinition.builder("receiving.Test")
            .addMethod(
                GrpcReceivingGuardRepro.METHOD,
                ServerCalls.asyncUnaryCall((byte[] b, StreamObserver<byte[]> o) -> {}))
            .addMethod(
                cancelMethod,
                ServerCalls.asyncUnaryCall(
                    (byte[] b, StreamObserver<byte[]> o) -> {
                      o.onNext(new byte[1]);
                      o.onCompleted();
                    }))
            .build();
    var server =
        NettyServerBuilder.forPort(0)
            .directExecutor()
            .callExecutor(guard)
            .addStreamTracerFactory(guard)
            .intercept(guard)
            .addService(service)
            .flowControlWindow(32768)
            .withChildOption(ChannelOption.ALLOCATOR, alloc)
            .withChildOption(ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
            .build()
            .start();
    List<GrpcReceivingGuardMatrix.Raw> sockets = new ArrayList<>();
    try {
      var a = new GrpcReceivingGuardMatrix.Raw(server.getPort(), null);
      var b = new GrpcReceivingGuardMatrix.Raw(server.getPort(), null);
      sockets.add(a);
      sockets.add(b);
      GrpcReceivingGuardRepro.until(
          () -> pool.getCompletedTaskCount() >= 2, "request two messages first");
      var blocked = new CountDownLatch(1);
      pool.execute(
          () -> {
            blocked.countDown();
            try {
              release.await();
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
          });
      assertTrue(blocked.await(2, TimeUnit.SECONDS));
      partial(a, 65536);
      GrpcReceivingGuardRepro.until(
          () -> guard.receivingBytes() == 65531,
          "all A partial body bytes charged before B competes for the shared budget");
      GrpcReceivingGuardRepro.until(
          () -> alloc.pinnedDirectMemory() >= 65531,
          "A incomplete body must really retain pooled buffers before B is sent");
      partial(b, 32768);
      GrpcReceivingGuardRepro.until(
          () -> guard.rejected() > 0, "second partial crosses shared budget");
      var controlChannel =
          NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
      try {
        GrpcReceivingGuardRepro.awaitReady(controlChannel);
        assertTrue(
            GrpcReceivingGuardMatrix.full(controlChannel, cancelMethod)
                .get(1, TimeUnit.SECONDS)
                .isOk());
        GrpcReceivingGuardRepro.until(
            () -> guard.controlReceivingBytes() == 0, "control lane bytes returned");
      } finally {
        controlChannel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      }
      long held = guard.receivingBytes();
      assertTrue(held <= 64L << 10);
      assertEquals(2, guard.receiving());
      assertTrue(
          alloc.pinnedDirectMemory() >= 65531,
          "A remains open with incomplete body while B is closed and its lease conservatively"
              + " held");
      for (int i = 0; i < 10; i++) {
        var raw = new GrpcReceivingGuardMatrix.Raw(server.getPort(), null);
        sockets.add(raw);
      }
      GrpcReceivingGuardRepro.until(
          () -> guard.rejected() >= 11,
          "closed-but-not-cleaned receiving permit cannot be reused by new TCP");
      assertEquals(held, guard.receivingBytes());
      System.out.println(
          "WIRE_MULTI_TCP blockedReceiving="
              + guard.receiving()
              + " charged="
              + held
              + " rejected="
              + guard.rejected()
              + " pinned="
              + alloc.pinnedDirectMemory());
      a.reset();
      release.countDown();
      GrpcReceivingGuardRepro.until(
          () ->
              guard.receiving() == 0
                  && guard.receivingBytes() == 0
                  && alloc.pinnedDirectMemory() == 0,
          "wire terminal cleanup while TCP open");
      System.out.println(
          "WIRE_MULTI_TCP_CLEAN charged="
              + guard.receivingBytes()
              + " pinned="
              + alloc.pinnedDirectMemory());
    } finally {
      release.countDown();
      for (var s : sockets) s.close();
      server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      guard.close();
      pool.shutdownNow();
      pool.awaitTermination(3, TimeUnit.SECONDS);
      control.shutdown();
      control.awaitTermination(3, TimeUnit.SECONDS);
    }
  }

  static void partial(GrpcReceivingGuardMatrix.Raw raw, int bytes) throws Exception {
    var all = new ByteArrayOutputStream();
    var d = new DataOutputStream(all);
    d.writeByte(0);
    d.writeInt(1048576);
    d.write(new byte[bytes - 5]);
    byte[] b = all.toByteArray();
    for (int i = 0; i < b.length; i += 16384)
      raw.frame(0, 0, Arrays.copyOfRange(b, i, Math.min(b.length, i + 16384)));
    raw.out.flush();
  }

  @Test
  void queuedCompleteProducerRetainsChargedBytesUntilActualCleanup() throws Exception {
    // Isolate this ownership fence from shared loops closing sockets in prior raw-frame tests.
    var transport = new io.netty.channel.nio.NioEventLoopGroup(1);
    var pool = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
    var release = new CountDownLatch(1);
    var handlerReady = new CountDownLatch(2);
    var guard = new GrpcReceivingGuard(2, 1, 5000, pool, pool, 48L << 10, 64L << 10);
    var alloc = new PooledByteBufAllocator(true);
    var service =
        ServerServiceDefinition.builder("receiving.Test")
            .addMethod(
                GrpcReceivingGuardRepro.METHOD,
                ServerCalls.asyncUnaryCall((byte[] b, StreamObserver<byte[]> o) -> {}))
            .build();
    var server =
        NettyServerBuilder.forAddress(new java.net.InetSocketAddress("127.0.0.1", 0))
            .channelType(io.netty.channel.socket.nio.NioServerSocketChannel.class)
            .bossEventLoopGroup(transport)
            .workerEventLoopGroup(transport)
            .directExecutor()
            .callExecutor(guard)
            .addStreamTracerFactory(guard)
            .intercept(new ServerInterceptor() {
              @Override public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                  ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
                var listener = next.startCall(call, headers);
                handlerReady.countDown(); // Unary handler has installed request(2).
                return listener;
              }
            })
            .intercept(guard)
            .addService(service)
            .withChildOption(ChannelOption.ALLOCATOR, alloc)
            .withChildOption(ChannelOption.RCVBUF_ALLOCATOR, new FixedRecvByteBufAllocator(4096))
            .build()
            .start();
    try (var a = new GrpcReceivingGuardMatrix.Raw(server.getPort(), null);
        var b = new GrpcReceivingGuardMatrix.Raw(server.getPort(), null)) {
      assertTrue(handlerReady.await(5, TimeUnit.SECONDS), "handler requested before blocking");
      var blocked = new CountDownLatch(1);
      pool.execute(
          () -> {
            blocked.countDown();
            try {
              release.await();
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            }
          });
      assertTrue(blocked.await(2, TimeUnit.SECONDS));
      a.message(new byte[32768], false);
      GrpcReceivingGuardRepro.until(
          () -> alloc.pinnedDirectMemory() >= 32768 && guard.receivingBytes() == 32768,
          "complete queued producer charged");
      a.reset();
      Thread.sleep(50);
      assertEquals(32768, guard.receivingBytes());
      assertTrue(alloc.pinnedDirectMemory() >= 32768);
      b.message(new byte[32768], false);
      GrpcReceivingGuardRepro.until(
          () -> guard.rejected() > 0, "uncleaned producer bytes prevent another body from fitting");
      assertTrue(guard.receivingBytes() >= 32768);
      assertTrue(guard.receivingBytes() <= 48L << 10);
      assertEquals(2, guard.receiving());
      release.countDown();
      GrpcReceivingGuardRepro.until(
          () -> guard.receivingBytes() == 0 && alloc.pinnedDirectMemory() == 0,
          "same cleanup fence for charged and pooled bytes");
      System.out.println(
          "WIRE_QUEUED_TCP_OPEN_CLEAN charged="
              + guard.receivingBytes()
              + " pinned="
              + alloc.pinnedDirectMemory());
    } finally {
      release.countDown();
      server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      guard.close();
      pool.shutdownNow();
      pool.awaitTermination(3, TimeUnit.SECONDS);
      transport.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
    }
  }

  @Test
  void parallelSmallMessagesUseBytesInsteadOfMaximumMessageSlots() throws Exception {
    parallelSmallMessages(64L << 30);
    parallelSmallMessages(2L << 30);
  }

  void parallelSmallMessages(long directLimit) throws Exception {
    var pool =
        new ThreadPoolExecutor(
            8,
            8,
            0,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(128),
            new FlexlbGrpcServer.CountingCallerRunsHandler());
    var control = Executors.newSingleThreadExecutor();
    var limits =
        GrpcReceivingBudget.resolve(
            new org.flexlb.config.FlexlbConfig.GrpcServerConfig(), directLimit);
    var guard =
        new GrpcReceivingGuard(
            limits.scheduleCalls(),
            limits.controlCalls(),
            5000,
            pool,
            control,
            limits.scheduleBudgetBytes(),
            limits.controlBudgetBytes());
    var alloc = new PooledByteBufAllocator(true);
    var service =
        ServerServiceDefinition.builder("receiving.Test")
            .addMethod(
                GrpcReceivingGuardRepro.METHOD,
                ServerCalls.asyncUnaryCall(
                    (byte[] b, StreamObserver<byte[]> o) -> {
                      o.onNext(new byte[1]);
                      o.onCompleted();
                    }))
            .build();
    var server =
        NettyServerBuilder.forPort(0)
            .directExecutor()
            .callExecutor(guard)
            .addStreamTracerFactory(guard)
            .intercept(new GrpcOverloadInterceptor(pool))
            .intercept(guard)
            .addService(service)
            .withChildOption(ChannelOption.ALLOCATOR, alloc)
            .build()
            .start();
    var ch = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
    try {
      GrpcReceivingGuardRepro.awaitReady(ch);
      List<ClientCall<byte[], byte[]>> calls = new ArrayList<>();
      List<CompletableFuture<Status>> result = new ArrayList<>();
      for (int i = 0; i < 32; i++) {
        var d = new CompletableFuture<Status>();
        calls.add(begin(ch, d));
        result.add(d);
      }
      GrpcReceivingGuardRepro.until(
          () -> guard.receiving() == 32, "all32 headers receive concurrently");
      assertEquals(0, guard.receivingBytes(), "HEADERS must not reserve hypothetical body bytes");
      for (var c : calls) {
        c.sendMessage(new byte[8192]);
        c.halfClose();
      }
      for (var d : result) assertTrue(d.get(5, TimeUnit.SECONDS).isOk());
      var sem = new Semaphore(32);
      result.clear();
      long start = System.nanoTime();
      for (int i = 0; i < 512; i++) {
        sem.acquire();
        var d = new CompletableFuture<Status>();
        d.whenComplete((s, t) -> sem.release());
        result.add(d);
        var c = begin(ch, d);
        c.sendMessage(new byte[8192]);
        c.halfClose();
      }
      for (var d : result) assertTrue(d.get(5, TimeUnit.SECONDS).isOk());
      GrpcReceivingGuardRepro.until(
          () ->
              guard.receivingBytes() == 0
                  && guard.pendingTimers() == 0
                  && alloc.pinnedDirectMemory() == 0,
          "wire parallel cleanup TCP still open");
      assertEquals(0, guard.rejected());
      System.out.println(
          "WIRE_PARALLEL_8K DirectGiB="
              + (directLimit >> 30)
              + " concurrency=32 total=512 ok=512 rejected="
              + guard.rejected()
              + " elapsedMs="
              + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start)
              + " charged="
              + guard.receivingBytes()
              + " pinned="
              + alloc.pinnedDirectMemory());
    } finally {
      ch.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
      guard.close();
      pool.shutdown();
      control.shutdown();
      pool.awaitTermination(3, TimeUnit.SECONDS);
      control.awaitTermination(3, TimeUnit.SECONDS);
    }
  }

  private static ClientCall<byte[], byte[]> begin(
      ManagedChannel channel, CompletableFuture<Status> done) {
    var method =
        GrpcReceivingGuardRepro.METHOD.toBuilder()
            .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
            .build();
    var call = channel.newCall(method, CallOptions.DEFAULT.withDeadlineAfter(5, TimeUnit.SECONDS));
    call.start(
        new ClientCall.Listener<>() {
          @Override
          public void onClose(Status status, Metadata trailers) {
            done.complete(status);
          }
        },
        new Metadata());
    call.request(1);
    return call;
  }
}
