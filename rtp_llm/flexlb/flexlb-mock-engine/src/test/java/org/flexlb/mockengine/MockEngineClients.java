package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.engine.grpc.EngineGrpcClient.CancelReply;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.engine.grpc.EngineRpcService.RequestCancelReasonPB;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Test fixtures at the Worker client boundary, backed by the mock engine's control endpoints. */
final class MockEngineClients {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Duration HTTP_TIMEOUT = Duration.ofMillis(500);

    private MockEngineClients() { }

    static EngineGrpcClient inProcess(Map<Integer, JavaMockEngineCluster.FastRpcService> services) {
        var client = mock(EngineGrpcClient.class);
        stubCancellation(client, services, true);
        return client;
    }

    static void stubCancellation(EngineGrpcClient client,
            Map<Integer, JavaMockEngineCluster.FastRpcService> services, boolean preemptionSupported) {
        when(client.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> {
            CancelRequestPB request = call.getArgument(2);
            if (!preemptionSupported && request.getReason()
                    == RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED) {
                return CompletableFuture.completedFuture(CancelReply.UNSUPPORTED);
            }
            return cancel(services.get(call.<Integer>getArgument(1)), request);
        });
    }

    private static CompletableFuture<CancelReply> cancel(
            JavaMockEngineCluster.FastRpcService service, CancelRequestPB request) {
        if (service == null) { return CompletableFuture.completedFuture(CancelReply.UNSUPPORTED); }
        if (service.isStopped()) { return CompletableFuture.failedFuture(new IllegalStateException("engine stopped")); }
        // Run fault injection before changing engine state, just like the real mock RPC handler.
        var fault = service.arriveCancelRpc();
        if (fault != null) {
            return switch (fault) {
                case NO_RESPOND -> new CompletableFuture<>();
                case ERROR -> CompletableFuture.failedFuture(new IllegalStateException("mock cancel error injection"));
                case UNEXPECTED_STATUS -> CompletableFuture.failedFuture(new IllegalStateException("unexpected cancel ack status"));
            };
        }
        try {
            if (request.getReason() != RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED) {
                CancelReason reason = switch (request.getReason()) {
                    case REQUEST_CANCEL_REASON_CLIENT_CANCELLED -> CancelReason.CLIENT_CANCELLED;
                    case REQUEST_CANCEL_REASON_DEADLINE_EXCEEDED -> CancelReason.DEADLINE_EXCEEDED;
                    case REQUEST_CANCEL_REASON_SHUTDOWN -> CancelReason.SHUTDOWN;
                    default -> throw new IllegalArgumentException("invalid cleanup reason: " + request.getReason());
                };
                service.cleanUpRequest(request.getRequestId(), reason);
                return CompletableFuture.completedFuture(CancelReply.REQUEST_CLEANED);
            }
            var result = service.cancelRequest(request.getRequestId());
            return CompletableFuture.completedFuture(result.found() ? CancelReply.ACCEPTED
                    : result.alreadyFinished() ? CancelReply.NOT_FOUND : CancelReply.REQUEST_FENCED);
        } catch (RuntimeException failure) {
            return CompletableFuture.failedFuture(failure);
        }
    }

    static EngineGrpcClient overHttp(String controlUrl) {
        var client = mock(EngineGrpcClient.class);
        var http = HttpClient.newBuilder().connectTimeout(HTTP_TIMEOUT).build();
        var uri = URI.create(controlUrl.replaceAll("/+$", "") + "/cancel_request");
        when(client.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call ->
                cancelOverHttp(http, uri, call.getArgument(0), call.getArgument(1), call.getArgument(2)));
        return client;
    }

    private static CompletableFuture<CancelReply> cancelOverHttp(
            HttpClient http, URI uri, String ip, int port, CancelRequestPB request) {
        if (request.getReason() != RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED
                || ip == null || ip.isBlank() || port <= 0) {
            return CompletableFuture.completedFuture(CancelReply.UNSUPPORTED);
        }
        try {
            String body = MAPPER.createObjectNode().put("port", port)
                    .put("request_id", request.getRequestId()).toString();
            var httpRequest = HttpRequest.newBuilder(uri).timeout(HTTP_TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body)).build();
            return http.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString())
                    .thenApply(MockEngineClients::readHttpReply);
        } catch (RuntimeException failure) {
            return CompletableFuture.failedFuture(failure);
        }
    }

    private static CancelReply readHttpReply(HttpResponse<String> response) {
        if (response.statusCode() == 404) { return CancelReply.UNSUPPORTED; }
        if (response.statusCode() != 200) {
            throw new IllegalStateException("mock cancel control plane returned HTTP "
                    + response.statusCode() + ": " + response.body());
        }
        try {
            String status = MAPPER.readTree(response.body()).path("status").asText();
            return switch (status) {
                case "ACCEPTED" -> CancelReply.ACCEPTED;
                case "NOT_FOUND" -> CancelReply.NOT_FOUND;
                case "TOMBSTONED" -> CancelReply.REQUEST_FENCED;
                default -> throw new IllegalStateException("unknown cancel status: " + status);
            };
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("invalid cancel response", failure);
        }
    }
}
