package org.flexlb.mock.protocol;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.*;
import java.io.IOException;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.net.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Supplier;

/**
 * Exposes component control operations over HTTP with typed JSON requests and responses.
 */
public final class ControlHttpServer implements AutoCloseable {
    private final Map<String, Binding> bindings = new HashMap<>();
    private final HttpServer server;
    private final ExecutorService handlers = Executors.newVirtualThreadPerTaskExecutor();
    private final ExecutorService calls = Executors.newVirtualThreadPerTaskExecutor();

    private boolean started;

    private Runnable health = () -> {};

    public ControlHttpServer(InetSocketAddress address) throws IOException {
        server = HttpServer.create(address, 0);
        server.setExecutor(handlers);
        server.createContext("/health", exchange -> {
            try (exchange) {
                if (!exchange.getRequestMethod().equals("GET") || !exchange.getRequestURI().toString().equals("/health")) {
                    send(exchange, 404, Map.of("error", "Unknown endpoint")); return;
                }
                try { health.run(); }
                catch (RuntimeException error) { send(exchange, 503, Map.of("status", "unavailable")); return; }
                send(exchange, 200, Map.of("status", "ok"));
            }
        });
    }

    public <A, R> ControlHttpServer route(String name, Class<A> input, Class<R> output, Operation<A, R> operation) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        if (!name.matches("[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)+") || !input.isRecord())
            throw new IllegalArgumentException("Invalid control route: " + name);
        var binding = new Binding(ControlJson.schema(input), ControlJson.schema(output),
            args -> operation.call(ControlJson.JSON.treeToValue(args, input)));
        if (bindings.putIfAbsent(name, binding) != null) throw new IllegalArgumentException("Duplicate binding: " + name);
        return this;
    }

    public ControlHttpServer health(Runnable check) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        health = Objects.requireNonNull(check);
        return this;
    }

    public ControlHttpServer start() {
        if (!bindings.isEmpty()) server.createContext("/invoke", this::invoke);
        server.start(); started = true;
        return this;
    }

    /** Optional pull endpoint; exposition and metric definitions belong to the component. */
    public ControlHttpServer metrics(Supplier<String> scrape) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        Objects.requireNonNull(scrape);
        server.createContext("/metrics", exchange -> {
            try (exchange) {
                if (!exchange.getRequestMethod().equals("GET") || !exchange.getRequestURI().toString().equals("/metrics")) {
                    send(exchange, 404, Map.of("error", "Unknown endpoint")); return;
                }
                String text;
                try { text = scrape.get(); }
                catch (RuntimeException error) { send(exchange, 500, Map.of("error", "Metric collection failed")); return; }
                byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            }
        });
        return this;
    }

    public int port() { return server.getAddress().getPort(); }

    /** Launcher variables are supplied by Python; publish only after initialization and API binding. */
    public void publishReady(Path path) throws IOException {
        if (!started) throw new IllegalStateException("API server is not started");
        JsonFiles.writeAtomically(path, Map.of("url", url()));
    }

    public String url() {
        String host = server.getAddress().getAddress().getHostAddress();
        return "http://" + (host.contains(":") ? "[" + host + "]" : host) + ":" + server.getAddress().getPort();
    }

    @Override public void close() {
        server.stop(0);
        handlers.shutdownNow(); calls.shutdownNow();
    }

    private void invoke(HttpExchange exchange) throws IOException {
        try (exchange) {
            if (!exchange.getRequestMethod().equals("POST") || !exchange.getRequestURI().toString().equals("/invoke")) {
                send(exchange, 404, Map.of("error", "Unknown endpoint")); return;
            }
            Future<Object> pending = null;
            try {
                byte[] bytes = exchange.getRequestBody().readNBytes(1024 * 1024 + 1);
                ControlJson.require(bytes.length <= 1024 * 1024, "Request is too large");
                JsonNode request = ControlJson.JSON.readTree(bytes);
                keys(request, Set.of("interface", "args", "timeout_ms"));
                ControlJson.require(request.get("interface").isTextual(), "Invalid interface name");
                var binding = bindings.get(request.get("interface").textValue());
                ControlJson.require(binding != null, "Interface is not implemented by this component");
                ControlJson.require(request.get("timeout_ms").isIntegralNumber() && request.get("timeout_ms").canConvertToLong(), "Invalid timeout");
                long timeout = request.get("timeout_ms").longValue();
                ControlJson.require(timeout > 0, "Timeout must be positive");
                ControlJson.validate(request.get("args"), binding.input());
                pending = calls.submit(() -> binding.operation().call(request.get("args")));
                Object result = pending.get(timeout, TimeUnit.MILLISECONDS);
                JsonNode encoded = ControlJson.JSON.valueToTree(result);
                try { ControlJson.validate(encoded, binding.output()); }
                catch (IllegalArgumentException error) { throw new IllegalStateException("Invalid component result: " + error.getMessage()); }
                send(exchange, 200, Map.of("result", encoded));
            } catch (TimeoutException error) {
                send(exchange, 504, Map.of("error", "Invocation timed out; action outcome may be unknown"));
            } catch (IllegalArgumentException | com.fasterxml.jackson.core.JsonProcessingException error) {
                send(exchange, 400, Map.of("error", error.toString()));
            } catch (Exception error) {
                send(exchange, 500, Map.of("error", error.toString()));
            } finally {
                if (pending != null && !pending.isDone()) pending.cancel(true);
            }
        }
    }

    private static void send(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] bytes = ControlJson.JSON.writeValueAsBytes(value);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
    }

    private static void keys(JsonNode value, Set<String> expected) {
        ControlJson.require(value != null && value.isObject(), "Expected object envelope");
        Set<String> actual = new HashSet<>(); value.fieldNames().forEachRemaining(actual::add);
        ControlJson.require(actual.equals(expected), "Unexpected/missing envelope fields");
    }

    /**
     * Marks a record component that may be omitted from the JSON message.
     */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.RECORD_COMPONENT)
    public @interface OptionalArgument {}

    /**
     * Invokes the component operation for one decoded request.
     */
    @FunctionalInterface
    public interface Operation<A, R> { R call(A args) throws Exception; }

    private record Binding(Map<String, Object> input, Map<String, Object> output,
                           Operation<JsonNode, Object> operation) {}
}
