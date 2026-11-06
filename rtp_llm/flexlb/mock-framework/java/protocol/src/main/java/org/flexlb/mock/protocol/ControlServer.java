package org.flexlb.mock.protocol;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.*;
import java.io.IOException;
import java.net.*;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Supplier;

/** A transport bound to component objects; no case, gate or engine behavior lives here. */
public final class ControlServer implements AutoCloseable {
    public record Context(String run, Instant queryTime, Map<String, Object> observation) {}

    @FunctionalInterface
    public interface Operation<A, R> { R call(A args) throws Exception; }

    @FunctionalInterface
    public interface ContextOperation<A, R> { R call(A args, Context context) throws Exception; }

    private record Binding(boolean background, Map<String, Object> input, Map<String, Object> output,
                           ContextOperation<JsonNode, Object> operation) {}
    private final Map<String, Binding> bindings = new HashMap<>();
    private final HttpServer server;
    private final ExecutorService handlers = Executors.newVirtualThreadPerTaskExecutor();
    private final ExecutorService calls = Executors.newVirtualThreadPerTaskExecutor();
    private boolean started;
    private Runnable health = () -> {};

    public ControlServer(InetSocketAddress address) throws IOException {
        server = HttpServer.create(address, 0);
        server.setExecutor(handlers);
        server.createContext("/invoke", this::invoke);
        server.createContext("/health", exchange -> {
            try (exchange) {
                if (!exchange.getRequestMethod().equals("GET") || !exchange.getRequestURI().toString().equals("/health")) {
                    send(exchange, 404, Map.of("error", "Unknown endpoint")); return;
                }
                try { health.run(); }
                catch (RuntimeException error) { send(exchange, 503, Map.of("status", "unavailable")); return; }
                send(exchange, 200, Map.of("status", "ok", "interfaces", bindings.keySet()));
            }
        });
    }

    public <A, R> ControlServer route(String name, Class<A> input, Class<R> output, Operation<A, R> operation) {
        return route(name, input, output, false, (args, context) -> operation.call(args));
    }

    public <A, R> ControlServer route(String name, Class<A> input, Class<R> output, boolean background,
                                 ContextOperation<A, R> operation) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        if (!name.matches("[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)+") || !input.isRecord())
            throw new IllegalArgumentException("Invalid control route: " + name);
        var binding = new Binding(background, RecordSchema.schema(input), RecordSchema.schema(output),
            (args, context) -> operation.call(input.cast(JsonWire.decode(args, input)), context));
        if (bindings.putIfAbsent(name, binding) != null) throw new IllegalArgumentException("Duplicate binding: " + name);
        return this;
    }

    public ControlServer health(Runnable check) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        health = Objects.requireNonNull(check);
        return this;
    }

    public ControlServer start() {
        // A native-protocol node can publish transport health before business adapters are bound.
        server.start(); started = true;
        return this;
    }

    /** Optional pull endpoint; exposition and metric definitions belong to the component. */
    public ControlServer metrics(Supplier<String> scrape) {
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

    public String url() {
        String host = server.getAddress().getAddress().getHostAddress();
        return "http://" + (host.contains(":") ? "[" + host + "]" : host) + ":" + server.getAddress().getPort();
    }

    /** Launcher variables are supplied by Python; publish only after initialization and API binding. */
    public void publishReady(Path path) throws IOException {
        if (!started) throw new IllegalStateException("API server is not started");
        Path temporary = Files.createTempFile(path.toAbsolutePath().getParent(), ".ready-", ".json");
        Files.write(temporary, JsonWire.JSON.writeValueAsBytes(Map.of("url", url())));
        Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE);
    }

    @SuppressWarnings("unchecked")
    private void invoke(HttpExchange exchange) throws IOException {
        try (exchange) {
            if (!exchange.getRequestMethod().equals("POST") || !exchange.getRequestURI().toString().equals("/invoke")) {
                send(exchange, 404, Map.of("error", "Unknown endpoint")); return;
            }
            Future<Object> pending = null;
            try {
                byte[] bytes = exchange.getRequestBody().readNBytes(1024 * 1024 + 1);
                JsonWire.require(bytes.length <= 1024 * 1024, "Request is too large");
                JsonNode request = JsonWire.JSON.readTree(bytes);
                keys(request, Set.of("interface", "args", "context"));
                JsonWire.require(request.get("interface").isTextual(), "Invalid interface name");
                var binding = bindings.get(request.get("interface").textValue());
                JsonWire.require(binding != null, "Interface is not implemented by this component");
                var context = request.get("context");
                keys(context, Set.of("run", "query_time", "observation", "timeout_ms"));
                String run = context.path("run").asText();
                JsonWire.require(run.equals("foreground") || (run.equals("background") && binding.background()), "Unsupported run mode");
                JsonWire.require(context.get("timeout_ms").isIntegralNumber() && context.get("timeout_ms").canConvertToLong(), "Invalid timeout");
                long timeout = context.get("timeout_ms").longValue();
                JsonWire.require(timeout > 0, "Timeout must be positive");
                JsonWire.require(context.get("query_time").isTextual(), "Invalid query time");
                Instant time;
                try { time = Instant.parse(context.get("query_time").textValue()); }
                catch (java.time.DateTimeException error) { throw new IllegalArgumentException("Invalid query time", error); }
                JsonWire.require(context.get("observation").isNull() || context.get("observation").isObject(), "Invalid observation context");
                var observation = context.get("observation").isNull() ? null :
                    (Map<String, Object>) JsonWire.JSON.convertValue(context.get("observation"), Map.class);
                var invocation = new Context(run, time, observation);
                JsonWire.validate(request.get("args"), binding.input());
                pending = calls.submit(() -> binding.operation().call(request.get("args"), invocation));
                Object result = pending.get(timeout, TimeUnit.MILLISECONDS);
                JsonNode encoded = JsonWire.JSON.valueToTree(result);
                try { JsonWire.validate(encoded, binding.output()); }
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

    private static void keys(JsonNode value, Set<String> expected) {
        JsonWire.require(value != null && value.isObject(), "Expected object envelope");
        Set<String> actual = new HashSet<>(); value.fieldNames().forEachRemaining(actual::add);
        JsonWire.require(actual.equals(expected), "Unexpected/missing envelope fields");
    }

    private static void send(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] bytes = JsonWire.JSON.writeValueAsBytes(value);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
    }

    @Override public void close() {
        server.stop(0);
        handlers.shutdownNow(); calls.shutdownNow();
    }
}
