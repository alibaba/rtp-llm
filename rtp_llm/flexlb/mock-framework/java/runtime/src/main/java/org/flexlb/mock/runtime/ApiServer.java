package org.flexlb.mock.runtime;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.*;
import java.io.IOException;
import java.lang.reflect.*;
import java.net.*;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.api.*;

/** A transport bound to component objects; no case, gate or engine behavior lives here. */
public final class ApiServer implements AutoCloseable {
    private record Binding(Object target, Method method, Map<String, Object> schema) {}
    private final Map<String, Binding> bindings = new HashMap<>();
    private final HttpServer server;
    private final ExecutorService handlers = Executors.newVirtualThreadPerTaskExecutor();
    private final ExecutorService calls = Executors.newVirtualThreadPerTaskExecutor();
    private boolean started;

    public ApiServer(InetSocketAddress address) throws IOException {
        server = HttpServer.create(address, 0);
        server.setExecutor(handlers);
        server.createContext("/invoke", this::invoke);
        server.createContext("/health", exchange -> {
            try (exchange) {
                if (exchange.getRequestMethod().equals("GET") && exchange.getRequestURI().toString().equals("/health"))
                    send(exchange, 200, Map.of("interfaces", bindings.keySet()));
                else send(exchange, 404, Map.of("error", "Unknown endpoint"));
            }
        });
    }

    @SuppressWarnings("unchecked")
    public <T> ApiServer bind(Class<T> api, T implementation) {
        if (started) throw new IllegalStateException("Cannot bind after start");
        if (!api.isInstance(implementation)) throw new IllegalArgumentException("Implementation does not implement " + api);
        var schemas = InterfaceRegistry.describe(api);
        for (var method : api.getMethods()) {
            String name = method.getAnnotation(Endpoint.class).value();
            if (bindings.putIfAbsent(name, new Binding(implementation, method, (Map<String, Object>) schemas.get(name))) != null)
                throw new IllegalArgumentException("Duplicate binding: " + name);
        }
        return this;
    }

    public ApiServer start() {
        // A native-protocol node can publish transport health before business adapters are bound.
        server.start(); started = true;
        return this;
    }

    public String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }

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
                JsonWire.require(run.equals("foreground") || (run.equals("background") && Boolean.TRUE.equals(binding.schema.get("background"))), "Unsupported run mode");
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
                var invocation = new InvocationContext(run, time, observation, timeout);
                JsonWire.validate(request.get("args"), (Map<String, Object>) binding.schema.get("args"));
                Object args = JsonWire.decode(request.get("args"), binding.method.getGenericParameterTypes()[0]);
                pending = calls.submit(() -> {
                    InvocationContext.enter(invocation);
                    try { return binding.method.invoke(binding.target, args); }
                    catch (InvocationTargetException error) {
                        if (error.getCause() instanceof Exception cause) throw cause;
                        throw error;
                    } finally { InvocationContext.leave(); }
                });
                Object result = pending.get(timeout, TimeUnit.MILLISECONDS);
                JsonNode encoded = JsonWire.JSON.valueToTree(result);
                try { JsonWire.validate(encoded, (Map<String, Object>) binding.schema.get("returns")); }
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
