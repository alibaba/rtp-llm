import java.net.URI;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.regex.Pattern;

/** Bounded-memory protocol load: no duplicate billion-key map in the generator. */
public class KvcmCapacityLoad {
    static Path errorPrefix;
    static final long KEY_BASE = 1_000_000_000_000_000_000L;
    static final long KEY_STRIDE = 10_000_000_000L;
    static final Pattern OK = Pattern.compile("\\\"code\\\"\\s*:\\s*(\\\"OK\\\"|\\\"1\\\"|1)([,}])");
    static final Pattern ITEMS = Pattern.compile("\\\"item_results\\\"\\s*:\\s*\\[([^]]*)]");
    static void post(HttpClient client, String endpoint, String body) throws Exception {
        var request = HttpRequest.newBuilder(URI.create(endpoint + "/api/reportEvent"))
                .timeout(Duration.ofSeconds(60)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        String result = response.body();
        var items = ITEMS.matcher(result);
        if (response.statusCode() != 200 || !OK.matcher(result).find()
                || !items.find() || !items.group(1).replaceAll("\\\"OK\\\"|\\\"1\\\"|1|,|\\s", "").isEmpty()
                || result.matches("(?s).*\\\"snapshot_required\\\"\\s*:\\s*true.*")) {
            Files.writeString(Path.of(errorPrefix + "-failure-" + Thread.currentThread().threadId() + ".json"), result);
            throw new IllegalStateException("HTTP=" + response.statusCode() + " " + result.substring(0, Math.min(400, result.length())));
        }
    }
    public static void main(String[] args) throws Exception {
        String endpoint = args[0], instance = args[1];
        int reporters = Integer.parseInt(args[2]), batch = Integer.parseInt(args[5]);
        long start = Long.parseLong(args[3]), end = Long.parseLong(args[4]);
        Path progress = Path.of(args[6]);
        errorPrefix = progress;
        int concurrency = args.length > 7 ? Integer.parseInt(args[7]) : reporters;
        var accepted = new AtomicLong();
        var requests = new AtomicLong();
        var bytes = new AtomicLong();
        var stopped = new AtomicBoolean();
        var error = new AtomicReference<String>("");
        long began = System.nanoTime();
        var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3))
                .version(HttpClient.Version.HTTP_1_1).build();
        var pool = Executors.newFixedThreadPool(concurrency);
        var tasks = new ArrayList<Future<?>>();
        for (int reporter = 0; reporter < reporters; reporter++) {
            final int id = reporter;
            tasks.add(pool.submit(() -> {
                String host = "127.2.0." + (id + 1) + ":" + (30000 + id);
                for (long cursor = start; cursor < end && !stopped.get(); cursor += batch) {
                    int count = (int) Math.min(batch, end - cursor);
                    var body = new StringBuilder(count * 195);
                    body.append("{\"trace_id\":\"capacity-").append(id).append('-').append(cursor)
                            .append("\",\"instance_id\":\"").append(instance).append("\",\"host_ip_port\":\"")
                            .append(host).append("\",\"storage_type\":\"ST_EVENT_REPORT_L1P5\",\"events\":[");
                    for (int i = 0; i < count; i++) {
                        if (i > 0) body.append(',');
                        body.append("{\"event_type\":\"EVENT_BLOCK_ADD\",\"block_add\":{\"block_key\":\"")
                                .append(KEY_BASE + id * KEY_STRIDE + cursor + i)
                                .append("\",\"medium\":\"hbm\",\"specs\":[{\"name\":\"mock_hbm\",\"uri\":\"rtp-llm://")
                                .append(host).append("/hbm\"}]}}");
                    }
                    body.append("]}");
                    try {
                        post(client, endpoint, body.toString());
                        accepted.addAndGet(count); requests.incrementAndGet(); bytes.addAndGet(body.length());
                    } catch (Exception ex) {
                        error.compareAndSet("", ex.toString()); stopped.set(true);
                    }
                }
            }));
        }
        pool.shutdown();
        while (!pool.awaitTermination(2, TimeUnit.SECONDS))
            save(progress, accepted.get(), requests.get(), bytes.get(), began, false, error.get());
        for (var task : tasks) task.get();
        save(progress, accepted.get(), requests.get(), bytes.get(), began, true, error.get());
        client.close();
        if (!error.get().isEmpty()) throw new IllegalStateException(error.get());
        if (accepted.get() != (end - start) * reporters) throw new IllegalStateException("incomplete load");
    }
    static void save(Path path, long keys, long requests, long bytes, long began, boolean done, String error) throws Exception {
        double seconds = (System.nanoTime() - began) / 1e9;
        String json = String.format(Locale.ROOT,
                "{\"accepted_keys\":%d,\"requests\":%d,\"request_bytes\":%d,\"elapsed_s\":%.3f,\"keys_per_s\":%.3f,\"done\":%s,\"error\":\"%s\"}",
                keys, requests, bytes, seconds, keys / seconds, done,
                error.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " "));
        Path temp = Path.of(path + ".tmp");
        Files.writeString(temp, json);
        Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        System.out.println(json);
    }
}
