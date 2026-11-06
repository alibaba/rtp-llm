package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.*;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/** One bounded, nonblocking producer queue; only the writer touches disk. */
final class RequestJournal {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final int capacity;
    private final ArrayDeque<byte[]> queue = new ArrayDeque<>();
    private final ObjectNode header;
    private final FileChannel file;
    private final Thread writer;
    private int queuedBytes;
    private long sequence;
    private ObjectNode seal;
    private boolean closing;
    private volatile String failure;

    RequestJournal(Path path, String client, ClientProtocol.Shard plan) throws IOException {
        this(path, client, plan, 8 * 1024 * 1024);
    }
    RequestJournal(Path path, String client, ClientProtocol.Shard plan, int capacity) throws IOException {
        this.capacity = capacity;
        header = JSON.createObjectNode().put("version", 1).put("client", client)
            .put("playback", Long.toString(plan.sequence()))
            .put("first_request_id", Long.toString(plan.first_request_id()))
            .put("first_index", Long.toString(plan.first_index())).put("stride", plan.stride())
            .put("planned", Long.toString(plan.count())).put("concurrency", plan.concurrency());
        Files.createDirectories(path.getParent());
        file = FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        writer = Thread.ofPlatform().daemon().name("client-evidence").start(this::write);
    }
    synchronized void record(ObjectNode event) {
        if (failure != null) return;
        if (closing) { failure = "Event after evidence close"; return; }
        byte[] row;
        try {
            row = JSON.writeValueAsBytes(event.put("sequence", Long.toString(sequence + 1)));
        } catch (IOException error) {
            failure = "Client evidence serialization failed: " + error;
            notifyAll();
            return;
        }
        int size = row.length + 16;
        if (size > 64 * 1024 || size > capacity - queuedBytes) {
            failure = "Client evidence queue full or oversized event"; notifyAll(); return;
        }
        sequence++; queue.addLast(row); queuedBytes += size; notifyAll();
    }
    String failure() { return failure; }
    void finish(long issued, long finished) throws IOException {
        synchronized (this) {
            if (issued != finished) failure = "Client evidence has unfinished requests";
            seal = JSON.createObjectNode();
            seal.putObject("seal").put("issued", Long.toString(issued)).put("finished", Long.toString(finished));
            closing = true;
            notifyAll();
        }
        try { writer.join(5000); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); failure = "Evidence close interrupted"; }
        if (writer.isAlive()) {
            failure = "Evidence writer did not finish"; writer.interrupt(); file.close();
        }
        if (failure != null) throw new IOException(failure);
    }
    private void write() {
        try (file; var out = new BufferedOutputStream(Channels.newOutputStream(file))) {
            boolean headerWritten = false;
            while (true) {
                var rows = new ArrayList<byte[]>();
                ObjectNode finalSeal;
                synchronized (this) {
                    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
                    while (!closing && failure == null && queuedBytes < 1024 * 1024 && System.nanoTime() < deadline)
                        TimeUnit.NANOSECONDS.timedWait(this, Math.max(1, deadline - System.nanoTime()));
                    if (failure != null) return;
                    int bytes = 0;
                    while (!queue.isEmpty() && bytes < 1024 * 1024) {
                        var row = queue.removeFirst();
                        int size = row.length + 16;
                        queuedBytes -= size; bytes += size; rows.add(row);
                    }
                    finalSeal = closing && queue.isEmpty() ? seal : null;
                }
                if (rows.isEmpty() && finalSeal == null) continue;
                if (!headerWritten) {
                    writeLine(out, JSON.writeValueAsBytes(header));
                    headerWritten = true;
                }
                for (var row : rows) writeLine(out, row);
                if (finalSeal != null) writeLine(out, JSON.writeValueAsBytes(finalSeal));
                out.flush();
                // Only a forced final seal certifies completeness.
                file.force(false);
                if (finalSeal != null) return;
            }
        } catch (Exception error) { failure = "Client evidence write failed: " + error; }
    }

    private static void writeLine(BufferedOutputStream out, byte[] row) throws IOException {
        out.write(row);
        out.write('\n');
    }
}
