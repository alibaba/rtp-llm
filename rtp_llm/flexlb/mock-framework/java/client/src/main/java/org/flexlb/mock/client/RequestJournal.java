package org.flexlb.mock.client;

import com.github.luben.zstd.ZstdOutputStream;
import java.io.*;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.ArrayDeque;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.client.evidence.ClientEvidence.*;

/** One bounded, nonblocking producer queue; only the writer touches disk. */
final class RequestJournal {
    static final int MAX_FRAME = 2 * 1024 * 1024;
    private final int capacity;
    private final ArrayDeque<Event> queue = new ArrayDeque<>();
    private final EventBatch template;
    private final FileChannel file;
    private final Thread writer;
    private int queuedBytes;
    private long sequence;
    private Seal seal;
    private boolean closing;
    private volatile String failure;

    RequestJournal(Path path, String client, ClientProtocol.Shard plan) throws IOException {
        this(path, client, plan, 8 * 1024 * 1024);
    }
    RequestJournal(Path path, String client, ClientProtocol.Shard plan, int capacity) throws IOException {
        this.capacity = capacity;
        template = EventBatch.newBuilder().setVersion(1).setClient(client).setPlayback(plan.sequence())
            .setFirstRequestId(plan.first_request_id()).setFirstIndex(plan.first_index()).setStride(plan.stride())
            .setPlanned(plan.count()).setConcurrency(plan.concurrency()).build();
        Files.createDirectories(path.getParent());
        file = FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        writer = Thread.ofPlatform().daemon().name("client-evidence").start(this::write);
    }
    synchronized void record(Event.Builder builder) {
        if (failure != null) return;
        if (closing) { failure = "Event after evidence close"; return; }
        var event = builder.setSequence(sequence + 1).build();
        int size = event.getSerializedSize() + 16;
        if (size > 64 * 1024 || size > capacity - queuedBytes) {
            failure = "Client evidence queue full or oversized event"; notifyAll(); return;
        }
        sequence++; queue.addLast(event); queuedBytes += size; notifyAll();
    }
    String failure() { return failure; }
    void finish(long issued, long finished) throws IOException {
        synchronized (this) {
            if (issued != finished) failure = "Client evidence has unfinished requests";
            seal = Seal.newBuilder().setIssued(issued).setFinished(finished).build(); closing = true; notifyAll();
        }
        try { writer.join(5000); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); failure = "Evidence close interrupted"; }
        if (writer.isAlive()) {
            failure = "Evidence writer did not finish"; writer.interrupt(); file.close();
        }
        if (failure != null) throw new IOException(failure);
    }
    private void write() {
        try (file; var out = new DataOutputStream(Channels.newOutputStream(file))) {
            long frame = 0;
            while (true) {
                var batch = template.toBuilder().setFrame(++frame);
                synchronized (this) {
                    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
                    while (!closing && failure == null && queuedBytes < 1024 * 1024 && System.nanoTime() < deadline)
                        TimeUnit.NANOSECONDS.timedWait(this, Math.max(1, deadline - System.nanoTime()));
                    if (failure != null) return;
                    int bytes = 0;
                    while (!queue.isEmpty() && bytes < 1024 * 1024) {
                        var event = queue.removeFirst(); int size = event.getSerializedSize() + 16;
                        queuedBytes -= size; bytes += size; batch.addEvents(event);
                    }
                    if (closing && queue.isEmpty()) batch.setSeal(seal);
                }
                if (batch.getEventsCount() == 0 && !batch.hasSeal()) { frame--; continue; }
                var compressed = new ByteArrayOutputStream();
                try (var zstd = new ZstdOutputStream(compressed, 3).setChecksum(true)) { batch.build().writeDelimitedTo(zstd); }
                byte[] encoded = compressed.toByteArray();
                if (encoded.length > MAX_FRAME) throw new IOException("Oversized evidence frame");
                out.writeInt(encoded.length); out.write(encoded);
                // Completed frames survive a later crash; only a forced final seal certifies completeness.
                file.force(false);
                if (batch.hasSeal()) return;
            }
        } catch (Exception error) { failure = "Client evidence write failed: " + error; }
    }
}
