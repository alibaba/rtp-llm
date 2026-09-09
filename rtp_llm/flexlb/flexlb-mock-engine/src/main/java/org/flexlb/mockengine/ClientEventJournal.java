package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Optional, flushed lifecycle evidence for bounded functional checkpoints. */
final class ClientEventJournal implements AutoCloseable {
    private final BufferedWriter writer;
    private long sequence;

    ClientEventJournal(Path path) throws IOException {
        writer = Files.newBufferedWriter(path);
    }

    synchronized void record(String event, ObjectNode request) {
        ObjectNode row = request.deepCopy();
        row.put("event", event);
        row.put("sequence", ++sequence);
        row.put("recorded_epoch_ms", System.currentTimeMillis());
        try {
            writer.write(row.toString());
            writer.newLine();
            writer.flush();
        } catch (IOException error) {
            throw new UncheckedIOException("Cannot persist live client evidence", error);
        }
    }

    @Override
    public synchronized void close() {
        try {
            writer.close();
        } catch (IOException error) {
            throw new UncheckedIOException(error);
        }
    }
}
