package org.flexlb.mock.client;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;
import org.flexlb.mock.client.ClientProtocol.*;

/** The only rate controller; shard timestamps retain their common source origin. */
final class Player {
    static long delayNanos(long index, org.flexlb.mock.client.input.ClientInput.Request input, Double qps, Double speed) throws IOException {
        if (speed != null && !input.hasArrivalOffsetMs()) throw new IOException("Replay requires arrival_offset_ms on every request");
        double seconds = speed == null ? index / qps : input.getArrivalOffsetMs() / (1000.0 * speed);
        if (!Double.isFinite(seconds) || seconds < 0 || seconds > 86400) throw new IOException("Playback exceeds 24 hours");
        return (long) (seconds * 1_000_000_000.0);
    }
    static void play(Shard plan, long startNanos, Consumer<Item> submit) throws IOException, InterruptedException {
        try (var reader = new InputFile.Reader(Path.of(plan.requests_file()))) {
            for (long position = 0; position < plan.count(); position++) {
                var input = reader.next();
                if (input == null) throw new IOException("Truncated request shard: " + plan.requests_file());
                long index = Math.addExact(plan.first_index(), Math.multiplyExact(position, plan.stride()));
                long due = startNanos + delayNanos(index, input, plan.qps(), plan.replay_speed());
                long left;
                while ((left = due - System.nanoTime()) > 0) java.util.concurrent.TimeUnit.NANOSECONDS.sleep(left);
                if (Thread.interrupted()) throw new InterruptedException();
                submit.accept(new Item(Math.addExact(plan.first_request_id(), index), index, input));
            }
            if (reader.next() != null) throw new IOException("Extra records in request shard: " + plan.requests_file());
            reader.verify(plan.sha256());
        }
    }
}
