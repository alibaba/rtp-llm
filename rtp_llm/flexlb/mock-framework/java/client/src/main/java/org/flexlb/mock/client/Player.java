package org.flexlb.mock.client;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;
import org.flexlb.mock.client.ClientProtocol.*;

/** The only rate controller. Requests carry no planned send timestamp. */
final class Player {
    static void play(Shard plan, long startNanos, Consumer<Item> submit) throws IOException, InterruptedException {
        try (var reader = new RequestFiles.Reader(Path.of(plan.requests_file()))) {
            for (long position = 0; position < plan.count(); position++) {
                Input input = reader.next();
                if (input == null) throw new IOException("Truncated request shard: " + plan.requests_file());
                long index = Math.addExact(plan.first_index(), Math.multiplyExact(position, plan.stride()));
                long due = startNanos + (long) (index * 1_000_000_000.0 / plan.qps());
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
