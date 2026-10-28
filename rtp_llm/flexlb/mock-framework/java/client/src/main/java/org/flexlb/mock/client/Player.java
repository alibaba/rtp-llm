package org.flexlb.mock.client;

import java.util.List;
import java.util.function.Consumer;
import org.flexlb.mock.client.ClientProtocol.Item;

/** The only rate controller. Requests carry no planned send timestamp. */
final class Player {
    static void play(List<Item> input, double qps, long startNanos, Consumer<Item> submit) throws InterruptedException {
        for (var item : input) {
            long due = startNanos + (long) (item.index() * 1_000_000_000.0 / qps);
            long left;
            while ((left = due - System.nanoTime()) > 0) java.util.concurrent.TimeUnit.NANOSECONDS.sleep(left);
            if (Thread.interrupted()) throw new InterruptedException();
            submit.accept(item);
        }
    }
}
