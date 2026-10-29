package org.flexlb.mock.client;

import java.nio.file.*;
import java.util.*;
import org.flexlb.mock.client.input.ClientInput.Request;
import org.flexlb.mock.client.ClientProtocol.Start;

final class InputFixture {
    static Request request() { return Request.newBuilder().addAllTokenIds(List.of(1, 2, 3)).setMaxNewTokens(3).setPriority(20).build(); }
    static Path write(Path path, Iterable<Request> requests) throws Exception {
        try (var writer = new InputFile.Writer(Files.newOutputStream(path, StandardOpenOption.CREATE_NEW))) {
            for (var request : requests) writer.write(request);
            writer.finish();
        }
        return path;
    }
    static Start plan(Path directory, List<String> masters, int count, double qps, int slots, int queue, long timeout, long drain) throws Exception {
        var path = write(directory.resolve(UUID.randomUUID() + ".binpb.zst"), Collections.nCopies(count, request()));
        return new Start(masters, path.toString(), qps, null, slots, queue, timeout, drain);
    }
}
