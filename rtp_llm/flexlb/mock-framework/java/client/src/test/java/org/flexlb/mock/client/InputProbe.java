package org.flexlb.mock.client;

import java.nio.file.*;
import java.util.stream.IntStream;
import org.flexlb.mock.client.input.ClientInput.Request;

/** Cross-language fixture producer/reader; no second production input path. */
public final class InputProbe {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[1]);
        if (args[0].equals("write")) {
            InputFixture.write(path, IntStream.range(0, 2050).mapToObj(i -> {
                var request = Request.newBuilder().addTokenIds(i).setMaxNewTokens(3);
                if (i % 2 == 0) request.setPriority(i % 100);
                return request.build();
            }).toList());
        } else if (args[0].equals("read")) {
            int count = 0;
            try (var reader = new InputFile.Reader(path)) {
                Request request;
                while ((request = reader.next()) != null) {
                    if (request.getTokenIds(0) != count || request.getMaxNewTokens() != 3
                            || request.hasPriority() != (count % 2 == 0) || request.hasPriority() && request.getPriority() != count % 100)
                        throw new AssertionError("Fields differ at request " + count);
                    count++;
                }
            }
            if (count != 2050) throw new AssertionError("Wrong request count: " + count);
        } else throw new IllegalArgumentException("Expected read or write");
    }
}
