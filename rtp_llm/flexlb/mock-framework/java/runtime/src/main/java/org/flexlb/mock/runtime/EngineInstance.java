package org.flexlb.mock.runtime;

import java.util.Map;

/** Host ownership only; role protocols and scheduling stay in their implementations. */
public interface EngineInstance extends AutoCloseable {
    Map<String,Object> endpoints();
    void checkHealthy();
    @Override void close();
}
