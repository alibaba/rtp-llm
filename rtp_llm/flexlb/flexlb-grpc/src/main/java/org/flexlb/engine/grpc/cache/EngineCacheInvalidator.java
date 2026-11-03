package org.flexlb.engine.grpc.cache;

import java.util.Collection;

/**
 * Removes cache metadata for engines that are no longer present in service discovery.
 *
 * <p>The interface lives in the transport module so gRPC clients do not depend on a
 * particular cache implementation. The cache module supplies the Spring bean.</p>
 */
@FunctionalInterface
public interface EngineCacheInvalidator {

    void removeStaleEngineCaches(Collection<String> activeEngineIpPorts);
}
