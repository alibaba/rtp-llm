package org.flexlb.engine.grpc.nameresolver;

import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * Compatibility adapter for the legacy engine gRPC client.
 *
 * <p>{@link EngineAddressResolver} owns service discovery, polling, address
 * aggregation, and listener notification. This adapter only bridges the
 * legacy {@link CustomNameResolver} listener API to that shared resolver.</p>
 *
 * @author saichen.sm
 * date: 2025/9/19
 */
@Component
public class EngineAddressNameResolver implements CustomNameResolver {

    private final EngineAddressResolver engineAddressResolver;

    public EngineAddressNameResolver(EngineAddressResolver engineAddressResolver) {
        this.engineAddressResolver = engineAddressResolver;
    }

    @Override
    public void start(Listener listener) {
        Objects.requireNonNull(listener, "listener");
        engineAddressResolver.subscribe(listener::onAddressUpdate);
    }
}
