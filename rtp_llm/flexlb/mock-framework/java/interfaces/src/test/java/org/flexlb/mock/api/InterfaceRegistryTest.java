package org.flexlb.mock.api;

import org.junit.jupiter.api.Test;
import org.flexlb.mock.api.model.Types;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class InterfaceRegistryTest {
    interface Extension {
        record Args(int threshold) {}
        @Endpoint(value="extension/queue", kind=Endpoint.Kind.read)
        Types.MetricValue queue(Args args);
    }
    interface Duplicate {
        @Endpoint(value="extension/queue", kind=Endpoint.Kind.action)
        Types.None queue(Types.None args);
    }
    interface Malformed { Types.None missing(Types.None args); }

    @Test void newMethodDefinesItsArgumentSchema() {
        var extra = InterfaceRegistry.describe(Extension.class);
        var spec = (Map<?, ?>) extra.get("extension/queue");
        var args = (Map<?, ?>) spec.get("args");
        var fields = (Map<?, ?>) args.get("properties");
        assertTrue(fields.containsKey("threshold"));
    }
    @Test void rejectsDuplicateAndUnannotatedMethods() {
        assertThrows(IllegalArgumentException.class, () -> InterfaceRegistry.describe(Extension.class, Duplicate.class));
        assertThrows(IllegalArgumentException.class, () -> InterfaceRegistry.describe(Malformed.class));
    }
}
