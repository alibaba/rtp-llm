package org.flexlb.mock.master;

import java.util.*;
import java.util.function.*;
import org.flexlb.mock.api.master.MasterStateApi;
import org.flexlb.mock.api.model.Types.*;

/** Translate observed Master state to stable handles; never synthesize membership from deployment. */
public final class MasterReads implements MasterStateApi {
    private final String id;
    private final BooleanSupplier leader;
    private final Function<Role, Set<String>> observed;
    private final Supplier<Map<String, String>> identities;
    public MasterReads(String id, BooleanSupplier leader, Function<Role, Set<String>> observed, Supplier<Map<String, String>> identities) {
        this.id = Objects.requireNonNull(id); this.leader = Objects.requireNonNull(leader);
        this.observed = Objects.requireNonNull(observed); this.identities = Objects.requireNonNull(identities);
    }
    @Override public MasterSet leadership(None args) {
        return new MasterSet(leader.getAsBoolean() ? List.of(new MasterRef(id)) : List.of());
    }
    @Override public PerMasterEngines engines(EngineSelection args) {
        var known = identities.get();
        var instances = observed.apply(args.role()).stream().map(address -> {
            var engine = known.get(address);
            if (engine == null) throw new IllegalStateException("Master observed an unregistered Engine address: " + address);
            return new EngineRef(engine);
        }).sorted(Comparator.comparing(EngineRef::id)).toList();
        return new PerMasterEngines(Map.of(id, new EngineSet(instances)));
    }
}
