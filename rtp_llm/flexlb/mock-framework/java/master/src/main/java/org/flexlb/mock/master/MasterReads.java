package org.flexlb.mock.master;

import java.util.*;
import java.util.function.*;
import org.flexlb.mock.api.master.MasterStateApi;
import org.flexlb.mock.api.model.Types.*;

/** Return observed Master state and addresses; never synthesize membership from deployment. */
public final class MasterReads implements MasterStateApi {
    private final String id;
    private final BooleanSupplier leader;
    private final Function<Role, Set<String>> observed;
    public MasterReads(String id, BooleanSupplier leader, Function<Role, Set<String>> observed) {
        this.id = Objects.requireNonNull(id); this.leader = Objects.requireNonNull(leader);
        this.observed = Objects.requireNonNull(observed);
    }
    @Override public MasterSet leadership(None args) {
        return new MasterSet(leader.getAsBoolean() ? List.of(new MasterRef(id)) : List.of());
    }
    @Override public PerMasterEngines engines(EngineSelection args) {
        return new PerMasterEngines(Map.of(id, observed.apply(args.role()).stream().sorted().toList()));
    }
}
