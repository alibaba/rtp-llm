package org.flexlb.engine.grpc.nameresolver;

import java.util.List;

/**
 * @author zjw
 * description:
 * date: 2025/4/18
 */
public interface CustomNameResolver {

    void start(Listener listener);

    interface Listener {

        void onAddressUpdate(List<String/*ip:port*/> ipPortList);
    }

}
