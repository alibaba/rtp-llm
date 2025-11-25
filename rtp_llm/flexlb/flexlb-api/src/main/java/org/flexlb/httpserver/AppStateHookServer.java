package org.flexlb.httpserver;

import lombok.NoArgsConstructor;
import org.flexlb.listener.OnlineListener;
import org.flexlb.util.LoggingUtils;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.concurrent.CopyOnWriteArrayList;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
@Component
@NoArgsConstructor
public class AppStateHookServer {

    private final CopyOnWriteArrayList<OnlineListener> onlineListeners = new CopyOnWriteArrayList<>();

    @Bean
    public RouterFunction<ServerResponse> appStateHook() {
        return route().GET("/hook/afterStart", accept(MediaType.ALL),
                        this::handleAppStartUp)
                .build();
    }

    public Mono<ServerResponse> handleAppStartUp(ServerRequest request) {
        LoggingUtils.warn("recv /hook/afterStart request.");
        if (request.remoteAddress().isPresent() && request.remoteAddress().get().getAddress().isLoopbackAddress()) {
            try {
                for (OnlineListener onlineListener : onlineListeners) {
                    onlineListener.afterStartUp();
                }
                return ServerResponse.ok().body(Mono.just("success"), String.class);
            } catch (Exception e) {
                LoggingUtils.warn("handleOnline error.", e);
                return ServerResponse.status(500).body(Mono.just("error"), String.class);
            }
        } else {
            return ServerResponse.status(403).body(Mono.just("remote request is not allowed."), String.class);
        }
    }

    public void addOnlineHandler(OnlineListener onlineListener) {
        onlineListeners.add(onlineListener);
    }

}
