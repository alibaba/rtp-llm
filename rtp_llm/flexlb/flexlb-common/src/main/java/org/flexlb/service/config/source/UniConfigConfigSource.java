package org.flexlb.service.config.source;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.service.config.ConfigSource;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/** Fetches raw configuration documents from the local Turbo UniConfig agent. */
@Slf4j
@Component
final class UniConfigConfigSource implements ConfigSource {

    private static final int CONFIG_READ_TIMEOUT_MS = 3000;
    private static final int POLL_INTERVAL_SECONDS = 30;
    private static final int PRIORITY = 3;

    private final URI configUri;
    private ScheduledExecutorService pollExecutor;
    private volatile String configContent;
    private volatile Consumer<String> updateListener;
    private volatile boolean closed;

    UniConfigConfigSource(DeploymentIdentity deploymentIdentity) {
        if (ConfigSourceSelection.fromEnvironment() != ConfigSourceSelection.UNICONFIG) {
            configUri = null;
            return;
        }
        if (!deploymentIdentity.isSpectrum()) {
            throw new IllegalStateException("UniConfig requires a Spectrum deployment identity: "
                    + "SPECTRUM_WORKSPACE_ID, SPECTRUM_APPLICATION_NAME and SPECTRUM_DEPLOYMENT_NAME");
        }
        configUri = URI.create("http://127.0.0.1:18080/v2/configs/modelstudio.spectrum.deployment."
                + deploymentIdentity.getWorkspaceId() + "." + deploymentIdentity.getDeploymentName()
                + ".runtime.meta");
    }

    @Override
    public String name() {
        return "UniConfig";
    }

    @Override
    public int priority() {
        return PRIORITY;
    }

    @PostConstruct
    void initialize() {
        if (configUri == null) {
            log.info("UniConfig configuration source is disabled");
            return;
        }
        try {
            configContent = fetchConfig();
            if (pollExecutor == null) {
                pollExecutor = Executors.newSingleThreadScheduledExecutor(task -> {
                    Thread thread = new Thread(task, "flexlb-uniconfig-poll");
                    thread.setDaemon(true);
                    return thread;
                });
            }
            pollExecutor.scheduleWithFixedDelay(this::poll, POLL_INTERVAL_SECONDS,
                    POLL_INTERVAL_SECONDS, TimeUnit.SECONDS);
            ConfigService.register(this);
        } catch (Exception error) {
            close();
            throw new IllegalStateException("Failed to initialize UniConfig configuration source", error);
        }
    }

    @Override
    public String load() {
        return configContent;
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {
        updateListener = listener;
    }

    @Override
    @PreDestroy
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        updateListener = null;
        if (pollExecutor != null) {
            pollExecutor.shutdownNow();
        }
    }

    private String fetchConfig() throws IOException {
        HttpURLConnection connection = (HttpURLConnection) configUri.toURL().openConnection(Proxy.NO_PROXY);
        try {
            connection.setConnectTimeout(CONFIG_READ_TIMEOUT_MS);
            connection.setReadTimeout(CONFIG_READ_TIMEOUT_MS);
            connection.setInstanceFollowRedirects(false);
            connection.setRequestMethod("GET");
            int status = connection.getResponseCode();
            if (status != HttpURLConnection.HTTP_OK) {
                throw new IOException("UniConfig returned HTTP " + status + " for " + configUri);
            }
            try (InputStream response = connection.getInputStream()) {
                return new String(response.readAllBytes(), StandardCharsets.UTF_8);
            }
        } finally {
            connection.disconnect();
        }
    }

    private void poll() {
        if (closed) {
            return;
        }
        try {
            String updated = fetchConfig();
            if (closed || Objects.equals(configContent, updated)) {
                return;
            }
            configContent = updated;
            Consumer<String> listener = updateListener;
            if (listener != null && !closed) {
                listener.accept(updated);
            }
        } catch (Exception error) {
            if (!closed) {
                log.warn("Failed to refresh UniConfig configuration; keeping last-known-good configuration: {}",
                        error.toString());
            }
        }
    }
}
