package org.flexlb.service.config.source;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.util.JsonUtils;
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
    private static final int STARTUP_RETRY_INTERVAL_SECONDS = 1;
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
            configContent = fetchInitialConfig();
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

    private String fetchInitialConfig() throws IOException {
        for (int attempt = 1; ; attempt++) {
            try {
                return fetchConfig();
            } catch (IOException error) {
                log.warn("UniConfig configuration is not ready; retrying startup (attempt {}): {}",
                        attempt, error.toString());
                try {
                    waitForStartupRetry();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted while waiting for the UniConfig agent", interrupted);
                }
            }
        }
    }

    void waitForStartupRetry() throws InterruptedException {
        TimeUnit.SECONDS.sleep(STARTUP_RETRY_INTERVAL_SECONDS);
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
                String content = new String(response.readAllBytes(), StandardCharsets.UTF_8);
                validateConfigDocument(content);
                return content;
            }
        } finally {
            connection.disconnect();
        }
    }

    private static void validateConfigDocument(String content) throws IOException {
        JsonNode document;
        try {
            document = JsonUtils.readStrictTree(content);
        } catch (Exception error) {
            throw new IOException("UniConfig returned invalid configuration JSON", error);
        }
        if (document == null || !document.isObject()) {
            throw new IOException("UniConfig configuration must be a JSON object");
        }
        if (!document.has("consistency") && !document.has("flexlbSyncConsistencyConfig")) {
            throw new IOException("UniConfig configuration must contain consistency or flexlbSyncConsistencyConfig");
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
