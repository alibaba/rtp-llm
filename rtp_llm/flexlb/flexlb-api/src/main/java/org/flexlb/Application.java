package org.flexlb;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Pandora Boot应用的入口类
 */
@Slf4j
@EnableScheduling
@SpringBootApplication(scanBasePackages = {"org.flexlb"})
public class Application {

    public static void main(String[] args) {
        // 打印启动参数
        log.info("Application start with args: {}", (Object[]) args);
        ConfigurableApplicationContext context = SpringApplication.run(Application.class, args);
        Environment env = context.getEnvironment();
        String port = env.getProperty("server.port");
        log.info("flex-lb server started on port {}", port);
    }
}