package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONReader;

import java.util.Random;

/**
 * Micro-bench: how much does parse() alone of a 752KB envelope (500 CJK prompts of 500 chars
 * each) cost under different fastjson2 entry points? Run as `java -cp ... ParseMicroBench`.
 *
 * <p>If the alternatives are within noise of {@code JSON.parse(byte[])} we can stop pursuing
 * C-light optimization — fastjson2's main entry is already streaming and there's no library-side
 * lift to be had short of skipping field materialization (which is C-medium).
 */
public class ParseMicroBench {

    public static void main(String[] args) throws Exception {
        byte[] payload = makePayload(500, 500);
        System.out.println("payload bytes: " + payload.length);

        int warmup = 200;
        int reps = 500;

        // Warm up each path independently — JIT compiles per-method.
        for (int i = 0; i < warmup; i++) jsonParseCast(payload);
        for (int i = 0; i < warmup; i++) jsonParseObject(payload);
        for (int i = 0; i < warmup; i++) readerOfRead(payload);
        for (int i = 0; i < warmup; i++) readerOfReadTyped(payload);
        for (int i = 0; i < warmup; i++) readerOfReadNative(payload);

        long t1 = time(reps, () -> jsonParseCast(payload));
        long t2 = time(reps, () -> jsonParseObject(payload));
        long t3 = time(reps, () -> readerOfRead(payload));
        long t4 = time(reps, () -> readerOfReadTyped(payload));
        long t5 = time(reps, () -> readerOfReadNative(payload));

        System.out.printf("JSON.parse cast        : %6.2f us/op%n", t1 / 1000.0 / reps);
        System.out.printf("JSON.parseObject       : %6.2f us/op%n", t2 / 1000.0 / reps);
        System.out.printf("JSONReader.of readObj  : %6.2f us/op%n", t3 / 1000.0 / reps);
        System.out.printf("JSONReader read(typed) : %6.2f us/op%n", t4 / 1000.0 / reps);
        System.out.printf("Reader UseNativeObject : %6.2f us/op%n", t5 / 1000.0 / reps);
    }

    static JSONObject jsonParseCast(byte[] body) {
        Object root = JSON.parse(body);
        return root instanceof JSONObject obj ? obj : null;
    }

    static JSONObject jsonParseObject(byte[] body) {
        return JSON.parseObject(body);
    }

    static JSONObject readerOfRead(byte[] body) {
        try (JSONReader r = JSONReader.of(body)) {
            return (JSONObject) r.readObject();
        }
    }

    static JSONObject readerOfReadTyped(byte[] body) {
        try (JSONReader r = JSONReader.of(body)) {
            return r.read(JSONObject.class);
        }
    }

    static Object readerOfReadNative(byte[] body) {
        try (JSONReader r = JSONReader.of(body)) {
            r.getContext().config(JSONReader.Feature.UseNativeObject);
            return r.readObject();
        }
    }

    static long time(int reps, Runnable r) {
        long start = System.nanoTime();
        for (int i = 0; i < reps; i++) r.run();
        return System.nanoTime() - start;
    }

    static byte[] makePayload(int batchSize, int charsPerPrompt) {
        Random rng = new Random(42);
        JSONObject body = new JSONObject();
        body.put("model", "qwen");
        JSONObject gc = new JSONObject();
        gc.put("max_new_tokens", 1);
        gc.put("temperature", 0.0);
        body.put("generate_config", gc);
        JSONArray arr = new JSONArray(batchSize);
        StringBuilder sb = new StringBuilder(charsPerPrompt);
        for (int i = 0; i < batchSize; i++) {
            sb.setLength(0);
            for (int j = 0; j < charsPerPrompt; j++) {
                sb.append((char) (0x4E00 + rng.nextInt(0x9FFF - 0x4E00 + 1)));
            }
            arr.add(sb.toString());
        }
        body.put("prompt_batch", arr);
        return JSON.toJSONBytes(body);
    }
}
