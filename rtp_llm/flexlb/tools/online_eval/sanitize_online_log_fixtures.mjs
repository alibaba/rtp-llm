#!/usr/bin/env node

import {randomBytes, randomInt} from "node:crypto";
import {mkdirSync, readFileSync, writeFileSync} from "node:fs";
import {resolve} from "node:path";

const sourceDir = resolve(process.argv[2] ?? "tools/online_eval/data/online_logs");
const outputDir = resolve(process.argv[3] ?? sourceDir);

const access = JSON.parse(readFileSync(resolve(sourceDir, "sample_access.json"), "utf8"));
const trace = readJsonLines(resolve(sourceDir, "trace_30min.jsonl"));
const arrivals = readFileSync(resolve(sourceDir, "pod1_arrivals.tsv"), "utf8");

const sanitizedAccess = sanitizeAccess(access);
const sanitizedTrace = sanitizeTrace(trace);
const sanitizedArrivals = sanitizeArrivals(arrivals);

mkdirSync(outputDir, {recursive: true});
writeFileSync(resolve(outputDir, "sample_access.json"),
        `${JSON.stringify(sanitizedAccess)}\n`, "utf8");
writeFileSync(resolve(outputDir, "trace_30min.jsonl"),
        `${sanitizedTrace.map((record) => JSON.stringify(record)).join("\n")}\n`, "utf8");
writeFileSync(resolve(outputDir, "pod1_arrivals.tsv"), sanitizedArrivals, "utf8");

console.log(JSON.stringify({
    accessTokenCount: sanitizedAccess.input_ids.length,
    traceRecordCount: sanitizedTrace.length,
    arrivalRecordCount: sanitizedArrivals.split("\n")
            .filter((line) => line && !line.startsWith("#")).length,
}));

function sanitizeAccess(raw) {
    if (!Array.isArray(raw.input_ids) || raw.input_ids.length === 0) {
        throw new Error("sample_access.json has no input_ids");
    }

    const sourceTokens = raw.input_ids.map(Number);
    const vocabulary = [...new Set(sourceTokens)];
    shuffle(vocabulary);
    const pseudonymBase = Math.max(1_000_000, Math.max(...sourceTokens) + 100_000);
    const tokenMap = new Map(vocabulary.map((token, index) =>
        [token, pseudonymBase + index]));
    const inputIds = sourceTokens.map((token) => tokenMap.get(token));
    shuffle(inputIds);

    const config = raw.generate_config ?? {};
    return {
        schema_version: 1,
        fixture_type: "sanitized_access_shape",
        sanitized: true,
        request_controls: {
            ds_header_attributes: {model: "mock-model"},
        },
        output_token_len: positiveInt(raw.output_token_len, 1),
        generate_config: {
            max_new_tokens: positiveInt(config.max_new_tokens, 1),
            num_return_sequences: positiveInt(config.num_return_sequences, 1),
            top_p: finiteNumber(config.top_p, 1.0),
            top_k: nonNegativeInt(config.top_k, 0),
            temperature: finiteNumber(config.temperature, 1.0),
            min_new_tokens: nonNegativeInt(config.min_new_tokens, 0),
            repetition_penalty: finiteNumber(config.repetition_penalty, 1.0),
            frequency_penalty: finiteNumber(config.frequency_penalty, 0.0),
            presence_penalty: finiteNumber(config.presence_penalty, 0.0),
            max_new_think_tokens: nonNegativeInt(config.max_new_think_tokens, 0),
            response_format: "",
            enable_thinking: Boolean(config.enable_thinking),
            timeout_ms: positiveInt(config.timeout_ms, 120_000),
        },
        input_ids: inputIds,
    };
}

function sanitizeTrace(records) {
    if (records.length === 0) {
        throw new Error("trace_30min.jsonl is empty");
    }
    const firstTimestamp = Math.min(...records.map((record) => Number(record.ts) || 0));
    const blockHashMap = new Map();
    const usedBlockHashes = new Set();
    const prefillMap = new Map();
    const decodeMap = new Map();

    return records.map((record) => ({
        ts: Math.max(0, (Number(record.ts) || firstTimestamp) - firstTimestamp),
        il: nonNegativeInt(record.il, 0),
        ol: nonNegativeInt(record.ol, 0),
        cached: nonNegativeInt(record.cached, 0),
        ttfb: nonNegativeNumber(record.ttfb, 0.0),
        total: nonNegativeNumber(record.total, 0.0),
        pep: endpointPseudonym(prefillMap, record.pep, "prefill"),
        dep: endpointPseudonym(decodeMap, record.dep, "decode"),
        bh: Array.isArray(record.bh)
                ? record.bh.map((value) => blockPseudonym(
                    blockHashMap, usedBlockHashes, String(value)))
                : [],
        shard: nonNegativeInt(record.shard, 0),
    }));
}

function sanitizeArrivals(content) {
    const rows = content.split("\n")
            .map((line) => line.trim())
            .filter(Boolean)
            .map((line) => line.split("\t"))
            .filter((columns) => columns.length === 2
                    && Number.isFinite(Number(columns[0]))
                    && Number.isFinite(Number(columns[1])));
    if (rows.length === 0) {
        throw new Error("pod1_arrivals.tsv has no numeric records");
    }
    const firstTimestamp = Math.min(...rows.map((columns) => Number(columns[0])));
    const lines = [
        "# sanitized arrival analysis fixture",
        "# columns: relative_timestamp_ms\\tinput_tokens",
        ...rows.map((columns) =>
            `${Number(columns[0]) - firstTimestamp}\t${nonNegativeInt(columns[1], 0)}`),
    ];
    return `${lines.join("\n")}\n`;
}

function endpointPseudonym(mapping, value, prefix) {
    const key = String(value ?? "");
    if (!mapping.has(key)) {
        mapping.set(key, `${prefix}-${mapping.size + 1}`);
    }
    return mapping.get(key);
}

function blockPseudonym(mapping, used, value) {
    if (mapping.has(value)) {
        return mapping.get(value);
    }
    let pseudonym;
    do {
        pseudonym = (randomBytes(8).readBigUInt64BE() & 0x7fff_ffff_ffff_ffffn).toString();
    } while (used.has(pseudonym));
    mapping.set(value, pseudonym);
    used.add(pseudonym);
    return pseudonym;
}

function shuffle(values) {
    for (let index = values.length - 1; index > 0; index--) {
        const other = randomInt(index + 1);
        [values[index], values[other]] = [values[other], values[index]];
    }
}

function readJsonLines(path) {
    return readFileSync(path, "utf8").split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
}

function positiveInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function nonNegativeInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function finiteNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function nonNegativeNumber(value, fallback) {
    return Math.max(0, finiteNumber(value, fallback));
}
