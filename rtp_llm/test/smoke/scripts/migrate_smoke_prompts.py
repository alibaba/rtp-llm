#!/usr/bin/env python3
"""Replace inline prompts in golden JSONs with $prompt:xxx references.

Usage:
    python3 migrate_smoke_prompts.py --dry-run          # preview changes
    python3 migrate_smoke_prompts.py                     # apply changes
    python3 migrate_smoke_prompts.py --clear-results     # also clear result fields
"""

import argparse
import glob
import json
import os
import re
import sys

SMOKE_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "model")

# Mapping: (prompt_pattern, prompt_type) -> prompt_id
# prompt_type: "single" for query.prompt, "batch_item" for items in query.prompt_batch,
#              "chat_content" for messages[].content
PROMPT_MAPPINGS = [
    # === Tiny (≤50 chars) ===
    # EN single: hello, hello gg, etc.
    (lambda t, l: l == "en" and len(t) <= 30 and t.strip().lower().startswith("hello"), "s1"),
    (lambda t, l: l == "en" and len(t) <= 30 and "what is" in t.lower(), "s1"),
    # ZH single: 你是谁, 请介绍下自己
    (lambda t, l: l == "zh" and len(t) <= 30, "s2"),
    # EN ~50: who are you, how old are you, what is your name
    (lambda t, l: l == "en" and 30 < len(t) <= 65 and not t.startswith("<|"), "s3"),
    # ZH ~50
    (lambda t, l: l == "zh" and 30 < len(t) <= 65, "s4"),

    # === Short (51-100 chars) ===
    # EN ~80: Write a detailed analogy..., etc.
    (lambda t, l: l == "en" and 65 < len(t) <= 120 and not t.startswith("<|"), "s5"),
    # ZH ~80
    (lambda t, l: l == "zh" and 65 < len(t) <= 120, "s6"),

    # === Medium (200-1000 chars) ===
    (lambda t, l: l == "en" and 200 < len(t) <= 600, "m1"),
    (lambda t, l: l == "en" and 600 < len(t) <= 1200, "m2"),
    (lambda t, l: l == "zh" and 200 < len(t) <= 1200, "m2"),

    # === Long (1-3K chars) ===
    (lambda t, l: l == "en" and 1200 < len(t) <= 3500, "l1"),
    (lambda t, l: l == "zh" and 1200 < len(t) <= 3500, "l2"),

    # === Very long (3-8K chars) ===
    (lambda t, l: l == "en" and 3500 < len(t) <= 8000, "x1"),
    (lambda t, l: l == "zh" and 3500 < len(t) <= 8000, "x2"),

    # === Extra long (8K+ chars) ===
    (lambda t, l: l == "en" and len(t) > 8000, "x3"),
    (lambda t, l: l == "zh" and 8000 < len(t) <= 14000, "x4"),
    (lambda t, l: l == "en" and len(t) > 14000, "x5"),
]

# Special generate_config keys that indicate a prompt should NOT be replaced
SPECIAL_KEYS = {
    "stop_words_str", "calculate_loss", "return_logits", "return_hidden_states",
    "return_softmax_probs", "logits_index",
}


def detect_lang(text):
    if any("一" <= c <= "鿿" for c in text[:100]):
        return "zh"
    return "en"


def has_special_config(gc):
    for key in SPECIAL_KEYS:
        if gc.get(key):
            return True
    if gc.get("random_seed") is not None:
        return True
    if gc.get("num_beams", 1) > 1 or gc.get("num_return_sequences", 1) > 1:
        return True
    return False


def extract_text(prompt):
    """Extract the core user text from a prompt that may have chat template wrapping."""
    if not isinstance(prompt, str):
        return None
    # Strip chat template markers
    text = prompt
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>\w*\s*", "", text)
    text = re.sub(r"<\|begin_of_text\|>.*?<\|end_header_id\|>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>\s*", "", text)
    return text.strip()


def match_prompt(text, lang):
    for matcher, pid in PROMPT_MAPPINGS:
        if matcher(text, lang):
            return pid
    return None


def migrate_file(filepath, dry_run=False, clear_results=False):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "query_result" not in data:
        return 0

    changes = 0
    for qi, qr in enumerate(data["query_result"]):
        query = qr.get("query", {})
        gc = query.get("generate_config", {})

        if has_special_config(gc):
            continue

        # Check single prompt
        prompt = query.get("prompt")
        if isinstance(prompt, str) and not prompt.startswith("$prompt:"):
            core_text = extract_text(prompt)
            if core_text:
                lang = detect_lang(core_text)
                pid = match_prompt(core_text, lang)
                if pid:
                    query["prompt"] = f"$prompt:{pid}"
                    changes += 1
                    if clear_results and "result" in qr:
                        qr["result"] = {}

        # Check prompt as list (chat_list format)
        if isinstance(prompt, list):
            full_text = " ".join(str(p) for p in prompt)
            core_text = extract_text(full_text)
            if core_text and not full_text.startswith("$prompt:"):
                lang = detect_lang(core_text)
                pid = match_prompt(core_text, lang)
                if pid:
                    query["prompt"] = f"$prompt:{pid}"
                    changes += 1
                    if clear_results and "result" in qr:
                        qr["result"] = {}

        # Check prompt_batch
        prompt_batch = query.get("prompt_batch")
        if isinstance(prompt_batch, list):
            new_batch = []
            batch_changed = False
            for item in prompt_batch:
                if isinstance(item, str) and not item.startswith("$prompt:"):
                    core = extract_text(item)
                    if core:
                        lang = detect_lang(core)
                        pid = match_prompt(core, lang)
                        if pid:
                            new_batch.append(f"$prompt:{pid}")
                            batch_changed = True
                            continue
                new_batch.append(item)
            if batch_changed:
                query["prompt_batch"] = new_batch
                changes += 1
                if clear_results and "result" in qr:
                    qr["result"] = {}

        # Check messages (OpenAI chat format)
        messages = query.get("messages")
        if isinstance(messages, list):
            msg_changed = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str) and not content.startswith("$prompt:"):
                    lang = detect_lang(content)
                    pid = match_prompt(content, lang)
                    if pid:
                        msg["content"] = f"$prompt:{pid}"
                        msg_changed = True
            if msg_changed:
                changes += 1
                if clear_results and "result" in qr:
                    qr["result"] = {}

    if changes > 0 and not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, separators=(",", ": "), ensure_ascii=False)

    return changes


def main():
    parser = argparse.ArgumentParser(description="Migrate smoke prompts to $prompt:xxx references")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--clear-results", action="store_true", help="Clear result fields for changed queries")
    parser.add_argument("--data-dir", default=SMOKE_DATA, help="Path to smoke data/model directory")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.json")))
    total_changes = 0
    changed_files = 0

    for filepath in json_files:
        rel = os.path.relpath(filepath, args.data_dir)
        changes = migrate_file(filepath, dry_run=args.dry_run, clear_results=args.clear_results)
        if changes > 0:
            action = "would change" if args.dry_run else "changed"
            print(f"  {rel}: {action} {changes} queries")
            total_changes += changes
            changed_files += 1

    print(f"\nTotal: {total_changes} queries in {changed_files} files {'(dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
