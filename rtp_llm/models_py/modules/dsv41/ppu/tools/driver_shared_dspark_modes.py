"""Evidence contracts for the bounded shared DSpark native role.

No model loop, sampler, budget allocator or runtime mutation. The selected
configuration is fixed before the ordinary launcher creates either model.
"""

from rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_modes import SharedDSparkMode


def selected_mode(env):
    anchor = env.get("RTP_V41_TEST_DSPARK_ANCHOR", "1")
    if anchor not in ("0", "1"):
        raise ValueError("test anchor selection must be0 or1")
    return SharedDSparkMode(
        gamma=int(env.get("RTP_V41_TEST_DSPARK_GAMMA", "3")),
        confidence=env.get("RTP_V41_TEST_DSPARK_CONFIDENCE", "bounded"),
        verify=env.get("RTP_V41_TEST_DSPARK_VERIFY_MODE", "packed"),
        budget=int(env.get("RTP_V41_TEST_DSPARK_BUDGET", "1")),
        budget_mode=env.get("RTP_V41_TEST_DSPARK_BUDGET_MODE", "adaptive"),
        sample_from_anchor=anchor == "1",
    ).validate()


def validate_decode_commit_layout(verifies, commits):
    """Match the complete target verify payload and draft writes for each round."""
    verifies = sorted(verifies, key=lambda item: (item["rank"], item["call"]))
    commits = sorted(commits, key=lambda item: (item["rank"], item["call"]))
    assert len(verifies) == len(commits) and verifies, (verifies, commits)
    for verify, commit in zip(verifies, commits):
        layout = verify["row_layout"]
        assert commit["rank"] == verify["rank"], (verify, commit)
        assert commit["prefixes"] == layout["prefixes"] and commit["lengths"] == layout["lengths"], (verify, commit)
        assert commit["feature_rows"] == verify["aux_feature_rows"] == sum(layout["lengths"]), (verify, commit)
        assert commit["input_feature_sha256"] == verify["aux_feature_sha256"], (verify, commit)
        expected = {
            (layer, request): list(range(prefix, prefix + length))
            for layer in range(3)
            for request, (prefix, length) in enumerate(zip(layout["prefixes"], layout["lengths"]))
        }
        actual = {(item["layer"], item["request"]): item["positions"] for item in commit["native_writes"]}
        assert len(actual) == len(commit["native_writes"]) and actual == expected, (expected, actual)


def validate_mode_reports(role, mode, reports, drafts, accepts, *, expected_requests=1):
    profile = {"mode": "compact", "keep_tokens": 140, "window": 128,
               "max_rollback_tokens": mode.max_verified_rows, "encoder_replay": False}
    assert reports and drafts, "actual target/draft evidence required"
    live = [item for item in reports if not item["idle"]]
    assert live
    for item in reports:
        assert item["profile"] == profile and item["model_layers"] == 40, item
        assert item["yoco"]["executed_layers"] == list(range(40)), item
        assert item["geometry"] == {"hidden_size": 5120, "hc_mult": 4, "routed_experts": 384, "top_k": 6}, item
        assert item["output_finite"] and all(p["native_alias"] for p in item["native_pool_bindings"]), item
        assert item["cache_protocol"]["speculative_state"] == mode.protocol(), item
        assert item["native_config_cache_key_hash_seed"] == item["cache_protocol"]["cache_key_hash_seed"], item
        assert item["cache_protocol"]["source_layout"] == "per-tag-native-cp-sharded", item
        if item["idle"]:
            assert not item["native_pool_bindings"] and not item["native_writes"], item
            assert len(item["routed_active_rows"]) == 40 and not any(item["routed_active_rows"].values()), item
        else:
            norm = item["normalization"]
            assert norm["attention_norm_calls"] == norm["ffn_calls"] == 40, item
            assert norm["ffn_norm_calls"] + norm["mega_transitions"] == 40, item
    for item in drafts:
        assert item["dspark_mode"] == mode.protocol() and item["profile"] == profile, item
        assert item["geometry"] == {"hidden_size": 5120, "hc_mult": 4, "routed_experts": 128, "top_k": 3}, item
        assert item["native_aliases"] and item["vocab_framework_alias"] and all(item["target_vocab_aliases"].values()), item
        assert item["output_finite"] and all(n == 1 for n in item["resident_conversions"].values()), item
        if item["idle"]:
            assert not item["native_writes"] and not any(item["routed_active_rows"].values()), item
    commits = [x for x in drafts if not x["idle"] and x["event"] == "commit" and x["feature_rows"]]
    assert commits and all(x["feature_finite"] and x["executed_layers"] == [0, 1, 2] and x["native_writes"] for x in commits), commits
    if role == "prefill":
        compact = [x for x in live if x["prefill"] and x["yoco"]["input_lengths"] == [257]]
        assert {x["rank"] for x in compact} == set(range(8)), compact
        positions = set()
        for item in compact:
            yoco = item["yoco"]
            assert yoco["compacted"] and yoco["restored"] and yoco["logical_retained_rows"] == 140, item
            assert yoco["cp_size"] == 8 and yoco["cp_padding_rows"] == 4, item
            assert item["cache_placement"] and all(p["sharded"] for p in item["cache_placement"]), item
            assert any(x["rank"] == item["rank"] and x["input_feature_sha256"] == item["aux_feature_sha256"] for x in commits), item
            positions.update(yoco["local_positions"])
        assert positions == set(range(117, 257)), positions
        assert all(x["commit_only"] and x["event"] == "commit" for x in drafts), drafts
        return {"CP_ranks": list(range(8)), "logical_rows": 140, "padding_rows": 4,
                "aux_feature_handoff": "EXACT_BYTES", "actual_max_backtracking": mode.max_verified_rows,
                "draft_prompt_fingerprints": [{"rank": x["rank"], "state": x["native_state"]} for x in commits]}
    assert not any(x["prefill"] for x in reports), reports
    assert {x["rank"] for x in reports} == set(range(8)), reports
    verifies = sorted((x for x in live if x["target_verify"]), key=lambda x: (x["rank"], x["call"]))
    proposals = sorted((x for x in drafts if not x["idle"] and x["event"] == "proposal"),
                       key=lambda x: (x["rank"], x["call"]))
    assert proposals and verifies and all(x["verify_native_unchanged"] for x in verifies), verifies
    for item in proposals:
        assert item["executed_layers"] == [0, 1, 2] and item["proposal_native_unchanged"] and not item["native_writes"], item
        assert item["confidence_hidden"] == (mode.confidence != "off"), item
        assert all(n == mode.query_width for n in item["lengths"]), item
        assert len(item["normalization"]) == 3 and all(
            x["attention_norm_calls"] == x["ffn_norm_calls"] == x["ffn_calls"] == 1
            for x in item["normalization"]), item
    # Eligibility remains public: configured packed may legally use dense at
    # full budget. A packed branch smoke must nevertheless observe it once.
    if mode.verify == "packed" and (mode.budget_mode == "adaptive" or mode.budget < mode.gamma):
        assert any(x["packed_target_verify"] for x in verifies), "packed branch not exercised; do not count dense fallback as coverage"
    else:
        assert not any(x["packed_target_verify"] for x in verifies), verifies
    accepted = sorted((x for x in accepts if not x.get("idle", False)), key=lambda x: (x["rank"], x["call"]))
    assert len(accepted) == len(verifies), (accepted, verifies)
    validate_decode_commit_layout(verifies, commits)
    requests = {}
    assert len(proposals) == len(accepted)
    for event, verify, proposal in zip(accepted, verifies, proposals):
        assert event["rank"] == verify["rank"] == proposal["rank"]
        assert event["call"] == verify["call"]
        request_id, = event["request_ids"]
        chain = requests.setdefault(request_id, {
            "request_id": request_id, "rank": event["rank"], "accepted_end": 257,
            "accepted_input_token_ids": [],
        })
        assert event["rank"] == chain["rank"]
        widths = verify["row_layout"]["lengths"]
        assert event["prefixes"] == proposal["prefixes"] == [chain["accepted_end"]]
        assert event["verified_rows"] == widths, (event, verify)
        assert all(1 <= n <= mode.max_verified_rows for n in widths), widths
        if not verify["packed_target_verify"]:
            assert widths == [mode.max_verified_rows], verify
        assert event["status"] == "ACCEPTED" and event["native_aliases_preserved"], event
        assert event["accepted_feature_rows"] == sum(event["accepted_input_rows"]), event
        assert event["maximum_backtracking"] == mode.max_verified_rows, event
        chain["accepted_input_token_ids"].extend(event["accepted_input_token_ids"][0])
        chain["accepted_end"] = event["accepted_ends"][0]
    assert len(requests) == expected_requests and all(chain["accepted_end"] > 257 for chain in requests.values())
    first = next(iter(requests.values()))
    return {"target_verify_cycles": len(verifies), "draft_proposals": len(proposals),
            "actual_max_backtracking": mode.max_verified_rows, "accepted_end": first["accepted_end"],
            "accepted_input_token_ids": first["accepted_input_token_ids"], "D_prefill": False,
            "request_chains": list(requests.values()),
            "actual_verify_layouts": [x["row_layout"] for x in verifies],
            "packed_calls": sum(x["packed_target_verify"] for x in verifies),
            "draft_initial_native_fingerprint": proposals[0]["native_state"],
            "aux_feature_handoff": "EXACT_VERIFY_BYTES", "mode": mode.protocol()}
