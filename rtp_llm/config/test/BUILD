
py_test(
    name = "server_config_setup_test",
    srcs = [
        "server_config_setup_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

# Fail-closed release-profile validation: every rejection rule, the fail-closed posture for a knob
# that cannot be resolved, the sanitized manifest and its digest, and the startup cross-rank
# consistency comparison.
py_test(
    name = "release_profile_test",
    srcs = [
        "release_profile_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
        # The canonical runnable settings the profile test checks for drift.
        "//rtp_llm/config/release_profiles:sm120_dp4ep4_n2",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

