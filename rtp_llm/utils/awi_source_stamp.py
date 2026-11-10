"""Packaging stamp for the AWI evaluator build.source field.

BUILD_SOURCE records the paired branch state the current packaging round is
built from. The stamp is updated before each pipeline-18410 trigger; the
authoritative binding image<->SHA is the CI receipt (github_commit resolves
to the public head at trigger time) plus the evaluator's git HEAD check of
launch.source_dirs. Launch-env overrides (AWI_BUILD_INTERNAL_SHA /
AWI_BUILD_PUBLIC_SHA) take precedence and must equal the candidate's
declared source pair.
"""

BUILD_SOURCE = {
    "internal": "1652dde216335fb3b14076964eb44ba3fc9c83c0",
    "public": "382c239d8e7749e3f3a51d4eaeb28ad453e8fbc3",
    "branch": "feature/adapt-dsv41-1017-mi308x-cc",
    "baseline": {
        "internal": "03036dcdf1b70ab8145169d013e2475e4a71a35a",
        "public": "f5c5feffebd856ddaf44cacc462adb04dd13f615",
    },
}
