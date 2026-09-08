#!/usr/bin/env bash
# Run inside the repository's development container as a non-root user.
# A native test Worker must already be built following the test-execution skill.
set -euo pipefail

if [[ ! -f /.dockerenv || $(id -u) == 0 ]]; then
    echo 'Run inside the development container as the repository owner, not root.' >&2
    exit 2
fi
: "${CONSTRAINT_TREE_CPP_WORKER_BINARY:?Set the absolute path to the compiled constraint_tree_test_server}"
if [[ ! -x "$CONSTRAINT_TREE_CPP_WORKER_BINARY" ]]; then
    echo 'C++ test Worker missing; refusing to silently skip the HTTP E2E.' >&2
    exit 2
fi

flexlb_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "$flexlb_dir/../.." && pwd)
maven_cmd=${MAVEN_CMD:-mvn}
maven_args=(-B)
if [[ -n ${MAVEN_SETTINGS:-} ]]; then
    maven_args+=(-s "$MAVEN_SETTINGS")
fi
export CONSTRAINT_TREE_RUN_IGRAPH_SCALE_TEST=1
export CONSTRAINT_TREE_RUN_SCALE_TEST=1

cd "$flexlb_dir"
"$maven_cmd" "${maven_args[@]}" -pl flexlb-common -am -DskipTests install
cd "$repo_dir/internal_source/java"
"$maven_cmd" "${maven_args[@]}" -pl igraph -am \
    -Dtest=IgraphSidBucketClientTest -Dsurefire.failIfNoSpecifiedTests=false install
cd "$flexlb_dir"
"$maven_cmd" "${maven_args[@]}" -pl flexlb-api -am \
    -Dtest=BucketSidReaderTest,IgraphConstraintTreePollerTest,IgraphConstraintTreeServerTest,ConstraintTreeMappedE2ETest,ConstraintTreeCrossLanguageE2ETest,ConstraintTreeScaleTest,ConstraintTreeBuilderTest,ConstraintTreeBuildServiceTest,ConstraintTreeSidMappingTest,WhaleConstraintTreePublisherTest,ConstraintTreeServerTest \
    -Dsurefire.failIfNoSpecifiedTests=false -DargLine=-Xmx4g test
