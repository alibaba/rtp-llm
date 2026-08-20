#!/bin/bash
# The committed surface must not contain plaintext long-lived credentials.
# Only inspects git-tracked content -- untracked local .env / skill files are the legitimate
# place for dev-time plaintext; once git add'ed, this check catches them. Any hit is FAIL,
# no exception allowlist: anywhere needing credentials goes through CI secrets or environment variables.
set -u
cd "$(dirname "$0")"
sub="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "FAIL: not inside a git repository"; exit 1; }
# The overlay is two repos: the open-source submodule + the internal superproject. The gate
# runs from inside the submodule; scanning only the submodule would miss internal_source/
# (skills, CI scripts, Dockerfiles all live there -- exactly where credentials like to sit).
# Use git's own submodule-ownership judgement: empty on a standalone clone, so an unrelated
# repo that happens to contain us is never grabbed by mistake.
super="$(git -C "$sub" rev-parse --show-superproject-working-tree 2>/dev/null || true)"

# AccessKeyId, plus the argument shapes ossutil/aliyun CLI use to pass secrets
patterns='LTAI[0-9A-Za-z]{12,}|(-k|--access-key-secret)[[:space:]]+[0-9A-Za-z]{25,}'
# Presigned URLs (`OSSAccessKeyId=…&…Signature=…`) are a different class: the signature is
# valid only for a single object + expiry time, the secret is not in the URL, and holding
# one grants no other privilege. It is not the "long-lived credential" this gate blocks;
# count it separately -- blocking it would only force people to replace external teams'
# RPM direct links with secrets they do not have, buying no security.
presigned_re='OSSAccessKeyId=[^&]+&.*Signature='
hits=""
notes=0
for root in "$sub" ${super:+"$super"}; do
  found="$(git -C "$root" grep -nIE "$patterns" -- . ':(exclude)*/check_no_secrets.sh' ':(exclude)check_no_secrets.sh' 2>/dev/null || true)"
  [ -z "$found" ] && continue
  notes=$((notes + $(printf '%s\n' "$found" | grep -cE "$presigned_re" || true)))
  keys="$(printf '%s\n' "$found" | grep -vE "$presigned_re" || true)"
  [ -n "${keys//[$'\n' ]/}" ] && hits="$hits$(printf '%s' "$keys" | sed "s#^#${root}/#")"$'\n'
done
if [ -n "${hits//[$'\n' ]/}" ]; then
  echo "FAIL: plaintext credentials in tracked files:"
  printf '%s' "$hits" | sed '/^$/d;s/^/  /'
  echo "  Fix: replace with \${OSS_ACCESS_KEY_ID} / \${{secrets.*}}, and rotate the leaked key"
  exit 1
fi
echo "OK: no plaintext AccessKey in tracked files (scanned $([ -n "$super" ] && echo 2 || echo 1) repo(s), plus $notes presigned URLs which are single-object temporary grants)"
