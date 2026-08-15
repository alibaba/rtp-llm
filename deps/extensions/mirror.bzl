"""The single download channel for our own OSS bucket: the bucket prefix appears once, in this file.

Archive declarations only write the "in-bucket path" or the "upstream coordinate".
The upstream URL is same-byte redundancy only: sha256 is pinned at each declaration site, so
switching URLs cannot substitute different bytes — it only covers runtime network flakiness.
A missing bucket object is caught before commit by `scripts/rtpcli deps verify`, not by
silently falling back to upstream.
"""

# Public anonymously-readable open-source bucket. Changing bucket/endpoint touches only this line.
_BUCKET = "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com"

def rtp_mirror_urls(path, upstream = None):
    """In-bucket relative path -> urls list (our bucket first, known upstream as same-byte backstop).

    When upstream is None only the bucket URL is returned — for some archives
    (boost/jsoncpp/curl/zlib and intranet rpms) the original source is no longer reliable or
    unsuitable as a backstop; in that case we prefer going red on a missing artifact over
    guessing an upstream.
    """
    if path.startswith("/") or path.startswith("http"):
        fail("rtp_mirror_urls expects an in-bucket relative path, not a full URL: %r" % path)
    urls = [_BUCKET + "/" + path]
    if upstream:
        urls.append(upstream)
    return urls

def rtp_github_archive_urls(owner, repo, ref):
    """GitHub ref archive: the upstream coordinate (owner/repo/commit or tag) is both the in-bucket path and the backstop URL.

    The in-bucket object is byte-identical to codeload.github.com/<owner>/<repo>/tar.gz/<ref>
    (sha256 pinned at each declaration site), so codeload is used directly as the second
    candidate — the coordinate is both on record and usable.
    Release artifacts (release assets, e.g. rules_pkg-0.6.0.tar.gz) do not go through here:
    they are not ref archives, and shoehorning them in would make the coordinate look like a
    commit when it is not; use rtp_mirror_urls with an explicit path.
    """
    return rtp_mirror_urls(
        "archives/github.com/%s/%s/%s.tar.gz" % (owner, repo, ref),
        upstream = "https://codeload.github.com/%s/%s/tar.gz/%s" % (owner, repo, ref),
    )
