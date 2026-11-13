if (
  window.parent !== window &&
  location.pathname.startsWith("/d/mock-experiment-embedded/") &&
  /^http:\/\/(127\.0\.0\.1|localhost):\d+\/?$/.test(document.referrer)
) {
  const parentOrigin = new URL(document.referrer).origin;
  window.addEventListener("message", (event) => {
    if (event.source !== window.parent || event.origin !== parentOrigin) return;
    if (
      event.data?.type !== "mock-view-query" ||
      typeof event.data.query !== "string"
    )
      return;

    const url = new URL(location.href);
    const query = new URLSearchParams(event.data.query);
    for (const key of new Set(query.keys())) {
      if (key !== "from" && key !== "to" && !key.startsWith("var-")) continue;
      url.searchParams.delete(key);
      for (const value of query.getAll(key))
        url.searchParams.append(key, value);
    }
    if (url.href === location.href) return;

    // Grafana's browser history listens for popstate and applies URL state.
    history.replaceState(history.state, "", url);
    window.dispatchEvent(
      new PopStateEvent("popstate", { state: history.state }),
    );
  });
}
