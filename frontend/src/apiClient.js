function detailMessage(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") return value.message || value.code || "The service rejected the request.";
  return "The service is unavailable.";
}

export async function fetchJson(url, options = {}, timeoutMs = 20000, fetchImpl = fetch) {
  const controller = new AbortController();
  let didTimeout = false;
  const timer = setTimeout(() => { didTimeout = true; controller.abort(); }, timeoutMs);
  const callerSignal = options.signal;
  const abortFromCaller = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener("abort", abortFromCaller, { once: true });
  }
  try {
    const response = await fetchImpl(url, { ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); }
    catch { throw new Error("The service returned an unreadable response."); }
    if (!response.ok) throw new Error(detailMessage(data?.detail) || `Request failed (${response.status}).`);
    return data;
  } catch (error) {
    if (didTimeout) throw new Error("The request timed out. Please try again.");
    if (error?.name === "AbortError") throw new Error("The request was cancelled.");
    throw error;
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener("abort", abortFromCaller);
  }
}
