export const NASA_CYCLE_INDEX_BOUNDS = Object.freeze({ minimum: 2, maximum: 614 });
export const NASA_RESEARCH_ENDPOINT = "/api/predict/nasa-capacity-retention";
export const NASA_RESEARCH_TARGET = "first_positive_measurement_capacity_retention_percent";

export async function requestNasaCapacityRetention({ apiBase, cycleIndex, fetchImpl = fetch }) {
  const value = Number(cycleIndex);
  if (!Number.isInteger(value) || value < NASA_CYCLE_INDEX_BOUNDS.minimum || value > NASA_CYCLE_INDEX_BOUNDS.maximum) {
    throw new Error(`Enter an integer cycle index from ${NASA_CYCLE_INDEX_BOUNDS.minimum} to ${NASA_CYCLE_INDEX_BOUNDS.maximum}.`);
  }
  const base = String(apiBase || "").replace(/\/+$/, "");
  const response = await fetchImpl(`${base}${NASA_RESEARCH_ENDPOINT}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cycle_index_in_file: value }),
  });
  let data;
  let httpOk = true;
  if (response && typeof response.json === "function") {
    try {
      data = await response.json();
      httpOk = response.ok;
    } catch {
      throw new Error("The NASA research endpoint returned an unreadable response.");
    }
  } else if (response && typeof response === "object" && !Array.isArray(response)) {
    // The shared fetchJson client already parsed the HTTP response and throws
    // clear errors for non-2xx responses before returning this payload.
    data = response;
  } else {
    throw new Error("The NASA research endpoint returned an unreadable response.");
  }
  if (!httpOk) {
    const detail = data?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || "NASA capacity-retention inference is unavailable.");
  }
  if (data?.status !== "prediction" || data?.target !== NASA_RESEARCH_TARGET || data?.provisional_proxy !== true || !Number.isFinite(data?.value)) {
    throw new Error("The NASA research endpoint returned an incompatible prediction response.");
  }
  return data;
}
