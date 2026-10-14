import test from "node:test";
import assert from "node:assert/strict";
import { requestNasaCapacityRetention, NASA_RESEARCH_ENDPOINT } from "../src/nasaResearch.js";
import { fetchJson } from "../src/apiClient.js";

const validResponse = {
  status: "prediction",
  config_id: "nasa_fy08q4:nasa_first_positive:xgboost",
  target: "first_positive_measurement_capacity_retention_percent",
  value: 97.25,
  unit: "% capacity retention",
  provisional_proxy: true,
};

test("sends only the registered NASA cycle feature to the dedicated endpoint", async () => {
  let captured;
  const result = await requestNasaCapacityRetention({
    apiBase: "http://localhost:8000/",
    cycleIndex: "180",
    fetchImpl: async (url, options) => {
      captured = { url, options };
      return { ok: true, json: async () => validResponse };
    },
  });
  assert.equal(captured.url, `http://localhost:8000${NASA_RESEARCH_ENDPOINT}`);
  assert.deepEqual(JSON.parse(captured.options.body), { cycle_index_in_file: 180 });
  assert.equal(result.value, 97.25);
});

test("accepts the already-decoded payload returned by the shared API client", async () => {
  let captured;
  const result = await requestNasaCapacityRetention({
    apiBase: "http://127.0.0.1:8000/",
    cycleIndex: 180,
    fetchImpl: (url, options) => fetchJson(url, options, 12000, async (requestUrl, requestOptions) => {
      captured = { url: requestUrl, options: requestOptions };
      return { ok: true, headers: { get: () => "application/json" }, json: async () => validResponse };
    }),
  });

  assert.equal(captured.url, `http://127.0.0.1:8000${NASA_RESEARCH_ENDPOINT}`);
  assert.equal(captured.options.method, "POST");
  assert.equal(captured.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(captured.options.body), { cycle_index_in_file: 180 });
  assert.equal(result.status, "prediction");
  assert.equal(result.target, "first_positive_measurement_capacity_retention_percent");
  assert.equal(result.provisional_proxy, true);
});

test("preserves the API client's clear error for a failed NASA response", async () => {
  await assert.rejects(requestNasaCapacityRetention({
    apiBase: "http://127.0.0.1:8000",
    cycleIndex: 180,
    fetchImpl: (url, options) => fetchJson(url, options, 12000, async () => ({
      ok: false,
      json: async () => ({ detail: { code: "nasa_model_artifact_unavailable" } }),
    })),
  }), /nasa_model_artifact_unavailable/);
});

test("rejects invalid cycle indices before making a request", async () => {
  let calls = 0;
  for (const cycleIndex of ["", 1, 615, 10.5, "NaN"]) {
    await assert.rejects(requestNasaCapacityRetention({
      apiBase: "http://localhost:8000",
      cycleIndex,
      fetchImpl: async () => { calls += 1; },
    }), /integer cycle index/);
  }
  assert.equal(calls, 0);
});

test("shows API errors and refuses mismatched or non-proxy response metadata", async () => {
  await assert.rejects(requestNasaCapacityRetention({
    apiBase: "http://localhost:8000", cycleIndex: 180,
    fetchImpl: async () => ({ ok: false, json: async () => ({ detail: "artifact unavailable" }) }),
  }), /artifact unavailable/);
  await assert.rejects(requestNasaCapacityRetention({
    apiBase: "http://localhost:8000", cycleIndex: 180,
    fetchImpl: async () => ({ ok: true, json: async () => ({ ...validResponse, target: "soh_percent" }) }),
  }), /incompatible prediction response/);
});
