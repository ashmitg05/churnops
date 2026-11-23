"use client";

import { useCallback, useEffect, useState } from "react";

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Contribution = { feature: string; value: string | number; contribution: number; text: string };
export type Prediction = {
  churn_probability: number;
  churn_label: boolean;
  risk_level: "low" | "medium" | "high";
  threshold: number;
  model_version: string;
  explanation: string;
  top_contributions: Contribution[];
};
export type ModelInfo = {
  family: string;
  imbalance: string;
  params: Record<string, number>;
  run_id: string;
  registry_version: string;
  trained_at: string;
  threshold: number;
};
type Run = Record<string, number | string> & { family: string; imbalance: string; run_id: string };
export type Reports = {
  metrics: Record<string, number> | null;
  train: { best: { family: string; imbalance: string; run_id: string }; comparison: Run[] } | null;
  promotion: {
    promoted: boolean;
    reason: string;
    decided_at: string;
    candidate: { roc_auc: number; latency_p95_ms: number; group_auc_gap: number; positive_rate: number; failed: string[] };
    production: { roc_auc: number } | null;
  } | null;
  drift: {
    share_drifted: number;
    n_drifted: number;
    drifted_columns: string[];
    n_current: number;
    mean_predicted_churn: number;
    drift_detected: boolean;
    checked_at: string;
  } | null;
  curves: {
    calibration: { predicted: number; observed: number }[];
    cost: { threshold: number; cost: number }[];
    threshold: number;
    cost_params: { churn_loss: number; offer_cost: number };
  } | null;
};
export type Stats = {
  total: number;
  mean_probability?: number;
  series: { ts: string; volume: number; churn_rate: number }[];
};
/** Input schema read from the API's OpenAPI doc, so the form never drifts from Pydantic/Pandera. */
export type FieldSpec = { name: string; options?: (string | number)[]; min?: number; max?: number; nullable: boolean };

export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json" } });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status}: ${body}`);
  }
  return (await r.json()) as T;
}

export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    setError(null);
    getJSON<T>(path).then(setData, (e: Error) => setError(e.message));
  }, [path]);
  useEffect(load, [load]);
  return { data, error, reload: load };
}

type OpenApiProp = {
  enum?: (string | number)[];
  anyOf?: { type?: string; minimum?: number; maximum?: number }[];
  minimum?: number;
  maximum?: number;
};

export async function getSchema(): Promise<FieldSpec[]> {
  const doc = await getJSON<{ components: { schemas: { Customer: { properties: Record<string, OpenApiProp> } } } }>(
    "/openapi.json",
  );
  return Object.entries(doc.components.schemas.Customer.properties)
    .filter(([name]) => name !== "customerID")
    .map(([name, p]) => {
      const num = p.anyOf?.find((a) => a.type === "number") ?? p;
      return {
        name,
        options: p.enum,
        min: num.minimum,
        max: num.maximum,
        nullable: Boolean(p.anyOf?.some((a) => a.type === "null")),
      };
    });
}

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const num = (x: number, d = 4) => x.toFixed(d);
