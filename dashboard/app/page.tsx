"use client";

import { ArrowClockwise, Lightning } from "@phosphor-icons/react";
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { Button, ErrorNote, PageHeader, Panel, Reveal } from "@/components/ui";
import { getJSON, getSchema, pct, type FieldSpec, type Prediction } from "@/lib/api";

type Row = Record<string, string | number | null>;

const LABELS: Record<string, string> = {
  gender: "Gender",
  SeniorCitizen: "Senior citizen",
  Partner: "Partner",
  Dependents: "Dependents",
  PhoneService: "Phone",
  MultipleLines: "Multiple lines",
  InternetService: "Internet",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Tech support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
  tenure: "Tenure (months)",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
};
const GROUPS: { title: string; fields: string[] }[] = [
  { title: "Customer", fields: ["gender", "SeniorCitizen", "Partner", "Dependents"] },
  {
    title: "Services",
    fields: [
      "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ],
  },
  { title: "Billing", fields: ["Contract", "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges"] },
];

// Two sample profiles for quick demos (hand-made, not real customers).
const SAMPLES: Record<string, Row> = {
  "New fiber customer": {
    gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", PhoneService: "Yes",
    MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No",
    DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "No",
    Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check",
    tenure: 3, MonthlyCharges: 89.1, TotalCharges: 267.3,
  },
  "Long-term two-year": {
    gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", PhoneService: "Yes",
    MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes",
    DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No",
    Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)",
    tenure: 61, MonthlyCharges: 64.35, TotalCharges: 3921.1,
  },
}; // prettier-ignore

export default function PredictPage() {
  const [schema, setSchema] = useState<FieldSpec[] | null>(null);
  const [row, setRow] = useState<Row>(SAMPLES["New fiber customer"]);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadSchema = () => {
    setError(null);
    getSchema().then(setSchema, (e: Error) => setError(e.message));
  };
  useEffect(loadSchema, []);
  const spec = useMemo(() => Object.fromEntries((schema ?? []).map((f) => [f.name, f])), [schema]);

  const set = (k: string, v: string | number | null) => {
    setRow((r) => {
      const next = { ...r, [k]: v };
      if (k === "tenure" && v === 0) next.TotalCharges = null; // API rule: no bill yet
      return next;
    });
  };

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await getJSON<Prediction>("/predict", { method: "POST", body: JSON.stringify(row) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Score a customer"
        sub="Churn probability at the business-cost threshold, with the five features that moved it most."
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <Reveal delay={0.05}>
          <Panel>
            <div className="mb-6 flex flex-wrap items-center gap-2">
              <span className="mr-1 text-sm text-muted">Samples</span>
              {Object.keys(SAMPLES).map((s) => (
                <button
                  key={s}
                  onClick={() => setRow(SAMPLES[s])}
                  className="rounded-full border border-line px-3 py-1 text-[13px] text-muted transition-colors hover:border-line-strong hover:text-fg"
                >
                  {s}
                </button>
              ))}
            </div>

            {!schema && !error && <FormSkeleton />}
            {schema && (
              <div className="flex flex-col gap-8">
                {GROUPS.map((g) => (
                  <fieldset key={g.title} className="flex flex-col gap-5">
                    <legend className="mb-4 text-sm font-medium text-fg">{g.title}</legend>
                    <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2">
                      {g.fields.map((f) =>
                        spec[f]?.options ? (
                          <Segmented
                            key={f}
                            label={LABELS[f]}
                            options={spec[f].options!}
                            value={row[f]}
                            onChange={(v) => set(f, v)}
                            wide={spec[f].options!.join("").length > 24}
                          />
                        ) : (
                          <NumberField
                            key={f}
                            label={LABELS[f]}
                            spec={spec[f]}
                            value={row[f]}
                            disabled={f === "TotalCharges" && row.tenure === 0}
                            onChange={(v) => set(f, v)}
                          />
                        ),
                      )}
                    </div>
                  </fieldset>
                ))}
                <div className="flex items-center gap-3 border-t border-line pt-6">
                  <Button onClick={submit} disabled={busy}>
                    <Lightning size={16} weight="fill" />
                    {busy ? "Scoring" : "Predict"}
                  </Button>
                  <span className="text-xs text-faint">Logged for drift monitoring.</span>
                </div>
              </div>
            )}
            {error && !schema && <ErrorNote message={error} onRetry={loadSchema} />}
          </Panel>
        </Reveal>

        <Reveal delay={0.12} className="lg:sticky lg:top-24 lg:self-start">
          <ResultPanel result={result} error={schema ? error : null} />
        </Reveal>
      </div>
    </>
  );
}

function Segmented({
  label,
  options,
  value,
  onChange,
  wide,
}: {
  label: string;
  options: (string | number)[];
  value: string | number | null;
  onChange: (v: string | number) => void;
  wide: boolean;
}) {
  const display = (o: string | number) => (typeof o === "number" ? (o ? "Yes" : "No") : o.replace(" (automatic)", " (auto)"));
  return (
    <div className={`flex flex-col gap-2 ${wide ? "sm:col-span-2" : ""}`}>
      <span className="text-[13px] text-muted">{label}</span>
      <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1 rounded-2xl border border-line bg-bg/40 p-1">
        {options.map((o) => {
          const on = o === value;
          return (
            <button
              key={String(o)}
              role="radio"
              aria-checked={on}
              onClick={() => onChange(o)}
              className={`flex-1 whitespace-nowrap rounded-full px-3 py-1.5 text-[13px] transition-colors ${
                on ? "bg-surface-2 text-fg shadow-[inset_0_0_0_1px_var(--line-strong)]" : "text-muted hover:text-fg"
              }`}
            >
              {display(o)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function NumberField({
  label,
  spec,
  value,
  disabled,
  onChange,
}: {
  label: string;
  spec?: FieldSpec;
  value: string | number | null;
  disabled: boolean;
  onChange: (v: number | null) => void;
}) {
  const id = `f-${label}`;
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-[13px] text-muted">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={spec?.min}
        max={spec?.max}
        step="any"
        disabled={disabled}
        value={value ?? ""}
        placeholder={disabled ? "No bill yet" : undefined}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className="h-10 rounded-full border border-line bg-bg/40 px-4 font-mono text-sm text-fg tabular-nums outline-none transition-colors placeholder:text-faint focus:border-accent disabled:opacity-60"
      />
      {spec?.max !== undefined && <span className="text-xs text-faint">Range {spec.min ?? 0} to {spec.max}</span>}
    </div>
  );
}

function ResultPanel({ result, error }: { result: Prediction | null; error: string | null }) {
  if (error) return <ErrorNote message={error} />;
  if (!result)
    return (
      <Panel className="flex min-h-[420px] flex-col items-center justify-center gap-3 text-center">
        <Gauge value={0} threshold={0} idle />
        <p className="max-w-[28ch] text-sm text-muted">Pick a sample or edit the form, then run Predict.</p>
      </Panel>
    );
  const maxAbs = Math.max(...result.top_contributions.map((c) => Math.abs(c.contribution)));
  return (
    <Panel className="flex flex-col gap-6">
      <div className="flex flex-col items-center gap-1">
        <Gauge value={result.churn_probability} threshold={result.threshold} />
        <span
          className={`-mt-2 rounded-full px-3 py-1 text-xs font-medium ${
            result.risk_level === "high"
              ? "bg-accent text-accent-fg"
              : result.risk_level === "medium"
                ? "border border-accent/60 text-accent"
                : "border border-line-strong text-muted"
          }`}
        >
          {result.risk_level} risk
        </span>
      </div>
      <p className="text-[15px] leading-relaxed text-fg">{result.explanation}</p>
      <div className="flex flex-col gap-3">
        <span className="text-[13px] text-muted">Feature contributions (log-odds)</span>
        {result.top_contributions.map((c, i) => (
          <ContributionBar key={c.feature} c={c} maxAbs={maxAbs} i={i} />
        ))}
      </div>
      <div className="flex items-center justify-between border-t border-line pt-4 font-mono text-xs text-faint">
        <span>threshold {result.threshold.toFixed(2)}</span>
        <span className="flex items-center gap-1.5">
          <ArrowClockwise size={12} /> model v{result.model_version}
        </span>
      </div>
    </Panel>
  );
}

function ContributionBar({ c, maxAbs, i }: { c: Prediction["top_contributions"][number]; maxAbs: number; i: number }) {
  const reduce = useReducedMotion();
  const w = (Math.abs(c.contribution) / maxAbs) * 50;
  const up = c.contribution > 0;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between gap-3 text-[13px]">
        <span className="truncate text-fg">{c.text.replace(/ \((raises|lowers) risk\)$/, "")}</span>
        <span className={`font-mono tabular-nums ${up ? "text-accent" : "text-muted"}`}>
          {up ? "+" : ""}
          {c.contribution.toFixed(2)}
        </span>
      </div>
      <div className="relative h-1.5">
        <div className="absolute left-1/2 top-[-3px] h-3 w-px bg-line-strong" />
        <motion.div
          className={`absolute top-0 h-full rounded-full ${up ? "left-1/2 bg-accent" : "right-1/2 bg-faint"}`}
          initial={reduce ? false : { width: 0 }}
          animate={{ width: `${w}%` }}
          transition={{ duration: 0.7, delay: 0.1 + i * 0.06, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  );
}

/** Semicircle gauge: arc = probability, tick = business threshold. */
function Gauge({ value, threshold, idle = false }: { value: number; threshold: number; idle?: boolean }) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(0);
  const label = useTransform(mv, (v) => pct(v));
  useEffect(() => {
    if (reduce) {
      mv.set(value);
      return;
    }
    const a = animate(mv, value, { duration: 0.9, ease: [0.16, 1, 0.3, 1] });
    return () => a.stop();
  }, [value, mv, reduce]);
  const R = 88;
  const C = Math.PI * R;
  const dash = useTransform(mv, (v) => `${v * C} ${C}`);
  const t = Math.PI * (1 - threshold);
  return (
    <svg viewBox="0 0 220 128" className="w-full max-w-[280px]" role="img" aria-label={idle ? "No prediction yet" : `Churn probability ${pct(value)}`}>
      <path d={`M ${110 - R} 112 A ${R} ${R} 0 0 1 ${110 + R} 112`} fill="none" stroke="var(--line-strong)" strokeWidth="10" strokeLinecap="round" />
      <motion.path
        d={`M ${110 - R} 112 A ${R} ${R} 0 0 1 ${110 + R} 112`}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="10"
        strokeLinecap="round"
        style={{ strokeDasharray: dash }}
      />
      {!idle && (
        <line
          x1={110 + (R - 14) * Math.cos(t)}
          y1={112 - (R - 14) * Math.sin(t)}
          x2={110 + (R + 10) * Math.cos(t)}
          y2={112 - (R + 10) * Math.sin(t)}
          stroke="var(--fg)"
          strokeWidth="2"
        />
      )}
      <motion.text x="110" y="100" textAnchor="middle" className="fill-fg font-mono text-[30px] tracking-tight">
        {idle ? "--" : label}
      </motion.text>
      <text x="110" y="122" textAnchor="middle" className="fill-[var(--faint)] text-[10px]">
        churn probability
      </text>
    </svg>
  );
}

function FormSkeleton() {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex flex-col gap-2">
          <div className="skeleton h-3 w-20 rounded" />
          <div className="skeleton h-10 rounded-full" />
        </div>
      ))}
    </div>
  );
}
