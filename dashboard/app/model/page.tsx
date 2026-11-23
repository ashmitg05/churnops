"use client";

import { CheckCircle, XCircle } from "@phosphor-icons/react";
import { CalibrationChart, CostChart } from "@/components/charts";
import { ErrorNote, PageHeader, Panel, Reveal, Stat } from "@/components/ui";
import { num, pct, useApi, type ModelInfo, type Reports } from "@/lib/api";

export default function ModelPage() {
  const info = useApi<ModelInfo>("/model-info");
  const rep = useApi<Reports>("/reports");
  const err = info.error ?? rep.error;
  const m = rep.data?.metrics;
  const c = rep.data?.curves;
  const p = rep.data?.promotion;
  const runs = [...(rep.data?.train?.comparison ?? [])].sort((a, b) => Number(b.val_roc_auc) - Number(a.val_roc_auc));

  return (
    <>
      <PageHeader
        title="Production model"
        sub="Held-out test metrics, the business-cost threshold and every candidate the pipeline compared. All numbers come from logged MLflow runs."
      />
      {err && <ErrorNote message={err} onRetry={() => (info.reload(), rep.reload())} />}

      {!err && (
        <div className="flex flex-col gap-6">
          <Reveal>
            <Panel className="grid grid-cols-2 gap-6 md:grid-cols-3 lg:grid-cols-6">
              {m ? (
                <>
                  <Stat label="ROC-AUC" value={num(m.test_roc_auc)} hint="test set" />
                  <Stat label="PR-AUC" value={num(m.test_pr_auc)} />
                  <Stat label="Recall" value={pct(m.test_recall)} hint={`at t = ${m.test_threshold}`} />
                  <Stat label="Precision" value={pct(m.test_precision)} />
                  <Stat label="Brier" value={num(m.test_brier)} hint="lower is better" />
                  <Stat label="Cost / customer" value={m.test_cost_per_customer.toFixed(2)} hint="expected" />
                </>
              ) : (
                Array.from({ length: 6 }).map((_, i) => <div key={i} className="skeleton h-14 rounded-lg" />)
              )}
            </Panel>
          </Reveal>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Reveal delay={0.06}>
              <Panel className="h-full">
                <h2 className="font-medium">Expected cost by threshold</h2>
                <p className="mt-1 mb-4 max-w-[60ch] text-sm text-muted">
                  {c
                    ? `A missed churner costs ${c.cost_params.churn_loss}, an offer costs ${c.cost_params.offer_cost}. The cheapest threshold sits far below 0.5.`
                    : "Loading"}
                </p>
                {c ? <CostChart data={c.cost} threshold={c.threshold} /> : <div className="skeleton h-[260px] rounded-xl" />}
              </Panel>
            </Reveal>
            <Reveal delay={0.1}>
              <Panel className="h-full">
                <h2 className="font-medium">Calibration</h2>
                <p className="mt-1 mb-4 text-sm text-muted">Observed churn vs predicted, 10 quantile bins. Dashed line is perfect.</p>
                {c ? <CalibrationChart data={c.calibration} /> : <div className="skeleton h-[260px] rounded-xl" />}
              </Panel>
            </Reveal>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
            <Reveal delay={0.08}>
              <Panel className="flex h-full flex-col gap-5">
                <h2 className="font-medium">Last promotion decision</h2>
                {p ? (
                  <>
                    <div className="flex items-start gap-3">
                      {p.promoted ? (
                        <CheckCircle size={22} weight="fill" className="mt-0.5 shrink-0 text-accent" />
                      ) : (
                        <XCircle size={22} weight="fill" className="mt-0.5 shrink-0 text-faint" />
                      )}
                      <div>
                        <p className="font-medium">{p.promoted ? "Promoted" : "Rejected"}</p>
                        <p className="mt-1 text-sm text-muted">{p.reason}</p>
                      </div>
                    </div>
                    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 font-mono text-[13px]">
                      <dt className="text-muted">Candidate AUC</dt>
                      <dd className="text-right tabular-nums">{num(p.candidate.roc_auc)}</dd>
                      <dt className="text-muted">Production AUC</dt>
                      <dd className="text-right tabular-nums">{p.production ? num(p.production.roc_auc) : "none"}</dd>
                      <dt className="text-muted">p95 latency</dt>
                      <dd className="text-right tabular-nums">{p.candidate.latency_p95_ms.toFixed(1)} ms</dd>
                      <dt className="text-muted">Gender AUC gap</dt>
                      <dd className="text-right tabular-nums">{num(p.candidate.group_auc_gap)}</dd>
                    </dl>
                    <p className="mt-auto font-mono text-xs text-faint">{new Date(p.decided_at).toLocaleString()}</p>
                  </>
                ) : (
                  <div className="skeleton h-40 rounded-xl" />
                )}
                {info.data && (
                  <div className="border-t border-line pt-4 font-mono text-xs text-muted">
                    {info.data.family} / {info.data.imbalance}, registry v{info.data.registry_version}, run{" "}
                    {info.data.run_id.slice(0, 8)}
                  </div>
                )}
              </Panel>
            </Reveal>

            <Reveal delay={0.12}>
              <Panel className="h-full overflow-hidden !p-0">
                <h2 className="px-5 pt-5 font-medium md:px-6 md:pt-6">Candidates compared</h2>
                <p className="mt-1 px-5 text-sm text-muted md:px-6">Validation set, each at its own cost-optimal threshold.</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[560px] text-left text-[13px]">
                    <thead className="text-muted">
                      <tr className="border-b border-line">
                        {["Model", "Imbalance", "ROC-AUC", "PR-AUC", "F1", "Brier", "Run"].map((h) => (
                          <th key={h} className="px-5 py-2.5 font-normal first:pl-5 md:first:pl-6">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="font-mono tabular-nums">
                      {runs.map((r, i) => (
                        <tr key={r.run_id} className={i === 0 ? "bg-accent/[0.07]" : ""}>
                          <td className="px-5 py-2.5 font-sans md:pl-6">{r.family.replace("_", " ")}</td>
                          <td className="px-5 py-2.5 font-sans text-muted">{r.imbalance.replace("_", " ")}</td>
                          <td className={`px-5 py-2.5 ${i === 0 ? "text-accent" : ""}`}>{num(Number(r.val_roc_auc))}</td>
                          <td className="px-5 py-2.5">{num(Number(r.val_pr_auc))}</td>
                          <td className="px-5 py-2.5">{num(Number(r.val_f1))}</td>
                          <td className="px-5 py-2.5">{num(Number(r.val_brier))}</td>
                          <td className="px-5 py-2.5 text-faint">{r.run_id.slice(0, 8)}</td>
                        </tr>
                      ))}
                      {!runs.length &&
                        Array.from({ length: 6 }).map((_, i) => (
                          <tr key={i}>
                            <td colSpan={7} className="px-5 py-2">
                              <div className="skeleton h-4 rounded" />
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </Reveal>
          </div>
        </div>
      )}
    </>
  );
}
