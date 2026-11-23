"use client";

import { ChartLineUp } from "@phosphor-icons/react";
import { RateChart, VolumeChart } from "@/components/charts";
import { ErrorNote, PageHeader, Panel, Reveal, Stat } from "@/components/ui";
import { pct, useApi, type Reports, type Stats } from "@/lib/api";

export default function MonitoringPage() {
  const rep = useApi<Reports>("/reports");
  const stats = useApi<Stats>("/monitoring/stats");
  const err = rep.error ?? stats.error;
  const d = rep.data?.drift;
  const s = stats.data;

  return (
    <>
      <PageHeader
        title="Monitoring"
        sub="Input drift against the training data, plus the volume and predicted churn rate of logged predictions."
      />
      {err && <ErrorNote message={err} onRetry={() => (rep.reload(), stats.reload())} />}

      {!err && (
        <div className="flex flex-col gap-6">
          <Reveal>
            <Panel className="grid gap-8 md:grid-cols-[minmax(0,1.2fr)_minmax(0,2fr)]">
              {rep.data && !d && (
                <p className="text-sm text-muted md:col-span-2">
                  No drift report yet. Run <code className="font-mono text-fg">python -m src.monitoring.drift</code> after some
                  predictions are logged.
                </p>
              )}
              {!rep.data && <div className="skeleton h-28 rounded-xl md:col-span-2" />}
              {d && (
                <>
                  <div className="flex flex-col gap-3">
                    <span
                      className={`w-fit rounded-full px-3 py-1 text-xs font-medium ${
                        d.drift_detected ? "bg-accent text-accent-fg" : "border border-line-strong text-muted"
                      }`}
                    >
                      {d.drift_detected ? "Drift detected" : "No drift"}
                    </span>
                    <p className="font-mono text-5xl tracking-tighter tabular-nums">{pct(d.share_drifted, 0)}</p>
                    <p className="text-sm text-muted">
                      of input columns drifted ({d.n_drifted} columns) over the last{" "}
                      {d.n_current} predictions.
                    </p>
                  </div>
                  <div className="flex flex-col justify-between gap-6">
                    <div>
                      <span className="text-[13px] text-muted">Drifted columns</span>
                      <div className="mt-2.5 flex flex-wrap gap-2">
                        {d.drifted_columns.length ? (
                          d.drifted_columns.map((c) => (
                            <span key={c} className="rounded-lg border border-accent/40 bg-accent/10 px-2.5 py-1 font-mono text-xs text-fg">
                              {c}
                            </span>
                          ))
                        ) : (
                          <span className="text-sm text-faint">none</span>
                        )}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-6 border-t border-line pt-5 sm:grid-cols-3">
                      <Stat label="Mean predicted churn" value={pct(d.mean_predicted_churn)} hint="drift window" />
                      <Stat label="Logged predictions" value={s ? s.total.toLocaleString() : "..."} />
                      <Stat label="Checked" value={new Date(d.checked_at).toLocaleDateString()} />
                    </div>
                  </div>
                </>
              )}
            </Panel>
          </Reveal>

          {s && s.series.length === 0 ? (
            <Panel className="flex flex-col items-center gap-3 py-14 text-center">
              <ChartLineUp size={28} weight="duotone" className="text-faint" />
              <p className="max-w-[44ch] text-sm text-muted">
                No predictions logged yet. Score a customer, or send a batch with{" "}
                <code className="font-mono text-fg">python -m src.monitoring.simulate_drift</code>.
              </p>
            </Panel>
          ) : (
            <div className="grid gap-6 lg:grid-cols-2">
              <Reveal delay={0.06}>
                <Panel>
                  <h2 className="mb-4 font-medium">Prediction volume per hour</h2>
                  {s ? <VolumeChart data={s.series} /> : <div className="skeleton h-[220px] rounded-xl" />}
                </Panel>
              </Reveal>
              <Reveal delay={0.1}>
                <Panel>
                  <h2 className="mb-4 font-medium">Predicted churn rate per hour</h2>
                  {s ? <RateChart data={s.series} /> : <div className="skeleton h-[220px] rounded-xl" />}
                </Panel>
              </Reveal>
            </div>
          )}
        </div>
      )}
    </>
  );
}
