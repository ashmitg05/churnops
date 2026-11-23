"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const axis = { stroke: "var(--faint)", fontSize: 11, tickLine: false, axisLine: false } as const;
const grid = <CartesianGrid stroke="var(--line)" vertical={false} />;
const tip = {
  contentStyle: {
    background: "var(--surface-2)",
    border: "1px solid var(--line-strong)",
    borderRadius: 12,
    fontSize: 12,
    fontFamily: "var(--font-geist-mono)",
  },
  labelStyle: { color: "var(--muted)" },
  itemStyle: { color: "var(--fg)" },
  cursor: { stroke: "var(--line-strong)" },
};

export function CostChart({ data, threshold }: { data: { threshold: number; cost: number }[]; threshold: number }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="costfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.28} />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
          </linearGradient>
        </defs>
        {grid}
        <XAxis dataKey="threshold" type="number" domain={[0, 1]} {...axis} />
        <YAxis {...axis} width={44} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(2)} labelFormatter={(l: number) => `t = ${l.toFixed(2)}`} />
        <ReferenceLine
          x={threshold}
          stroke="var(--fg)"
          strokeDasharray="3 3"
          label={{ value: `chosen ${threshold}`, fill: "var(--muted)", fontSize: 11, position: "insideTopRight" }}
        />
        <ReferenceLine x={0.5} stroke="var(--faint)" strokeDasharray="2 4" />
        <Area dataKey="cost" stroke="var(--accent)" strokeWidth={2} fill="url(#costfill)" isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function CalibrationChart({ data }: { data: { predicted: number; observed: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        {grid}
        <XAxis dataKey="predicted" type="number" domain={[0, 1]} {...axis} tickFormatter={(v: number) => v.toFixed(1)} />
        <YAxis domain={[0, 1]} {...axis} width={44} tickFormatter={(v: number) => v.toFixed(1)} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} labelFormatter={(l: number) => `predicted ${l.toFixed(3)}`} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--faint)" strokeDasharray="3 3" />
        <Line dataKey="observed" stroke="var(--accent)" strokeWidth={2} dot={{ r: 3, fill: "var(--accent)" }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

const fmtHour = (s: string) => new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit" });

export function VolumeChart({ data }: { data: { ts: string; volume: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ts" {...axis} tickFormatter={fmtHour} minTickGap={24} />
        <YAxis {...axis} width={44} allowDecimals={false} />
        <Tooltip {...tip} labelFormatter={fmtHour} cursor={{ fill: "var(--line)" }} />
        <Bar dataKey="volume" fill="var(--accent)" radius={[4, 4, 0, 0]} maxBarSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RateChart({ data }: { data: { ts: string; churn_rate: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ts" {...axis} tickFormatter={fmtHour} minTickGap={24} />
        <YAxis domain={[0, 1]} {...axis} width={44} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
        <Tooltip {...tip} labelFormatter={fmtHour} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
        <Line dataKey="churn_rate" stroke="var(--accent)" strokeWidth={2} dot={{ r: 3, fill: "var(--accent)" }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
