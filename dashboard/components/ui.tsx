"use client";

import { Warning } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import type { ButtonHTMLAttributes, PointerEvent, ReactNode } from "react";

/* Radius system: controls/buttons = full pill, panels = 16px (rounded-2xl), chips = 8px. */

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  // Cursor position goes straight to CSS vars: no React state on pointer move.
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
  };
  return (
    <div onPointerMove={onMove} className={`spotlight rounded-2xl p-5 md:p-6 ${className}`}>
      {children}
    </div>
  );
}

export function Button({
  children,
  variant = "primary",
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" }) {
  const base =
    "inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap rounded-full px-5 text-sm font-medium transition-[transform,background-color,border-color] duration-200 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";
  const styles =
    variant === "primary"
      ? "btn-shine bg-accent text-accent-fg shadow-[inset_0_1px_0_rgb(255_255_255/0.35),0_8px_24px_-8px_color-mix(in_oklab,var(--accent)_60%,transparent)] hover:-translate-y-px"
      : "border border-line-strong bg-surface/60 text-fg hover:border-faint";
  return (
    <button className={`${base} ${styles} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[13px] text-muted">{label}</span>
      <span className="font-mono text-2xl tracking-tight text-fg tabular-nums">{value}</span>
      {hint && <span className="text-xs text-faint">{hint}</span>}
    </div>
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-2xl border border-line bg-surface p-5 text-sm sm:flex-row sm:items-center">
      <Warning size={18} weight="duotone" className="shrink-0 text-accent" />
      <p className="flex-1 text-muted">
        Could not reach the API. Start it with <code className="font-mono text-fg">uvicorn src.serving.app:app</code>.
        <span className="mt-1 block font-mono text-xs text-faint">{message}</span>
      </p>
      {onRetry && (
        <Button variant="ghost" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    // Not animated: the page title is the LCP element and must paint before hydration.
    <div className="mb-8 md:mb-10">
      <h1 className="text-3xl font-semibold tracking-tighter md:text-[2.75rem] md:leading-[1.05]">{title}</h1>
      <p className="mt-3 max-w-[60ch] text-base leading-relaxed text-muted">{sub}</p>
    </div>
  );
}
