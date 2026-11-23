"use client";

import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useApi, type ModelInfo } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Predict" },
  { href: "/model", label: "Model" },
  { href: "/monitoring", label: "Monitoring" },
];

export function Nav() {
  const path = usePathname();
  const reduce = useReducedMotion();
  const { data } = useApi<ModelInfo>("/model-info");

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/70 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1280px] items-center justify-between gap-4 px-4 md:px-8">
        <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
          <span className="grid size-7 place-items-center rounded-lg bg-accent font-mono text-[13px] text-accent-fg">
            C
          </span>
          <span className="hidden sm:inline">ChurnOps</span>
        </Link>

        {/* Segmented nav with a shared sliding indicator (layoutId). */}
        <nav className="flex rounded-full border border-line bg-surface/80 p-1 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`relative rounded-full px-3.5 py-1.5 transition-colors ${active ? "text-fg" : "text-muted hover:text-fg"}`}
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-full border border-line-strong bg-surface-2"
                    transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                <span className="relative">{l.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="hidden min-w-[150px] justify-end font-mono text-xs text-muted md:flex">
          {data ? (
            <span>
              {data.family} <span className="text-fg">v{data.registry_version}</span>
            </span>
          ) : (
            <span className="skeleton h-3 w-24 rounded" />
          )}
        </div>
      </div>
    </header>
  );
}
