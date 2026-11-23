"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";

/** Fade-and-rise when the element scrolls into view. */
export function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.8, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function PageHeader({ eyebrow, title, children, korean }: { eyebrow: string; title: string; children?: ReactNode; korean?: string }) {
  return (
    <header className="relative overflow-hidden pt-32 pb-12 sm:pt-40">
      <div aria-hidden className="absolute inset-0 bg-[radial-gradient(70%_90%_at_80%_0%,rgba(155,36,76,0.38),transparent_60%),radial-gradient(50%_70%_at_10%_10%,rgba(167,154,203,0.12),transparent_60%)]" />
      {korean && (
        <span aria-hidden className="pointer-events-none absolute -right-4 top-20 select-none font-kr text-[9rem] leading-none text-paper/[0.035] sm:text-[15rem]">
          {korean}
        </span>
      )}
      <div className="relative mx-auto max-w-[1500px] px-5 sm:px-10">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="display mt-5 max-w-4xl text-5xl sm:text-7xl">{title}</h1>
        {children && <div className="prose-soft mt-6 max-w-2xl text-base sm:text-lg">{children}</div>}
      </div>
    </header>
  );
}

export function Loading({ label = "Asking the Python backend…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-3 py-10 text-sm text-mist">
      <span className="h-2 w-2 animate-ping rounded-full bg-rose" />
      {label}
    </div>
  );
}

/** Honest failure state: we show the problem instead of placeholder numbers. */
export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-lg border border-blush/30 bg-wine/25 p-6">
      <p className="eyebrow">Backend unavailable</p>
      <p className="mt-3 text-sm leading-relaxed text-blush">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="btn-ghost mt-5 !py-2.5 !text-[0.7rem]">
          Try again
        </button>
      )}
    </div>
  );
}

export function Stat({ value, label, note }: { value: ReactNode; label: string; note?: string }) {
  return (
    <div>
      <p className="display text-4xl text-paper sm:text-5xl">{value}</p>
      <p className="mt-2 text-[0.7rem] font-semibold uppercase tracking-[0.18em] text-rose">{label}</p>
      {note && <p className="mt-1 text-xs leading-relaxed text-dim">{note}</p>}
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange, label }: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex max-w-full gap-1 overflow-x-auto rounded-lg border border-line bg-coal p-1">
      {options.map((o) => (
        <button key={o.value} type="button" className="segment" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
