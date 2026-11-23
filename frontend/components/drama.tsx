"use client";

import Link from "next/link";
import { useRef, useState, type ReactNode } from "react";
import { img, rating, type DramaCard } from "@/lib/catalog";
import { useWatchlist } from "@/lib/watchlist";

/* ------------------------------------------------------------------ icons (inline SVG, no emoji) */

export const Icon = {
  star: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden className={p.className}><path d="M10 1.6l2.47 5.3 5.8.7-4.28 3.98 1.13 5.74L10 14.47 4.88 17.32l1.13-5.74L1.73 7.6l5.8-.7L10 1.6z" /></svg>
  ),
  plus: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden className={p.className}><path d="M10 4v12M4 10h12" /></svg>
  ),
  check: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className={p.className}><path d="M4 10.5l4 4 8-9" /></svg>
  ),
  chevron: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className={p.className}><path d="M7.5 4l6 6-6 6" /></svg>
  ),
  play: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden className={p.className}><path d="M6 3.8v12.4a.8.8 0 001.22.68l9.9-6.2a.8.8 0 000-1.36l-9.9-6.2A.8.8 0 006 3.8z" /></svg>
  ),
  close: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden className={p.className}><path d="M5 5l10 10M15 5L5 15" /></svg>
  ),
  search: (p: { className?: string }) => (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden className={p.className}><circle cx="9" cy="9" r="5.5" /><path d="M13.5 13.5L17 17" /></svg>
  ),
};

/* ------------------------------------------------------------------ poster image with an honest fallback */

export function Poster({ path, title, size = "w342", className = "" }: { path: string | null; title: string | null; size?: string; className?: string }) {
  const [failed, setFailed] = useState(false);
  const src = img(path, size);
  if (!src || failed) {
    return (
      <div className={`flex h-full w-full flex-col items-center justify-center bg-panel-2 p-4 text-center ${className}`}>
        <span className="text-[0.6rem] font-bold uppercase tracking-[0.2em] text-dim">No poster available</span>
        <span className="display mt-2 text-lg leading-tight text-mist">{title}</span>
      </div>
    );
  }
  return <img src={src} alt={title ? `${title} poster` : "Drama poster"} loading="lazy" decoding="async" onError={() => setFailed(true)} className={`h-full w-full object-cover ${className}`} />;
}

export function RatingBadge({ value, className = "" }: { value: number | null; className?: string }) {
  if (value === null) return null;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-bold text-paper ${className}`} title="User rating from the catalogue source">
      <Icon.star className="h-3 w-3 text-gold" />
      {rating(value)}
    </span>
  );
}

/* ------------------------------------------------------------------ poster card */

export function DramaPosterCard({ drama, className = "" }: { drama: DramaCard; className?: string }) {
  const { statusOf, setStatus, remove } = useWatchlist();
  const saved = statusOf(drama.id) !== null;
  return (
    <article className={`group relative ${className}`}>
      <Link href={`/drama/${drama.id}`} className="block" aria-label={`${drama.name}${drama.year ? `, ${drama.year}` : ""}`}>
        <div className="relative aspect-[2/3] overflow-hidden rounded-md bg-panel ring-1 ring-line transition-shadow duration-500 group-hover:shadow-[0_24px_50px_-20px_rgba(155,36,76,0.75)] group-hover:ring-blush/40">
          <Poster path={drama.poster_path} title={drama.name} className="transition-transform duration-700 ease-[var(--ease-cinema)] group-hover:scale-[1.07]" />
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-ink/90 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
          <RatingBadge value={drama.vote_average} className="absolute left-2 top-2 rounded bg-ink/75 px-1.5 py-1 backdrop-blur-sm" />
          <span className="pointer-events-none absolute inset-x-0 bottom-0 translate-y-2 p-3 text-[0.68rem] font-bold uppercase tracking-[0.16em] text-paper opacity-0 transition-all duration-500 group-hover:translate-y-0 group-hover:opacity-100">
            View details
          </span>
        </div>
        <h3 className="mt-2.5 line-clamp-2 text-sm font-semibold leading-snug text-paper transition-colors group-hover:text-blush">{drama.name}</h3>
        <p className="mt-0.5 truncate text-xs text-dim">
          {[drama.year, drama.genres.slice(0, 2).join(", ")].filter(Boolean).join(" · ")}
        </p>
      </Link>
      <button
        type="button"
        onClick={() => (saved ? remove(drama.id) : setStatus(drama, "planning"))}
        aria-pressed={saved}
        aria-label={saved ? `Remove ${drama.name} from watchlist` : `Add ${drama.name} to watchlist`}
        title={saved ? "In your watchlist — click to remove" : "Add to watchlist"}
        className={`absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full backdrop-blur-sm transition-all duration-300 ${
          saved ? "bg-blush text-ink" : "bg-ink/70 text-paper opacity-0 hover:bg-burgundy focus-visible:opacity-100 group-hover:opacity-100 max-md:opacity-100"
        }`}
      >
        {saved ? <Icon.check className="h-4 w-4" /> : <Icon.plus className="h-4 w-4" />}
      </button>
    </article>
  );
}

export function PosterSkeleton({ className = "" }: { className?: string }) {
  return (
    <div className={className} aria-hidden>
      <div className="skeleton aspect-[2/3] rounded-md" />
      <div className="skeleton mt-2.5 h-3.5 w-4/5 rounded" />
      <div className="skeleton mt-1.5 h-3 w-1/2 rounded" />
    </div>
  );
}

/* ------------------------------------------------------------------ horizontally scrollable collection */

export function DramaRow({ title, criterion, children, href }: { title: string; criterion?: string; children: ReactNode; href?: string }) {
  const scroller = useRef<HTMLDivElement>(null);
  const move = (direction: 1 | -1) => scroller.current?.scrollBy({ left: direction * scroller.current.clientWidth * 0.85, behavior: "smooth" });
  return (
    <section className="group/row relative">
      <div className="mx-auto flex max-w-[1500px] items-end justify-between gap-4 px-5 sm:px-10">
        <div>
          <h2 className="display text-2xl sm:text-3xl">{title}</h2>
          {criterion && <p className="mt-1 text-xs text-dim">{criterion}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {href && <Link href={href} className="mr-2 hidden text-xs font-bold uppercase tracking-[0.14em] text-mist transition-colors hover:text-blush sm:block">See all</Link>}
          <button type="button" onClick={() => move(-1)} aria-label={`Scroll ${title} left`} className="hidden h-9 w-9 items-center justify-center rounded-full border border-line text-mist transition-colors hover:border-blush hover:text-paper sm:flex">
            <Icon.chevron className="h-4 w-4 rotate-180" />
          </button>
          <button type="button" onClick={() => move(1)} aria-label={`Scroll ${title} right`} className="hidden h-9 w-9 items-center justify-center rounded-full border border-line text-mist transition-colors hover:border-blush hover:text-paper sm:flex">
            <Icon.chevron className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div ref={scroller} className="row-scroll mx-auto max-w-[1500px] px-5 sm:px-10">{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------ catalogue problems (never placeholder data) */

export function CatalogProblem({ message, code, onRetry }: { message: string; code?: string; onRetry?: () => void }) {
  const setup = code === "catalog_not_configured" || code === "tmdb_not_configured" || code === "tmdb_bad_credential";
  return (
    <div role="alert" className="mx-auto max-w-2xl rounded-lg border border-blush/30 bg-wine/25 p-7">
      <p className="eyebrow">{setup ? "Drama catalogue is not connected" : "The catalogue could not be loaded"}</p>
      <p className="mt-3 text-sm leading-relaxed text-paper">{message}</p>
      {setup && (
        <ol className="mt-4 list-decimal space-y-1 pl-5 text-sm text-mist">
          <li>No key needed: run <code className="text-blush">python backend/scripts/build_tvmaze_index.py</code> once (TVmaze).</li>
          <li>Or add a TMDB credential to <code className="text-blush">backend/.env</code> (see <code className="text-blush">backend/.env.example</code>).</li>
          <li>Restart the Python backend and reload this page.</li>
        </ol>
      )}
      {onRetry && <button type="button" onClick={onRetry} className="btn-ghost mt-5 !py-2 !text-[0.7rem]">Try again</button>}
    </div>
  );
}
