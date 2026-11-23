"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api, ApiError, num, useApi } from "@/lib/api";
import type { DiscoverData, DramaCard } from "@/lib/catalog";
import { CatalogProblem, DramaPosterCard, Icon, PosterSkeleton } from "@/components/drama";

type Genre = { id: number | string; name: string };
const SORTS = [
  { value: "popularity", label: "Popularity" },
  { value: "rating", label: "Rating" },
  { value: "release", label: "Release date" },
];
const THIS_YEAR = new Date().getFullYear();
const YEARS = Array.from({ length: THIS_YEAR - 1999 }, (_, i) => THIS_YEAR - i);
const GRID = "grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6";

export default function Explore() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  // The URL is the single source of truth, so filters survive reloads and the Back button.
  const q = params.get("q") ?? "";
  const genre = params.get("genre") ?? "";
  const year = params.get("year") ?? "";
  const sort = params.get("sort") ?? "popularity";
  const filterKey = `${q}|${genre}|${year}|${sort}`;

  const { data: genreData } = useApi<{ genres: Genre[] }>("/catalog/genres");
  const [text, setText] = useState(q);
  const [items, setItems] = useState<DramaCard[]>([]);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const request = useRef(0);
  const sentinel = useRef<HTMLDivElement>(null);

  const update = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params.toString());
    Object.entries(patch).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)));
    router.replace(`${pathname}${next.toString() ? `?${next}` : ""}`, { scroll: false });
  };

  const load = (pageToLoad: number, replace: boolean) => {
    const id = ++request.current;
    setLoading(true);
    setError(null);
    const query = new URLSearchParams({ sort, page: String(pageToLoad) });
    if (q) query.set("q", q);
    if (genre) query.set("genre", genre);
    if (year) query.set("year", year);
    api<DiscoverData>(`/catalog/discover?${query}`)
      .then((d) => {
        if (id !== request.current) return;               // a newer request has started
        setItems((old) => {
          const base = replace ? [] : old;
          const seen = new Set(base.map((i) => i.id));
          return [...base, ...d.items.filter((i) => !seen.has(i.id))];
        });
        setPage(d.page);
        setTotalPages(d.total_pages);
        setTotal(d.total_results);
      })
      .catch((e: ApiError) => id === request.current && setError(e))
      .finally(() => id === request.current && setLoading(false));
  };

  // new filters -> start again from page 1
  useEffect(() => {
    setText(q);
    setItems([]);
    setTotal(null);
    load(1, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey]);

  // typing in the search box updates the URL after a short pause
  useEffect(() => {
    if (text === q) return;
    const timer = setTimeout(() => update({ q: text.trim() }), 350);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  // infinite scrolling: load the next page when the sentinel becomes visible
  const hasMore = page < totalPages;
  useEffect(() => {
    const node = sentinel.current;
    if (!node || !hasMore || loading || error) return;
    const observer = new IntersectionObserver((entries) => entries[0].isIntersecting && load(page + 1, false), { rootMargin: "600px" });
    observer.observe(node);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasMore, loading, error, page, filterKey]);

  const activeFilters = Boolean(q || genre || year || sort !== "popularity");

  return (
    <div className="mx-auto max-w-[1500px] px-5 pt-28 sm:px-10">
      <p className="eyebrow">Explore</p>
      <h1 className="display mt-3 text-5xl sm:text-6xl">All Korean dramas</h1>

      {/* ---------------- controls ---------------- */}
      <div className="sticky top-16 z-30 -mx-5 mt-8 border-y border-line bg-ink/92 px-5 py-4 backdrop-blur-xl sm:-mx-10 sm:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Icon.search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-dim" />
            <input type="search" value={text} onChange={(e) => setText(e.target.value)} placeholder="Search by title — e.g. Queen of Tears" aria-label="Search by title" className="field !pl-10" />
          </div>
          <div className="flex gap-3">
            <label className="flex-1 lg:w-40">
              <span className="sr-only">Release year</span>
              <select value={year} onChange={(e) => update({ year: e.target.value })} className="field" aria-label="Release year">
                <option value="">Any year</option>
                {YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </label>
            <label className="flex-1 lg:w-48">
              <span className="sr-only">Sort by</span>
              <select value={sort} onChange={(e) => update({ sort: e.target.value === "popularity" ? "" : e.target.value })} className="field" aria-label="Sort by">
                {SORTS.map((s) => <option key={s.value} value={s.value}>Sort: {s.label}</option>)}
              </select>
            </label>
          </div>
        </div>
        <div className="row-scroll !gap-2 !py-0 !pt-3" role="group" aria-label="Genre">
          <button type="button" className="chip" aria-pressed={!genre} onClick={() => update({ genre: "" })}>All genres</button>
          {genreData?.genres.map((g) => (
            <button key={g.id} type="button" className="chip" aria-pressed={genre === String(g.id)} onClick={() => update({ genre: genre === String(g.id) ? "" : String(g.id) })}>{g.name}</button>
          ))}
        </div>
      </div>

      {/* ---------------- results ---------------- */}
      <div className="flex min-h-10 items-center justify-between py-5 text-sm text-mist" aria-live="polite">
        <span>
          {total !== null && !error && (<><b className="text-paper">{num(total)}</b> {total === 1 ? "drama" : "dramas"}{q && <> matching “{q}”</>}</>)}
        </span>
        {activeFilters && <button type="button" onClick={() => router.replace(pathname, { scroll: false })} className="text-xs font-bold uppercase tracking-[0.14em] text-blush hover:underline">Clear filters</button>}
      </div>

      {error && items.length === 0 ? (
        <CatalogProblem message={error.message} code={error.code} onRetry={() => load(1, true)} />
      ) : total === 0 && !loading ? (
        <div className="mx-auto max-w-md py-20 text-center">
          <p className="display text-4xl">No dramas found</p>
          <p className="prose-soft mt-3 text-sm">The catalogue has no Korean drama for this combination of search and filters. Try another title or remove a filter.</p>
          <button type="button" onClick={() => router.replace(pathname, { scroll: false })} className="btn-ghost mt-6">Clear filters</button>
        </div>
      ) : (
        <>
          <div className={GRID}>
            {items.map((d) => <DramaPosterCard key={d.id} drama={d} />)}
            {loading && Array.from({ length: items.length ? 6 : 18 }, (_, i) => <PosterSkeleton key={`s${i}`} />)}
          </div>
          {error && <div className="mt-8"><CatalogProblem message={error.message} code={error.code} onRetry={() => load(page + 1, false)} /></div>}
          <div ref={sentinel} className="h-10" />
          {!hasMore && !loading && items.length > 0 && <p className="pb-6 text-center text-xs text-dim">End of the list · {num(items.length)} dramas shown</p>}
        </>
      )}
    </div>
  );
}
