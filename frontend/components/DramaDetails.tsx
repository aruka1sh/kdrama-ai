"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { num, useApi } from "@/lib/api";
import { formatDate, img, type DramaDetails as Details, type GalleryImage } from "@/lib/catalog";
import { STATUS_LABEL, useWatchlist, type WatchStatus } from "@/lib/watchlist";
import { CatalogProblem, DramaPosterCard, DramaRow, Icon, Poster, RatingBadge } from "@/components/drama";
import DramaPredictor from "@/components/DramaPredictor";

/* ------------------------------------------------------------------ watchlist control */

function WatchlistControl({ drama }: { drama: Details }) {
  const { statusOf, setStatus, remove } = useWatchlist();
  const status = statusOf(drama.id);
  return (
    <div>
      <div role="group" aria-label="Watchlist status" className="inline-flex flex-wrap gap-1 rounded-lg border border-line bg-ink/60 p-1 backdrop-blur">
        {(Object.keys(STATUS_LABEL) as WatchStatus[]).map((s) => (
          <button key={s} type="button" className="segment" aria-pressed={status === s} onClick={() => setStatus(drama, s)}>
            {status === s && <Icon.check className="mr-1 inline h-3.5 w-3.5" />}
            {STATUS_LABEL[s]}
          </button>
        ))}
      </div>
      <p className="mt-2 h-4 text-xs text-dim">
        {status ? (
          <>In your watchlist · <button type="button" onClick={() => remove(drama.id)} className="font-bold text-blush hover:underline">Remove</button></>
        ) : "Add this drama to your watchlist by choosing a status."}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ gallery with fullscreen lightbox */

function Gallery({ images, title, source }: { images: GalleryImage[]; title: string | null; source: string }) {
  const [open, setOpen] = useState<number | null>(null);
  const step = useCallback((d: number) => setOpen((i) => (i === null ? i : (i + d + images.length) % images.length)), [images.length]);

  useEffect(() => {
    if (open === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = ""; };
  }, [open, step]);

  return (
    <>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {images.map((image, i) => (
          <button key={image.file_path} type="button" onClick={() => setOpen(i)} aria-label={`Open image ${i + 1} of ${images.length}`}
            className={`group relative overflow-hidden rounded-md bg-panel ${i === 0 ? "col-span-2 row-span-2" : ""}`}>
            <img src={img(image.file_path, i === 0 ? "w1280" : "w780")!} alt={`${title ?? "Drama"} — image ${i + 1}`} loading="lazy"
              className="aspect-video h-full w-full object-cover transition-transform duration-700 ease-[var(--ease-cinema)] group-hover:scale-105" />
            <span className="absolute inset-0 bg-ink/0 transition-colors duration-300 group-hover:bg-ink/25" />
          </button>
        ))}
      </div>

      <AnimatePresence>
        {open !== null && (
          <motion.div role="dialog" aria-modal aria-label={`Image ${open + 1} of ${images.length}`} className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/95 backdrop-blur-sm"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setOpen(null)}>
            <motion.img key={images[open].file_path} src={img(images[open].file_path, "original")!} alt={`${title ?? "Drama"} — image ${open + 1}`}
              className="max-h-[86vh] max-w-[92vw] rounded object-contain shadow-2xl" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3 }}
              onClick={(e) => e.stopPropagation()} />
            <button type="button" autoFocus onClick={() => setOpen(null)} aria-label="Close gallery" className="absolute right-4 top-4 flex h-11 w-11 items-center justify-center rounded-full bg-panel text-paper hover:bg-burgundy"><Icon.close className="h-5 w-5" /></button>
            {images.length > 1 && (
              <>
                <button type="button" onClick={(e) => { e.stopPropagation(); step(-1); }} aria-label="Previous image" className="absolute left-3 top-1/2 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-panel/90 text-paper hover:bg-burgundy sm:left-6"><Icon.chevron className="h-5 w-5 rotate-180" /></button>
                <button type="button" onClick={(e) => { e.stopPropagation(); step(1); }} aria-label="Next image" className="absolute right-3 top-1/2 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-panel/90 text-paper hover:bg-burgundy sm:right-6"><Icon.chevron className="h-5 w-5" /></button>
              </>
            )}
            <p className="absolute bottom-4 left-0 right-0 px-16 text-center text-xs text-mist">{open + 1} / {images.length} · {images[open].caption} · Image: {source}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/* ------------------------------------------------------------------ trailer (YouTube is loaded only after a click) */

function Trailer({ trailer, backdrop }: { trailer: NonNullable<Details["trailer"]>; backdrop: string | null }) {
  const [playing, setPlaying] = useState(false);
  return (
    <div className="relative aspect-video overflow-hidden rounded-lg bg-panel ring-1 ring-line">
      {playing ? (
        <iframe className="h-full w-full" src={`https://www.youtube-nocookie.com/embed/${trailer.key}?autoplay=1&rel=0`} title={trailer.name ?? "Trailer"} allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowFullScreen />
      ) : (
        <button type="button" onClick={() => setPlaying(true)} className="group absolute inset-0" aria-label={`Play ${trailer.name ?? "trailer"}`}>
          <img src={backdrop ?? `https://i.ytimg.com/vi/${trailer.key}/hqdefault.jpg`} alt="" className="h-full w-full object-cover opacity-70 transition-all duration-700 group-hover:scale-105 group-hover:opacity-90" />
          <span className="absolute inset-0 bg-gradient-to-t from-ink/80 to-transparent" />
          <span className="absolute left-1/2 top-1/2 flex h-16 w-16 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-burgundy text-paper shadow-2xl transition-transform duration-300 group-hover:scale-110"><Icon.play className="ml-1 h-6 w-6" /></span>
          <span className="absolute bottom-4 left-5 right-5 text-left text-sm font-semibold">{trailer.name} <span className="font-normal text-mist">· {trailer.type} on YouTube</span></span>
        </button>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ page */

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-[0.66rem] font-bold uppercase tracking-[0.18em] text-dim">{label}</dt>
      <dd className="mt-1 text-sm text-paper">{children}</dd>
    </div>
  );
}

function DetailsSkeleton() {
  return (
    <div aria-busy className="pt-16">
      <div className="skeleton h-[52svh] opacity-40" />
      <div className="mx-auto -mt-32 flex max-w-[1500px] gap-8 px-5 sm:px-10">
        <div className="skeleton hidden aspect-[2/3] w-64 shrink-0 rounded-md md:block" />
        <div className="flex-1 space-y-4 pt-10"><div className="skeleton h-14 w-2/3 rounded" /><div className="skeleton h-4 w-full rounded" /><div className="skeleton h-4 w-5/6 rounded" /></div>
      </div>
    </div>
  );
}

export default function DramaDetails() {
  const { id } = useParams<{ id: string }>();
  const { data: d, error, errorCode, loading, retry } = useApi<Details>(`/catalog/drama/${id}`);

  useEffect(() => { if (d?.name) document.title = `${d.name} · K-Drama AI`; }, [d?.name]);

  if (loading) return <DetailsSkeleton />;
  if (error || !d) {
    return (
      <div className="px-5 pt-36">
        {errorCode === "not_found" ? (
          <div className="mx-auto max-w-md text-center">
            <p className="display text-5xl">Drama not found</p>
            <p className="prose-soft mt-3 text-sm">The catalogue has no series with this id.</p>
            <Link href="/explore" className="btn-primary mt-6">Explore Dramas</Link>
          </div>
        ) : <CatalogProblem message={error ?? "Unknown error"} code={errorCode} onRetry={retry} />}
      </div>
    );
  }

  const backdrop = img(d.backdrop_path, "original");
  const facts: [string, React.ReactNode][] = [];
  if (d.first_air_date) facts.push(["First aired", formatDate(d.first_air_date)]);
  if (d.status) facts.push(["Status", d.status]);
  if (d.number_of_episodes) facts.push(["Episodes", `${d.number_of_episodes}${d.number_of_seasons && d.number_of_seasons > 1 ? ` · ${d.number_of_seasons} seasons` : ""}`]);
  if (d.runtime) facts.push(["Runtime", `${d.runtime} min per episode`]);
  if (d.networks.length) facts.push(["Network", d.networks.map((n) => n.name).join(", ")]);
  if (d.content_rating_kr) facts.push(["Age rating (KR)", d.content_rating_kr]);

  return (
    <article>
      {/* ---------------- cinematic header ---------------- */}
      <header className="relative">
        <div className="absolute inset-x-0 top-0 h-[78svh] overflow-hidden">
          {backdrop ? (
            <motion.img src={backdrop} alt="" aria-hidden className="h-full w-full object-cover object-[center_20%]" initial={{ scale: 1.06, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 1.4, ease: "easeOut" }} />
          ) : <div className="h-full w-full bg-[radial-gradient(80%_80%_at_70%_10%,rgba(155,36,76,0.45),transparent_70%)]" />}
          <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/65 to-ink/30" />
          <div className="absolute inset-0 bg-gradient-to-r from-ink/90 via-ink/30 to-transparent" />
        </div>

        <div className="relative mx-auto flex max-w-[1500px] flex-col gap-8 px-5 pt-[30svh] sm:px-10 md:flex-row md:items-end md:pt-[34svh]">
          <motion.div className="w-40 shrink-0 overflow-hidden rounded-md shadow-[0_30px_70px_-20px_rgba(0,0,0,0.9)] ring-1 ring-line sm:w-52 md:w-64" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}>
            <div className="aspect-[2/3]"><Poster path={d.poster_path} title={d.name} size="w500" /></div>
          </motion.div>

          <motion.div className="min-w-0 flex-1 pb-2" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}>
            <h1 className="display text-[clamp(2.4rem,6vw,5rem)] leading-[0.98]">{d.name}</h1>
            {d.original_name && d.original_name !== d.name && <p className="mt-2 font-kr text-xl text-mist" lang="ko">{d.original_name}</p>}
            {d.tagline && <p className="display mt-3 text-xl italic text-blush">{d.tagline}</p>}
            <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-mist">
              {d.vote_average !== null && (
                <span className="flex items-center gap-2"><RatingBadge value={d.vote_average} className="!text-base" /><span className="text-xs">{d.vote_count > 0 ? `${num(d.vote_count)} votes on ${d.source}` : `${d.source} rating`}</span></span>
              )}
              {d.year && <span>{d.year}</span>}
              {d.genres.map((g) => <span key={g} className="rounded-sm border border-line px-2 py-0.5 text-xs text-paper">{g}</span>)}
            </div>
            <div className="mt-6 flex flex-wrap items-start gap-4">
              <WatchlistControl drama={d} />
              {d.trailer && <a href="#trailer" className="btn-ghost !py-2.5"><Icon.play className="h-3.5 w-3.5" /> Trailer</a>}
            </div>
          </motion.div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] space-y-16 px-5 pt-12 sm:px-10">
        {/* ---------------- synopsis and facts ---------------- */}
        <section className="grid gap-10 lg:grid-cols-[1.7fr_1fr]">
          <div>
            <h2 className="eyebrow">Synopsis</h2>
            {d.overview ? <p className="mt-4 max-w-3xl text-base leading-relaxed text-paper/90 sm:text-lg">{d.overview}</p>
              : <p className="mt-4 text-sm text-dim">{d.source} has no English synopsis for this drama yet.</p>}
          </div>
          {facts.length > 0 && (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-5 self-start rounded-lg border border-line bg-panel p-6">
              {facts.map(([label, value]) => <Fact key={label} label={label}>{value}</Fact>)}
            </dl>
          )}
        </section>

        {/* ---------------- AI predictor (Project 1) ---------------- */}
        <DramaPredictor drama={d} />

        {/* ---------------- cast ---------------- */}
        {d.cast.length > 0 && (
          <section>
            <h2 className="display text-3xl">Cast</h2>
            <ul className="row-scroll">
              {d.cast.map((person) => (
                <li key={`${person.id}-${person.name}`} className="w-28 shrink-0 sm:w-32">
                  <div className="aspect-[2/3] overflow-hidden rounded-md bg-panel-2">
                    {person.profile_path
                      ? <img src={img(person.profile_path, "w185")!} alt={person.name ?? ""} loading="lazy" className="h-full w-full object-cover" />
                      : <div className="flex h-full items-center justify-center p-2 text-center text-[0.62rem] font-bold uppercase tracking-wider text-dim">No photo available</div>}
                  </div>
                  <p className="mt-2 text-sm font-semibold leading-snug">{person.name}</p>
                  {person.character && <p className="text-xs leading-snug text-mist">{person.character}</p>}
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* ---------------- trailer ---------------- */}
        {d.trailer && (
          <section id="trailer" className="scroll-mt-24">
            <h2 className="display text-3xl">Trailer</h2>
            <div className="mt-4 max-w-4xl"><Trailer trailer={d.trailer} backdrop={img(d.backdrop_path, "w1280")} /></div>
          </section>
        )}

        {/* ---------------- gallery ---------------- */}
        <section>
          <h2 className="display text-3xl">Gallery</h2>
          {d.gallery.length > 0 ? (
            <><p className="mb-4 mt-1 text-xs text-dim">{d.gallery.length} real images from {d.source} (official artwork and episode stills). Click an image to view it full screen.</p><Gallery images={d.gallery} title={d.name} source={d.source} /></>
          ) : <p className="mt-3 text-sm text-dim">{d.source} has no images for this drama.</p>}
        </section>
      </div>

      {/* ---------------- related ---------------- */}
      {d.related.length > 0 && (
        <div className="mt-16">
          <DramaRow title="Related dramas" criterion={d.related_criterion}>
            {d.related.map((r) => <DramaPosterCard key={r.id} drama={r} className="w-36 shrink-0 sm:w-44" />)}
          </DramaRow>
        </div>
      )}

      <p className="mx-auto mt-10 max-w-[1500px] px-5 text-xs text-dim sm:px-10">
        Drama information and images: {d.source_url ? <a href={d.source_url} target="_blank" rel="noreferrer" className="text-blush hover:underline">{d.source}</a> : d.source}
        {d.source === "TVmaze" && " (CC BY-SA 4.0)"}.
      </p>
    </article>
  );
}
