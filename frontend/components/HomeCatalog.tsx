"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useApi } from "@/lib/api";
import { img, type DramaCard, type HomeData } from "@/lib/catalog";
import { CatalogProblem, DramaPosterCard, DramaRow, PosterSkeleton, RatingBadge } from "@/components/drama";

const ROTATE_MS = 8000;

function Hero({ dramas }: { dramas: DramaCard[] }) {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const drama = dramas[index];

  useEffect(() => {
    if (paused || dramas.length < 2) return;
    const timer = setTimeout(() => setIndex((i) => (i + 1) % dramas.length), ROTATE_MS);
    return () => clearTimeout(timer);
  }, [index, paused, dramas.length]);

  return (
    <section className="relative h-[92svh] min-h-[560px] overflow-hidden" onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)} aria-roledescription="carousel" aria-label="Featured dramas">
      <AnimatePresence initial={false}>
        <motion.div key={drama.id} className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 1.2, ease: "easeInOut" }}>
          <motion.img
            src={img(drama.backdrop_path, "original")!} alt="" aria-hidden
            className="h-full w-full object-cover object-[center_20%]"
            initial={{ scale: 1.08 }} animate={{ scale: 1 }} transition={{ duration: ROTATE_MS / 1000 + 1.2, ease: "linear" }}
          />
        </motion.div>
      </AnimatePresence>

      {/* cinematic gradients */}
      <div className="absolute inset-0 bg-gradient-to-r from-ink via-ink/70 to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/10 to-ink/40" />
      <div className="absolute inset-0 bg-[radial-gradient(60%_60%_at_15%_70%,rgba(155,36,76,0.28),transparent_70%)]" />

      <div className="relative mx-auto flex h-full max-w-[1500px] items-end px-5 pb-24 sm:px-10 sm:pb-28">
        <AnimatePresence mode="wait">
          <motion.div key={drama.id} className="max-w-2xl" initial={{ opacity: 0, y: 26 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>
            <p className="eyebrow">Your Next Obsession Starts Here</p>
            <h1 className="display mt-4 text-[clamp(2.8rem,8vw,6.2rem)] leading-[0.95]">{drama.name}</h1>
            {drama.original_name && drama.original_name !== drama.name && <p className="mt-2 font-kr text-lg text-mist">{drama.original_name}</p>}
            <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-mist">
              <RatingBadge value={drama.vote_average} className="!text-sm" />
              {drama.year && <span>{drama.year}</span>}
              {drama.genres.slice(0, 3).map((g) => <span key={g} className="rounded-sm border border-line px-2 py-0.5 text-xs text-paper">{g}</span>)}
            </div>
            <p className="mt-5 line-clamp-3 max-w-xl text-base leading-relaxed text-paper/85 sm:line-clamp-4">{drama.overview}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href={`/drama/${drama.id}`} className="btn-primary">View Details</Link>
              <Link href="/explore" className="btn-ghost">Explore Dramas</Link>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {dramas.length > 1 && (
        <div className="absolute bottom-10 left-0 right-0 mx-auto flex max-w-[1500px] items-center gap-2 px-5 sm:px-10">
          {dramas.map((d, i) => (
            <button key={d.id} type="button" onClick={() => setIndex(i)} aria-label={`Show ${d.name}`} aria-current={i === index}
              className="group relative h-6 w-10 sm:w-14">
              <span className="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 overflow-hidden rounded-full bg-paper/25">
                {i === index && <motion.span key={`${index}-${paused}`} className="block h-full bg-blush" initial={{ width: paused ? "100%" : "0%" }} animate={{ width: "100%" }} transition={{ duration: paused ? 0 : ROTATE_MS / 1000, ease: "linear" }} />}
              </span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function HeroSkeleton() {
  return (
    <section className="relative h-[92svh] min-h-[560px] overflow-hidden" aria-busy>
      <div className="skeleton absolute inset-0 opacity-40" />
      <div className="relative mx-auto flex h-full max-w-[1500px] items-end px-5 pb-28 sm:px-10">
        <div className="w-full max-w-xl space-y-4">
          <div className="skeleton h-16 w-4/5 rounded" />
          <div className="skeleton h-4 w-full rounded" />
          <div className="skeleton h-4 w-2/3 rounded" />
        </div>
      </div>
    </section>
  );
}

export default function HomeCatalog() {
  const { data, error, errorCode, loading, retry } = useApi<HomeData>("/catalog/home");

  if (error) {
    return (
      <section className="flex min-h-[80svh] items-center px-5 pt-24">
        <div className="w-full">
          <h1 className="display mb-8 text-center text-5xl sm:text-7xl">K-Drama <span className="text-blush">AI</span></h1>
          <CatalogProblem message={error} code={errorCode} onRetry={retry} />
          <p className="mt-8 text-center text-sm text-mist">
            The machine-learning pages do not need the catalogue:{" "}
            <Link href="/predictor" className="text-blush hover:underline">AI Predictor</Link> ·{" "}
            <Link href="/insights" className="text-blush hover:underline">AI Insights</Link>
          </p>
        </div>
      </section>
    );
  }

  return (
    <>
      {loading || !data ? <HeroSkeleton /> : data.hero.length > 0 ? <Hero dramas={data.hero} /> : <div className="h-24" />}

      <div className="relative z-10 -mt-10 space-y-10 sm:space-y-12">
        {loading || !data
          ? Array.from({ length: 3 }, (_, r) => (
              <div key={r} className="mx-auto max-w-[1500px] px-5 sm:px-10">
                <div className="skeleton h-7 w-56 rounded" />
                <div className="row-scroll">{Array.from({ length: 8 }, (_, i) => <PosterSkeleton key={i} className="w-36 shrink-0 sm:w-44" />)}</div>
              </div>
            ))
          : data.collections.map((c, i) => (
              <motion.div key={c.key} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-40px" }} transition={{ duration: 0.7, delay: i < 2 ? i * 0.1 : 0, ease: [0.22, 1, 0.36, 1] }}>
                <DramaRow title={c.title} criterion={c.criterion}>
                  {c.items.map((d) => <DramaPosterCard key={d.id} drama={d} className="w-36 shrink-0 sm:w-44" />)}
                </DramaRow>
              </motion.div>
            ))}
      </div>

      {/* AI teaser: the two academic projects, presented as product features */}
      <section className="mx-auto mt-20 grid max-w-[1500px] gap-5 px-5 sm:px-10 lg:grid-cols-2">
        <Link href="/predictor" className="group relative overflow-hidden rounded-lg border border-line bg-panel p-8 transition-colors hover:border-blush/50 sm:p-10">
          <div aria-hidden className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-burgundy/40 blur-3xl transition-transform duration-700 group-hover:scale-125" />
          <p className="eyebrow">Supervised learning</p>
          <h2 className="display mt-3 text-4xl">Will you love it?</h2>
          <p className="prose-soft mt-3 max-w-md text-sm">Open any drama and ask a model trained on thousands of real viewer reviews. It answers with a probability — and tells you how sure it is.</p>
          <span className="mt-6 inline-block text-xs font-bold uppercase tracking-[0.16em] text-blush">Try the AI Predictor →</span>
        </Link>
        <Link href="/insights" className="group relative overflow-hidden rounded-lg border border-line bg-panel p-8 transition-colors hover:border-lavender/50 sm:p-10">
          <div aria-hidden className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-lavender/25 blur-3xl transition-transform duration-700 group-hover:scale-125" />
          <p className="eyebrow !text-lavender">Unsupervised learning</p>
          <h2 className="display mt-3 text-4xl">The hidden map of K-Dramas</h2>
          <p className="prose-soft mt-3 max-w-md text-sm">Six algorithms sort 1,094 dramas into groups and find the ones that stand apart — without ever seeing a label.</p>
          <span className="mt-6 inline-block text-xs font-bold uppercase tracking-[0.16em] text-lavender">Open AI Insights →</span>
        </Link>
      </section>
    </>
  );
}
