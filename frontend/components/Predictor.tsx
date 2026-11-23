"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { api, fixed, num, pct, useApi, type Drama, type ModelInfo, type ModelInput, type Prediction, type PredictionRequest } from "@/lib/api";
import { ErrorNote, Loading, Reveal } from "@/components/ui";

type FormState = { genre: number; genreUnknown: boolean; runtime: number; rank: number; rating: number; episodes: number; year: number };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/* ------------------------------------------------------------------ small form pieces */

function Slider({ id, label, hint, value, min, max, step, onChange, display, disabled }: {
  id: string; label: string; hint: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; display: string; disabled?: boolean;
}) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-4">
        <label htmlFor={id} className="text-sm font-semibold">{label}</label>
        <output htmlFor={id} className={`display text-2xl ${disabled ? "text-dim" : "text-blush"}`}>{display}</output>
      </div>
      <input
        id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ["--fill" as string]: `${fill}%` }}
        aria-describedby={`${id}-hint`}
      />
      <p id={`${id}-hint`} className="text-xs leading-relaxed text-dim">{hint}</p>
    </div>
  );
}

function DramaSearch({ onPick }: { onPick: (d: Drama) => void }) {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<Drama[]>([]);
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      api<{ items: Drama[] }>(`/dramas?limit=6&q=${encodeURIComponent(query)}`)
        .then((r) => { setItems(r.items); setFailed(false); })
        .catch(() => setFailed(true));
    }, 180);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  return (
    <div ref={box} className="relative">
      <label htmlFor="drama-search" className="text-sm font-semibold">Start from a real drama <span className="font-normal text-dim">(optional)</span></label>
      <input
        id="drama-search" type="search" autoComplete="off" value={query} placeholder="Search 1,094 titles — e.g. Vincenzo"
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        className="mt-2 w-full rounded-md border border-line bg-ink/60 px-4 py-3 text-sm outline-none transition-colors placeholder:text-dim focus:border-rose"
      />
      <AnimatePresence>
        {open && (items.length > 0 || failed) && (
          <motion.ul
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
            className="absolute z-20 mt-2 w-full overflow-hidden rounded-md border border-line bg-panel-2 shadow-2xl shadow-black/60"
          >
            {failed && <li className="px-4 py-3 text-xs text-blush">The catalogue could not be loaded from the backend.</li>}
            {!failed && items.map((d) => (
              <li key={d.title + d.year}>
                <button
                  type="button"
                  className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition-colors hover:bg-burgundy/30"
                  onClick={() => { onPick(d); setQuery(d.title); setOpen(false); }}
                >
                  <span>
                    <span className="block text-sm font-semibold">{d.title}</span>
                    <span className="block text-xs text-mist">{d.year} · {d.genres.split("|").join(", ")}</span>
                  </span>
                  <span className="shrink-0 text-xs text-dim">#{num(d.popularity_rank)}</span>
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ result pieces */

function ProbabilityRing({ probability, positive }: { probability: number; positive: boolean }) {
  const r = 86;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative mx-auto h-56 w-56">
      <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
        <defs>
          <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={positive ? "#e8a1ba" : "#a79acb"} />
            <stop offset="1" stopColor={positive ? "#9b244c" : "#4b3a7a"} />
          </linearGradient>
        </defs>
        <circle cx="100" cy="100" r={r} fill="none" stroke="rgba(248,247,250,0.08)" strokeWidth="7" />
        <motion.circle
          key={probability}
          cx="100" cy="100" r={r} fill="none" stroke="url(#ring)" strokeWidth="7" strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - probability) }}
          transition={{ duration: 1.3, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span key={probability} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.25, duration: 0.6 }} className="display text-6xl">
          {pct(probability)}
        </motion.span>
        <span className="mt-1 text-[0.62rem] font-semibold uppercase tracking-[0.2em] text-mist">estimated chance</span>
      </div>
    </div>
  );
}

function Result({ result }: { result: Prediction }) {
  const positive = result.predicted_class === 1;
  const biggest = Math.max(0.01, ...result.feature_effects.map((e) => Math.abs(e.effect)));
  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} aria-live="polite">
      <p className="eyebrow text-center">{result.model_name} says</p>
      <h2 className={`display mt-3 text-center text-4xl ${positive ? "text-blush" : "text-lavender"}`}>{result.predicted_label}</h2>

      <div className="mt-8"><ProbabilityRing probability={result.probability_like} positive={positive} /></div>

      {/* probability against the decision threshold */}
      <div className="mt-8">
        <div className="relative h-1.5 rounded-full bg-paper/10">
          <motion.div
            className={`absolute inset-y-0 left-0 rounded-full ${positive ? "bg-rose" : "bg-lavender"}`}
            initial={{ width: 0 }} animate={{ width: pct(result.probability_like, 1) }} transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
          />
          <span className="absolute -top-1.5 h-4.5 w-px bg-paper" style={{ left: pct(result.threshold) }} />
        </div>
        <div className="mt-2 flex justify-between text-[0.68rem] text-dim">
          <span>0%</span>
          <span className="text-mist">decision threshold {pct(result.threshold)}: at or above → “like”</span>
          <span>100%</span>
        </div>
      </div>

      <p className="mt-6 rounded-md border border-line bg-ink/50 p-4 text-sm leading-relaxed text-mist">
        <span className="font-semibold text-paper">How sure is the model? </span>{result.confidence_note}
      </p>

      {result.warnings.map((w) => (
        <p key={w} className="mt-3 rounded-md border border-gold/30 bg-gold/5 p-4 text-xs leading-relaxed text-gold">{w}</p>
      ))}

      <div className="mt-8">
        <h3 className="text-sm font-semibold">What moved this estimate</h3>
        <p className="mt-1 text-xs leading-relaxed text-dim">
          Each bar shows how the probability changes when your value replaces a typical value (the training median). It describes the model, not a real-world cause.
        </p>
        <ul className="mt-4 space-y-3">
          {result.feature_effects.map((e) => {
            const width = (Math.abs(e.effect) / biggest) * 50;
            const flat = Math.abs(e.effect) < 0.0005;
            const up = e.effect >= 0;
            return (
              <li key={e.feature} className="text-xs">
                <div className="flex justify-between gap-3">
                  <span className="text-mist">{e.label} <span className="text-dim">({e.value} vs typical {fixed(e.typical_value, e.feature === "public_rating" || e.feature === "genre_score" ? 1 : 0)})</span></span>
                  <span className={`shrink-0 font-semibold ${flat ? "text-dim" : up ? "text-blush" : "text-lavender"}`}>{flat ? "no change" : `${up ? "+" : "−"}${Math.abs(e.effect * 100).toFixed(1)} pts`}</span>
                </div>
                <div className="relative mt-1.5 h-1.5 rounded-full bg-paper/5">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-paper/25" />
                  <motion.span
                    className={`absolute inset-y-0 rounded-full ${up ? "left-1/2 bg-rose" : "right-1/2 bg-lavender"}`}
                    initial={{ width: 0 }} animate={{ width: `${width}%` }} transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <p className="mt-8 border-t border-line pt-5 text-xs leading-relaxed text-dim">{result.disclosure}</p>
    </motion.div>
  );
}

function ModelCard({ info }: { info: ModelInfo }) {
  const { test } = info;
  const cm = test.confusion_matrix;
  const cv = info.cross_validation;
  const importance = Object.entries(info.interpretation.permutation_importance_test_auc_drop).sort((a, b) => b[1].mean - a[1].mean);
  const maxImp = importance[0][1].mean;
  const labelOf = (name: string) => (name === "popularity_log" ? "Popularity rank" : info.inputs.find((i) => i.name === name)?.label ?? name);

  return (
    <section className="mx-auto max-w-[1500px] px-5 pt-24 sm:px-8">
      <Reveal>
        <p className="eyebrow">Model card</p>
        <h2 className="display mt-4 text-4xl sm:text-5xl">How good is this model, honestly?</h2>
        <p className="prose-soft mt-4 max-w-3xl text-sm sm:text-base">
          Trained on {num(info.split.train_rows)} real reviews and tested once on {num(info.split.test_rows)} reviews written by viewers the model never saw.
          {" "}{pct(info.dataset.positive_rate)} of all reviews are “liked” (score 8+), so a model that always answers “like” already reaches an F1 of {fixed(cv.baseline_always_like_f1_mean, 2)} in cross-validation.
        </p>
      </Reveal>

      <div className="mt-10 grid gap-6 lg:grid-cols-3">
        <Reveal className="card p-7">
          <h3 className="text-sm font-semibold">Test-set metrics <span className="font-normal text-dim">(threshold {test.threshold})</span></h3>
          <dl className="mt-5 grid grid-cols-2 gap-5">
            {[
              ["Precision", test.precision, "of the predicted “likes” were right"],
              ["Recall", test.recall, "of the real “likes” were found"],
              ["F1 score", test.f1, "balance of precision and recall"],
              ["ROC-AUC", test.roc_auc, "ranking quality; 0.5 = guessing"],
            ].map(([name, value, text]) => (
              <div key={name as string}>
                <dd className="display text-4xl">{fixed(value as number, 3)}</dd>
                <dt className="mt-1 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-rose">{name}</dt>
                <p className="mt-1 text-xs text-dim">{text}</p>
              </div>
            ))}
          </dl>
        </Reveal>

        <Reveal delay={0.08} className="card p-7">
          <h3 className="text-sm font-semibold">Confusion matrix <span className="font-normal text-dim">(test set)</span></h3>
          <div className="mt-5 grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-xs">
            <span />
            <span className="pb-1 text-dim">predicted: not like</span>
            <span className="pb-1 text-dim">predicted: like</span>
            <span className="flex items-center pr-2 text-dim [writing-mode:vertical-rl] rotate-180">really not</span>
            <Cell n={cm.tn} name="True negative" tone="bg-lavender/15" />
            <Cell n={cm.fp} name="False positive" tone="bg-wine/50" />
            <span className="flex items-center pr-2 text-dim [writing-mode:vertical-rl] rotate-180">really liked</span>
            <Cell n={cm.fn} name="False negative" tone="bg-wine/50" />
            <Cell n={cm.tp} name="True positive" tone="bg-rose/25" />
          </div>
          <p className="mt-4 text-xs leading-relaxed text-dim">
            False positive: we promise a favourite and it disappoints. False negative: we hide a drama the viewer would have loved.
          </p>
        </Reveal>

        <Reveal delay={0.16} className="card p-7">
          <h3 className="text-sm font-semibold">What the model relies on</h3>
          <p className="mt-1 text-xs text-dim">Drop in test ROC-AUC when one feature is shuffled (permutation importance).</p>
          <ul className="mt-5 space-y-3">
            {importance.map(([name, v]) => (
              <li key={name} className="text-xs">
                <div className="flex justify-between"><span className="text-mist">{labelOf(name)}</span><span className="text-dim">{fixed(v.mean, 3)}</span></div>
                <div className="mt-1 h-1.5 rounded-full bg-paper/5">
                  <div className="h-full rounded-full bg-gradient-to-r from-burgundy to-rose" style={{ width: `${Math.max(1, (v.mean / maxImp) * 100)}%` }} />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs leading-relaxed text-dim">Importance shows what the model uses. It does not prove what causes people to like a drama.</p>
        </Reveal>
      </div>

      <Reveal className="card mt-6 overflow-x-auto p-7">
        <h3 className="text-sm font-semibold">Model selection — same stratified 3-fold cross-validation on training data</h3>
        <table className="mt-4 w-full min-w-[420px] text-left text-sm">
          <thead className="text-[0.68rem] uppercase tracking-[0.14em] text-dim">
            <tr><th className="pb-2 font-semibold">Model</th><th className="pb-2 font-semibold">Mean CV F1</th><th className="pb-2 font-semibold">SD</th><th className="pb-2 font-semibold">Mean CV ROC-AUC</th></tr>
          </thead>
          <tbody>
            {Object.entries(cv.models).map(([name, m]) => (
              <tr key={name} className="border-t border-line">
                <td className="py-2.5">{name}{name === info.model_name && <span className="ml-2 rounded-full bg-rose/20 px-2 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-blush">selected</span>}</td>
                <td>{fixed(m.f1_mean)}</td><td>{fixed(m.f1_std)}</td><td>{fixed(m.roc_auc_mean)}</td>
              </tr>
            ))}
            <tr className="border-t border-line text-mist"><td className="py-2.5">Baseline: always “like”</td><td>{fixed(cv.baseline_always_like_f1_mean)}</td><td>—</td><td>0.500</td></tr>
          </tbody>
        </table>
      </Reveal>
    </section>
  );
}

function Cell({ n, name, tone }: { n: number; name: string; tone: string }) {
  return (
    <div className={`rounded-lg ${tone} px-2 py-5`}>
      <p className="display text-3xl">{n}</p>
      <p className="mt-1 text-[0.62rem] uppercase tracking-wider text-mist">{name}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ page body */

export default function Predictor() {
  const { data: info, error: infoError, loading, retry } = useApi<ModelInfo>("/model");
  const [form, setForm] = useState<FormState>({ genre: 8, genreUnknown: false, runtime: 60, rank: 500, rating: 8, episodes: 16, year: 2020 });
  const [picked, setPicked] = useState<Drama | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const range = useMemo(() => {
    const map: Record<string, ModelInput> = {};
    info?.inputs.forEach((i) => (map[i.name] = i));
    return map;
  }, [info]);

  const set = (patch: Partial<FormState>) => { setForm((f) => ({ ...f, ...patch })); setResult(null); setError(null); };

  const pick = (d: Drama) => {
    setPicked(d);
    set({
      runtime: clamp(Math.round(d.runtime_min), range.runtime_min.train_min, range.runtime_min.train_max),
      rank: d.popularity_rank, rating: d.public_rating, episodes: d.episodes, year: d.year,
    });
  };

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const body: PredictionRequest = {
      genre_score: form.genreUnknown ? null : form.genre,
      runtime_min: form.runtime, popularity_rank: form.rank, public_rating: form.rating,
      episodes: form.episodes, year: form.year,
    };
    try {
      setResult(await api<Prediction>("/predict", { method: "POST", body: JSON.stringify(body) }));
    } catch (err) {
      setResult(null);
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <div className="mx-auto max-w-[1500px] px-5 sm:px-10"><Loading label="Loading the model's input schema from the Python backend…" /></div>;
  if (infoError || !info) return <div className="mx-auto max-w-3xl px-5 sm:px-10"><ErrorNote message={infoError ?? "Unknown error"} onRetry={retry} /></div>;

  const maxLogRank = Math.log10(range.popularity_rank.train_max);
  const genreNames = picked ? picked.genres.split("|").join(", ") : "this genre";

  return (
    <>
      <section className="mx-auto grid max-w-[1500px] gap-8 px-5 sm:px-10 lg:grid-cols-[1.15fr_1fr]">
        <form onSubmit={submit} className="card p-6 sm:p-10">
          <p className="eyebrow">Step 1 — the drama</p>
          <div className="mt-5"><DramaSearch onPick={pick} /></div>
          {picked && (
            <p className="mt-3 text-xs text-mist">
              Filled in from <span className="font-semibold text-blush">{picked.title}</span> ({picked.year}). You can still change any value below.
            </p>
          )}

          <div className="mt-8 grid gap-7 sm:grid-cols-2">
            <Slider id="runtime" label="Episode runtime" hint="Minutes per episode." value={form.runtime}
              min={range.runtime_min.train_min} max={range.runtime_min.train_max} step={1}
              onChange={(v) => set({ runtime: v })} display={`${form.runtime} min`} />
            <Slider id="popularity" label="Popularity rank" hint="MyDramaList rank. #1 is the most popular drama."
              value={Math.log10(form.rank)} min={0} max={maxLogRank} step={0.01}
              onChange={(v) => set({ rank: clamp(Math.round(10 ** v), 1, range.popularity_rank.train_max) })} display={`#${num(form.rank)}`} />
            <Slider id="rating" label="Public rating" hint="Average MyDramaList rating of the drama." value={form.rating}
              min={range.public_rating.train_min} max={range.public_rating.train_max} step={0.1}
              onChange={(v) => set({ rating: Math.round(v * 10) / 10 })} display={form.rating.toFixed(1)} />
            <Slider id="episodes" label="Episodes" hint="Number of episodes." value={form.episodes}
              min={range.episodes.train_min} max={range.episodes.train_max} step={1}
              onChange={(v) => set({ episodes: v })} display={`${form.episodes}`} />
            <Slider id="year" label="Release year" hint="The model was trained on 2015–2023 releases." value={form.year}
              min={range.year.train_min} max={range.year.train_max} step={1}
              onChange={(v) => set({ year: v })} display={`${form.year}`} />
          </div>

          <div className="hairline my-9" />

          <p className="eyebrow">Step 2 — you</p>
          <div className="mt-5">
            <Slider id="genre" label="Your genre preference score"
              hint={`The score you usually give to dramas like this (${genreNames}). 1 = you rarely enjoy them, 10 = you love them.`}
              value={form.genre} min={1} max={10} step={0.5} disabled={form.genreUnknown}
              onChange={(v) => set({ genre: v })} display={form.genreUnknown ? "unknown" : form.genre.toFixed(1)} />
            <label className="mt-3 flex cursor-pointer items-center gap-3 text-xs text-mist">
              <input type="checkbox" checked={form.genreUnknown} onChange={(e) => set({ genreUnknown: e.target.checked })} className="h-4 w-4 accent-[#e8a1ba]" />
              I am not sure — treat me like a viewer with no history
            </label>
          </div>

          <button type="submit" disabled={busy} className="btn-primary mt-10 w-full sm:w-auto">
            {busy ? "Asking the model…" : "Predict My Preference"}
          </button>
        </form>

        <aside className="glass relative min-h-[28rem] overflow-hidden rounded-lg p-6 sm:p-10 lg:sticky lg:top-24 lg:self-start">
          <div aria-hidden className="absolute -right-24 -top-24 h-64 w-64 rounded-full bg-burgundy/40 blur-3xl" />
          <div className="relative">
            {error && <ErrorNote message={error} />}
            {!error && result && <Result result={result} />}
            {!error && !result && (
              <div className="flex min-h-[24rem] flex-col items-center justify-center text-center">
                                <p className="display mt-4 text-3xl">Your result will appear here</p>
                <p className="prose-soft mt-3 max-w-xs text-sm">
                  Set the values and press <span className="text-blush">Predict My Preference</span>. The answer is computed by the trained {info.model_name} in Python.
                </p>
              </div>
            )}
          </div>
        </aside>
      </section>

      <ModelCard info={info} />
    </>
  );
}
