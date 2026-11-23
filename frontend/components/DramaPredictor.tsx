"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { api, num, pct, useApi, type ModelInfo, type Prediction, type PredictionRequest } from "@/lib/api";
import type { DramaDetails } from "@/lib/catalog";

/**
 * "Will You Love This Drama?" - Project 1 inside the drama page.
 *
 * The model was trained on MyDramaList data, so its inputs are MyDramaList values. When the drama
 * exists in that dataset we use its real row. Otherwise each input says where its value comes from,
 * and the visitor can correct it. The feature schema of the model is never changed.
 */

type Field = { key: keyof Omit<PredictionRequest, "genre_score">; label: string; value: number; source: string; exact: boolean; step: number; min: number; max: number };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export default function DramaPredictor({ drama }: { drama: DramaDetails }) {
  const { data: info, error: infoError } = useApi<ModelInfo>("/model");
  const [genre, setGenre] = useState(8);
  const [unknown, setUnknown] = useState(false);
  const [overrides, setOverrides] = useState<Partial<Record<Field["key"], number>>>({});
  const [editing, setEditing] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { setResult(null); setOverrides({}); setError(null); }, [drama.id]);

  const fields = useMemo<Field[] | null>(() => {
    if (!info) return null;
    const median = Object.fromEntries(info.inputs.map((i) => [i.name, i.train_median])) as Record<string, number>;
    const m = drama.mdl_match;
    const dataset = "MyDramaList dataset (the model's training source)";
    const typical = "not available — typical value from the training data";
    return [
      { key: "runtime_min", label: "Episode runtime (min)", step: 1, min: 1, max: 180,
        ...(m ? { value: m.runtime_min, source: dataset, exact: true } : drama.runtime ? { value: drama.runtime, source: `${drama.source} runtime`, exact: true } : { value: median.runtime_min, source: typical, exact: false }) },
      { key: "episodes", label: "Episodes", step: 1, min: 1, max: 300,
        ...(m ? { value: m.episodes, source: dataset, exact: true } : drama.number_of_episodes ? { value: drama.number_of_episodes, source: `${drama.source} episode count`, exact: true } : { value: median.episodes, source: typical, exact: false }) },
      { key: "year", label: "Release year", step: 1, min: 1990, max: 2035,
        ...(m ? { value: m.year, source: dataset, exact: true } : drama.year ? { value: drama.year, source: `${drama.source} premiere date`, exact: true } : { value: median.year, source: typical, exact: false }) },
      { key: "public_rating", label: "Public rating (0–10)", step: 0.1, min: 1, max: 10,
        // A rating from another site is on a different scale (MyDramaList scores run much higher), so it is NOT substituted.
        ...(m ? { value: m.public_rating, source: dataset, exact: true } : { value: median.public_rating, source: `MyDramaList rating unknown — typical value from the training data${drama.vote_average ? ` (the ${drama.source} rating ${drama.vote_average.toFixed(1)} uses a different scale)` : ""}`, exact: false }) },
      { key: "popularity_rank", label: "MyDramaList popularity rank", step: 1, min: 1, max: 100000,
        ...(m ? { value: m.popularity_rank, source: dataset, exact: true } : { value: median.popularity_rank, source: typical + ` (${drama.source} has no comparable rank)`, exact: false }) },
    ].map((f) => ({ ...f, value: clamp(overrides[f.key as Field["key"]] ?? f.value, f.min, f.max) })) as Field[];
  }, [info, drama, overrides]);

  async function predict() {
    if (!fields) return;
    setBusy(true);
    setError(null);
    const body = { genre_score: unknown ? null : genre, ...Object.fromEntries(fields.map((f) => [f.key, f.value])) } as PredictionRequest;
    try {
      setResult(await api<Prediction>("/predict", { method: "POST", body: JSON.stringify(body) }));
    } catch (e) {
      setResult(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const positive = result?.predicted_class === 1;
  const approximations = fields?.filter((f) => !f.exact && overrides[f.key] === undefined) ?? [];

  return (
    <section className="relative overflow-hidden rounded-lg border border-line bg-panel">
      <div aria-hidden className="absolute -right-24 -top-24 h-72 w-72 rounded-full bg-burgundy/35 blur-3xl" />
      <div className="relative grid gap-8 p-6 sm:p-9 lg:grid-cols-2">
        {/* ---------------- inputs ---------------- */}
        <div>
          <p className="eyebrow">AI Preference Predictor</p>
          <h2 className="display mt-3 text-4xl">Will You Love This Drama?</h2>
          <p className="prose-soft mt-3 text-sm">
            A {info?.model_name ?? "machine-learning"} model trained on {info ? num(info.dataset.rows) : "thousands of"} real MyDramaList reviews estimates the chance that a viewer like you scores this drama 8 or higher.
          </p>

          {infoError && <p role="alert" className="mt-5 rounded-md border border-blush/30 bg-wine/25 p-4 text-sm text-blush">{infoError}</p>}

          {fields && (
            <>
              <div className="mt-6">
                <div className="flex items-baseline justify-between gap-4">
                  <label htmlFor="dp-genre" className="text-sm font-semibold">Your genre preference score</label>
                  <output htmlFor="dp-genre" className={`display text-3xl ${unknown ? "text-dim" : "text-blush"}`}>{unknown ? "—" : genre.toFixed(1)}</output>
                </div>
                <input id="dp-genre" type="range" min={1} max={10} step={0.5} value={genre} disabled={unknown}
                  onChange={(e) => { setGenre(Number(e.target.value)); setResult(null); }} style={{ ["--fill" as string]: `${((genre - 1) / 9) * 100}%` }} />
                <p className="text-xs leading-relaxed text-dim">
                  The score you usually give to {drama.genres.length ? drama.genres.join(" / ") : "this kind of"} dramas. 1 = rarely enjoy them, 10 = love them.
                </p>
                <label className="mt-2 flex cursor-pointer items-center gap-2.5 text-xs text-mist">
                  <input type="checkbox" checked={unknown} onChange={(e) => { setUnknown(e.target.checked); setResult(null); }} className="h-4 w-4 accent-[#e8a1ba]" />
                  I am not sure — treat me like a viewer with no history
                </label>
              </div>

              <div className="mt-6 rounded-md border border-line bg-ink/50">
                <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
                  <span className="text-xs font-bold uppercase tracking-[0.14em] text-mist">Drama inputs sent to the model</span>
                  <button type="button" onClick={() => setEditing((v) => !v)} aria-expanded={editing} className="text-xs font-bold text-blush hover:underline">{editing ? "Done" : "Adjust"}</button>
                </div>
                <dl>
                  {fields.map((f) => (
                    <div key={f.key} className="grid grid-cols-[1fr_auto] items-center gap-x-4 border-b border-line/60 px-4 py-2.5 last:border-0">
                      <dt className="text-sm">
                        <label htmlFor={`dp-${f.key}`}>{f.label}</label>
                        <span className={`block text-[0.7rem] leading-snug ${f.exact || overrides[f.key] !== undefined ? "text-dim" : "text-gold"}`}>
                          {overrides[f.key] !== undefined ? "entered by you" : f.source}
                        </span>
                      </dt>
                      <dd>
                        {editing ? (
                          <input id={`dp-${f.key}`} type="number" value={f.value} min={f.min} max={f.max} step={f.step}
                            onChange={(e) => { setOverrides((o) => ({ ...o, [f.key]: Number(e.target.value) })); setResult(null); }}
                            className="w-24 rounded border border-line bg-panel px-2 py-1 text-right text-sm outline-none focus:border-blush" />
                        ) : (
                          <span className="text-sm font-bold">{f.key === "popularity_rank" ? `#${num(f.value)}` : f.value}</span>
                        )}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
              {drama.mdl_match ? (
                <p className="mt-2 text-xs text-dim">This drama is in the training dataset as “{drama.mdl_match.title}” ({drama.mdl_match.year}); its real MyDramaList values are used.</p>
              ) : (
                <p className="mt-2 text-xs text-gold">This drama is not in the 2015–2023 MyDramaList dataset the model was trained on, so some inputs are approximations. Use “Adjust” if you know the real values.</p>
              )}

              <button type="button" onClick={predict} disabled={busy} className="btn-primary mt-6 w-full sm:w-auto">{busy ? "Asking the model…" : "Predict My Preference"}</button>
            </>
          )}
        </div>

        {/* ---------------- result ---------------- */}
        <div className="flex min-h-[20rem] flex-col justify-center rounded-md border border-line bg-ink/55 p-6 sm:p-8" aria-live="polite">
          {error && <p role="alert" className="text-sm leading-relaxed text-blush">{error}</p>}
          {!error && !result && (
            <div className="text-center">
              <p className="display text-3xl text-mist">Your result appears here</p>
              <p className="mt-3 text-sm text-dim">The answer is computed by the saved scikit-learn model in our Python backend — never in the browser.</p>
            </div>
          )}
          {!error && result && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
              <p className="text-[0.68rem] font-bold uppercase tracking-[0.2em] text-mist">{result.model_name} · predicted preference</p>
              <div className="mt-2 flex items-end justify-between gap-4">
                <h3 className={`display text-4xl ${positive ? "text-blush" : "text-lavender"}`}>{result.predicted_label}</h3>
                <span className="display text-6xl leading-none">{pct(result.probability_like)}</span>
              </div>
              <div className="relative mt-5 h-1.5 rounded-full bg-paper/10">
                <motion.div className={`absolute inset-y-0 left-0 rounded-full ${positive ? "bg-blush" : "bg-lavender"}`} initial={{ width: 0 }} animate={{ width: pct(result.probability_like, 1) }} transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }} />
                <span className="absolute -top-1.5 h-4 w-0.5 bg-paper" style={{ left: pct(result.threshold) }} />
              </div>
              <p className="mt-2 text-[0.7rem] text-dim">Model probability of “like”. White mark = decision threshold {pct(result.threshold)}.</p>

              <p className="mt-5 text-sm leading-relaxed text-mist"><b className="text-paper">How sure? </b>{result.confidence_note}</p>

              <p className="mt-4 text-xs font-bold uppercase tracking-[0.14em] text-mist">What moved the estimate</p>
              <ul className="mt-2 space-y-1.5 text-sm">
                {result.feature_effects.filter((e) => Math.abs(e.effect) >= 0.005).slice(0, 4).map((e) => (
                  <li key={e.feature} className="flex justify-between gap-4">
                    <span className="text-mist">{e.label} <span className="text-dim">({e.value} vs typical {Number(e.typical_value.toFixed(1))})</span></span>
                    <span className={`shrink-0 font-bold ${e.effect >= 0 ? "text-blush" : "text-lavender"}`}>{e.effect >= 0 ? "+" : "−"}{Math.abs(e.effect * 100).toFixed(1)} pts</span>
                  </li>
                ))}
              </ul>

              {[...result.warnings, ...(approximations.length ? [`Approximated inputs: ${approximations.map((f) => f.label).join(", ")}.`] : [])].map((w) => (
                <p key={w} className="mt-3 rounded border border-gold/30 bg-gold/5 p-3 text-xs leading-relaxed text-gold">{w}</p>
              ))}

              <div className="mt-5 border-t border-line pt-4 text-xs leading-relaxed text-dim">
                <p className="font-bold uppercase tracking-[0.14em] text-mist">Model limitations</p>
                <p className="mt-1.5">{result.disclosure}</p>
                {info && <p className="mt-1.5">On unseen viewers it reached F1 {info.test.f1.toFixed(2)} and ROC-AUC {info.test.roc_auc.toFixed(2)}; a model that always answers “like” gets F1 {info.cross_validation.baseline_always_like_f1_mean.toFixed(2)}. The explanation describes the model, not real-world causes.</p>}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </section>
  );
}
