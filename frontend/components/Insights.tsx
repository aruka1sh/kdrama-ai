"use client";

import { useMemo, useState } from "react";
import { CLUSTER_COLORS, NOISE_COLOR, fixed, num, pct, useApi, type InsightPoint, type Insights as InsightsData } from "@/lib/api";
import Plot from "@/components/Plot";
import { ErrorNote, Loading, Reveal, Segmented } from "@/components/ui";

type Projection = "pca" | "tsne" | "umap";
type Colouring = "kmeans" | "dbscan" | "iso";

const hover = (p: InsightPoint) =>
  `<b>${p.title}</b> (${p.year})<br>${p.genres.split("|").join(", ")}<br>` +
  `${p.runtime_min} min · ${p.episodes} episodes<br>popularity #${p.popularity_rank} · rating ${p.public_rating}`;

function buildTraces(points: InsightPoint[], projection: Projection, colouring: Colouring, contamination: string) {
  const groups = new Map<string, { name: string; color: string; symbol: string; size: number; opacity: number; items: InsightPoint[]; order: number }>();
  for (const p of points) {
    let key: string, name: string, color: string, symbol = "circle", size = 6, opacity = 0.85, order = 0;
    if (colouring === "kmeans") {
      key = `k${p.kmeans}`; name = `Cluster ${p.kmeans}`; color = CLUSTER_COLORS[p.kmeans]; order = p.kmeans;
    } else if (colouring === "dbscan") {
      if (p.dbscan === -1) { key = "noise"; name = "Noise"; color = NOISE_COLOR; symbol = "x"; size = 5; opacity = 0.7; order = -1; }
      else { key = `d${p.dbscan}`; name = `Dense group ${p.dbscan}`; color = CLUSTER_COLORS[p.dbscan % CLUSTER_COLORS.length]; order = p.dbscan; }
    } else if (p.iso[contamination]) {
      key = "flag"; name = "Flagged as unusual"; color = "#e8a1ba"; size = 9; opacity = 1; order = 1;
    } else {
      key = "normal"; name = "Typical"; color = "#7a7285"; opacity = 0.55; order = 0;
    }
    if (!groups.has(key)) groups.set(key, { name, color, symbol, size, opacity, items: [], order });
    groups.get(key)!.items.push(p);
  }
  return [...groups.values()].sort((a, b) => a.order - b.order).map((g) => ({
    type: "scatter" as const,
    mode: "markers" as const,
    name: `${g.name} (${g.items.length})`,
    x: g.items.map((p) => p[projection][0]),
    y: g.items.map((p) => p[projection][1]),
    text: g.items.map(hover),
    hovertemplate: "%{text}<extra></extra>",
    marker: { color: g.color, size: g.size, symbol: g.symbol, opacity: g.opacity, line: { width: 0 } },
  }));
}

export default function Insights() {
  const { data, error, loading, retry } = useApi<InsightsData>("/insights");
  const [projection, setProjection] = useState<Projection>("pca");
  const [colouring, setColouring] = useState<Colouring>("kmeans");
  const [contamination, setContamination] = useState("0.05");

  const traces = useMemo(
    () => (data ? buildTraces(data.points, projection, colouring, contamination) : []),
    [data, projection, colouring, contamination],
  );

  if (loading) return <div className="mx-auto max-w-[1500px] px-5 sm:px-10"><Loading label="Loading the analysis results from the Python backend…" /></div>;
  if (error || !data) return <div className="mx-auto max-w-3xl px-5 sm:px-10"><ErrorNote message={error ?? "Unknown error"} onRetry={retry} /></div>;

  const { kmeans, dbscan, pca, isolation_forest: iso } = data;
  const isoRow = iso.summary.find((r) => String(r.contamination) === contamination)!;
  const kOf = (k: number) => kmeans.k_curve.find((r) => r.k === k)!;
  const epsValues = [...new Set(dbscan.grid.map((g) => g.eps))];
  const msValues = [...new Set(dbscan.grid.map((g) => g.min_samples))];

  const projectionText: Record<Projection, { title: string; body: string }> = {
    pca: {
      title: "PCA — a straight, honest shadow",
      body: `PCA turns the five features into new straight-line axes that keep as much variation as possible. The two axes you see keep ${pct(pca.retained_2d, 1)} of it, so about ${pct(1 - pca.retained_2d)} is hidden. Distances here are meaningful: the horizontal axis runs from short, lesser-known dramas to long, popular, highly rated ones; the vertical axis is mostly the number of episodes.`,
    },
    tsne: {
      title: "t-SNE — keeping neighbours together",
      body: `t-SNE (perplexity ${data.tsne.perplexity}) places dramas that are similar in all five features next to each other. It is good at showing local neighbourhoods, but it bends space: the size of an island and the gap between two islands have no meaning. An island on this map is a hint, not proof of a real group.`,
    },
    umap: {
      title: "UMAP — a map of the neighbour graph",
      body: `UMAP (n_neighbors ${data.umap.n_neighbors}, min_dist ${data.umap.min_dist}) connects every drama to its nearest neighbours and draws that network in 2-D. It often looks cleaner than t-SNE, but the picture depends on its settings and on a random seed. Compare it with PCA before trusting any gap.`,
    },
  };

  const colourText: Record<Colouring, { title: string; body: string }> = {
    kmeans: {
      title: `K-Means — ${kmeans.chosen_k} segments`,
      body: `K-Means puts every drama into the group with the nearest centre. We tested K = 2 to 8. The silhouette score is best at K = 2 (${fixed(kOf(2).silhouette)}) and has a second small peak at K = ${kmeans.chosen_k} (${fixed(kmeans.chosen_silhouette)}), where the elbow curve also flattens. A score near 0.3 means the groups overlap: they are useful segments, not sharp natural classes.`,
    },
    dbscan: {
      title: `DBSCAN — ${dbscan.clusters} dense groups, ${dbscan.noise} noise points`,
      body: `DBSCAN (eps ${dbscan.eps}, min_samples ${dbscan.min_samples}) finds crowded regions and is allowed to leave lonely dramas outside as “noise” (grey crosses). It does not need the number of groups. Noise means “in a sparse region for these settings”, not “wrong data”. Agreement with K-Means on non-noise dramas: ARI ${fixed(dbscan.ari_with_kmeans_excluding_noise, 2)}.`,
    },
    iso: {
      title: `Isolation Forest — ${isoRow.flagged} unusual dramas`,
      body: `Isolation Forest cuts the data with random splits; dramas that are separated after very few cuts are unusual. Contamination ${pct(isoRow.contamination)} is our choice of how many to flag, not a measured rate. ${isoRow.also_dbscan_noise} of the ${isoRow.flagged} flagged dramas (${pct(isoRow.share_of_flags_that_are_noise)}) are also DBSCAN noise.`,
    },
  };

  return (
    <div className="mx-auto max-w-[1500px] px-5 sm:px-10">
      {/* ------------------------------------------------ explorer */}
      <section className="card overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-line p-5 sm:p-7 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-dim">2-D view</p>
            <Segmented label="Projection" value={projection} onChange={setProjection}
              options={[{ value: "pca", label: "PCA" }, { value: "tsne", label: "t-SNE" }, { value: "umap", label: "UMAP" }]} />
          </div>
          <div>
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-dim">Colour by</p>
            <Segmented label="Colouring" value={colouring} onChange={setColouring}
              options={[{ value: "kmeans", label: "K-Means clusters" }, { value: "dbscan", label: "DBSCAN + noise" }, { value: "iso", label: "Isolation Forest" }]} />
          </div>
          <div className={colouring === "iso" ? "" : "pointer-events-none opacity-30"}>
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-dim">Contamination</p>
            <Segmented label="Contamination" value={contamination} onChange={setContamination}
              options={iso.summary.map((r) => ({ value: String(r.contamination), label: pct(r.contamination) }))} />
          </div>
        </div>

        <div className="grid lg:grid-cols-[1.7fr_1fr]">
          <div className="p-2 sm:p-4">
            <Plot
              data={traces}
              height={520}
              layout={{
                xaxis: { title: { text: projection === "pca" ? "PC1" : `${projection === "tsne" ? "t-SNE" : "UMAP"} axis 1 (no unit)` }, showticklabels: projection === "pca" },
                yaxis: { title: { text: projection === "pca" ? "PC2" : `${projection === "tsne" ? "t-SNE" : "UMAP"} axis 2 (no unit)` }, showticklabels: projection === "pca" },
                margin: { l: 44, r: 12, t: 12, b: 70 },
                hovermode: "closest",
              }}
            />
            <p className="px-4 pb-4 text-xs text-dim">Each point is one real drama ({num(data.dataset.observations)} in total). Hover or tap a point to see it.</p>
          </div>
          <div className="space-y-7 border-t border-line p-6 sm:p-8 lg:border-l lg:border-t-0">
            {[projectionText[projection], colourText[colouring]].map((block) => (
              <div key={block.title}>
                <h2 className="display text-2xl text-blush">{block.title}</h2>
                <p className="prose-soft mt-3 text-sm">{block.body}</p>
              </div>
            ))}
            <p className="rounded-md border border-line bg-ink/50 p-4 text-xs leading-relaxed text-dim">
              All views use the same five standardized features: {data.dataset.features.join(", ")}. No “liked” label is used anywhere on this page.
            </p>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ choosing K */}
      <section className="pt-24">
        <Reveal>
          <p className="eyebrow">K-Means · choosing K</p>
          <h2 className="display mt-4 text-4xl sm:text-5xl">How many groups are there?</h2>
          <p className="prose-soft mt-4 max-w-3xl text-sm sm:text-base">
            Inertia always falls when K grows, so we look for the point where the curve bends (the elbow). The silhouette
            score rewards compact, well separated groups. Neither gives a single “true” answer.
          </p>
        </Reveal>
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Reveal className="card p-5">
            <h3 className="px-2 text-sm font-semibold">Elbow method — inertia</h3>
            <Plot height={280} data={[{ type: "scatter", mode: "lines+markers", x: kmeans.k_curve.map((r) => r.k), y: kmeans.k_curve.map((r) => r.inertia), line: { color: "#e8a1ba", width: 2.5 }, marker: { size: 8, color: kmeans.k_curve.map((r) => (r.k === kmeans.chosen_k ? "#f8f7fa" : "#e8a1ba")) }, hovertemplate: "K = %{x}<br>inertia %{y:.0f}<extra></extra>" }]}
              layout={{ xaxis: { title: { text: "K" }, dtick: 1 }, showlegend: false }} />
          </Reveal>
          <Reveal delay={0.1} className="card p-5">
            <h3 className="px-2 text-sm font-semibold">Silhouette score</h3>
            <Plot height={280} data={[{ type: "scatter", mode: "lines+markers", x: kmeans.k_curve.map((r) => r.k), y: kmeans.k_curve.map((r) => r.silhouette), line: { color: "#a79acb", width: 2.5 }, marker: { size: 8, color: kmeans.k_curve.map((r) => (r.k === kmeans.chosen_k ? "#f8f7fa" : "#a79acb")) }, hovertemplate: "K = %{x}<br>silhouette %{y:.3f}<extra></extra>" }]}
              layout={{ xaxis: { title: { text: "K" }, dtick: 1 }, showlegend: false }} />
          </Reveal>
        </div>
        <p className="mt-4 text-xs text-dim">
          White marker = our choice (K = {kmeans.chosen_k}). Stability check: five random seeds give ARI between{" "}
          {fixed(Math.min(...kmeans.seed_stability.map((s) => s.ari_vs_seed_42)), 2)} and 1.00 with the chosen solution.
        </p>
      </section>

      {/* ------------------------------------------------ cluster summaries */}
      <section className="pt-24">
        <Reveal>
          <p className="eyebrow">Cluster summaries</p>
          <h2 className="display mt-4 text-4xl sm:text-5xl">What distinguishes each group</h2>
          <p className="prose-soft mt-4 max-w-3xl text-sm sm:text-base">
            Averages are in original units. The one-line descriptions are generated from the data (features that differ
            from the overall mean by at least half a standard deviation) — the algorithm itself only returns numbers 0–{kmeans.chosen_k - 1}.
          </p>
        </Reveal>
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          {kmeans.profiles.map((p, i) => (
            <Reveal key={p.cluster} delay={i * 0.06} className="card relative overflow-hidden p-7">
              <span aria-hidden className="absolute inset-y-0 left-0 w-1" style={{ background: CLUSTER_COLORS[p.cluster] }} />
              <div className="flex items-baseline justify-between gap-4">
                <h3 className="display text-3xl" style={{ color: CLUSTER_COLORS[p.cluster] }}>Cluster {p.cluster}</h3>
                <span className="text-xs text-mist">{p.size} dramas · {pct(p.share)}</span>
              </div>
              <p className="mt-2 text-sm text-paper first-letter:uppercase">{p.description}</p>
              <dl className="mt-5 grid grid-cols-3 gap-4 text-center">
                {[
                  [`${Math.round(p.mean.runtime_min)} min`, "runtime"],
                  [`${Math.round(p.mean.episodes)}`, "episodes"],
                  [`#${num(Math.round(p.median.popularity_rank))}`, "median rank"],
                  [fixed(p.mean.public_rating, 2), "public rating"],
                  [fixed(p.mean.genre_score, 2), "genre score"],
                  [`${Math.round(p.mean.year)}`, "mean year"],
                ].map(([v, l]) => (
                  <div key={l} className="rounded-lg bg-ink/50 py-3">
                    <dd className="display text-2xl">{v}</dd>
                    <dt className="text-[0.62rem] uppercase tracking-wider text-dim">{l}</dt>
                  </div>
                ))}
              </dl>
              <p className="mt-5 text-xs text-mist">
                <span className="text-dim">Most common genres: </span>
                {p.top_genres.map((g) => `${g.genre} ${pct(g.share)}`).join(" · ")}
              </p>
              <p className="mt-2 text-xs text-mist"><span className="text-dim">Most popular members: </span>{p.examples.join(", ")}</p>
            </Reveal>
          ))}
        </div>
        <p className="mt-4 text-xs text-dim">Genre lists were not used for clustering; they only describe the groups afterwards.</p>
      </section>

      {/* ------------------------------------------------ DBSCAN sensitivity */}
      <section className="pt-24">
        <Reveal>
          <p className="eyebrow">DBSCAN · parameter sensitivity</p>
          <h2 className="display mt-4 text-4xl sm:text-5xl">Change two numbers, change the story</h2>
          <p className="prose-soft mt-4 max-w-3xl text-sm sm:text-base">
            <b className="text-paper">eps</b> is the neighbourhood radius; <b className="text-paper">min_samples</b> is how many neighbours make a region “dense”.
            Each cell shows clusters / noise points for one setting. The highlighted cell is the setting used in the explorer above.
          </p>
        </Reveal>
        <Reveal className="card mt-8 overflow-x-auto p-6">
          <table className="w-full min-w-[520px] text-center text-sm">
            <thead>
              <tr className="text-[0.68rem] uppercase tracking-[0.14em] text-dim">
                <th className="pb-3 text-left font-semibold">min_samples ↓ / eps →</th>
                {epsValues.map((e) => <th key={e} className="pb-3 font-semibold">{e}</th>)}
              </tr>
            </thead>
            <tbody>
              {msValues.map((ms) => (
                <tr key={ms} className="border-t border-line">
                  <th className="py-3 text-left font-semibold text-mist">{ms}</th>
                  {epsValues.map((e) => {
                    const g = dbscan.grid.find((r) => r.eps === e && r.min_samples === ms)!;
                    const main = e === dbscan.eps && ms === dbscan.min_samples;
                    return (
                      <td key={e} className="p-1.5">
                        <div className={`rounded-lg px-2 py-2.5 ${main ? "bg-rose/25 ring-1 ring-rose" : "bg-ink/50"}`}>
                          <span className="display text-xl">{g.clusters}</span>
                          <span className="text-dim"> / </span>
                          <span className="text-mist">{g.noise}</span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Reveal>
      </section>

      {/* ------------------------------------------------ PCA */}
      <section className="pt-24">
        <Reveal>
          <p className="eyebrow">PCA · explained variance</p>
          <h2 className="display mt-4 text-4xl sm:text-5xl">Five features, two main directions</h2>
        </Reveal>
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Reveal className="card p-5">
            <h3 className="px-2 text-sm font-semibold">Variance kept by each component</h3>
            <Plot height={290}
              data={[
                { type: "bar", name: "per component", x: pca.explained_variance_ratio.map((_, i) => `PC${i + 1}`), y: pca.explained_variance_ratio, marker: { color: "#9b244c" }, hovertemplate: "%{x}: %{y:.1%}<extra></extra>" },
                { type: "scatter", mode: "lines+markers", name: "cumulative", x: pca.cumulative.map((_, i) => `PC${i + 1}`), y: pca.cumulative, line: { color: "#e8a1ba", width: 2 }, hovertemplate: "up to %{x}: %{y:.1%}<extra></extra>" },
              ]}
              layout={{ yaxis: { tickformat: ".0%", range: [0, 1.05] }, margin: { l: 48, r: 16, t: 16, b: 60 } }} />
          </Reveal>
          <Reveal delay={0.1} className="card overflow-x-auto p-7">
            <h3 className="text-sm font-semibold">Loadings — what each axis is made of</h3>
            <table className="mt-4 w-full min-w-[320px] text-sm">
              <thead className="text-[0.68rem] uppercase tracking-[0.14em] text-dim">
                <tr><th className="pb-2 text-left font-semibold">Feature</th><th className="pb-2 text-right font-semibold">PC1</th><th className="pb-2 text-right font-semibold">PC2</th></tr>
              </thead>
              <tbody>
                {data.dataset.features.map((f) => (
                  <tr key={f} className="border-t border-line">
                    <td className="py-2.5 text-mist">{f}</td>
                    {(["PC1", "PC2"] as const).map((pc) => {
                      const v = pca.loadings[pc][f];
                      return <td key={pc} className={`text-right ${Math.abs(v) >= 0.4 ? "font-semibold text-blush" : "text-dim"}`}>{v >= 0 ? "+" : "−"}{Math.abs(v).toFixed(2)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-xs leading-relaxed text-dim">
              PC1 ({pct(pca.explained_variance_ratio[0], 1)}) mixes rating, runtime, genre score and popularity: a “mainstream success” direction.
              PC2 ({pct(pca.explained_variance_ratio[1], 1)}) is almost only the number of episodes. Loadings describe directions of variation, not causes.
            </p>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------ anomalies */}
      <section className="pt-24">
        <Reveal>
          <p className="eyebrow">Isolation Forest · anomalies</p>
          <h2 className="display mt-4 text-4xl sm:text-5xl">The dramas that stand apart</h2>
          <p className="prose-soft mt-4 max-w-3xl text-sm sm:text-base">
            “Unusual” means a rare combination of values — not a bad drama and not a data error. The list mixes global
            mega-hits with extreme formats such as 100-episode daily serials and tiny web dramas.
          </p>
        </Reveal>
        <div className="mt-8 grid gap-6 lg:grid-cols-[1.6fr_1fr]">
          <Reveal className="card overflow-x-auto p-6">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="text-[0.68rem] uppercase tracking-[0.14em] text-dim">
                <tr><th className="pb-3 font-semibold">Most unusual (contamination {pct(iso.main_contamination)})</th><th className="pb-3 font-semibold">Runtime</th><th className="pb-3 font-semibold">Eps.</th><th className="pb-3 font-semibold">Rank</th><th className="pb-3 font-semibold">Rating</th><th className="pb-3 font-semibold">DBSCAN</th></tr>
              </thead>
              <tbody>
                {iso.top_anomalies.map((a) => (
                  <tr key={a.title} className="border-t border-line">
                    <td className="py-2.5"><span className="font-semibold">{a.title}</span> <span className="text-dim">{a.year}</span></td>
                    <td className="text-mist">{a.runtime_min} min</td>
                    <td className="text-mist">{a.episodes}</td>
                    <td className="text-mist">#{num(a.popularity_rank)}</td>
                    <td className="text-mist">{a.public_rating}</td>
                    <td className={a.dbscan_noise ? "text-blush" : "text-dim"}>{a.dbscan_noise ? "noise" : "in a group"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Reveal>
          <Reveal delay={0.1} className="card p-7">
            <h3 className="text-sm font-semibold">Isolation Forest vs DBSCAN noise</h3>
            <table className="mt-4 w-full text-sm">
              <thead className="text-[0.62rem] uppercase tracking-[0.12em] text-dim">
                <tr><th className="pb-2 text-left font-semibold">Contam.</th><th className="pb-2 text-right font-semibold">Flagged</th><th className="pb-2 text-right font-semibold">Both</th><th className="pb-2 text-right font-semibold">Only noise</th></tr>
              </thead>
              <tbody>
                {iso.summary.map((r) => (
                  <tr key={r.contamination} className="border-t border-line">
                    <td className="py-2.5 text-mist">{pct(r.contamination)}</td>
                    <td className="text-right">{r.flagged}</td>
                    <td className="text-right text-blush">{r.also_dbscan_noise}</td>
                    <td className="text-right text-mist">{r.only_dbscan_noise}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-xs leading-relaxed text-dim">
              Most flagged dramas are also DBSCAN noise, but DBSCAN marks many more ({dbscan.noise}). The two methods define “unusual” differently:
              easy to isolate versus not in a dense region.
            </p>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------ cautions */}
      <section className="pt-24">
        <Reveal className="glass rounded-lg p-8 sm:p-10">
          <p className="eyebrow">Read with care</p>
          <ul className="prose-soft mt-5 grid gap-x-10 gap-y-4 text-sm md:grid-cols-2">
            <li>Clusters overlap (silhouette ≈ {fixed(kmeans.chosen_silhouette, 2)}). K = {kmeans.chosen_k} is a useful choice, not a discovered truth.</li>
            <li>Without scaling, runtime alone would decide the clusters (raw silhouette {fixed(data.preprocessing.scaling_experiment.raw_silhouette, 2)}, but only one feature matters).</li>
            <li>t-SNE and UMAP pictures change with their settings and seed; do not compare distances across maps.</li>
            <li>None of these outputs is a ground-truth class or a verified anomaly.</li>
          </ul>
        </Reveal>
      </section>
    </div>
  );
}
