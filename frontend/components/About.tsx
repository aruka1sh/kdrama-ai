"use client";

import type { ReactNode } from "react";
import { fixed, num, pct, useApi, type Summary } from "@/lib/api";
import { TEAM } from "@/lib/team";
import { ErrorNote, Loading, Reveal } from "@/components/ui";

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 border-t border-line py-14">
      <Reveal className="grid gap-8 lg:grid-cols-[1fr_2fr]">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="display mt-3 text-4xl">{title}</h2>
        </div>
        <div className="prose-soft space-y-4 text-sm sm:text-base">{children}</div>
      </Reveal>
    </section>
  );
}

export default function About() {
  const { data, error, loading, retry } = useApi<Summary>("/summary");
  const s = data?.supervised;
  const u = data?.unsupervised;

  return (
    <div className="mx-auto max-w-[1500px] px-5 sm:px-10">
      <Section id="objective" eyebrow="01 · Objective" title="Our academic objective">
        <p>
          K-Drama AI is our final exam project for the <b className="text-paper">AI Fundamentals</b> course at QAIRU. We had to deliver two
          connected machine-learning projects — one supervised, one unsupervised — and explain every step during an oral defence.
          This website is an extra: a drama discovery platform built around the same Python code and the same results.
        </p>
      </Section>

      <Section id="variant" eyebrow="02 · Variant 21" title="Movie preference, adapted to K-Dramas">
        <p>
          The worksheets define Variant 21 as <i>“predict whether a viewer likes a movie from genre score, runtime and popularity”</i> and,
          for the second project, <i>“identify groups, low-dimensional structure and unusual observations”</i> with the same characteristics.
          No exact dataset is given, so we chose Korean dramas, a topic we know and love, and kept the three required features.
        </p>
      </Section>

      <Section id="learning" eyebrow="03 · Two kinds of learning" title="Supervised vs unsupervised">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="card p-6">
            <h3 className="display text-2xl text-blush">Supervised</h3>
            <p className="mt-2 text-sm">We have the answer for every example (liked or not) and the model learns to predict it. We can measure how often it is right.</p>
          </div>
          <div className="card p-6">
            <h3 className="display text-2xl text-lavender">Unsupervised</h3>
            <p className="mt-2 text-sm">There is no answer column. The algorithms only look for structure: groups, simpler views of the data, unusual points. There is no “accuracy”.</p>
          </div>
        </div>
      </Section>

      <Section id="data" eyebrow="04 · Data" title="Dataset sources">
        <p>
          The site uses two kinds of data and keeps them apart. <b className="text-paper">The catalogue</b> (posters, images, cast, synopses) comes from{" "}
          <a className="text-blush hover:underline" href="https://www.tvmaze.com" target="_blank" rel="noreferrer">TVmaze</a> (CC BY-SA 4.0) through our backend, or from TMDB when a TMDB key is configured.{" "}
          <b className="text-paper">The machine-learning models</b> are trained only on the two MyDramaList datasets below, joined by title and year.
        </p>
        <ul className="space-y-3">
          <li className="card p-5">
            <b className="text-paper">Source A</b> — Chanon Charuchinda (2023), <i>Korean Drama from 2015–2023 with Actors &amp; Reviews</i>. Licence CC0 (public domain).
            Drama metadata (runtime, episodes, popularity rank) and 10,625 reviews with each viewer’s own score.
          </li>
          <li className="card p-5">
            <b className="text-paper">Source B</b> — Noor Rizki (2023), <i>Top Korean Drama List (~1500)</i>. Licence not stated; used for classroom purposes only.
            Genres and the public MyDramaList rating.
          </li>
        </ul>
        <p>
          <b className="text-paper">Nothing is simulated.</b> The target of Project 1 is a real viewer’s score, grouped into two classes: “liked” = 8 or more out of 10.
          A drama’s public rating is <i>not</i> used as anyone’s personal opinion — it is only an input feature.
          Our “genre score” is the viewer’s average score for <i>other</i> dramas of the same genres, so it never contains the answer we predict.
        </p>
      </Section>

      <Section id="algorithms" eyebrow="05 · Algorithms" title="What we used">
        <div className="grid gap-x-10 gap-y-3 sm:grid-cols-2">
          {[
            ["Logistic Regression", "a simple linear model with easy-to-read coefficients"],
            ["Random Forest", "many decision trees voting together; can learn non-linear rules"],
            ["K-Means", "groups dramas around K centres"],
            ["DBSCAN", "finds dense regions and marks lonely points as noise"],
            ["PCA", "a linear 2-D summary with measurable information loss"],
            ["t-SNE and UMAP", "non-linear maps that keep neighbours together"],
            ["Isolation Forest", "flags dramas that are easy to separate from the rest"],
          ].map(([name, text]) => (
            <p key={name}><b className="text-paper">{name}</b> — {text}.</p>
          ))}
        </div>
      </Section>

      <Section id="method" eyebrow="06 · Method" title="Evaluation methodology">
        <ol className="list-decimal space-y-2 pl-5">
          <li><b className="text-paper">Split first (80/20).</b> All reviews of one viewer stay on the same side, so the test viewers are truly new.</li>
          <li><b className="text-paper">Pipelines.</b> Missing values are filled and features are scaled inside the pipeline, using training data only — no leakage.</li>
          <li><b className="text-paper">Fair comparison.</b> Both models use the same stratified 3-fold cross-validation; we select by mean F1.</li>
          <li><b className="text-paper">One final test.</b> The selected model is evaluated once on the untouched test set: confusion matrix, precision, recall, F1, ROC-AUC.</li>
          <li><b className="text-paper">Unsupervised checks.</b> Elbow and silhouette for K, parameter grids for DBSCAN, several seeds and settings for t-SNE/UMAP, three contamination values.</li>
        </ol>
      </Section>

      <Section id="results" eyebrow="07 · Results" title="What we found">
        {loading && <Loading />}
        {error && <ErrorNote message={error} onRetry={retry} />}
        {s && u && (
          <>
            <p>
              <b className="text-paper">Project 1.</b> From {num(s.reviews)} real reviews by {num(s.viewers)} viewers, the selected model
              ({s.selected_model}, mean CV F1 {fixed(s.cv_f1[s.selected_model])}) reached F1 = {fixed(s.test.f1)}, precision = {fixed(s.test.precision)},
              recall = {fixed(s.test.recall)} and ROC-AUC = {fixed(s.test.roc_auc)} on {num(s.test_rows)} test reviews. A model that always says “like”
              gets F1 ≈ {fixed(s.baseline_f1, 2)}, so the gain is real but modest. The viewer’s genre score and the public rating matter most.
            </p>
            <p>
              <b className="text-paper">Project 2.</b> For {num(u.dramas)} dramas, K-Means with K = {u.chosen_k} (silhouette {fixed(u.silhouette, 2)}) separates short web dramas,
              long-running serials, popular hits and standard dramas. DBSCAN finds {u.dbscan_clusters} dense groups and {u.dbscan_noise} noise points.
              Two PCA axes keep {pct(u.pca_retained_2d, 1)} of the variance. Isolation Forest flags {u.iso_flagged} dramas at {pct(u.iso_contamination)} contamination;
              {" "}{u.iso_also_noise} of them are also DBSCAN noise.
            </p>
          </>
        )}
      </Section>

      <Section id="limits" eyebrow="08 · Limitations" title="What this project cannot claim">
        <ul className="list-disc space-y-2 pl-5">
          <li>Reviewers are self-selected and generous; they are not typical viewers.</li>
          <li>{s ? `For ${pct(s.missing_share_genre_score)} of reviews` : "For many reviews"} the viewer has no history, so the genre score is filled with a median (the cold-start problem).</li>
          <li>The data has no dates: a real system must build features only from past ratings.</li>
          <li>Clusters overlap and depend on our choices (K, eps, min_samples, contamination).</li>
          <li>Feature importance shows what the model uses, not what causes people to enjoy a drama.</li>
          <li>The predictor on this site is a classroom demonstration, not a validated personal recommender.</li>
        </ul>
      </Section>

      <Section id="team" eyebrow="09 · Team" title="Four students, one story">
        <div className="grid gap-4 sm:grid-cols-2">
          {TEAM.map((m, i) => (
            <div key={m.name} className="card flex gap-5 p-6">
              <span className="display text-5xl italic text-rose/60">{i + 1}</span>
              <div>
                <h3 className="text-base font-semibold text-paper">{m.name}</h3>
                <p className="text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-rose">{m.role}</p>
                <p className="mt-2 text-sm">{m.focus}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-dim">Every member studied both projects and can explain any step of the code.</p>
      </Section>
    </div>
  );
}
