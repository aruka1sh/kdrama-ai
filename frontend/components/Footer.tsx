"use client";

import Link from "next/link";
import { useApi } from "@/lib/api";

type Status = { configured: boolean; source: string | null; attribution: string | null; source_url: string | null };

export default function Footer() {
  const { data } = useApi<Status>("/catalog/status");
  return (
    <footer className="mt-24 border-t border-line bg-coal">
      <div className="mx-auto grid max-w-[1500px] gap-10 px-5 py-12 sm:px-10 md:grid-cols-[1.3fr_1fr_1fr_1.3fr]">
        <div>
          <p className="display text-3xl">K-Drama <span className="text-blush">AI</span></p>
          <p className="mt-2 text-sm text-mist">Your Next Obsession Starts Here.</p>
          <p className="mt-4 max-w-xs text-xs leading-relaxed text-dim">
            AI Fundamentals final project, QAIRU. Variant 21 — Movie Preference, adapted to Korean dramas.
          </p>
        </div>
        <div>
          <p className="eyebrow">Discover</p>
          <ul className="mt-4 space-y-2 text-sm text-mist">
            <li><Link className="hover:text-paper" href="/explore">Explore all dramas</Link></li>
            <li><Link className="hover:text-paper" href="/explore?sort=rating">Highly rated</Link></li>
            <li><Link className="hover:text-paper" href="/watchlist">My watchlist</Link></li>
          </ul>
        </div>
        <div>
          <p className="eyebrow">Machine learning</p>
          <ul className="mt-4 space-y-2 text-sm text-mist">
            <li><Link className="hover:text-paper" href="/predictor">AI Predictor</Link></li>
            <li><Link className="hover:text-paper" href="/insights">AI Insights</Link></li>
            <li><Link className="hover:text-paper" href="/about">About the project</Link></li>
          </ul>
        </div>
        <div>
          <p className="eyebrow">Data sources</p>
          <p className="mt-4 text-xs leading-relaxed text-mist">
            {data?.configured && data.attribution ? (
              <>
                {data.attribution}{" "}
                {data.source_url && <a className="text-blush hover:underline" href={data.source_url} target="_blank" rel="noreferrer">{data.source}</a>}
              </>
            ) : "The drama catalogue is not connected."}
          </p>
          <p className="mt-3 text-xs leading-relaxed text-dim">
            The machine-learning models are trained on MyDramaList data published on Kaggle (C. Charuchinda, CC0; N. Rizki).
          </p>
        </div>
      </div>
      <div className="border-t border-line py-5 text-center text-xs text-dim">Classroom demonstration — not a commercial streaming service.</div>
    </footer>
  );
}
