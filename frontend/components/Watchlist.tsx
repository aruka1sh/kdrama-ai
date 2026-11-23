"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { STATUS_LABEL, useWatchlist, type WatchStatus } from "@/lib/watchlist";
import { Icon, Poster, RatingBadge } from "@/components/drama";

type Tab = "all" | WatchStatus;

export default function Watchlist() {
  const { items, setStatus, remove } = useWatchlist();
  const [tab, setTab] = useState<Tab>("all");
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);          // localStorage is only available in the browser

  const count = (t: Tab) => (t === "all" ? items.length : items.filter((i) => i.status === t).length);
  const shown = tab === "all" ? items : items.filter((i) => i.status === tab);
  const tabs: Tab[] = ["all", "planning", "watching", "completed"];

  return (
    <div className="mx-auto max-w-[1500px] px-5 pt-28 sm:px-10">
      <p className="eyebrow">My Watchlist</p>
      <h1 className="display mt-3 text-5xl sm:text-6xl">Your dramas</h1>
      <p className="prose-soft mt-3 max-w-xl text-sm">Saved in this browser only — no account needed. It stays here after you refresh or close the page.</p>

      <div role="tablist" aria-label="Watchlist status" className="row-scroll !gap-1 mt-6 border-b border-line !py-0">
        {tabs.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} type="button" onClick={() => setTab(t)}
            className={`relative shrink-0 px-4 py-3 text-sm font-semibold transition-colors ${tab === t ? "text-paper" : "text-mist hover:text-paper"}`}>
            {t === "all" ? "All" : STATUS_LABEL[t]} <span className="ml-1 text-xs text-dim">{ready ? count(t) : ""}</span>
            {tab === t && <motion.span layoutId="watch-tab" className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-blush" />}
          </button>
        ))}
      </div>

      {!ready ? (
        <div className="grid gap-3 py-8 lg:grid-cols-2">{Array.from({ length: 4 }, (_, i) => <div key={i} className="skeleton h-36 rounded-lg" />)}</div>
      ) : shown.length === 0 ? (
        <div className="mx-auto max-w-md py-24 text-center">
          <div aria-hidden className="mx-auto flex h-28 w-20 items-center justify-center rounded-md border border-dashed border-line"><Icon.plus className="h-6 w-6 text-dim" /></div>
          <p className="display mt-6 text-4xl">{items.length === 0 ? "Your watchlist is empty" : `Nothing in “${STATUS_LABEL[tab as WatchStatus]}” yet`}</p>
          <p className="prose-soft mt-3 text-sm">
            {items.length === 0 ? "Press the + on any poster, or choose a status on a drama page, and it will appear here." : "Change the status of a drama to move it into this list."}
          </p>
          <Link href="/explore" className="btn-primary mt-7">Explore Dramas</Link>
        </div>
      ) : (
        <ul className="grid gap-3 py-8 lg:grid-cols-2">
          <AnimatePresence initial={false}>
            {shown.map((item) => (
              <motion.li key={item.id} layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97 }} transition={{ duration: 0.3 }}
                className="flex gap-4 rounded-lg border border-line bg-panel p-3">
                <Link href={`/drama/${item.id}`} className="group w-20 shrink-0 overflow-hidden rounded sm:w-24">
                  <div className="aspect-[2/3]"><Poster path={item.poster_path} title={item.name} size="w185" className="transition-transform duration-500 group-hover:scale-105" /></div>
                </Link>
                <div className="flex min-w-0 flex-1 flex-col py-1">
                  <Link href={`/drama/${item.id}`} className="display truncate text-2xl hover:text-blush">{item.name}</Link>
                  <p className="mt-1 flex items-center gap-3 truncate text-xs text-dim">
                    <RatingBadge value={item.vote_average} />
                    <span className="truncate">{[item.year, item.genres.slice(0, 2).join(", ")].filter(Boolean).join(" · ")}</span>
                  </p>
                  <div className="mt-auto flex flex-wrap items-center gap-2 pt-3">
                    <div role="group" aria-label={`Status of ${item.name}`} className="inline-flex gap-0.5 rounded-md border border-line bg-ink/60 p-0.5">
                      {(Object.keys(STATUS_LABEL) as WatchStatus[]).map((s) => (
                        <button key={s} type="button" className="segment !px-2.5 !py-1.5 !text-[0.7rem]" aria-pressed={item.status === s} onClick={() => setStatus(item, s)}>{STATUS_LABEL[s]}</button>
                      ))}
                    </div>
                    <button type="button" onClick={() => remove(item.id)} className="ml-auto text-xs font-bold text-mist hover:text-blush" aria-label={`Remove ${item.name} from watchlist`}>Remove</button>
                  </div>
                </div>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
