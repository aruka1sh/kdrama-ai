"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { DramaCard } from "@/lib/catalog";

/** The watchlist lives in the browser (localStorage). No account is needed. */

export type WatchStatus = "planning" | "watching" | "completed";

export const STATUS_LABEL: Record<WatchStatus, string> = {
  planning: "Plan to watch",
  watching: "Watching",
  completed: "Completed",
};

export type WatchItem = {
  id: number | string;
  name: string | null;
  poster_path: string | null;
  year: number | null;
  genres: string[];
  vote_average: number | null;
  status: WatchStatus;
  addedAt: number;
};

const KEY = "kdrama-ai:watchlist:v1";
const EMPTY: WatchItem[] = [];
const listeners = new Set<() => void>();
let cache: WatchItem[] = EMPTY;
let cacheRaw: string | null = null;

function read(): WatchItem[] {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(KEY);
  } catch {
    return cache;
  }
  if (raw !== cacheRaw) {
    cacheRaw = raw;
    try {
      const parsed = raw ? JSON.parse(raw) : [];
      cache = Array.isArray(parsed) ? parsed : EMPTY;
    } catch {
      cache = EMPTY;
    }
  }
  return cache;
}

function write(items: WatchItem[]) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(items));
  } catch {
    /* storage full or blocked: keep the in-memory copy for this visit */
    cache = items;
    cacheRaw = JSON.stringify(items);
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (e: StorageEvent) => e.key === KEY && listener();
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useWatchlist() {
  const items = useSyncExternalStore(subscribe, read, () => EMPTY);

  const setStatus = useCallback((drama: Pick<DramaCard, "id" | "name" | "poster_path" | "year" | "genres" | "vote_average">, status: WatchStatus) => {
    const current = read();
    const existing = current.find((i) => i.id === drama.id);
    write(
      existing
        ? current.map((i) => (i.id === drama.id ? { ...i, status } : i))
        : [{ id: drama.id, name: drama.name, poster_path: drama.poster_path, year: drama.year, genres: drama.genres, vote_average: drama.vote_average, status, addedAt: Date.now() }, ...current],
    );
  }, []);

  const remove = useCallback((id: number | string) => write(read().filter((i) => i.id !== id)), []);
  const statusOf = useCallback((id: number | string) => items.find((i) => i.id === id)?.status ?? null, [items]);

  return { items, setStatus, remove, statusOf };
}
