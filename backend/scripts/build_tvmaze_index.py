"""Build the local index of Korean dramas from the TVmaze API (no API key needed).

Run:  python backend/scripts/build_tvmaze_index.py        (about 4 minutes, ~370 small requests)
Writes backend/data/tvmaze_kdramas.json

TVmaze has no "list Korean shows" endpoint, so we walk its public show index once
(https://api.tvmaze.com/shows?page=N), keep the scripted Korean-language series and store only the
fields the website needs. Every value is copied from TVmaze; nothing is invented.
Data licence: CC BY-SA 4.0, https://www.tvmaze.com/api#licensing
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "data" / "tvmaze_kdramas.json"
API = "https://api.tvmaze.com/shows"


def clean(summary: str | None) -> str | None:
    if not summary:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", "", summary)).strip()
    return text or None


def compact(show: dict) -> dict:
    image = show.get("image") or {}
    return {
        "id": show["id"],
        "name": show.get("name"),
        "genres": show.get("genres") or [],
        "status": show.get("status"),
        "premiered": show.get("premiered"),
        "ended": show.get("ended"),
        "runtime": show.get("averageRuntime") or show.get("runtime"),
        "rating": (show.get("rating") or {}).get("average"),
        "weight": show.get("weight"),                      # TVmaze popularity score, 0-100
        "poster": image.get("original"),
        "poster_small": image.get("medium"),
        "summary": clean(show.get("summary")),
        "network": (show.get("network") or {}).get("name") or (show.get("webChannel") or {}).get("name"),
        "url": show.get("url"),
    }


def main() -> None:
    dramas, page, total = [], 0, 0
    with httpx.Client(timeout=30, headers={"User-Agent": "kdrama-ai-classroom-project"}) as client:
        while True:
            response = client.get(API, params={"page": page})
            if response.status_code == 404:                # past the last page
                break
            if response.status_code == 429:                # rate limit: wait and repeat the same page
                time.sleep(6)
                continue
            response.raise_for_status()
            shows = response.json()
            total += len(shows)
            dramas += [compact(s) for s in shows if s.get("language") == "Korean" and s.get("type") == "Scripted"]
            if page % 40 == 0:
                print(f"page {page}: {total} shows scanned, {len(dramas)} Korean dramas", flush=True)
            page += 1
            time.sleep(0.5)                                # stay well under 20 calls per 10 seconds
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "TVmaze API (https://www.tvmaze.com/api), licence CC BY-SA 4.0",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shows_scanned": total,
        "dramas": dramas,
    }, ensure_ascii=False))
    print(f"done: {total} shows scanned, {len(dramas)} Korean dramas -> {OUT}")


if __name__ == "__main__":
    main()
