"""TVmaze catalogue provider (no API key needed).

Lists, search, filters and sorting run on a local index of Korean dramas built once by
backend/scripts/build_tvmaze_index.py. Cast, images, episode stills and alternative titles of one
drama are fetched live from the TVmaze API and cached in memory.

Every value comes from TVmaze; missing fields stay missing. TVmaze has no trailers and no vote
counts, so those are simply absent. Data licence: CC BY-SA 4.0 - https://www.tvmaze.com/api
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date
from pathlib import Path

import httpx
from fastapi import HTTPException

API = "https://api.tvmaze.com"
INDEX_FILE = Path(__file__).resolve().parents[1] / "data" / "tvmaze_kdramas.json"
PREFIX = "tvmaze-"
SOURCE = "TVmaze"
ATTRIBUTION = "Drama information and images are provided by TVmaze (tvmaze.com) under the CC BY-SA 4.0 licence."
CACHE_SECONDS = 6 * 60 * 60
PAGE_SIZE = 24
# Search terms from the team that TVmaze lists under its official English title.
TITLE_ALIASES = {"Goblin": "Guardian: The Lonely and Great God"}

_index: list[dict] | None = None
_cache: dict[str, tuple[float, object]] = {}
_client: httpx.AsyncClient | None = None
transport: httpx.AsyncBaseTransport | None = None      # replaced by a mock transport in the tests


def available() -> bool:
    return INDEX_FILE.exists()


def index() -> list[dict]:
    global _index
    if _index is None:
        if not INDEX_FILE.exists():
            raise HTTPException(status_code=503, detail={
                "code": "catalog_not_configured",
                "message": "The drama catalogue is not set up. Run 'python backend/scripts/build_tvmaze_index.py' "
                           "(no key needed) or add a TMDB credential to backend/.env, then restart the backend."})
        _index = json.loads(INDEX_FILE.read_text())["dramas"]
    return _index


def clear_cache() -> None:
    global _index, _client
    _index, _client = None, None
    _cache.clear()


async def get(path: str) -> object:
    global _client
    hit = _cache.get(path)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    if _client is None:
        _client = httpx.AsyncClient(timeout=15, transport=transport, headers={"User-Agent": "kdrama-ai-classroom-project"})
    try:
        response = await _client.get(API + path)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail={
            "code": "catalog_unreachable", "message": f"TVmaze could not be reached ({type(error).__name__}). Please try again."})
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "TVmaze has no entry for this id."})
    if response.status_code == 429:
        raise HTTPException(status_code=503, detail={"code": "catalog_rate_limited", "message": "TVmaze rate limit reached. Try again in a moment."})
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail={"code": "catalog_error", "message": f"TVmaze answered with HTTP {response.status_code}."})
    data = response.json()
    _cache[path] = (time.time(), data)
    return data


# --------------------------------------------------------------------------- shaping
def card(d: dict, backdrop: str | None = None) -> dict:
    premiered = d.get("premiered") or None
    return {
        "id": f"{PREFIX}{d['id']}", "source": SOURCE,
        "name": d.get("name"), "original_name": None,
        "overview": d.get("summary"),
        "poster_path": d.get("poster"), "backdrop_path": backdrop,
        "first_air_date": premiered, "year": int(premiered[:4]) if premiered else None,
        "genres": d.get("genres") or [],
        "vote_average": d.get("rating"), "vote_count": 0,       # TVmaze does not publish vote counts
        "popularity": d.get("weight"),
    }


def _key(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _top(items: list[dict], key, n: int = 20) -> list[dict]:
    return [card(d) for d in sorted(items, key=key, reverse=True)[:n]]


def genres() -> dict:
    counts: dict[str, int] = {}
    for d in index():
        for g in d["genres"]:
            counts[g] = counts.get(g, 0) + 1
    names = sorted(g for g, c in counts.items() if c >= 10)
    return {"genres": [{"id": g, "name": g} for g in names]}


def _backdrop_from(show: dict) -> str | None:
    """A real landscape image of the drama: an official background, otherwise an episode still."""
    embedded = show.get("_embedded", {})
    for image in embedded.get("images") or []:
        if image.get("type") == "background":
            return image["resolutions"]["original"]["url"]
    for episode in embedded.get("episodes") or []:
        if episode.get("image"):
            return episode["image"]["original"]
    return None


async def _details(tv_id: int) -> dict:
    return await get(f"/shows/{tv_id}?embed[]=cast&embed[]=images&embed[]=episodes&embed[]=akas")   # type: ignore[return-value]


async def home(iconic_titles: list[str]) -> dict:
    today = date.today().isoformat()
    with_poster = [d for d in index() if d.get("poster")]
    weight = lambda d: (d.get("weight") or 0, d.get("rating") or 0)                    # noqa: E731
    has = lambda d, *names: any(g in d["genres"] for g in names)                       # noqa: E731

    by_name = {_key(d["name"]): d for d in sorted(index(), key=weight)}               # most popular wins a name clash

    async def find(title: str) -> dict | None:
        title = TITLE_ALIASES.get(title, title)
        if _key(title) in by_name:
            return by_name[_key(title)]
        results = await get("/search/shows?" + str(httpx.QueryParams({"q": title})))
        ids = {d["id"]: d for d in index()}
        korean = [ids[r["show"]["id"]] for r in results if r["show"]["id"] in ids]     # type: ignore[union-attr]
        return korean[0] if korean else None

    iconic_found = await asyncio.gather(*[find(t) for t in iconic_titles], return_exceptions=True)
    seen, iconic = set(), []
    for d in iconic_found:
        if isinstance(d, dict) and d["id"] not in seen:
            seen.add(d["id"])
            iconic.append(card(d))

    rated = [d for d in with_poster if d.get("rating")]
    collections = [
        {"key": "trending", "title": "Trending K-Dramas", "criterion": "Highest TVmaze popularity score (\"weight\")",
         "items": _top(with_poster, weight)},
        {"key": "iconic", "title": "Fan Favourites", "criterion": "Titles chosen by our team; all details come from TVmaze", "items": iconic},
        {"key": "romance", "title": "Popular Romance", "criterion": "TVmaze genre Romance, sorted by popularity score",
         "items": _top([d for d in with_poster if has(d, "Romance")], weight)},
        {"key": "thriller", "title": "Thriller & Mystery", "criterion": "TVmaze genres Thriller, Mystery or Crime, sorted by popularity score",
         "items": _top([d for d in with_poster if has(d, "Thriller", "Mystery", "Crime")], weight)},
        {"key": "fantasy", "title": "Fantasy & Supernatural", "criterion": "TVmaze genres Fantasy or Supernatural, sorted by popularity score",
         "items": _top([d for d in with_poster if has(d, "Fantasy", "Supernatural")], weight)},
        {"key": "top", "title": "Highly Rated", "criterion": "Highest TVmaze user rating among dramas with a popularity score of 60 or more",
         "items": _top([d for d in rated if (d.get("weight") or 0) >= 60], lambda d: (d["rating"], d.get("weight") or 0))},
        {"key": "gems", "title": "Hidden Gems", "criterion": "TVmaze rating of 7.5 or more with a popularity score below 60",
         "items": _top([d for d in rated if d["rating"] >= 7.5 and (d.get("weight") or 0) < 60], lambda d: (d["rating"], d.get("weight") or 0))},
        {"key": "recent", "title": "Recently Released", "criterion": "Newest premiere dates on TVmaze",
         "items": _top([d for d in with_poster if d.get("premiered") and d["premiered"] <= today], lambda d: d["premiered"])},
    ]

    # Hero: the most popular dramas that have a synopsis and a real landscape image.
    candidates = [d for d in sorted(with_poster, key=weight, reverse=True) if d.get("summary")][:12]
    details = await asyncio.gather(*[_details(d["id"]) for d in candidates], return_exceptions=True)
    hero = []
    for d, detail in zip(candidates, details):
        backdrop = _backdrop_from(detail) if isinstance(detail, dict) else None
        if backdrop and len(hero) < 6:
            hero.append(card(d, backdrop))
    return {"source": SOURCE, "hero": hero, "collections": [c for c in collections if c["items"]]}


async def discover(q: str, genre: str | None, year: int | None, sort: str, page: int) -> dict:
    items = index()
    if q.strip():
        needle = _key(q)
        local = [d for d in items if needle in _key(d["name"])]
        # TVmaze's own search also knows alternative and Korean titles ("Goblin", "도깨비"), so we add its hits.
        try:
            hits = await get("/search/shows?" + str(httpx.QueryParams({"q": q.strip()})))
            by_id = {d["id"]: d for d in items}
            found = {d["id"] for d in local}
            local += [by_id[h["show"]["id"]] for h in hits if h["show"]["id"] in by_id and h["show"]["id"] not in found]   # type: ignore[union-attr]
        except HTTPException:
            pass                                           # the local title match still works offline
        items = local
    if genre:
        items = [d for d in items if genre in d["genres"]]
    if year:
        items = [d for d in items if (d.get("premiered") or "").startswith(str(year))]
    today = date.today().isoformat()
    if sort == "rating":
        items = sorted([d for d in items if d.get("rating")], key=lambda d: (d["rating"], d.get("weight") or 0), reverse=True)
    elif sort == "release":
        items = sorted([d for d in items if d.get("premiered") and d["premiered"] <= today], key=lambda d: d["premiered"], reverse=True)
    else:
        items = sorted(items, key=lambda d: (d.get("weight") or 0, d.get("rating") or 0), reverse=True)
    pages = max(1, -(-len(items) // PAGE_SIZE))
    return {"source": SOURCE, "page": page, "total_pages": pages, "total_results": len(items),
            "items": [card(d) for d in items[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]]}


async def drama(drama_id: str, mdl_lookup) -> dict:
    tv_id = int(drama_id.removeprefix(PREFIX))
    show = await _details(tv_id)
    embedded = show.get("_embedded", {})
    in_index = next((d for d in index() if d["id"] == tv_id), None)

    akas = [a.get("name") for a in embedded.get("akas") or [] if a.get("name")]
    korean_title = next((a for a in akas if re.search(r"[가-힣]", a)), None)

    cast, seen = [], set()
    for member in embedded.get("cast") or []:
        person, character = member.get("person") or {}, member.get("character") or {}
        if person.get("id") in seen:
            continue
        seen.add(person.get("id"))
        image = person.get("image") or {}
        cast.append({"id": person.get("id"), "name": person.get("name"), "original_name": None,
                     "profile_path": image.get("medium"), "character": character.get("name"), "episodes": None})

    images = embedded.get("images") or []
    episodes = [e for e in embedded.get("episodes") or [] if e.get("type", "regular") == "regular"]
    gallery = [{"file_path": i["resolutions"]["original"]["url"], "width": i["resolutions"]["original"].get("width"),
                "height": i["resolutions"]["original"].get("height"), "kind": "backdrop", "caption": "Official background image"}
               for i in images if i.get("type") == "background"]
    gallery += [{"file_path": e["image"]["original"], "width": None, "height": None, "kind": "backdrop",
                 "caption": f"Episode {e.get('number')} still" + (f": {e['name']}" if e.get("name") and not e["name"].lower().startswith("episode") else "")}
                for e in episodes if e.get("image")]
    posters = [{"file_path": i["resolutions"]["original"]["url"], "width": None, "height": None, "kind": "poster", "caption": "Poster"}
               for i in images if i.get("type") == "poster"][:8]

    base = card(in_index) if in_index else card({
        "id": tv_id, "name": show.get("name"), "genres": show.get("genres"), "premiered": show.get("premiered"),
        "rating": (show.get("rating") or {}).get("average"), "weight": show.get("weight"),
        "poster": (show.get("image") or {}).get("original"),
        "summary": re.sub(r"<[^>]+>", "", show.get("summary") or "") or None})

    # "Related": TVmaze has no recommendations, so we list dramas sharing the most genres (stated on the page).
    mine = set(base["genres"])
    related = sorted(
        [d for d in index() if d["id"] != tv_id and d.get("poster") and mine & set(d["genres"])],
        key=lambda d: (len(mine & set(d["genres"])), d.get("weight") or 0), reverse=True)[:14] if mine else []

    network = (show.get("network") or {}).get("name") or (show.get("webChannel") or {}).get("name")
    return {
        **base,
        "original_name": korean_title,
        "backdrop_path": _backdrop_from(show),
        "tagline": None,
        "status": show.get("status") or None,
        "last_air_date": show.get("ended") or None,
        "number_of_episodes": len(episodes) or None,
        "number_of_seasons": len({e.get("season") for e in episodes}) or None,
        "runtime": show.get("averageRuntime") or show.get("runtime"),
        "networks": [{"name": network, "logo_path": None}] if network else [],
        "content_rating_kr": None,
        "homepage": show.get("officialSite") or None,
        "cast": cast[:18],
        "trailer": None,                                   # TVmaze does not provide trailers
        "gallery": gallery[:24],
        "posters": posters,
        "related": [card(d) for d in related],
        "related_criterion": "Korean dramas that share the most genres with this one, by TVmaze popularity score",
        "source_url": show.get("url"),
        "mdl_match": mdl_lookup([show.get("name"), *akas], base["year"]),
    }
