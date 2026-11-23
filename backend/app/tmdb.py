"""TMDB proxy for the K-Drama AI website.

The TMDB credential stays on the server: the browser only ever talks to /api/catalog/... on our
own backend. Credentials are read from the environment or from backend/.env:

    TMDB_ACCESS_TOKEN=...   (v4 "API Read Access Token", preferred)
    TMDB_API_KEY=...        (v3 API key, also supported)

Nothing in this file invents data. Every field returned to the website is copied from a TMDB
response; missing fields stay missing (None / empty list). Responses are cached in memory for a
few hours (TMDB allows caching for up to six months).

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from fastapi import HTTPException

API = "https://api.themoviedb.org/3"
CACHE_SECONDS = 6 * 60 * 60
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
ROOT = Path(__file__).resolve().parents[2]
_DEV_CSV = ROOT / "data" / "processed" / "dramas.csv"
DRAMAS_CSV = Path(os.environ.get("KDRAMA_DRAMAS_CSV") or
                  (_DEV_CSV if _DEV_CSV.exists() else Path(__file__).resolve().parents[1] / "data" / "ml" / "dramas.csv"))

# Scripted Korean-language series produced in South Korea. TMDB "type": 2 = miniseries, 4 = scripted.
KOREAN_DRAMA = {"with_origin_country": "KR", "with_original_language": "ko", "with_type": "2|4",
                "include_adult": "false", "language": "en-US"}

# Titles the team asked to feature. They are only search terms: all metadata comes from TMDB.
ICONIC_TITLES = [
    "Queen of Tears", "Crash Landing on You", "Goblin", "True Beauty", "Business Proposal",
    "Lovely Runner", "Twenty-Five Twenty-One", "Vincenzo", "The Glory", "My Demon",
    "Extraordinary Attorney Woo", "Weak Hero Class 1", "Alchemy of Souls", "Twinkling Watermelon",
    "Hotel Del Luna", "Hometown Cha-Cha-Cha", "Descendants of the Sun", "Reply 1988",
    "It's Okay to Not Be Okay", "When Life Gives You Tangerines",
]

SOURCE = "TMDB"
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
_cache: dict[str, tuple[float, dict]] = {}
_client: httpx.AsyncClient | None = None
transport: httpx.AsyncBaseTransport | None = None      # replaced by a mock transport in the tests


# --------------------------------------------------------------------------- credentials
def _read_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def credentials() -> tuple[dict, dict]:
    """Return (headers, query params) for TMDB, or raise 503 if no credential is configured."""
    file_values = _read_env_file()
    token = os.environ.get("TMDB_ACCESS_TOKEN") or file_values.get("TMDB_ACCESS_TOKEN")
    key = os.environ.get("TMDB_API_KEY") or file_values.get("TMDB_API_KEY")
    if token:
        return {"Authorization": f"Bearer {token}"}, {}
    if key:
        return {}, {"api_key": key}
    raise HTTPException(status_code=503, detail={
        "code": "tmdb_not_configured",
        "message": "The drama catalogue needs a TMDB API credential. Add TMDB_ACCESS_TOKEN (or "
                   "TMDB_API_KEY) to backend/.env and restart the backend. No drama data is shown "
                   "until then, because this site never invents titles, posters or ratings."})


def is_configured() -> bool:
    try:
        credentials()
        return True
    except HTTPException:
        return False


# --------------------------------------------------------------------------- HTTP + cache
async def tmdb(path: str, **params) -> dict:
    """GET one TMDB resource (cached). Raises an honest HTTP error when TMDB cannot answer."""
    global _client
    headers, auth_params = credentials()
    clean = {k: str(v) for k, v in params.items() if v not in (None, "")}
    cache_key = path + "?" + "&".join(f"{k}={clean[k]}" for k in sorted(clean))
    hit = _cache.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    if _client is None:
        _client = httpx.AsyncClient(timeout=12, transport=transport)
    try:
        response = await _client.get(API + path, params={**clean, **auth_params}, headers=headers)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail={
            "code": "tmdb_unreachable",
            "message": f"TMDB could not be reached ({type(error).__name__}). Please try again."})
    if response.status_code == 401:
        raise HTTPException(status_code=503, detail={
            "code": "tmdb_bad_credential",
            "message": "TMDB rejected the configured credential. Check TMDB_ACCESS_TOKEN / "
                       "TMDB_API_KEY in backend/.env."})
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail={"code": "not_found",
                                                     "message": "TMDB has no entry for this id."})
    if response.status_code == 429:
        raise HTTPException(status_code=503, detail={"code": "tmdb_rate_limited",
                                                     "message": "TMDB rate limit reached. Try again in a moment."})
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail={
            "code": "tmdb_error", "message": f"TMDB answered with HTTP {response.status_code}."})
    data = response.json()
    _cache[cache_key] = (time.time(), data)
    return data


def clear_cache() -> None:
    global _client
    _cache.clear()
    _client = None


# --------------------------------------------------------------------------- shaping helpers
def is_korean(item: dict) -> bool:
    return "KR" in (item.get("origin_country") or []) and item.get("original_language") == "ko"


def card(item: dict, genre_names: dict[int, str]) -> dict:
    """The fields a poster card needs. Values are copied from TMDB; nothing is filled in."""
    first_air = item.get("first_air_date") or None
    genre_ids = item.get("genre_ids") or [g["id"] for g in item.get("genres", [])]
    return {
        "id": item["id"], "source": SOURCE,
        "name": item.get("name"),
        "original_name": item.get("original_name"),
        "overview": item.get("overview") or None,
        "poster_path": item.get("poster_path"),
        "backdrop_path": item.get("backdrop_path"),
        "first_air_date": first_air,
        "year": int(first_air[:4]) if first_air else None,
        "genres": [genre_names[g] for g in genre_ids if g in genre_names],
        "vote_average": item.get("vote_average") if item.get("vote_count") else None,
        "vote_count": item.get("vote_count") or 0,
        "popularity": item.get("popularity"),
    }


async def genre_map() -> dict[int, str]:
    data = await tmdb("/genre/tv/list", language="en-US")
    return {g["id"]: g["name"] for g in data.get("genres", [])}


async def keyword_id(name: str) -> int | None:
    """Look up a TMDB keyword id by its exact name (for example 'romance')."""
    data = await tmdb("/search/keyword", query=name)
    for row in data.get("results", []):
        if row.get("name", "").lower() == name.lower():
            return row["id"]
    return None


async def discover(pages: int = 1, **params) -> list[dict]:
    results = await asyncio.gather(*[tmdb("/discover/tv", **KOREAN_DRAMA, **params, page=p)
                                     for p in range(1, pages + 1)])
    return [item for page in results for item in page.get("results", [])]


def _key(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_mdl_index: dict[str, dict] | None = None


def mdl_lookup(names: list[str], year: int | None) -> dict | None:
    """Find the same drama in our MyDramaList dataset (the data the ML model was trained on)."""
    global _mdl_index
    if _mdl_index is None:
        _mdl_index = {}
        if DRAMAS_CSV.exists():
            for row in pd.read_csv(DRAMAS_CSV).to_dict(orient="records"):
                _mdl_index[_key(row["title"]) + str(int(row["year"]))] = row
    if year is None:
        return None
    for name in names:
        for y in (year, year - 1, year + 1):         # release years sometimes differ by one
            row = _mdl_index.get(_key(name) + str(y))
            if row:
                return {k: row[k] for k in ("title", "year", "genres", "runtime_min", "episodes",
                                            "popularity_rank", "public_rating")}
    return None


# --------------------------------------------------------------------------- endpoints
async def genres() -> dict:
    names = await genre_map()
    wanted = ["Drama", "Comedy", "Mystery", "Crime", "Sci-Fi & Fantasy", "Action & Adventure",
              "Family", "War & Politics"]
    return {"genres": [{"id": i, "name": n} for i, n in names.items() if n in wanted]}


async def _iconic(names: dict[int, str]) -> list[dict]:
    async def find(title: str) -> dict | None:
        data = await tmdb("/search/tv", query=title, language="en-US", include_adult="false")
        korean = [r for r in data.get("results", []) if is_korean(r)]
        return max(korean, key=lambda r: r.get("popularity") or 0) if korean else None

    found = await asyncio.gather(*[find(t) for t in ICONIC_TITLES])
    seen, out = set(), []
    for item in found:
        if item and item["id"] not in seen:
            seen.add(item["id"])
            out.append(card(item, names))
    return out


async def home() -> dict:
    """Hero dramas and the collections of the homepage. Each collection states its real criterion."""
    names = await genre_map()
    today = date.today().isoformat()
    romance = await keyword_id("romance")
    no_talk = "10763|10764|10767|99"                  # news, reality, talk, documentary

    async def trending() -> list[dict]:
        pages = await asyncio.gather(*[tmdb("/trending/tv/week", language="en-US", page=p) for p in range(1, 6)])
        return [r for page in pages for r in page.get("results", []) if is_korean(r)]

    jobs = {
        "trending": trending(),
        "popular": discover(sort_by="popularity.desc", without_genres=no_talk),
        "romance": (discover(sort_by="popularity.desc", with_keywords=romance, without_genres=no_talk)
                    if romance else asyncio.sleep(0, result=[])),
        "thriller": discover(sort_by="popularity.desc", with_genres="9648|80", without_genres=no_talk),
        "fantasy": discover(sort_by="popularity.desc", with_genres="10765", without_genres=no_talk),
        "top": discover(sort_by="vote_average.desc", **{"vote_count.gte": 200}, without_genres=no_talk),
        "gems": discover(sort_by="vote_average.desc", without_genres=no_talk,
                         **{"vote_average.gte": 8, "vote_count.gte": 30, "vote_count.lte": 300}),
        "recent": discover(sort_by="first_air_date.desc", without_genres=no_talk,
                           **{"first_air_date.lte": today, "vote_count.gte": 10}),
        "iconic": _iconic(names),
    }
    done = dict(zip(jobs, await asyncio.gather(*jobs.values())))
    cards = {k: (v if k == "iconic" else [card(i, names) for i in v]) for k, v in done.items()}

    trending_cards = cards["trending"]
    trending_note = "TMDB weekly trending TV list, filtered to Korean-language series from South Korea"
    if len(trending_cards) < 6:                       # too few Korean titles in this week's list
        trending_cards = cards["popular"]
        trending_note = "Korean series with the highest current TMDB popularity score"

    collections = [
        {"key": "trending", "title": "Trending K-Dramas", "criterion": trending_note, "items": trending_cards},
        {"key": "iconic", "title": "Fan Favourites", "criterion": "Titles chosen by our team; all details come from TMDB", "items": cards["iconic"]},
        {"key": "romance", "title": "Popular Romance", "criterion": "TMDB keyword \"romance\", sorted by TMDB popularity", "items": cards["romance"]},
        {"key": "thriller", "title": "Thriller & Mystery", "criterion": "TMDB genres Mystery or Crime, sorted by TMDB popularity", "items": cards["thriller"]},
        {"key": "fantasy", "title": "Fantasy & Supernatural", "criterion": "TMDB genre Sci-Fi & Fantasy, sorted by TMDB popularity", "items": cards["fantasy"]},
        {"key": "top", "title": "Highly Rated", "criterion": "Highest TMDB user score among series with at least 200 votes", "items": cards["top"]},
        {"key": "gems", "title": "Hidden Gems", "criterion": "TMDB score of 8.0 or more with only 30 to 300 votes", "items": cards["gems"]},
        {"key": "recent", "title": "Recently Released", "criterion": "Newest first-air dates on TMDB (at least 10 votes)", "items": cards["recent"]},
    ]
    collections = [c for c in collections if c["items"]]
    hero = [c for c in trending_cards if c["backdrop_path"] and c["overview"]][:6]
    return {"source": SOURCE, "hero": hero, "collections": collections}


SORTS = {"popularity": "popularity.desc", "rating": "vote_average.desc", "release": "first_air_date.desc"}


async def explore(q: str, genre: int | None, year: int | None, sort: str, page: int) -> dict:
    """Catalogue page: title search, genre and year filters, sorting and paging - all done on real TMDB data."""
    names = await genre_map()
    if q.strip():
        # TMDB's title search cannot be combined with filters, so we filter and sort its results here.
        pages = await asyncio.gather(*[tmdb("/search/tv", query=q.strip(), language="en-US",
                                            include_adult="false", page=p) for p in (1, 2, 3)])
        items = [card(r, names) for pg in pages for r in pg.get("results", []) if is_korean(r)]
        seen: set[int] = set()
        items = [i for i in items if not (i["id"] in seen or seen.add(i["id"]))]
        if genre:
            items = [i for i in items if names.get(genre) in i["genres"]]
        if year:
            items = [i for i in items if i["year"] == year]
        sort_key = {"popularity": lambda i: i["popularity"] or 0, "rating": lambda i: i["vote_average"] or 0,
                    "release": lambda i: i["first_air_date"] or ""}[sort]
        items.sort(key=sort_key, reverse=True)
        size = 20
        return {"source": SOURCE, "page": page, "total_pages": max(1, -(-len(items) // size)), "total_results": len(items),
                "items": items[(page - 1) * size: page * size]}

    params: dict = {"sort_by": SORTS[sort], "without_genres": "10763|10764|10767|99"}
    if genre:
        params["with_genres"] = genre
    if year:
        params["first_air_date_year"] = year
    if sort == "rating":
        params["vote_count.gte"] = 50              # a score from a handful of votes is not a ranking
    if sort == "release":
        params["first_air_date.lte"] = date.today().isoformat()
    data = await tmdb("/discover/tv", **KOREAN_DRAMA, **params, page=page)
    return {"source": SOURCE, "page": data.get("page", page), "total_pages": min(data.get("total_pages", 1), 500),
            "total_results": data.get("total_results", 0),
            "items": [card(r, names) for r in data.get("results", [])]}


async def drama(drama_id: int) -> dict:
    """Everything the details page shows, copied from TMDB, plus the matching row of our ML dataset."""
    names = await genre_map()
    d = await tmdb(f"/tv/{drama_id}", language="en-US", include_image_language="en,ko,null",
                   include_video_language="en,ko",
                   append_to_response="aggregate_credits,videos,images,recommendations,similar,alternative_titles,content_ratings")

    cast = []
    for person in (d.get("aggregate_credits", {}).get("cast") or [])[:18]:
        roles = [r.get("character") for r in person.get("roles", []) if r.get("character")]
        cast.append({"id": person["id"], "name": person.get("name"), "original_name": person.get("original_name"),
                     "profile_path": person.get("profile_path"), "character": roles[0] if roles else None,
                     "episodes": person.get("total_episode_count")})

    videos = [v for v in d.get("videos", {}).get("results", []) if v.get("site") == "YouTube" and v.get("key")]
    rank = {"Trailer": 0, "Teaser": 1}
    videos.sort(key=lambda v: (rank.get(v.get("type"), 2), not v.get("official", False)))
    trailer = videos[0] if videos and videos[0].get("type") in rank else None

    images = d.get("images", {})
    gallery = [{"file_path": i["file_path"], "width": i.get("width"), "height": i.get("height"), "kind": "backdrop", "caption": "Official image"}
               for i in (images.get("backdrops") or [])[:24]]
    posters = [{"file_path": i["file_path"], "width": i.get("width"), "height": i.get("height"), "kind": "poster", "caption": "Poster"}
               for i in (images.get("posters") or [])[:8]]

    related = [r for r in d.get("recommendations", {}).get("results", []) if is_korean(r)]
    if len(related) < 6:
        ids = {r["id"] for r in related}
        related += [r for r in d.get("similar", {}).get("results", []) if is_korean(r) and r["id"] not in ids]

    runtimes = d.get("episode_run_time") or []
    runtime = runtimes[0] if runtimes else (d.get("last_episode_to_air") or {}).get("runtime")
    first_air = d.get("first_air_date") or None
    year = int(first_air[:4]) if first_air else None
    rating_kr = next((r.get("rating") for r in d.get("content_ratings", {}).get("results", [])
                      if r.get("iso_3166_1") == "KR" and r.get("rating")), None)
    alt_titles = [t.get("title") for t in d.get("alternative_titles", {}).get("results", []) if t.get("title")]

    return {
        **card(d, names),
        "tagline": d.get("tagline") or None,
        "status": d.get("status") or None,
        "last_air_date": d.get("last_air_date") or None,
        "number_of_episodes": d.get("number_of_episodes"),
        "number_of_seasons": d.get("number_of_seasons"),
        "runtime": runtime,
        "networks": [{"name": n.get("name"), "logo_path": n.get("logo_path")} for n in d.get("networks", [])],
        "content_rating_kr": rating_kr,
        "homepage": d.get("homepage") or None,
        "cast": cast,
        "trailer": {"key": trailer["key"], "name": trailer.get("name"), "type": trailer.get("type")} if trailer else None,
        "gallery": gallery,
        "posters": posters,
        "related": [card(r, names) for r in related[:14]],
        "related_criterion": "Korean series that TMDB recommends or lists as similar",
        "source_url": f"https://www.themoviedb.org/tv/{drama_id}",
        # The row of the MyDramaList dataset our model was trained on (None when the drama is not in it).
        "mdl_match": mdl_lookup([d.get("name"), d.get("original_name"), *alt_titles], year),
    }
