"""Drama catalogue API used by the website.

Two interchangeable providers return the same JSON shape:
  * TMDB   - used when a TMDB credential is configured in backend/.env (posters, cast, trailers, ...)
  * TVmaze - needs no key; used otherwise, from a local index plus live detail requests

The website never talks to either service directly, and neither provider invents data.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import tmdb, tvmaze

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


def provider() -> str:
    if tmdb.is_configured():
        return "tmdb"
    if tvmaze.available():
        return "tvmaze"
    raise HTTPException(status_code=503, detail={
        "code": "catalog_not_configured",
        "message": "The drama catalogue is not set up. Run 'python backend/scripts/build_tvmaze_index.py' "
                   "(no key needed) or add a TMDB credential to backend/.env, then restart the backend. "
                   "No drama data is shown until then, because this site never invents titles, posters or ratings."})


@router.get("/status")
def status() -> dict:
    try:
        name = provider()
    except HTTPException:
        return {"configured": False, "source": None, "attribution": None, "source_url": None}
    module = tmdb if name == "tmdb" else tvmaze
    return {"configured": True, "source": module.SOURCE, "attribution": module.ATTRIBUTION,
            "source_url": "https://www.themoviedb.org" if name == "tmdb" else "https://www.tvmaze.com"}


@router.get("/genres")
async def genres() -> dict:
    return await tmdb.genres() if provider() == "tmdb" else tvmaze.genres()


@router.get("/home")
async def home() -> dict:
    """Hero dramas and the homepage collections. Each collection states its real selection rule."""
    return await tmdb.home() if provider() == "tmdb" else await tvmaze.home(tmdb.ICONIC_TITLES)


@router.get("/discover")
async def discover(q: str = Query(default="", max_length=80), genre: str | None = Query(default=None, max_length=40),
                   year: int | None = Query(default=None, ge=1960, le=2100),
                   sort: str = Query(default="popularity", pattern="^(popularity|rating|release)$"),
                   page: int = Query(default=1, ge=1, le=500)) -> dict:
    """Catalogue page: title search, genre and year filters, sorting and paging on real data."""
    if provider() == "tmdb":
        if genre and not genre.isdigit():
            raise HTTPException(status_code=422, detail="For TMDB the genre must be a numeric genre id.")
        return await tmdb.explore(q, int(genre) if genre else None, year, sort, page)
    return await tvmaze.discover(q, genre or None, year, sort, page)


@router.get("/drama/{drama_id}")
async def drama(drama_id: str) -> dict:
    """Details of one drama. The id itself says which provider it belongs to (tvmaze-123 or a TMDB number)."""
    if drama_id.startswith(tvmaze.PREFIX) and drama_id.removeprefix(tvmaze.PREFIX).isdigit():
        return await tvmaze.drama(drama_id, tmdb.mdl_lookup)
    if drama_id.isdigit():
        return await tmdb.drama(int(drama_id))
    raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Unknown drama id."})
