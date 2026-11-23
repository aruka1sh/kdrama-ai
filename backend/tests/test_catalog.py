"""Tests of the TMDB proxy. TMDB itself is replaced by a small fake server (httpx.MockTransport),
so these tests check OUR logic (filtering, shaping, errors, secrecy) without a real API key.
The fake payloads exist only in this test file - the website never shows them."""
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import main, tmdb, tvmaze  # noqa: E402

client = TestClient(main.app)
SECRET = "test-secret-token"
calls: list[httpx.Request] = []


def show(i, name, korean=True, **extra):
    return {"id": i, "name": name, "original_name": name + " (ko)", "overview": "Synopsis of " + name,
            "poster_path": f"/p{i}.jpg", "backdrop_path": f"/b{i}.jpg", "first_air_date": "2021-02-20",
            "genre_ids": [18, 35], "vote_average": 8.5, "vote_count": 900, "popularity": 50.0 + i,
            "origin_country": ["KR"] if korean else ["US"], "original_language": "ko" if korean else "en", **extra}


def fake_tmdb(request: httpx.Request) -> httpx.Response:
    calls.append(request)
    if request.headers.get("authorization") != f"Bearer {SECRET}":
        return httpx.Response(401, json={"status_message": "Invalid API key"})
    path, q = request.url.path, dict(request.url.params)
    if path.endswith("/genre/tv/list"):
        return httpx.Response(200, json={"genres": [{"id": 18, "name": "Drama"}, {"id": 35, "name": "Comedy"},
                                                     {"id": 9648, "name": "Mystery"}, {"id": 10764, "name": "Reality"}]})
    if path.endswith("/search/keyword"):
        return httpx.Response(200, json={"results": [{"id": 9840, "name": "romance"}]})
    if path.endswith("/search/tv"):
        return httpx.Response(200, json={"results": [show(1, "Vincenzo"), show(2, "Vincenzo Remake", korean=False)]
                                         if q.get("page", "1") == "1" else []})
    if path.endswith("/trending/tv/week"):
        return httpx.Response(200, json={"results": [show(10 + int(q["page"]), f"Trend {q['page']}"), show(99, "US Show", korean=False)]})
    if path.endswith("/discover/tv"):
        return httpx.Response(200, json={"page": int(q.get("page", 1)), "total_pages": 7, "total_results": 140,
                                         "results": [show(20, "Discovered")]})
    if path.endswith("/tv/1"):
        return httpx.Response(200, json={**show(1, "Vincenzo"), "genres": [{"id": 18, "name": "Drama"}], "genre_ids": None,
            "number_of_episodes": 20, "number_of_seasons": 1, "episode_run_time": [], "last_episode_to_air": {"runtime": 85},
            "status": "Ended", "tagline": "", "networks": [{"name": "tvN", "logo_path": "/tvn.png"}],
            "aggregate_credits": {"cast": [{"id": 5, "name": "Song Joong-ki", "original_name": "송중기", "profile_path": "/sjk.jpg",
                                            "roles": [{"character": "Vincenzo Cassano"}], "total_episode_count": 20}]},
            "videos": {"results": [{"site": "YouTube", "key": "clip1", "type": "Clip", "name": "Clip"},
                                   {"site": "YouTube", "key": "trailer1", "type": "Trailer", "name": "Official Trailer", "official": True}]},
            "images": {"backdrops": [{"file_path": "/still1.jpg", "width": 1920, "height": 1080}], "posters": [{"file_path": "/poster1.jpg"}]},
            "recommendations": {"results": [show(3, "Related KR"), show(4, "Related US", korean=False)]},
            "similar": {"results": []}, "alternative_titles": {"results": []},
            "content_ratings": {"results": [{"iso_3166_1": "KR", "rating": "15"}]}})
    return httpx.Response(404, json={"status_message": "not found"})


@pytest.fixture(autouse=True)
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("TMDB_ACCESS_TOKEN", SECRET)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(tmdb, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(tmdb, "transport", httpx.MockTransport(fake_tmdb))
    tmdb.clear_cache()
    calls.clear()
    yield
    tmdb.clear_cache()


def test_not_configured_is_an_honest_503(monkeypatch, tmp_path):
    monkeypatch.delenv("TMDB_ACCESS_TOKEN")
    monkeypatch.setattr(tvmaze, "INDEX_FILE", tmp_path / "no-index.json")     # neither provider is available
    assert client.get("/api/catalog/status").json()["configured"] is False
    response = client.get("/api/catalog/home")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "catalog_not_configured"
    assert not calls                                   # TMDB is not even contacted


def test_bad_credential_is_reported(monkeypatch):
    monkeypatch.setenv("TMDB_ACCESS_TOKEN", "wrong")
    response = client.get("/api/catalog/discover")
    assert response.status_code == 503 and response.json()["detail"]["code"] == "tmdb_bad_credential"


def test_secret_never_reaches_the_browser():
    for path in ["/api/catalog/status", "/api/catalog/home", "/api/catalog/discover", "/api/catalog/drama/1"]:
        assert SECRET not in client.get(path).text


def test_home_has_real_collections_and_only_korean_titles():
    body = client.get("/api/catalog/home").json()
    titles = [c["title"] for c in body["collections"]]
    assert {"Trending K-Dramas", "Popular Romance", "Thriller & Mystery", "Fantasy & Supernatural",
            "Highly Rated", "Hidden Gems", "Recently Released"} <= set(titles)
    assert all(c["criterion"] for c in body["collections"])
    names = [i["name"] for c in body["collections"] for i in c["items"]]
    assert "US Show" not in names and "Vincenzo Remake" not in names
    assert body["hero"] and all(h["backdrop_path"] and h["overview"] for h in body["hero"])


def test_discover_passes_filters_to_tmdb():
    client.get("/api/catalog/discover", params={"genre": 9648, "year": 2021, "sort": "rating", "page": 3})
    q = dict(next(c for c in calls if c.url.path.endswith("/discover/tv")).url.params)
    assert q["with_genres"] == "9648" and q["first_air_date_year"] == "2021" and q["page"] == "3"
    assert q["sort_by"] == "vote_average.desc" and q["vote_count.gte"] == "50"
    assert q["with_origin_country"] == "KR" and q["with_original_language"] == "ko"


def test_search_keeps_korean_results_and_applies_filters():
    body = client.get("/api/catalog/discover", params={"q": "vincenzo"}).json()
    assert [i["name"] for i in body["items"]] == ["Vincenzo"]
    assert client.get("/api/catalog/discover", params={"q": "vincenzo", "year": 1999}).json()["items"] == []
    assert client.get("/api/catalog/discover", params={"q": "vincenzo", "genre": 9648}).json()["items"] == []


def test_drama_details_are_copied_from_tmdb():
    d = client.get("/api/catalog/drama/1").json()
    assert d["name"] == "Vincenzo" and d["original_name"] == "Vincenzo (ko)" and d["year"] == 2021
    assert d["number_of_episodes"] == 20 and d["runtime"] == 85 and d["content_rating_kr"] == "15"
    assert d["source"] == "TMDB" and d["source_url"].endswith("/tv/1")
    assert d["cast"][0] == {"id": 5, "name": "Song Joong-ki", "original_name": "송중기", "profile_path": "/sjk.jpg",
                            "character": "Vincenzo Cassano", "episodes": 20}
    assert d["trailer"]["key"] == "trailer1"            # the trailer wins over the clip
    assert d["gallery"][0]["file_path"] == "/still1.jpg"
    assert [r["name"] for r in d["related"]] == ["Related KR"]
    assert d["tagline"] is None                         # empty values stay empty


def test_drama_is_matched_with_the_training_dataset():
    match = client.get("/api/catalog/drama/1").json()["mdl_match"]
    assert match["title"] == "Vincenzo" and match["year"] == 2021
    assert match["popularity_rank"] == 17 and match["public_rating"] == 9.0


def test_unknown_drama_is_404():
    assert client.get("/api/catalog/drama/424242").status_code == 404


def test_responses_are_cached():
    client.get("/api/catalog/drama/1")
    first = len(calls)
    client.get("/api/catalog/drama/1")
    assert len(calls) == first
