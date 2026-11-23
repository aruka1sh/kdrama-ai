"""Tests of the key-free TVmaze provider. A tiny fake index and a fake TVmaze server are used,
so the tests check our filtering, sorting and shaping logic without the network."""
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


def entry(i, name, genres, premiered, rating, weight, poster=True):
    return {"id": i, "name": name, "genres": genres, "status": "Ended", "premiered": premiered, "ended": None,
            "runtime": 70, "rating": rating, "weight": weight, "summary": f"About {name}.",
            "poster": f"https://static.tvmaze.com/uploads/images/original_untouched/1/{i}.jpg" if poster else None,
            "poster_small": None, "network": "tvN", "url": f"https://www.tvmaze.com/shows/{i}"}


INDEX = [
    entry(1, "Vincenzo", ["Drama", "Comedy", "Crime"], "2021-02-20", 8.2, 95),
    entry(2, "Quiet Romance", ["Romance"], "2019-05-01", 8.9, 20),
    entry(3, "Ghost Hotel", ["Fantasy", "Romance"], "2019-07-13", 7.9, 90),
    entry(4, "No Poster Show", ["Drama"], "2018-01-01", None, 10, poster=False),
    entry(5, "Future Drama", ["Drama"], "2999-01-01", None, 50),
]


def fake_tvmaze(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/shows/1":
        return httpx.Response(200, json={
            "id": 1, "name": "Vincenzo", "url": "https://www.tvmaze.com/shows/1", "status": "Ended", "averageRuntime": 85,
            "ended": "2021-05-02", "officialSite": None, "network": {"name": "tvN"}, "_embedded": {
                "cast": [{"person": {"id": 9, "name": "Song Joong Ki", "image": {"medium": "https://img/m.jpg"}}, "character": {"name": "Vincenzo Cassano"}},
                         {"person": {"id": 9, "name": "Song Joong Ki", "image": None}, "character": {"name": "Park Joo Hyung"}}],
                "images": [{"type": "poster", "resolutions": {"original": {"url": "https://img/poster.jpg"}}}],
                "episodes": [{"type": "regular", "season": 1, "number": 1, "name": "Episode 1", "image": {"original": "https://img/still1.jpg"}},
                             {"type": "regular", "season": 1, "number": 2, "name": "Episode 2", "image": None}],
                "akas": [{"name": "빈센조"}, {"name": "Binsenjo"}]}})
    if request.url.path == "/search/shows":
        return httpx.Response(200, json=[])
    return httpx.Response(404, json={})


@pytest.fixture(autouse=True)
def tvmaze_only(monkeypatch, tmp_path):
    monkeypatch.delenv("TMDB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(tmdb, "ENV_FILE", tmp_path / "missing.env")
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"dramas": INDEX}))
    monkeypatch.setattr(tvmaze, "INDEX_FILE", index_file)
    monkeypatch.setattr(tvmaze, "transport", httpx.MockTransport(fake_tvmaze))
    tvmaze.clear_cache()
    yield
    tvmaze.clear_cache()


def test_status_names_tvmaze_and_its_licence():
    body = client.get("/api/catalog/status").json()
    assert body["configured"] and body["source"] == "TVmaze" and "CC BY-SA" in body["attribution"]


def test_search_filters_and_sorting_run_on_the_index():
    names = lambda **p: [i["name"] for i in client.get("/api/catalog/discover", params=p).json()["items"]]  # noqa: E731
    assert names()[:2] == ["Vincenzo", "Ghost Hotel"]                       # popularity (weight)
    assert names(sort="rating") == ["Quiet Romance", "Vincenzo", "Ghost Hotel"]   # unrated dramas are left out
    assert names(sort="release")[0] == "Vincenzo" and "Future Drama" not in names(sort="release")
    assert names(q="ghost") == ["Ghost Hotel"]
    assert names(genre="Romance") == ["Ghost Hotel", "Quiet Romance"]
    assert names(year=2019, genre="Romance", q="quiet") == ["Quiet Romance"]
    assert client.get("/api/catalog/discover", params={"q": "zzz"}).json()["total_results"] == 0


def test_cards_use_prefixed_ids_and_never_invent_votes():
    item = client.get("/api/catalog/discover").json()["items"][0]
    assert item["id"] == "tvmaze-1" and item["source"] == "TVmaze" and item["vote_count"] == 0
    assert item["poster_path"].startswith("https://static.tvmaze.com/")


def test_home_collections_have_criteria_and_real_backdrops():
    body = client.get("/api/catalog/home").json()
    titles = {c["title"]: c for c in body["collections"]}
    assert [i["name"] for i in titles["Popular Romance"]["items"]] == ["Ghost Hotel", "Quiet Romance"]
    assert [i["name"] for i in titles["Hidden Gems"]["items"]] == ["Quiet Romance"]
    assert all("No Poster Show" not in [i["name"] for i in c["items"]] for c in body["collections"])
    assert all(c["criterion"] for c in body["collections"])
    assert [h["name"] for h in body["hero"]] == ["Vincenzo"]               # the only drama with a landscape image
    assert body["hero"][0]["backdrop_path"] == "https://img/still1.jpg"


def test_drama_details_come_from_tvmaze():
    d = client.get("/api/catalog/drama/tvmaze-1").json()
    assert d["original_name"] == "빈센조" and d["runtime"] == 85 and d["number_of_episodes"] == 2
    assert d["cast"] == [{"id": 9, "name": "Song Joong Ki", "original_name": None, "profile_path": "https://img/m.jpg",
                          "character": "Vincenzo Cassano", "episodes": None}]
    assert [g["file_path"] for g in d["gallery"]] == ["https://img/still1.jpg"] and "still" in d["gallery"][0]["caption"]
    assert d["trailer"] is None                                             # TVmaze has no trailers: nothing is made up
    assert [r["name"] for r in d["related"]] == ["Future Drama"]            # shares "Drama"; no shared genre -> not related
    assert d["mdl_match"]["title"] == "Vincenzo" and d["mdl_match"]["popularity_rank"] == 17


def test_unknown_ids_are_404():
    assert client.get("/api/catalog/drama/tvmaze-999").status_code == 404
    assert client.get("/api/catalog/drama/not-an-id").status_code == 404
