from __future__ import annotations

from pathlib import Path

from finagent.llm.cache import DiskCache, make_key


def test_make_key_is_stable_and_order_independent() -> None:
    assert make_key(a=1, b="x") == make_key(b="x", a=1)
    assert make_key(a=1) != make_key(a=2)


def test_make_key_hashes_bytes() -> None:
    assert make_key(img=b"abc") == make_key(img=b"abc")
    assert make_key(img=b"abc") != make_key(img=b"abd")


def test_cache_roundtrip_and_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("deadbeef") is None
    cache.set("deadbeef", {"x": [1, 2]})
    assert cache.get("deadbeef") == {"x": [1, 2]}


def test_corrupt_entry_is_a_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("abcd", "ok")
    (tmp_path / "ab" / "abcd.json").write_text("{not json", encoding="utf-8")
    assert cache.get("abcd") is None
