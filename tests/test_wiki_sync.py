import hashlib
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import image_io
import wiki_sync


def test_grayscale_png_uses_trns_key_for_sprite_alpha():
    pixels = np.array([[41, 23, 0]], np.uint8)
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    # Insert a transparency key after IHDR. Only pixels with gray value 23 are clear.
    key_chunk = b"tRNS" + struct.pack(">H", 23)
    trns = struct.pack(">I", 2) + key_chunk + struct.pack(">I", zlib.crc32(key_chunk))
    png = encoded.tobytes()
    sprite = image_io.decode_sprite(png[:33] + trns + png[33:])
    np.testing.assert_array_equal(sprite[0, :, 3], [255, 0, 255])
    np.testing.assert_array_equal(sprite[:, :, 0], pixels)


def test_oversized_sprite_rejected_before_decoding(monkeypatch):
    # A valid grayscale PNG just above the 16-million-pixel sprite limit.
    success, encoded = cv2.imencode(".png", np.zeros((4001, 4000), np.uint8))
    assert success
    monkeypatch.setattr(
        cv2, "imdecode", lambda *args: pytest.fail("decoded oversized sprite")
    )
    with pytest.raises(ValueError):
        wiki_sync._verify_png(encoded.tobytes())


def wiki_page(second_file="second"):
    cards = []
    for pet_id, form, file in [
        ("pet_1", "", "first"),
        ("pet_2", "地域形态", second_file),
    ]:
        if file is None:
            continue
        cards.append(f"""
          <div class="npc-card" data-number="011" data-id="{pet_id}">
            <div class="npc-name">鸭吉吉</div><div class="npc-form">{form}</div>
            <div class="npc-art-normal"><img src="https://patchwiki.biligame.com/images/nrc/{file}.png"></div>
            <div class="npc-art-shiny"><img src="https://patchwiki.biligame.com/images/nrc/shiny.png"></div>
          </div>""")
    return (
        '<div class="nrc-pet-catalog"><span class="npc-total-number">2</span>'
        + "".join(cards)
        + "</div>"
    ).encode()


async def test_incremental_sync_and_failed_replacement(tmp_path, monkeypatch):
    image = np.zeros((3, 3, 4), np.uint8)
    image[1, 1, 3] = 255
    success, encoded = cv2.imencode(".png", image)
    assert success
    sprite_bytes = encoded.tobytes()
    page = wiki_page()
    requests = []

    async def fetch(session, url, limit):
        requests.append(url)
        if url == wiki_sync.CATALOG_URL:
            return page
        if url.endswith("/missing.png"):
            return b"<html>Image unavailable</html>"
        return sprite_bytes

    monkeypatch.setattr(wiki_sync, "_fetch", fetch)
    first = await wiki_sync.sync_sprites(tmp_path)
    assert (first.total, first.downloaded, first.failed) == (2, 2, 0)
    sprites = tmp_path / "sprites"
    manifest = sprites / "manifest.json"
    before = manifest.read_bytes()
    catalog = json.loads(before)
    rows = catalog["entries"]
    assert catalog["version"] == 2
    assert [row["form"] for row in rows] == ["", "地域形态"]
    assert rows[0]["path"] != rows[1]["path"]
    assert [row["id"] for row in rows] == ["pet_1", "pet_2"]
    assert all("-normal" not in row["path"] for row in rows)
    assert not any(url.endswith("shiny.png") for url in requests)

    # An interrupted first run can leave complete PNGs before manifest publication.
    manifest.unlink()
    legacy_files = []
    for row in rows:
        normal = wiki_sync.Sprite(
            f"{row['id']}-normal", row["number"], row["name"], row["form"], row["url"]
        )
        shiny = wiki_sync.Sprite(
            f"{row['id']}-shiny",
            row["number"],
            row["name"],
            row["form"],
            "https://patchwiki.biligame.com/images/nrc/shiny.png",
        )
        (sprites / row["path"]).rename(sprites / normal.path)
        (sprites / shiny.path).write_bytes(sprite_bytes)
        legacy_files.extend([normal.path, shiny.path])
    unknown = sprites / "011" / "pet_1-shiny-0000000000000000.png"
    unknown.write_bytes(b"user")
    requests.clear()
    resumed = await wiki_sync.sync_sprites(tmp_path)
    assert (resumed.total, resumed.downloaded) == (2, 0)
    assert requests == [wiki_sync.CATALOG_URL]
    assert not any((sprites / path).exists() for path in legacy_files)
    assert unknown.read_bytes() == b"user"

    requests.clear()
    unchanged = await wiki_sync.sync_sprites(tmp_path)
    assert requests == [wiki_sync.CATALOG_URL]
    assert not unchanged.changed

    page = wiki_page(second_file=None)
    with pytest.raises(ValueError):
        await wiki_sync.sync_sprites(tmp_path)
    assert manifest.read_bytes() == before

    page = wiki_page("missing")
    partial = await wiki_sync.sync_sprites(tmp_path)
    assert (partial.total, partial.failed) == (2, 1)
    assert manifest.read_bytes() == before
    assert (sprites / rows[1]["path"]).read_bytes() == sprite_bytes

    (sprites / rows[0]["path"]).unlink()
    page = wiki_page()
    recovered = await wiki_sync.sync_sprites(tmp_path)
    assert (recovered.total, recovered.downloaded) == (2, 1)
    assert recovered.changed


def test_offline_migration_resumes_after_publication(tmp_path, monkeypatch):
    sprites = tmp_path / "sprites"
    folder = sprites / "011"
    folder.mkdir(parents=True)
    manifest = sprites / "manifest.json"
    unmanaged = folder / "user.png"
    unmanaged.write_bytes(b"user")
    entries = []
    for kind, data in [("normal", b"original"), ("shiny", b"recolored")]:
        sprite = wiki_sync.Sprite(
            f"pet_1-{kind}", "011", "鸭吉吉", "", f"https://example.com/{kind}.png"
        )
        entries.append(sprite.entry(hashlib.sha256(data).hexdigest(), len(data)))
        (sprites / sprite.path).write_bytes(data)
    manifest.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf8")

    original_unlink = Path.unlink

    def interrupt_unlink(path, *args, **kwargs):
        if path == sprites / entries[1]["path"]:
            raise PermissionError("模拟中断")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt_unlink)
        with pytest.raises(PermissionError, match="模拟中断"):
            wiki_sync.migrate_sprites(tmp_path)
    upgraded = json.loads(manifest.read_text(encoding="utf8"))
    assert upgraded["version"] == 2
    (normal,) = upgraded["entries"]
    assert normal["id"] == "pet_1"
    assert (sprites / normal["path"]).read_bytes() == b"original"
    assert wiki_sync.migrate_sprites(tmp_path)
    assert not wiki_sync.migrate_sprites(tmp_path)
    assert not any((sprites / old["path"]).exists() for old in entries)
    assert unmanaged.read_bytes() == b"user"
