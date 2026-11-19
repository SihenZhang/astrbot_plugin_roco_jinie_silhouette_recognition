import hashlib
import json
import struct
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import image_io
import recognition
from reply import (
    NOT_FOUND_REPLIES,
    PAIR_REPLIES,
    SINGLE_REPLIES,
    build_reply,
)


def encode_image(image, extension=".png", params=()):
    success, encoded = cv2.imencode(extension, image, params)
    assert success
    return encoded.tobytes()


def test_relocated_jpeg_card_and_incremental_form_index(tmp_path):
    directory = tmp_path / "sprites" / "001"
    directory.mkdir(parents=True)
    entries = []
    for form in ("round", "two-parts"):
        sprite = np.zeros((96, 96, 4), np.uint8)
        cv2.ellipse(sprite, (39, 44), (24, 24), 0, 0, 360, (0, 0, 255, 255), -1)
        if form == "two-parts":
            triangle = np.array([(36, 30), (89, 10), (55, 42)], np.int32)
            cv2.fillConvexPoly(sprite, triangle, (0, 128, 0, 255))
            cv2.circle(sprite, (82, 72), 5, (0, 128, 0, 255), -1)
        path = directory / (form + ".png")
        path.write_bytes(encode_image(sprite))
        entries.append(
            {
                "number": "001",
                "name": "测试精灵",
                "form": form,
                "path": "001/" + path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    # The colored card can sit anywhere inside a forwarded screenshot.
    card = np.full((300, 420, 3), (123, 47, 51), np.uint8)
    cv2.rectangle(card, (205, 19), (359, 205), (232, 150, 54), -1)
    card[44:140, 232:328][sprite[:, :, 3] > 0] = (43, 33, 32)
    jpeg = encode_image(card, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 84])
    manifest = directory.parent / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "entries": entries}), encoding="utf-8"
    )
    index = recognition.load_index(tmp_path)
    match = index.match(jpeg)
    assert (match.name, match.form) == ("测试精灵", "two-parts")
    assert match.score > 0.85
    matches = index.match_candidates(jpeg)
    assert len(matches) == 1
    reply = build_reply(matches)
    assert reply.text.count("测试精灵") == 1
    assert "two-parts" not in reply.text
    assert reply.image == (directory / "two-parts.png").read_bytes()

    # Rename a form and drop another while keeping the same image hash.
    entries[1]["name"] = "新名字"
    manifest.write_text(
        json.dumps({"version": 1, "entries": entries[1:]}), encoding="utf-8"
    )
    with patch.object(recognition, "_sprite_mask", side_effect=AssertionError):
        updated = recognition.load_index(tmp_path)
    assert len(updated) == 1
    assert updated.match(jpeg).name == "新名字"
    assert index.match(jpeg).name == "测试精灵"
    (tmp_path / recognition.CACHE_NAME).write_bytes(b"interrupted")
    assert len(recognition.load_index(tmp_path)) == 1


def test_same_name_forms_leave_room_for_an_ambiguous_second_name(tmp_path):
    directory = tmp_path / "sprites" / "001"
    directory.mkdir(parents=True)
    entries = []
    for name, color in [("甲", (0, 0, 255, 255)), ("乙", (255, 0, 0, 255))]:
        art = np.zeros((120, 80, 4), np.uint8)
        cv2.ellipse(art, (39, 69), (25, 33), 0, 0, 360, color, -1)
        path = directory / f"{name}.png"
        path.write_bytes(encode_image(art))
        # The tie must survive both shortlists, even with many duplicate names.
        for form in range(51 if name == "甲" else 1):
            entries.append(
                {
                    "number": "001",
                    "name": name,
                    "form": str(form),
                    "path": f"001/{path.name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = directory.parent / "manifest.json"
    manifest.write_text(json.dumps({"version": 2, "entries": entries}), "utf-8")
    card = np.full((210, 180, 3), (128, 0, 128), np.uint8)
    cv2.rectangle(card, (18, 20), (162, 189), (232, 150, 54), -1)
    card[42:162, 61:141][art[:, :, 3] > 0] = (41, 32, 32)
    data = encode_image(card)
    index = recognition.load_index(tmp_path)
    matches = index.match_candidates(data)
    assert [m.name for m in matches] == ["甲", "乙"]
    assert matches[1].score > 0.95
    reply = build_reply(matches)
    assert 0 <= reply.text.find("甲") < reply.text.find("乙")
    combined = cv2.imdecode(np.frombuffer(reply.image, np.uint8), cv2.IMREAD_COLOR)
    h, w = combined.shape[:2]
    assert tuple(combined[h // 2, w // 4]) == (0, 0, 255)
    assert tuple(combined[h // 2, 3 * w // 4]) == (255, 0, 0)

    # Keep only the raw winner when both candidates score poorly.
    with patch.object(recognition, "_aligned_score", return_value=0.6):
        assert len(index.match_candidates(data)) == 1

    entries[0].update(id="demo-shiny", name="旧异色")
    manifest.write_text(json.dumps({"version": 1, "entries": entries}), "utf-8")
    assert len(recognition.load_index(tmp_path)) == len(entries) - 1


def test_reply_templates_display_candidate_names_in_order():
    for template in SINGLE_REPLIES:
        assert template.format(name="精灵甲").count("精灵甲") == 1, template
    for template in PAIR_REPLIES:
        text = template.format(first="精灵甲", second="精灵乙")
        assert 0 <= text.find("精灵甲") < text.find("精灵乙"), template


def test_below_threshold_reply_does_not_load_artwork(tmp_path):
    match = recognition.Match("009", "波波拉", "", tmp_path / "missing.png", 0.779)
    reply = build_reply([match])
    assert reply.text in NOT_FOUND_REPLIES
    assert reply.image is None
    assert not reply.matched


def test_score_at_threshold_is_accepted(tmp_path):
    match = recognition.Match("009", "波波拉", "", tmp_path / "missing.png", 0.78)
    reply = build_reply([match], send_image=False)
    assert "波波拉" in reply.text
    assert reply.matched


def test_blank_card_without_silhouette_is_rejected():
    blank = np.full((180, 240, 3), (232, 150, 54), np.uint8)
    with pytest.raises(recognition.RecognitionError):
        recognition.extract_silhouette(encode_image(blank))


@pytest.mark.parametrize(
    "extension,quality,alpha",
    [
        pytest.param(".bmp", None, False, id="bmp"),
        pytest.param(".gif", None, False, id="gif"),
        pytest.param(".webp", 85, False, id="lossy-webp"),
        pytest.param(".webp", 101, False, id="lossless-webp"),
        pytest.param(".webp", 85, True, id="extended-webp"),
    ],
)
def test_card_header_dimensions_match_decoded_pixels(extension, quality, alpha):
    image = np.zeros((17, 29, 4 if alpha else 3), np.uint8)
    image[3:11, 5:21] = (0, 0, 255, 255) if alpha else (0, 0, 255)
    params = [cv2.IMWRITE_WEBP_QUALITY, quality] if quality is not None else []
    data = encode_image(image, extension, params)
    assert image_io._card_dimensions(data) == (29, 17)
    decoded = image_io.decode_card(data)
    assert decoded.shape == (17, 29, 3)
    assert decoded[7, 13, 2] > 200


def test_oversized_jpeg_card_rejected_before_decoding(monkeypatch):
    jpeg = encode_image(np.zeros((4001, 5000), np.uint8), ".jpg")
    monkeypatch.setattr(
        cv2, "imdecode", lambda *args: pytest.fail("decoded oversized card")
    )
    with pytest.raises(recognition.RecognitionError):
        recognition.extract_silhouette(jpeg)


@pytest.mark.parametrize("extension", [".jpg", ".png", ".webp"])
def test_exif_orientation_rotates_the_silhouette(extension):
    card = np.full((150, 160, 3), (232, 150, 54), np.uint8)
    cv2.rectangle(card, (48, 36), (60, 98), (43, 33, 32), -1)
    cv2.rectangle(card, (60, 86), (93, 98), (43, 33, 32), -1)
    raw_mask = recognition.extract_silhouette(encode_image(card, extension))
    # A little-endian TIFF IFD with EXIF orientation 6 (90 degrees clockwise).
    exif = b"II" + struct.pack("<HIHHHIHHI", 42, 8, 1, 0x112, 3, 1, 6, 0, 0)
    ok, tagged = cv2.imencodeWithMetadata(
        extension, card, [cv2.IMAGE_METADATA_EXIF], [np.frombuffer(exif, np.uint8)]
    )
    assert ok
    np.testing.assert_array_equal(
        recognition.extract_silhouette(tagged.tobytes()), np.rot90(raw_mask, -1)
    )


def test_translucent_reply_art_blends_over_white(tmp_path):
    sprite = np.zeros((6, 10, 4), np.uint8)
    sprite[1:5, 2:8] = (0, 0, 255, 128)
    path = tmp_path / "半透明.png"
    path.write_bytes(encode_image(sprite))
    matches = [recognition.Match("001", name, "", path, 1.0) for name in ("甲", "乙")]
    image = cv2.imdecode(
        np.frombuffer(build_reply(matches).image, np.uint8), cv2.IMREAD_COLOR
    )
    assert image.shape == (512, 1024, 3)
    for x in (256, 768):
        assert tuple(image[256, x]) == (127, 127, 255)
        assert tuple(image[256, x - 4]) == (255, 255, 255)


def test_translucent_appendage_survives_segmentation():
    card = np.full((180, 160, 3), (123, 47, 51), np.uint8)
    cv2.rectangle(card, (15, 10), (140, 160), (230, 150, 52), -1)
    cv2.ellipse(card, ((64, 84.5), (62, 69), 0), (44, 34, 32), -1)
    # 60% opacity dark fill over the blue background, as on the sample card.
    cv2.rectangle(card, (105, 74), (122, 86), (118, 80, 40), -1)
    mask = recognition.extract_silhouette(encode_image(card))
    assert mask.shape == (70, 90)
