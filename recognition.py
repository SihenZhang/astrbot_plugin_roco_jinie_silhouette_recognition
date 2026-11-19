"""Extract the game card silhouette and match it against the Wiki sprite alpha masks."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np

if __package__:
    from .image_io import decode_card, decode_sprite
else:
    from image_io import decode_card, decode_sprite

logger = logging.getLogger(__name__)
SIZE = 128
LOW_SIZE = 32
PADDING = 4  # Leave space for the final ±2 pixel translation.
# Bump the filename when changing alpha extraction or normalization.
CACHE_NAME = "silhouettes-v3.npz"
MIN_ALTERNATIVE_SCORE = 0.86
MAX_ALTERNATIVE_GAP = 0.035


class RecognitionError(ValueError):
    """The image or library cannot be used for matching."""


@dataclass(frozen=True)
class Sprite:
    number: str
    name: str
    form: str
    path: Path


@dataclass(frozen=True)
class Match(Sprite):
    score: float


def _trim(mask: np.ndarray) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        raise RecognitionError("图片中没有可用的轮廓")
    return mask[y : y + height, x : x + width]


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    # Float area coverage prevents a uint8 rounding bias on thin diagonals.
    coverage = cv2.resize(
        mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    return (coverage >= 0.5).astype(np.uint8)


def _normalize(mask: np.ndarray) -> np.ndarray:
    cropped = _trim(mask)
    # Tiny sparkles and thin translucent tips can disappear after downsampling.
    # Use central mass bounds for scale, discarding at most 0.3% per side.
    margin_mass = cropped.sum() * 0.003
    y_mass = cropped.sum(axis=1).cumsum()
    x_mass = cropped.sum(axis=0).cumsum()
    y0, y1 = y_mass.searchsorted([margin_mass, y_mass[-1] - margin_mass])
    x0, x1 = x_mass.searchsorted([margin_mass, x_mass[-1] - margin_mass])
    cropped = cropped[y0 : y1 + 1, x0 : x1 + 1]
    height, width = cropped.shape
    scale = (SIZE - 2 * PADDING) / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (SIZE - w) // 2, (SIZE - h) // 2
    normalized = np.zeros((SIZE, SIZE), dtype=np.uint8)
    normalized[y : y + h, x : x + w] = _resize_mask(cropped, w, h)
    if not np.any(normalized):
        raise RecognitionError("轮廓太细，无法标准化")
    return normalized


def _sprite_mask(path: Path) -> np.ndarray:
    alpha = decode_sprite(path.read_bytes())[:, :, 3]
    mask = (alpha >= 128).astype(np.uint8)
    # Some Wiki renders contain isolated 1–3 px specks in the outer margin of
    # a 1024 px canvas. Those disappear at game resolution but shift the box.
    _, labels, parts, _ = cv2.connectedComponentsWithStats(mask)
    if len(parts) > 1:
        areas = parts[:, cv2.CC_STAT_AREA]
        keep = areas >= areas[1:].max() * 0.0001
        keep[0] = False
        mask = keep[labels].astype(np.uint8)
    return _normalize(mask)


def extract_silhouette(image: bytes) -> np.ndarray:
    """Return the tightly cropped, 0/1 mask from an encoded game card."""
    try:
        bgr = decode_card(image)
    except ValueError as exc:
        raise RecognitionError(str(exc)) from exc

    # The cyan panel is brighter and greener than the purple card background.
    # Locate its connected area so a forwarded, resized or padded card still works.
    b, g, r = bgr.astype(np.int16).transpose(2, 0, 1)
    blue = ((b > 145) & (g - r > 40) & (b - g > 25)).astype(np.uint8)
    _, blue_labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    for label in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:5] + 1:
        x, y, w, h, area = stats[label]
        if min(w, h) < 24 or not 0.45 <= w / h <= 1.7 or area < w * h * 0.15:
            continue
        # Decorative corner rivets can join the panel after JPEG compression.
        # Solid frame rows and columns have much more blue coverage than a rivet.
        region = blue_labels[y : y + h, x : x + w] == label
        cols = np.flatnonzero(region.mean(axis=0) >= 0.30)
        rows = np.flatnonzero(region.mean(axis=1) >= 0.30)
        if not cols.size or not rows.size:
            continue
        panel = bgr[y + rows[0] : y + rows[-1] + 1, x + cols[0] : x + cols[-1] + 1]
        h, w = panel.shape[:2]
        # A translucent sprite blends its dark fill into the blue background.
        # Estimate this opacity so the query uses the same 50% cutoff as the PNG.
        light = panel.max(axis=2)
        solid = (light < 94) & (light - panel.min(axis=2) <= 42)
        if not np.any(solid):
            continue
        blue_panel = region[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        background = np.median(panel[blue_panel], axis=0)
        fill = np.median(panel[solid], axis=0)
        direction = (background - fill).astype(np.float32)
        midpoint = (background + fill) * 0.5
        dark = ((panel @ direction) <= midpoint @ direction).astype(np.uint8)
        count, labels, parts, _ = cv2.connectedComponentsWithStats(dark)
        # Rivets at the frame corners can be dark blue, even deep inside a JPEG.
        # Require nearby dark fill for corner features. Interior features (steam
        # and magic effects in particular) can be translucent and disconnected.
        radius = max(1, round(min(w, h) * 0.04))
        near_solid = cv2.dilate(
            solid.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)
        )
        near_labels = set(np.unique(labels[near_solid != 0]))
        keep = np.zeros(count, dtype=bool)
        for i in range(1, count):
            left, top, width, height, pixels = parts[i]
            touches_frame = (
                left == 0 or top == 0 or left + width == w or top + height == h
            )
            if touches_frame and pixels < w * h * 0.03:
                continue
            in_corner = (
                min(left, w - left - width) < 2 * radius
                and min(top, h - top - height) < 2 * radius
            )
            keep[i] = (not in_corner or i in near_labels) and pixels >= max(
                2, w * h * 0.00015
            )
        mask = keep[labels].astype(np.uint8)
        if 0.008 < mask.mean() < 0.85:
            return _trim(mask)
    raise RecognitionError("未在题图的蓝色画框内找到剪影")


def _iou(mask: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = np.count_nonzero(candidates & mask, axis=(1, 2))
    union = np.count_nonzero(candidates | mask, axis=(1, 2))
    return intersection / np.maximum(union, 1)


def _edge_distances(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge = mask - cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_5)
    return edge.astype(bool), distance


def _aligned_score(
    query: np.ndarray,
    query_edge: np.ndarray,
    query_distance: np.ndarray,
    candidate: np.ndarray,
) -> float:
    padded = np.pad(candidate, 2)
    # On binary masks cross-correlation gives intersection area at each offset.
    # The padding in each normalized mask makes the union computable from that.
    overlap = np.rint(cv2.matchTemplate(padded, query, cv2.TM_CCORR)).astype(int)
    y, x = np.unravel_index(overlap.argmax(), overlap.shape)
    intersection = int(overlap[y, x])
    union = int(query.sum()) + int(candidate.sum()) - intersection
    aligned = padded[y : y + SIZE, x : x + SIZE]
    edge, distance = _edge_distances(aligned)
    chamfer = (query_distance[edge].mean() + distance[query_edge].mean()) / 2
    return 0.80 * intersection / union + 0.20 * math.exp(-chamfer / 3.0)


class MaskIndex:
    """A read-only snapshot. An in-flight match can finish while a new index loads."""

    def __init__(self, sprites: list[Sprite], masks: list[np.ndarray]):
        self._sprites = tuple(sprites)
        self._masks = np.asarray(masks, dtype=np.uint8).reshape((-1, SIZE, SIZE))
        self._low = np.asarray(
            [_resize_mask(m, LOW_SIZE, LOW_SIZE) for m in masks], dtype=np.uint8
        ).reshape((-1, LOW_SIZE, LOW_SIZE))

    def __len__(self) -> int:
        return len(self._sprites)

    def _shortlist(self, ordered: np.ndarray, name_limit: int) -> np.ndarray:
        """Keep every pose of the highest-ranked names for the next scoring stage."""
        names = set()
        indices = []
        for i in ordered:
            name = self._sprites[i].name
            if name in names or len(names) < name_limit:
                names.add(name)
                indices.append(i)
        return np.asarray(indices, dtype=np.intp)

    def match_candidates(self, image: bytes) -> list[Match]:
        if not self._sprites:
            raise RecognitionError("精灵轮廓库为空")
        mask = _normalize(extract_silhouette(image))
        coarse = _iou(_resize_mask(mask, LOW_SIZE, LOW_SIZE), self._low)
        top = self._shortlist(np.argsort(-coarse, kind="stable"), 50)
        refined = _iou(mask, self._masks[top])
        finalists = self._shortlist(top[np.argsort(-refined, kind="stable")], 16)
        edge, distance = _edge_distances(mask)
        best_by_name: dict[str, Match] = {}
        for i in finalists:
            sprite = self._sprites[i]
            score = _aligned_score(mask, edge, distance, self._masks[i])
            previous = best_by_name.get(sprite.name)
            if previous is None or score > previous.score:
                best_by_name[sprite.name] = Match(
                    sprite.number, sprite.name, sprite.form, sprite.path, score
                )
        ranked = sorted(best_by_name.values(), key=lambda m: -m.score)
        if len(ranked) >= 2:
            best, second = ranked[:2]
            if (
                second.score >= MIN_ALTERNATIVE_SCORE
                and best.score - second.score <= MAX_ALTERNATIVE_GAP
            ):
                return [best, second]
        return ranked[:1]

    def match(self, image: bytes) -> Match:
        return self.match_candidates(image)[0]


def _read_cache(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as cache:
            keys, packed = cache["keys"], cache["masks"]
            if (
                packed.shape != (len(keys), SIZE * SIZE // 8)
                or packed.dtype != np.uint8
            ):
                return {}
            masks = np.unpackbits(packed, axis=1).reshape((-1, SIZE, SIZE))
            return dict(zip(keys.tolist(), masks))
    except (FileNotFoundError, OSError, ValueError, KeyError, BadZipFile, EOFError):
        return {}


def _write_cache(path: Path, masks: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as f:
            packed = np.packbits(
                np.asarray(list(masks.values())).reshape(-1, SIZE * SIZE), axis=1
            )
            np.savez_compressed(f, keys=np.array(list(masks)), masks=packed)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_index(data_dir: Path) -> MaskIndex:
    """Load the manifest, reusing cached masks by source digest across Wiki revisions."""
    root = (Path(data_dir) / "sprites").resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] not in (1, 2) or not isinstance(
            manifest["entries"], list
        ):
            raise RecognitionError("不支持的精灵清单格式")
        entries = manifest["entries"]
    except FileNotFoundError:
        return MaskIndex([], [])
    except (KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecognitionError("无法读取精灵清单") from exc

    cache_path = root.parent / CACHE_NAME
    cached = _read_cache(cache_path)
    masks_by_key = {}
    sprites, masks = [], []
    for entry in entries:
        try:
            if not isinstance(entry, dict):
                raise RecognitionError("精灵素材记录无效")
            if manifest["version"] == 1 and str(entry.get("id", "")).endswith("-shiny"):
                continue
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RecognitionError("精灵图片不存在或路径无效")
            sprite = Sprite(str(entry["number"]), entry["name"], entry["form"], path)
            key = entry["sha256"]
            mask = masks_by_key.get(key, cached.get(key))
            if mask is None:
                mask = _sprite_mask(path)
            masks_by_key[key] = mask
        except (KeyError, TypeError, OSError, ValueError) as exc:
            logger.warning("跳过无效精灵素材 %s (%s)", entry, exc)
            continue
        sprites.append(sprite)
        masks.append(mask)
    if masks_by_key and masks_by_key.keys() != cached.keys():
        _write_cache(cache_path, masks_by_key)
    return MaskIndex(sprites, masks)
