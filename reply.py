"""Compose a quiz reply from the ranked candidates."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if __package__:
    from .image_io import decode_sprite
else:
    from image_io import decode_sprite

if TYPE_CHECKING:
    from .recognition import Match


MIN_MATCH_SCORE = 0.78

SINGLE_REPLIES = (
    "这应该是{name}",
    "我猜是{name}",
    "我投{name}一票",
    "这题我猜{name}",
    "看着挺眼熟，是{name}吧",
    "{name}！感觉挺像它的",
    "这只看着像{name}",
    "{name}吧",
    "这不是{name}吗",
    "一眼{name}",
    "哦这个我熟，{name}",
    "{name}，别问，问就是直觉",
    "{name}？我赌一包辣条",
    "{name}！这题送分吧",
    "我选{name}，错了算你赢",
    "这感觉，{name}没跑了",
    "{name}吧，我先报答案了",
    "不是吧，这都不认识，{name}啊",
    "我说{name}，没人反对吧",
)

PAIR_REPLIES = (
    "我偏向{first}，不过{second}也挺像",
    "有点难分，我会在{first}和{second}之间犹豫",
    "我先猜{first}，如果猜错了那就是{second}",
    "我想到的就是{first}和{second}这两只",
    "第一眼像{first}，可{second}也说得通",
    "{first}？但也可能是{second}",
    "像{first}吧？再看一眼，{second}也有可能",
    "{first}？不对，也像{second}",
    "{first}和{second}，我纠结了",
    "不是{first}就是{second}，二选一太难了",
    "先押{first}，{second}当备胎",
    "{first}吧，但{second}我也不敢排除",
    "这俩长得太像了，{first}还是{second}啊",
    "{first}，要翻车了就是{second}",
    "难绷，{first}和{second}我都想投",
    "{first}？{second}？你给个提示",
    "我脑子里蹦出来俩，{first}和{second}",
)

NOT_FOUND_REPLIES = (
    "这只精灵我暂时没认出来",
    "？这只我真不认识",
    "没见过，换一张",
    "这啥，我库里好像没有",
    "认不出来，别难为我了",
    "草，这只我卡壳了",
    "这个我真没印象",
    "看不出来，再来一张？",
)


@dataclass(frozen=True)
class RecognitionReply:
    text: str
    image: bytes | None
    matched: bool


def _pair_image(matches: list[Match]) -> bytes:
    side = 512
    margin = 16
    max_side = side - 2 * margin
    canvas = np.full((side, side * 2, 3), 255, dtype=np.uint8)
    for i, match in enumerate(matches):
        art = decode_sprite(match.path.read_bytes())
        left, top, width, height = cv2.boundingRect(art[:, :, 3])
        art = art[top : top + height, left : left + width]
        # Composite on white before resizing so hidden colors cannot bleed into edges.
        alpha = art[:, :, 3:4].astype(np.uint16)
        blended = art[:, :, :3] * alpha + 255 * (255 - alpha)
        thumbnail = ((blended + 127) // 255).astype(np.uint8)
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            width = max(1, round(width * scale))
            height = max(1, round(height * scale))
            thumbnail = cv2.resize(
                thumbnail, (width, height), interpolation=cv2.INTER_AREA
            )
        x = side * i + (side - width) // 2
        y = (side - height) // 2
        canvas[y : y + height, x : x + width] = thumbnail
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError("无法编码候选精灵立绘")
    return encoded.tobytes()


def build_reply(matches: list[Match], *, send_image: bool = True) -> RecognitionReply:
    if not matches or matches[0].score < MIN_MATCH_SCORE:
        return RecognitionReply(random.choice(NOT_FOUND_REPLIES), None, matched=False)
    if len(matches) == 1:
        first = matches[0]
        text = random.choice(SINGLE_REPLIES).format(name=first.name)
        image = first.path.read_bytes() if send_image else None
        return RecognitionReply(text, image, matched=True)

    first, second = matches
    text = random.choice(PAIR_REPLIES).format(first=first.name, second=second.name)
    if not send_image:
        return RecognitionReply(text, None, matched=True)
    try:
        image = _pair_image(matches)
    except (ValueError, cv2.error) as exc:
        raise OSError("无法生成候选精灵立绘") from exc
    return RecognitionReply(text, image, matched=True)
