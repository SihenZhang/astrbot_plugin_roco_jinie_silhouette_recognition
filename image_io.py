"""Decode quiz cards and transparent Wiki artwork using OpenCV."""

import struct

import cv2
import numpy as np

PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def _card_dimensions(data: bytes) -> tuple[int, int]:
    # imdecode allocates the full image. Read the stored dimensions first.
    if data.startswith(PNG_HEADER):
        return struct.unpack_from(">II", data, 16)
    if data.startswith(b"\xff\xd8"):
        # JPEG stores its dimensions in a Start of Frame segment, after any
        # variable-length metadata. Skip entire segments, including EXIF thumbnails.
        offset = 2
        while offset + 4 <= len(data) and data[offset] == 0xFF:
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            marker, length = struct.unpack_from(">BH", data, offset)
            offset += 1
            if length < 2 or offset + length > len(data):
                break
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if length < 8:
                    break
                height, width = struct.unpack_from(">HH", data, offset + 3)
                return width, height
            if marker in (0xDA, 0xD9):
                break
            offset += length
    if data.startswith((b"GIF87a", b"GIF89a")):
        return struct.unpack_from("<HH", data, 6)
    if data.startswith(b"BM"):
        dib_size = struct.unpack_from("<I", data, 14)[0]
        if dib_size == 12:
            return struct.unpack_from("<HH", data, 18)
        width, height = struct.unpack_from("<ii", data, 18)
        return width, abs(height)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk == b"VP8L" and data[20] == 0x2F:
            size = struct.unpack_from("<I", data, 21)[0]
            return (size & 0x3FFF) + 1, ((size >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack_from("<HH", data, 26)
            return width & 0x3FFF, height & 0x3FFF
    raise ValueError("无法读取题图")


def _decode(data: bytes, flags: int) -> np.ndarray:
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    except cv2.error as exc:
        raise ValueError("无法解码图片") from exc
    if image is None:
        raise ValueError("无法解码图片")
    return image


def decode_card(data: bytes) -> np.ndarray:
    """Decode to BGR, honoring EXIF orientation, and fit inside a 1280px square."""
    try:
        width, height = _card_dimensions(data)
    except struct.error as exc:
        raise ValueError("无法读取题图") from exc
    if min(width, height) <= 0:
        raise ValueError("无法读取题图")
    if width * height > 20_000_000:
        raise ValueError("题图分辨率过大")
    image = _decode(data, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    scale = 1280 / max(width, height)
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def _gray_transparency(data: bytes) -> int | None:
    # OpenCV keeps palette and truecolor tRNS but drops the grayscale color key.
    # PNG stores that key in the original sample depth before the first IDAT.
    pos = 33  # Just past IHDR and its checksum.
    while pos + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        if kind == b"tRNS" and length == 2:
            key = struct.unpack_from(">H", data, pos + 8)[0]
            depth = data[24]
            if depth < 8:
                key *= 255 // ((1 << depth) - 1)
            return key
        if kind == b"IDAT":
            break
        pos += length + 12
    return None


def decode_sprite(data: bytes) -> np.ndarray:
    """Read a Wiki PNG as 8-bit BGRA and validate the transparency used for matching."""
    if not data.startswith(PNG_HEADER) or len(data) < 33:
        raise ValueError("立绘格式或尺寸异常")
    width, height = struct.unpack_from(">II", data, 16)
    if not width or not height or width * height > 16_000_000:
        raise ValueError("立绘格式或尺寸异常")
    image = _decode(data, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        key = _gray_transparency(data)
        if key is not None:
            transparent = image == key
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            image[transparent, 3] = 0
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("立绘缺少可用的透明通道")
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    alpha = image[:, :, 3]
    if alpha.min() >= 128 or alpha.max() < 128:
        raise ValueError("立绘缺少可用的透明通道")
    return image
