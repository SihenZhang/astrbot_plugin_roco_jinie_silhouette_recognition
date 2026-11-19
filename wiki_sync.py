"""Incrementally mirror the NRC Wiki's published full-body pet artwork."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from bs4 import BeautifulSoup
from filelock import FileLock
from filelock import Timeout as LockTimeout

if __package__:
    from .image_io import decode_sprite
else:  # Allow direct execution of the standalone synchronization command.
    from image_io import decode_sprite

CATALOG_URL = "https://wiki.biligame.com/nrc/精灵图鉴"
DEFAULT_USER_AGENT = "AstrBot-RocoSilhouette/0.1.0 (daily Wiki sprite index)"
MANIFEST_VERSION = 2
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncResult:
    total: int
    downloaded: int
    failed: int
    changed: bool


@dataclass(frozen=True)
class Sprite:
    id: str
    number: str
    name: str
    form: str
    url: str
    # Only used to identify old files left by an interrupted v1 bootstrap.
    legacy_shiny_url: str = ""

    @property
    def path(self) -> str:
        # Including the URL digest keeps existing manifests readable throughout an update.
        source = hashlib.sha256(self.url.encode()).hexdigest()[:16]
        return f"{self.number}/{self.id}-{source}.png"

    def entry(self, sha256: str, size: int) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "name": self.name,
            "form": self.form,
            "url": self.url,
            "path": self.path,
            "sha256": sha256,
            "size": size,
        }


def parse_catalog(html: bytes) -> list[Sprite]:
    """The Wiki renders the complete catalog server-side, including alternate forms."""
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    cards = soup.select(".nrc-pet-catalog .npc-card")
    total = soup.select_one(".nrc-pet-catalog .npc-total-number")
    if not cards or (total and total.get_text(strip=True) != str(len(cards))):
        raise ValueError("Wiki 图鉴卡片数量不完整，请检查页面结构")
    sprites = []
    seen = set()
    for card in cards:
        pet_id = card.get("data-id", "")
        number = card.get("data-number", "")
        name_node = card.select_one(".npc-name")
        name = name_node.get_text(strip=True) if name_node else ""
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", pet_id)
            or pet_id in seen
            or not re.fullmatch(r"[0-9]+", number)
            or not name
        ):
            raise ValueError(f"Wiki 精灵卡片编号或名称异常：{pet_id!r}")
        seen.add(pet_id)
        form_node = card.select_one(".npc-form")
        form = form_node.get_text(strip=True) if form_node else ""
        node = card.select_one(".npc-art-normal img")
        url = urljoin(CATALOG_URL, node.get("src", "")) if node else ""
        address = urlsplit(url)
        if (
            address.scheme != "https"
            or address.netloc != "patchwiki.biligame.com"
            or not address.path.startswith("/images/nrc/")
        ):
            raise ValueError(f"Wiki 图片地址异常：{name}")
        shiny = card.select_one(".npc-art-shiny img")
        shiny_url = urljoin(CATALOG_URL, shiny.get("src", "")) if shiny else ""
        sprites.append(Sprite(pet_id, number, name, form, url, shiny_url))
    return sprites


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as f:
            tmp = Path(f.name)
            f.write(data)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _write_manifest(path: Path, entries: list[dict], **extra) -> None:
    state = {"version": MANIFEST_VERSION, "entries": entries, **extra}
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    _atomic_write(path, (payload + "\n").encode())


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        _atomic_write(target, source.read_bytes())


def _migrate_locked(directory: Path) -> bool:
    """Publish normal artwork under v2 names before retiring files owned by v1."""
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return False
    state = json.loads(manifest.read_text("utf-8"))
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("entries"), list)
        or state.get("version") not in (1, MANIFEST_VERSION)
    ):
        raise ValueError("无法识别现有 Wiki 素材清单")
    if state["version"] == MANIFEST_VERSION and "retired_v1" not in state:
        return False

    # Include the old manifest records in the transaction. A crash between
    # publication and cleanup can resume without guessing which files we own.
    legacy = state.get("retired_v1", state["entries"])
    entries = []
    retired = []
    root = directory.resolve()
    for old in legacy:
        pet_id, _, kind = old["id"].rpartition("-")
        if (
            kind not in ("normal", "shiny")
            or not re.fullmatch(r"[A-Za-z0-9_-]+", pet_id)
            or not re.fullmatch(r"[0-9]+", old["number"])
        ):
            raise ValueError("旧 Wiki 清单的形态 ID 或编号无效")
        sprite = Sprite(pet_id, old["number"], old["name"], old["form"], old["url"])
        digest = hashlib.sha256(sprite.url.encode()).hexdigest()[:16]
        relative = f"{sprite.number}/{old['id']}-{digest}.png"
        if old["path"] != relative:
            raise ValueError("旧 Wiki 清单与已知的素材命名规则不符")
        source = root / relative
        target = root / sprite.path
        if source.resolve() != source or target.resolve() != target:
            raise ValueError("旧 Wiki 素材路径包含文件链接")
        retired.append(source)
        if kind == "shiny" or state["version"] == MANIFEST_VERSION:
            continue
        if source.is_file():
            if target.exists():
                if (
                    not source.samefile(target)
                    and hashlib.sha256(target.read_bytes()).hexdigest() != old["sha256"]
                ):
                    raise ValueError(f"迁移目标文件已存在且内容不符：{sprite.path}")
            else:
                _link_or_copy(source, target)
        entries.append(sprite.entry(old["sha256"], old["size"]))

    if state["version"] == 1:
        _write_manifest(manifest, entries, retired_v1=legacy)
    else:
        entries = state["entries"]
    for source in retired:
        source.unlink(missing_ok=True)
    _write_manifest(manifest, entries)
    return True


def migrate_sprites(data_dir: Path) -> bool:
    """Upgrade the installed library offline before opening its silhouette index."""
    directory = Path(data_dir) / "sprites"
    if not (directory / "manifest.json").is_file():
        return False
    try:
        with FileLock(directory / ".sync.lock", timeout=0):
            return _migrate_locked(directory)
    except LockTimeout as exc:
        raise RuntimeError("另一个进程正在同步 Wiki 素材") from exc
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError("旧 Wiki 素材清单字段有误") from exc


def _recover_unpublished_v1(directory: Path, catalog: list[Sprite]) -> None:
    """Recognize unfinished v1 downloads by exact names reconstructed from the Wiki."""
    root = directory.resolve()
    if next(root.glob("[0-9]*/*-*-????????????????.png"), None) is None:
        return
    for sprite in catalog:
        target = root / sprite.path
        for kind, url in (("normal", sprite.url), ("shiny", sprite.legacy_shiny_url)):
            if not url:
                continue
            suffix = hashlib.sha256(url.encode()).hexdigest()[:16]
            old = root / sprite.number / f"{sprite.id}-{kind}-{suffix}.png"
            if not old.is_file() or old.resolve() != old or target.resolve() != target:
                continue
            if kind == "normal":
                if not target.exists():
                    _link_or_copy(old, target)
                elif (
                    not target.samefile(old) and target.read_bytes() != old.read_bytes()
                ):
                    continue
            old.unlink()


def _verify_png(data: bytes) -> str:
    decode_sprite(data)
    return hashlib.sha256(data).hexdigest()


def _read_local(path: Path, expected_hash: str | None) -> tuple[bytes, str] | None:
    try:
        data = path.read_bytes()
        digest = _verify_png(data)
        if expected_hash is None or digest == expected_hash:
            return data, digest
    except (OSError, ValueError):
        pass
    return None


async def _fetch(session: aiohttp.ClientSession, url: str, limit: int) -> bytes:
    async with session.get(url) as response:
        response.raise_for_status()
        data = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            if len(data) + len(chunk) > limit:
                raise ValueError("Wiki 响应超过大小限制")
            data.extend(chunk)
        return bytes(data)


async def sync_sprites(
    data_dir: Path,
    user_agent: str = DEFAULT_USER_AGENT,
    *,
    logger: logging.Logger = log,
) -> SyncResult:
    """Fetch one catalog, then only missing artwork, serializing CLI and plugin syncs."""
    directory = Path(data_dir) / "sprites"
    directory.mkdir(parents=True, exist_ok=True)
    migrated = await asyncio.to_thread(migrate_sprites, data_dir)
    try:
        with FileLock(directory / ".sync.lock", timeout=0):
            return await _sync_locked(directory, user_agent, migrated, logger)
    except LockTimeout as exc:
        raise RuntimeError("另一个进程正在同步 Wiki 素材") from exc


async def _sync_locked(
    directory: Path, user_agent: str, migrated: bool, logger: logging.Logger
) -> SyncResult:
    manifest = directory / "manifest.json"
    old_entries = []
    if manifest.is_file():
        state = json.loads(manifest.read_text("utf-8"))
        if state.get("version") != MANIFEST_VERSION or not isinstance(
            state.get("entries"), list
        ):
            raise ValueError("无法识别现有 Wiki 素材清单")
        old_entries = state["entries"]
    previous = {row["id"]: row for row in old_entries}
    entries = []
    downloaded = failed = consecutive_failures = 0
    repaired_file = False
    # Reused illustrations occur across a few different forms.
    cached_urls: dict[str, Path] = {}
    headers = {
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Referer": "https://wiki.biligame.com/",
    }
    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=40),
        trust_env=True,
    ) as session:
        html = await _fetch(session, CATALOG_URL, 8 * 1024 * 1024)
        catalog = await asyncio.to_thread(parse_catalog, html)
        recovery = asyncio.create_task(
            asyncio.to_thread(_recover_unpublished_v1, directory, catalog)
        )
        try:
            await asyncio.shield(recovery)
        except asyncio.CancelledError:
            # Keep the filesystem lock until the last local rename is durable.
            await recovery
            raise
        for sprite in catalog:
            path = directory / sprite.path
            old = previous.get(sprite.id)
            if (
                old
                and old.get("path") == sprite.path
                and path.is_file()
                and path.stat().st_size == old["size"]
            ):
                entries.append(sprite.entry(old["sha256"], old["size"]))
                cached_urls[sprite.url] = path
                continue
            try:
                # Recover complete, previously verified files from an interrupted first run.
                source = path if path.is_file() else cached_urls.get(sprite.url)
                expected = (
                    old["sha256"] if old and old.get("path") == sprite.path else None
                )
                local = (
                    await asyncio.to_thread(_read_local, source, expected)
                    if source
                    else None
                )
                if local:
                    data, sha256 = local
                else:
                    if consecutive_failures >= 3:
                        raise ConnectionError("本轮 Wiki 请求已暂停")
                    await asyncio.sleep(0.15)
                    data = await _fetch(session, sprite.url, 16 * 1024 * 1024)
                    sha256 = await asyncio.to_thread(_verify_png, data)
                    downloaded += 1
                    source = None
                    consecutive_failures = 0
                if source != path:
                    await asyncio.to_thread(_atomic_write, path, data)
                    repaired_file = True
                entries.append(sprite.entry(sha256, len(data)))
                cached_urls[sprite.url] = path
                if downloaded and downloaded % 100 == 0 and source is None:
                    logger.info("Wiki 已下载 %d 张精灵立绘", downloaded)
            except (OSError, ValueError, aiohttp.ClientError) as exc:
                if isinstance(exc, OSError) and exc.errno in (
                    errno.ENOSPC,
                    errno.EACCES,
                    errno.EROFS,
                ):
                    raise
                if consecutive_failures < 3:
                    logger.warning(
                        "Wiki 素材更新失败 %s (%s)：%s", sprite.id, sprite.name, exc
                    )
                if isinstance(exc, aiohttp.ClientResponseError) and exc.status in (
                    403,
                    429,
                ):
                    consecutive_failures = 3
                elif (
                    isinstance(exc, (aiohttp.ClientConnectionError, TimeoutError))
                    or isinstance(exc, aiohttp.ClientResponseError)
                    and exc.status >= 500
                ):
                    consecutive_failures += 1
                failed += 1
                if old:
                    fallback = directory / old["path"]
                    if fallback.is_file() and fallback.stat().st_size == old["size"]:
                        entries.append(old)
    manifest_changed = entries != old_entries
    if manifest_changed:
        _write_manifest(manifest, entries)
    return SyncResult(
        len(entries), downloaded, failed, migrated or manifest_changed or repaired_file
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="增量同步洛克王国世界 Wiki 精灵立绘")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="插件数据目录，sprites 将建在其中"
    )
    parser.add_argument(
        "--user-agent", default=DEFAULT_USER_AGENT, help="用于访问 Wiki 的 User-Agent"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("Wiki 同步开始，检查精灵图鉴")
    try:
        result = asyncio.run(sync_sprites(args.data_dir, args.user_agent))
    except (OSError, ValueError, RuntimeError, aiohttp.ClientError) as exc:
        log.error("同步失败，请检查网络连接及图鉴页面后重试：%s", exc)
        return 1
    log.info(
        "Wiki 同步结束，可用素材 %d 条，本轮下载 %d 张，失败 %d 条。",
        result.total,
        result.downloaded,
        result.failed,
    )
    if result.failed:
        log.warning("下次同步会复用已下载的素材并补齐失败项。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
