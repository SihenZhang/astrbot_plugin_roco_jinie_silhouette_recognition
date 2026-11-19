import asyncio
import hashlib
import importlib
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import DEFAULT, AsyncMock, Mock

import cv2
import numpy as np
import pytest

from wiki_sync import Sprite, SyncResult

plugin_package = Path(__file__).resolve().parents[1].name
service = importlib.import_module(f"{plugin_package}.service")


def write_sprite_library(root, *names):
    art = np.zeros((64, 64, 4), np.uint8)
    art[12:52, 24:40] = (0, 0, 255, 255)
    data = cv2.imencode(".png", art)[1].tobytes()
    digest = hashlib.sha256(data).hexdigest()
    directory = root / "sprites"
    entries = []
    for number, name in enumerate(names, start=1):
        sprite = Sprite(
            f"pet_{number}", f"{number:03d}", name, "", "https://example.org/pet.png"
        )
        path = directory / sprite.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(sprite.entry(digest, len(data)))
    (directory / "manifest.json").write_text(
        json.dumps({"version": 2, "entries": entries}), encoding="utf-8"
    )


def quiz_card(*, horizontal=False):
    card = np.full((64, 64, 3), (232, 150, 54), np.uint8)
    card[12:52, 24:40] = (43, 33, 32)
    if horizontal:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    return cv2.imencode(".png", card)[1].tobytes()


@pytest.fixture
def logger(caplog):
    log = logging.getLogger("test.roco")
    caplog.set_level(logging.INFO, logger=log.name)
    return log


@asynccontextmanager
async def running(library):
    try:
        await library.start()
        yield
    finally:
        await library.close()


async def test_bootstrap_and_concurrent_timer_tick(
    tmp_path, logger, caplog, monkeypatch
):
    library = service.RecognitionService(tmp_path, {"sync_cron": ""}, logger=logger)

    async def download(root, *args, **kwargs):
        write_sprite_library(root, "波波拉")
        return SyncResult(1, 1, 0, True)

    sync = AsyncMock(side_effect=download)
    monkeypatch.setattr(service, "sync_sprites", sync)

    async with running(library):
        with pytest.raises(service.LibraryUnavailable):
            await library.recognize(quiz_card())
        await library._request_update()
        await library._update_task
        sync.assert_awaited_once()
        reply = await library.recognize(quiz_card())
        assert "波波拉" in reply.text
        (artwork,) = (tmp_path / "sprites").rglob("*.png")
        assert reply.image == artwork.read_bytes()
        assert "素材已更新" in caplog.text


async def test_text_only_recognizes_ambiguous_matches_without_art(
    tmp_path, logger, caplog
):
    write_sprite_library(tmp_path, "波波拉", "水灵")
    library = service.RecognitionService(tmp_path, {"send_image": False}, logger=logger)
    library.index = service.load_index(tmp_path)
    for art in (tmp_path / "sprites").rglob("*.png"):
        art.unlink()

    reply = await library.recognize(quiz_card())

    assert reply.matched
    assert reply.image is None
    assert "剪影识别成功" in caplog.text
    for name in ("波波拉", "水灵"):
        assert name in reply.text
        assert re.search(rf"{name} \d\.\d{{3}}\b", caplog.text)


async def test_low_score_and_unreadable_card(tmp_path, logger, caplog):
    write_sprite_library(tmp_path, "波波拉")
    library = service.RecognitionService(tmp_path, {}, logger=logger)
    library.index = service.load_index(tmp_path)

    reply = await library.recognize(quiz_card(horizontal=True))
    assert not reply.matched
    assert reply.image is None
    assert "剪影未认出" in caplog.text
    assert "波波拉" in caplog.text

    caplog.clear()
    assert await library.recognize(b"invalid image") is None
    assert "剪影未认出" in caplog.text


async def test_cached_answers_survive_interrupted_update(
    tmp_path, logger, caplog, monkeypatch
):
    write_sprite_library(tmp_path, "迪莫")
    library = service.RecognitionService(
        tmp_path, {}, timezone="Europe/Berlin", logger=logger
    )
    started = asyncio.Event()

    async def slow_sync(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "sync_sprites", slow_sync)

    async with running(library):
        await asyncio.wait_for(started.wait(), timeout=2)
        (job,) = library._scheduler.get_jobs()
        assert job.next_run_time.hour == 5
        assert str(job.next_run_time.tzinfo) == "Europe/Berlin"
        assert "迪莫" in (await library.recognize(quiz_card())).text

    assert "Wiki 同步已取消" in caplog.text
    assert library._update_task.cancelled()


async def test_unchanged_sync_recovers_failed_index_load_without_cron(
    tmp_path, logger, caplog, monkeypatch
):
    write_sprite_library(tmp_path, "波波拉")
    library = service.RecognitionService(tmp_path, {"sync_cron": ""}, logger=logger)
    monkeypatch.setattr(
        service, "sync_sprites", AsyncMock(return_value=SyncResult(1, 0, 0, False))
    )
    # Fail the startup read, then let the next load use the files on disk.
    monkeypatch.setattr(
        service,
        "load_index",
        Mock(wraps=service.load_index, side_effect=[OSError("缓存暂不可读"), DEFAULT]),
    )

    async with running(library):
        await library._update_task
        assert "波波拉" in (await library.recognize(quiz_card())).text
        assert "Wiki 同步开始（插件加载）" in caplog.text
        assert "素材无变化" in caplog.text
        assert "当前索引 1" in caplog.text


async def test_partial_download_and_failed_sync_keep_recognition_available(
    tmp_path, logger, caplog, monkeypatch
):
    library = service.RecognitionService(tmp_path, {"sync_cron": ""}, logger=logger)

    async def partial_download(root, *args, **kwargs):
        write_sprite_library(root, "波波拉")
        return SyncResult(1, 1, 1, True)

    monkeypatch.setattr(service, "sync_sprites", partial_download)

    async with running(library):
        await library._update_task
        assert "Wiki 同步未完成" in caplog.text
        assert "失败 1" in caplog.text
        assert "当前索引 1" in caplog.text

        caplog.clear()
        monkeypatch.setattr(
            service, "sync_sprites", AsyncMock(side_effect=ConnectionError("网络中断"))
        )
        await library._request_update()
        await library._update_task
        assert "Wiki 同步开始（定时任务）" in caplog.text
        assert "Wiki 同步失败" in caplog.text
        assert "网络中断" in caplog.text
        assert "波波拉" in (await library.recognize(quiz_card())).text


@pytest.mark.parametrize(
    ("original", "expected"),
    [("0 4 * * *", "0 5 * * *"), ("30 8 * * *", "30 8 * * *")],
    ids=["old-default", "custom"],
)
async def test_old_library_updates_only_previous_default_schedule(
    original, expected, tmp_path, monkeypatch
):
    config = {"sync_cron": original}
    library = service.RecognitionService(tmp_path, config)
    monkeypatch.setattr(service, "migrate_sprites", Mock(return_value=True))
    monkeypatch.setattr(
        service, "sync_sprites", AsyncMock(return_value=SyncResult(1, 0, 0, False))
    )

    async with running(library):
        assert config["sync_cron"] == expected
