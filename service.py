"""Manage the active silhouette index and the Wiki update schedule."""

import asyncio
import logging
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .recognition import MaskIndex, RecognitionError, load_index
from .reply import MIN_MATCH_SCORE, RecognitionReply, build_reply
from .wiki_sync import DEFAULT_USER_AGENT, migrate_sprites, sync_sprites

DEFAULT_CRON = "0 5 * * *"
DEFAULT_LOGGER = logging.getLogger("astrbot")


class LibraryUnavailable(RuntimeError):
    pass


class RecognitionService:
    def __init__(
        self,
        data_dir: Path,
        config,
        timezone: str | None = None,
        *,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.timezone = timezone
        self.logger = logger
        self.index = None
        self._scheduler = None
        self._update_task = None
        self._closing = False

    async def start(self):
        try:
            migrated = await asyncio.to_thread(migrate_sprites, self.data_dir)
        except (OSError, RuntimeError, ValueError):
            self.logger.exception("旧版立绘迁移未完成，尝试读取现有普通立绘")
            migrated = False
        if migrated and self.config.get("sync_cron") == "0 4 * * *":
            self.config["sync_cron"] = DEFAULT_CRON
            if hasattr(self.config, "save_config"):
                self.config.save_config()
        try:
            self.index = await asyncio.to_thread(load_index, self.data_dir)
        except (OSError, ValueError):
            self.logger.exception("无法载入精灵轮廓库")
        cron = self.config.get("sync_cron", DEFAULT_CRON).strip()
        if cron:
            try:
                scheduler = AsyncIOScheduler(timezone=self.timezone or None)
                trigger = CronTrigger.from_crontab(cron, timezone=scheduler.timezone)
            except (ValueError, ZoneInfoNotFoundError):
                self.logger.exception(
                    "无法建立 Wiki 同步计划，请检查 Cron 和 AstrBot 时区"
                )
            else:
                self._scheduler = scheduler
                self._scheduler.add_job(
                    self._request_update,
                    trigger,
                    coalesce=True,
                    misfire_grace_time=3600,
                )
                self._scheduler.start()
        await self._request_update("插件加载")

    async def _request_update(self, trigger: str = "定时任务"):
        if self._closing:
            return
        if self._update_task and not self._update_task.done():
            self.logger.info("Wiki 同步已在进行中（%s），复用本轮同步", trigger)
            return
        self._update_task = asyncio.create_task(
            self._update(trigger), name="roco-wiki-sync"
        )

    async def _update(self, trigger: str):
        self.logger.info("Wiki 同步开始（%s）", trigger)
        try:
            result = await sync_sprites(
                self.data_dir,
                self.config.get("user_agent", DEFAULT_USER_AGENT),
                logger=self.logger,
            )
            # Load even after a no-op sync so an earlier failed rebuild can recover.
            # Shield the thread while it publishes its on-disk cache during unload.
            rebuild = asyncio.create_task(asyncio.to_thread(load_index, self.data_dir))
            try:
                index = await asyncio.shield(rebuild)
            except asyncio.CancelledError:
                await rebuild
                raise
            if index:
                self.index = index
            if result.failed or not index or len(index) != result.total:
                self.logger.warning(
                    "Wiki 同步未完成：可用素材 %d，下载 %d，失败 %d，本轮建索引 %d，当前索引 %d",
                    result.total,
                    result.downloaded,
                    result.failed,
                    len(index),
                    len(self.index or ()),
                )
                return
            self.logger.info(
                "Wiki 同步完成（%s）：%s，可用素材 %d，下载 %d，当前索引 %d",
                trigger,
                "素材已更新" if result.changed else "素材无变化",
                result.total,
                result.downloaded,
                len(index),
            )
        except asyncio.CancelledError:
            self.logger.info("Wiki 同步已取消（%s）", trigger)
            raise
        except Exception:
            self.logger.exception(
                "Wiki 同步失败（%s），当前索引 %d", trigger, len(self.index or ())
            )

    def _recognize(
        self, index: MaskIndex, image_bytes: bytes, send_image: bool
    ) -> RecognitionReply:
        candidates = index.match_candidates(image_bytes)
        scores = "、".join(f"{m.name} {m.score:.3f}" for m in candidates) or "无候选"
        try:
            reply = build_reply(candidates, send_image=send_image)
        except OSError as exc:
            self.logger.exception("已匹配到 %s，读取候选精灵立绘失败", scores)
            names = "、".join(m.name for m in candidates)
            raise LibraryUnavailable(f"匹配到了{names}，但暂时无法读取立绘。") from exc
        if reply.matched:
            self.logger.info("剪影识别成功：%s", scores)
        else:
            self.logger.info(
                "剪影未认出（最低匹配分 %.3f）：%s", MIN_MATCH_SCORE, scores
            )
        return reply

    async def recognize(self, image_bytes: bytes) -> RecognitionReply | None:
        index = self.index
        if not index:
            if self._update_task and not self._update_task.done():
                reason = "精灵素材库正在初始化，完成后即可识别。"
            else:
                reason = "精灵素材库暂时不可用，请检查插件同步日志。"
            self.logger.info("剪影识别暂不可用：%s", reason)
            raise LibraryUnavailable(reason)
        try:
            send_image = self.config.get("send_image", True)
            return await asyncio.to_thread(
                self._recognize, index, image_bytes, send_image
            )
        except RecognitionError as exc:
            self.logger.info("剪影未认出：%s", exc)
            return None

    async def close(self):
        self._closing = True
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        if self._update_task:
            self._update_task.cancel()
            await asyncio.gather(self._update_task, return_exceptions=True)
