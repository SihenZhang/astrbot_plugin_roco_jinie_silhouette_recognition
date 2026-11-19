import asyncio
import re
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import BaseMessageComponent, Image, Plain
from astrbot.api.platform import MessageType
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.platform.message_session import MessageSession

from .reply import build_reply
from .service import LibraryUnavailable, RecognitionService

PLUGIN_NAME = "astrbot_plugin_roco_jinie_silhouette_recognition"
DEFAULT_BOT_SID = "qq:FriendMessage:2854203313"
QQ_ADAPTERS = {"aiocqhttp", "qq_official", "qq_official_webhook"}
QUESTION_PATTERN = (
    r"(?s)^(?=.*我是谁)(?=.*请回复[\s*]*是\s*[+＋]\s*答案[\s*]*或[\s*]*"
    r"这是\s*[+＋]\s*答案[\s*]*的形式作答)"
)
# OneBot exposes QQ bot Markdown cards as Plain text, including the picture.
QQ_MARKDOWN_IMAGE = re.compile(
    r"!\[[^\]\r\n]*\]\(\s*(https://qqbot\.ugcimg\.cn/[^\s()<>]+)\s*\)"
)


def _question_images(parts: list[BaseMessageComponent]) -> list[Image]:
    images = [part for part in parts if isinstance(part, Image)]
    seen = {image.url or image.file for image in images}
    for part in parts:
        if isinstance(part, Plain):
            for url in QQ_MARKDOWN_IMAGE.findall(part.text):
                if url not in seen:
                    images.append(Image.fromURL(url))
                    seen.add(url)
    return images


def _configured_sids(
    config: AstrBotConfig, key: str, message_types: set[MessageType]
) -> set[str] | None:
    """Return None for the default and an empty set for an invalid allowlist."""
    values = config.get(key, [])
    if values == []:
        return None
    accepted = set()
    if not isinstance(values, list):
        logger.warning("%s 需要填写 SID 列表", key)
        return accepted
    for value in values:
        try:
            sid = (
                MessageSession.from_str(value.strip())
                if isinstance(value, str)
                else None
            )
        except ValueError:
            sid = None
        if (
            sid
            and sid.platform_id.strip()
            and sid.session_id.strip()
            and sid.message_type in message_types
        ):
            accepted.add(str(sid))
        else:
            logger.warning("忽略 %s 中的无效 SID %r", key, value)
    return accepted


class SilhouettePlugin(Star):
    """在 QQ 收到洛克王国世界「我是谁」题目时识别剪影并回复。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # Older installations used an empty list for the built-in quiz bot.
        if config.get("bot_sids", []) == []:
            config["bot_sids"] = [DEFAULT_BOT_SID]
            if isinstance(config, AstrBotConfig):
                config.save_config()
        self.bot_sids = (
            _configured_sids(config, "bot_sids", {MessageType.FRIEND_MESSAGE}) or set()
        )
        self.allowed_sids = _configured_sids(
            config,
            "allowed_sids",
            {MessageType.FRIEND_MESSAGE, MessageType.GROUP_MESSAGE},
        )
        self.service = RecognitionService(
            StarTools.get_data_dir(PLUGIN_NAME),
            config,
            timezone=context.get_config().get("timezone"),
            logger=logger,
        )

    async def initialize(self):
        await self.service.start()

    def _accepts_source(self, event: AstrMessageEvent) -> bool:
        if event.get_platform_name() not in QQ_ADAPTERS:
            return False
        sender = str(event.get_sender_id())
        instance = event.get_platform_id()
        sender_sid = MessageSession(instance, MessageType.FRIEND_MESSAGE, sender)
        if str(sender_sid) not in self.bot_sids:
            return False
        if self.allowed_sids is None or event.unified_msg_origin in self.allowed_sids:
            return True
        # With per-member isolation, the event's session ID differs from the group ID.
        if (
            event.get_message_type() == MessageType.GROUP_MESSAGE
            and event.get_group_id()
        ):
            group_sid = MessageSession(
                instance, MessageType.GROUP_MESSAGE, str(event.get_group_id())
            )
            return str(group_sid) in self.allowed_sids
        return False

    @filter.regex(QUESTION_PATTERN)
    async def on_question(self, event: AstrMessageEvent):
        """匹配同一条 QQ 消息中的题目文本和图片。"""
        if not self._accepts_source(event):
            return

        images = _question_images(event.get_messages())
        if not images:
            return
        event.should_call_llm(False)
        fallback_text = None
        for image in images:
            try:
                path = await image.convert_to_file_path()
                data = await asyncio.to_thread(Path(path).read_bytes)
            except Exception:
                logger.exception("无法读取「我是谁」题图")
                continue
            try:
                answer = await self.service.recognize(data)
            except LibraryUnavailable as error:
                yield event.plain_result(str(error)).use_t2i(False)
                return
            if answer is None:
                continue
            if not answer.matched:
                fallback_text = answer.text
                continue
            if answer.image is None:
                yield event.plain_result(answer.text).use_t2i(False)
            elif self.config.get("image_in_same_message", True):
                parts = [Plain(answer.text), Image.fromBytes(answer.image)]
                yield event.chain_result(parts).use_t2i(False)
            else:
                yield event.plain_result(answer.text).use_t2i(False)
                yield event.chain_result([Image.fromBytes(answer.image)]).use_t2i(False)
            return
        yield event.plain_result(fallback_text or build_reply([]).text).use_t2i(False)

    async def terminate(self):
        await self.service.close()
