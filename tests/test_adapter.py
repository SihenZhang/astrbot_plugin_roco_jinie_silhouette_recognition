import asyncio
import base64
import importlib
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from recognition import Match
from reply import NOT_FOUND_REPLIES, RecognitionReply

QUESTION = "【我是谁】\n请回复 是 + 答案 或 这是 + 答案 的形式作答。"
CARD_URL = (
    "https://qqbot.ugcimg.cn/102055112/925a351120547785270a519087e1937d00be332e/"
    "5e737d7f31eb330c163b0d45429f7d3d"
)
MARKDOWN_QUESTION = (
    "[](%7B%22version%22%3A2%7D) ## 我是谁\n"
    f"![img #450px #239px]({CARD_URL})\n"
    ">请回复 **是 + 答案** 或 **这是 + 答案** 的形式作答。"
)


@pytest.fixture(scope="module")
def adapter():
    if importlib.util.find_spec("astrbot") is None:
        pytest.skip("requires AstrBot")
    package = Path(__file__).resolve().parents[1].name
    # Importing AstrBot creates its data directory.
    with tempfile.TemporaryDirectory() as root, pytest.MonkeyPatch.context() as env:
        env.setenv("ASTRBOT_ROOT", root)
        yield importlib.import_module(f"{package}.main")


@pytest.fixture
def attachment(tmp_path):
    path = tmp_path / "attachment"
    path.write_bytes(b"question")
    return path


@pytest.fixture
def make_plugin(adapter):
    def create(config, *, real_service=False):
        context = Mock()
        context.get_config.return_value = {"timezone": "Asia/Shanghai"}
        plugin = adapter.SilhouettePlugin(context, config)
        if not real_service:
            plugin.service = AsyncMock()
            plugin.service.recognize.return_value = RecognitionReply(
                "这应该是喵喵", b"art", matched=True
            )
        return plugin

    return create


@pytest.fixture
def plugin(make_plugin):
    return make_plugin(
        {
            "bot_sids": ["qq-a:FriendMessage:123"],
            "allowed_sids": ["qq-a:GroupMessage:777"],
        }
    )


async def dispatch_event(plugin, event):
    from astrbot.core.star.star_handler import star_handlers_registry

    handler_name = f"{type(plugin).__module__}_on_question"
    handler = star_handlers_registry.get_handler_by_full_name(handler_name)
    if not all(f.filter(event, {}) for f in handler.event_filters):
        return []
    return [reply async for reply in plugin.on_question(event)]


@pytest.fixture
def dispatch(adapter, attachment):
    from astrbot.api.platform import (
        AstrBotMessage,
        MessageMember,
        MessageType,
        PlatformMetadata,
    )

    async def send(
        plugin,
        text=QUESTION,
        sender="123",
        platform="aiocqhttp",
        image_count=1,
        instance="qq-a",
        group="777",
        session_id=None,
    ):
        message = AstrBotMessage()
        message.type = (
            MessageType.GROUP_MESSAGE if group else MessageType.FRIEND_MESSAGE
        )
        message.sender = MessageMember(user_id=sender)
        message.group_id = group
        message.message = [adapter.Plain(text)]
        for _ in range(image_count):
            message.message.append(adapter.Image.fromFileSystem(str(attachment)))
        event = adapter.AstrMessageEvent(
            text,
            message,
            PlatformMetadata(platform, "test", id=instance),
            session_id or group or sender,
        )
        return await dispatch_event(plugin, event)

    return send


async def onebot_event(segments, sender=2854203313):
    from aiocqhttp import Event
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
        AiocqhttpAdapter,
    )

    events = asyncio.Queue()
    platform = AiocqhttpAdapter(
        {"id": "qq", "ws_reverse_host": "127.0.0.1", "ws_reverse_port": 0}, {}, events
    )
    raw = Event.from_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "self_id": 101,
            "user_id": sender,
            "group_id": 777,
            "message_id": 42,
            "time": 0,
            "sender": {"user_id": sender, "nickname": "小呱呱"},
            "message": segments,
        }
    )
    await platform.handle_msg(await platform.convert_message(raw))
    return events.get_nowait()


@pytest.fixture
def resolve_image(adapter, attachment):
    with patch.object(
        adapter.Image,
        "convert_to_file_path",
        autospec=True,
        return_value=str(attachment),
    ) as resolve:
        yield resolve


async def test_onebot_markdown_quiz_with_inline_keyboard(make_plugin, resolve_image):
    plugin = make_plugin({})
    event = await onebot_event(
        [
            {"type": "markdown", "data": {"content": MARKDOWN_QUESTION}},
            {"type": "inline_keyboard", "data": {"bot_appid": "102055112", "rows": []}},
        ]
    )
    (reply,) = await dispatch_event(plugin, event)
    text, art = reply.chain
    assert text.text == "这应该是喵喵"
    assert base64.b64decode(await art.convert_to_base64()) == b"art"
    assert reply.use_t2i_ is False
    resolve_image.assert_awaited_once()
    assert resolve_image.await_args.args[0].file == CARD_URL
    plugin.service.recognize.assert_awaited_once_with(b"question")


@pytest.mark.parametrize(
    ("extra_segments", "expected_file"),
    [
        ([], CARD_URL),
        (
            [{"type": "image", "data": {"file": "native.png", "url": CARD_URL}}],
            "native.png",
        ),
    ],
    ids=["markdown-only", "with-native-image"],
)
async def test_duplicate_card_text_is_resolved_once(
    make_plugin, adapter, resolve_image, extra_segments, expected_file
):
    plugin = make_plugin({})
    plugin.service.recognize.return_value = None  # Examine every candidate.
    text = (
        MARKDOWN_QUESTION
        + f" [查看答案]({CARD_URL}/answer) ![图](https://example.org/other)"
    )
    event = await onebot_event(
        [
            {"type": "markdown", "data": {"markdown": text}},
            {"type": "text", "data": {"text": text}},
            *extra_segments,
        ]
    )
    (reply,) = await dispatch_event(plugin, event)
    (component,) = reply.chain
    assert isinstance(component, adapter.Plain)
    assert component.text in NOT_FOUND_REPLIES
    resolve_image.assert_awaited_once()
    assert resolve_image.await_args.args[0].file == expected_file
    plugin.service.recognize.assert_awaited_once()


async def test_other_senders_markdown_image_is_ignored(make_plugin, resolve_image):
    plugin = make_plugin({})
    card = {"type": "markdown", "data": {"content": MARKDOWN_QUESTION}}
    event = await onebot_event([card], sender=456)
    assert await dispatch_event(plugin, event) == []
    resolve_image.assert_not_awaited()
    plugin.service.recognize.assert_not_awaited()


async def test_separate_art_is_a_second_image_only_message(make_plugin, dispatch):
    plugin = make_plugin({"image_in_same_message": False})
    text_reply, image_reply = await dispatch(
        plugin, QUESTION.replace(" + ", "＋"), instance="qq", sender="2854203313"
    )
    (text,) = text_reply.chain
    (art,) = image_reply.chain
    assert text.text == "这应该是喵喵"
    assert base64.b64decode(await art.convert_to_base64()) == b"art"
    assert text_reply.use_t2i_ is False
    assert image_reply.use_t2i_ is False
    plugin.service.recognize.assert_awaited_once_with(b"question")


async def test_text_only_stops_at_first_match(make_plugin, adapter, dispatch, tmp_path):
    match = Match("009", "波波拉", "", tmp_path / "missing.png", 0.95)
    config = {"send_image": False, "image_in_same_message": False}
    plugin = make_plugin(config, real_service=True)
    index = Mock()
    plugin.service.index = index
    index.match_candidates.side_effect = [[], [match]]
    (reply,) = await dispatch(plugin, instance="qq", sender="2854203313", image_count=3)
    (part,) = reply.chain
    assert isinstance(part, adapter.Plain)
    assert "波波拉" in part.text
    assert reply.use_t2i_ is False
    assert index.match_candidates.call_count == 2


@pytest.mark.parametrize(
    "case",
    [
        {"text": "我是谁"},
        {"text": QUESTION.splitlines()[1]},
        {"platform": "telegram"},
        {"image_count": 0},
        {"group": "778"},
    ],
    ids=["title-only", "prompt-only", "platform", "no-image", "group"],
)
async def test_only_process_matching_images_from_game_bot(plugin, dispatch, case):
    assert await dispatch(plugin, **case) == []
    plugin.service.recognize.assert_not_awaited()


async def test_low_scores_reply_once_with_only_text(make_plugin, adapter, dispatch):
    plugin = make_plugin({"image_in_same_message": False})
    plugin.service.recognize.return_value = RecognitionReply(
        "这个我真没印象", None, matched=False
    )
    (reply,) = await dispatch(plugin, instance="qq", sender="2854203313", image_count=2)
    assert reply.use_t2i_ is False
    (component,) = reply.chain
    assert isinstance(component, adapter.Plain)
    assert component.text == "这个我真没印象"
    assert plugin.service.recognize.await_count == 2


async def test_group_isolation_and_private_session(plugin, make_plugin, dispatch):
    assert await dispatch(plugin, session_id="123_777")
    scoped = make_plugin(
        {
            "bot_sids": ["qq-a:FriendMessage:123", "qq-a:FriendMessage:456"],
            "allowed_sids": ["qq-a:GroupMessage:123_777", "qq-a:FriendMessage:123"],
        }
    )
    assert await dispatch(scoped, session_id="123_777")
    assert not await dispatch(scoped, sender="456", session_id="456_777")
    assert await dispatch(scoped, group=None)
    assert not await dispatch(scoped, group=None, sender="456")


async def test_sender_sid_is_scoped_to_platform_instance(make_plugin, dispatch):
    plugin = make_plugin({"bot_sids": ["qq-a:FriendMessage:123"]})
    assert await dispatch(plugin)
    assert not await dispatch(plugin, instance="qq-b")
    assert not await dispatch(plugin, sender="2854203313", instance="qq")


async def test_legacy_empty_bot_list_restores_default_in_group_and_private_messages(
    make_plugin, adapter, dispatch, tmp_path
):
    schema_path = Path(adapter.__file__).with_name("_conf_schema.json")
    schema = json.loads(schema_path.read_text("utf-8"))
    assert schema["image_in_same_message"]["condition"] == {"send_image": True}
    path = tmp_path / "config.json"
    config = adapter.AstrBotConfig(str(path), schema=schema)
    expected = ["qq:FriendMessage:2854203313"]
    assert config["bot_sids"] == expected
    assert config["send_image"] is True
    assert config["image_in_same_message"] is True

    config["bot_sids"] = []
    config.save_config()
    plugin = make_plugin(config)
    assert json.loads(path.read_text("utf-8-sig"))["bot_sids"] == expected
    assert await dispatch(plugin, sender="2854203313", instance="qq")
    assert await dispatch(plugin, sender="2854203313", instance="qq", group=None)
    assert not await dispatch(plugin, sender="2854203313", instance="qq-a")
    assert not await dispatch(plugin, sender="123", instance="qq")


@pytest.mark.parametrize(
    "config",
    [
        {"bot_sids": ["2854203313", "qq-a:GroupMessage:777", "qq-a:FriendMessage:"]},
        {"allowed_sids": ["777"]},
        {"allowed_sids": ":GroupMessage:777"},
    ],
    ids=["invalid-bot-sids", "invalid-session", "non-list"],
)
async def test_invalid_sid_does_not_broaden_scope(make_plugin, dispatch, config):
    plugin = make_plugin(config)
    assert not await dispatch(plugin, sender="2854203313", instance="qq")
