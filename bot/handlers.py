"""
bot/handlers.py
Command and message handlers.

Commands:
  /start, /help
  /watch                 toggle real-time monitoring (chat admins only)
  /analyze [N|text]      analyse last N watched messages, ad-hoc text, or the replied-to message
  /report [N]            last N flagged messages with receipts
  /cluster               narrative clusters for this chat

Works in private chats, groups and channels. Channel posts don't go through CommandHandler,
so `channel_command` parses them and dispatches to the same functions.
"""

from __future__ import annotations

import logging
import re

from telegram import (
    Message,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    Update,
)
from telegram.constants import ChatMemberStatus, ChatType
from telegram.ext import ContextTypes

from bot.formatter import (
    format_analyze_result,
    format_clusters,
    format_flag_alert,
    format_report,
    format_watch_off,
    format_watch_on,
)
from services.classifier import classify
from storage import db

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10
MAX_LIMIT = 50


# ── Helpers ──────────────────────────────────────────────────────────────────


def message_text(msg: Message) -> str | None:
    """Text of a message or the caption of a media post."""
    return msg.text or msg.caption


def parse_analyze_args(args: list[str] | None) -> tuple[int | None, str | None]:
    """`[]` -> (10, None); `["20"]` -> (20, None); anything else -> (None, joined text)."""
    if not args:
        return DEFAULT_LIMIT, None
    if len(args) == 1 and args[0].isdigit():
        return max(1, min(int(args[0]), MAX_LIMIT)), None
    return None, " ".join(args)


def extract_origin(msg: Message) -> dict:
    """Origin fields for storage/receipts: channel, forward source, t.me link."""
    origin: dict = {
        "chat_title": msg.chat.title or msg.chat.full_name,
        "tg_message_id": msg.message_id,
        "link": msg.link,
    }
    if msg.sender_chat:
        origin["sender_chat_id"] = msg.sender_chat.id
        origin["sender_chat_title"] = msg.sender_chat.title or msg.sender_chat.username
    fwd = msg.forward_origin
    if isinstance(fwd, MessageOriginChannel):
        origin.update(
            fwd_chat_id=fwd.chat.id, fwd_chat_title=fwd.chat.title, fwd_chat_username=fwd.chat.username
        )
    elif isinstance(fwd, MessageOriginChat):
        origin.update(
            fwd_chat_id=fwd.sender_chat.id,
            fwd_chat_title=fwd.sender_chat.title,
            fwd_chat_username=fwd.sender_chat.username,
        )
    elif isinstance(fwd, MessageOriginUser):
        origin["fwd_sender_name"] = fwd.sender_user.full_name
    elif isinstance(fwd, MessageOriginHiddenUser):
        origin["fwd_sender_name"] = fwd.sender_user_name
    return origin


async def _reply(update: Update, html: str) -> None:
    await update.effective_message.reply_html(html, disable_web_page_preview=True)


async def _is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    if chat.type in (ChatType.PRIVATE, ChatType.CHANNEL):
        return True  # only admins can post in a channel anyway
    msg = update.effective_message
    if msg.sender_chat and msg.sender_chat.id == chat.id:
        return True  # anonymous admin posting as the group
    user = update.effective_user
    if not user:
        return False
    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in (ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER)


# ── Commands ─────────────────────────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(
        update,
        "👁‍🗨 <b>Propaganda Watchdog</b>  —  DIAL 2026\n" + "━" * 25 + "\n"
        "I detect propaganda narratives in Telegram chats and channels.\n\n"
        "🔧 <b>Quick start:</b>\n"
        "  1. Add me to your group/channel as admin\n"
        "  2. Run /watch to start real-time monitoring\n"
        "  3. Use /report to see flagged messages\n\n"
        "Type /help for all commands.",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(
        update,
        "🤖 <b>Commands</b>\n" + "━" * 25 + "\n"
        "/watch       — Toggle real-time monitoring on/off (admins)\n"
        "/analyze     — Analyse last 10 watched messages\n"
        "/analyze 20  — Analyse last 20 watched messages\n"
        "/analyze &lt;text&gt; — Analyse a specific text\n"
        "/analyze (as a reply) — Analyse the replied-to message\n"
        "/report      — Show last 10 flagged messages (receipts)\n"
        "/report 20   — Show last 20 flagged messages\n"
        "/cluster     — Map narrative clusters\n"
        "/help        — Show this message\n" + "━" * 25 + "\n"
        "⚠️ Watch mode must be enabled to collect messages automatically.",
    )


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not await _is_admin(update, context):
        await _reply(update, "🔒 Only chat admins can toggle watch mode.")
        return
    if db.is_watch_enabled(chat_id):
        db.disable_watch(chat_id)
        await _reply(update, format_watch_off())
        logger.info("Watch disabled for chat %s", chat_id)
    else:
        db.enable_watch(chat_id)
        await _reply(update, format_watch_on())
        logger.info("Watch enabled for chat %s", chat_id)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    limit, text = parse_analyze_args(context.args)

    # /analyze as a reply → analyse the replied-to message
    replied = msg.reply_to_message
    if text is None and replied and message_text(replied):
        text = message_text(replied)

    if text is not None:
        result = await classify(text)
        user = update.effective_user
        msg_id = db.save_message(
            chat.id, user.id if user else None, user.username if user else None, text, source=db.SOURCE_ADHOC
        )
        if result.is_propaganda:
            db.save_flagged(
                msg_id, chat.id, result.narrative_label, result.confidence, result.cluster_id, result.backend
            )
        await _reply(update, format_analyze_result(text, result))
        return

    messages = db.get_recent_messages(chat.id, limit or DEFAULT_LIMIT)
    if not messages:
        await _reply(
            update,
            "📭 <b>No stored messages found.</b>\nEnable /watch first so I can collect messages, "
            "or use /analyze &lt;text&gt; to analyse a specific message.",
        )
        return

    await _reply(update, f"🔍 Analysing {len(messages)} stored message(s)…")
    hits = 0
    for row in messages:
        result = await classify(row["text"])
        if result.is_propaganda:
            hits += 1
            db.save_flagged(
                row["id"],
                chat.id,
                result.narrative_label,
                result.confidence,
                result.cluster_id,
                result.backend,
            )
            await _reply(update, format_flag_alert(row["text"], row["username"], result, row))
        else:
            db.unflag(row["id"])
    await _reply(
        update,
        f"✅ <b>Analysis complete.</b>\nChecked {len(messages)} message(s).  "
        f"Found <b>{hits}</b> propaganda hit(s).\nUse /report to see all receipts.",
    )


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    limit, _ = parse_analyze_args(context.args)
    rows = db.get_flagged_for_chat(update.effective_chat.id, limit or DEFAULT_LIMIT)
    await _reply(update, format_report(list(rows)))


async def cluster_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clusters = db.get_clusters_for_chat(update.effective_chat.id)
    await _reply(update, format_clusters(clusters))


COMMANDS = {
    "start": start_command,
    "help": help_command,
    "watch": watch_command,
    "analyze": analyze_command,
    "report": report_command,
    "cluster": cluster_command,
}

_CMD_RE = re.compile(r"^/(\w+)(?:@(\w+))?(?:\s+(.*))?$", re.DOTALL)


async def channel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch slash commands posted in channels (CommandHandler ignores channel posts)."""
    text = message_text(update.effective_message) or ""
    m = _CMD_RE.match(text.strip())
    if not m:
        return
    cmd, bot_name, rest = m.group(1).lower(), m.group(2), m.group(3)
    if bot_name and context.bot.username and bot_name.lower() != context.bot.username.lower():
        return
    handler = COMMANDS.get(cmd)
    if handler is None:
        return
    context.args = rest.split() if rest else []
    await handler(update, context)


# ── Real-time watcher ────────────────────────────────────────────────────────


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Every non-command text/caption in a watched chat: store, classify, alert if flagged."""
    msg = update.effective_message
    text = message_text(msg) if msg else None
    if not text:
        return
    chat_id = update.effective_chat.id
    if not db.is_watch_enabled(chat_id):
        return

    user = update.effective_user
    username = user.username if user else None
    origin = extract_origin(msg)
    msg_id = db.save_message(chat_id, user.id if user else None, username, text, **origin)

    result = await classify(text)
    logger.info("chat=%s msg=%s %s", chat_id, msg_id, result)
    if result.is_propaganda:
        db.save_flagged(
            msg_id, chat_id, result.narrative_label, result.confidence, result.cluster_id, result.backend
        )
        await _reply(update, format_flag_alert(text, username, result, origin))
