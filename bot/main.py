"""
bot/main.py — entry point.  `python bot/main.py`

Env (.env): TELEGRAM_BOT_TOKEN (required), CLASSIFIER_BACKEND (default: mock), BOT_DB_PATH (optional)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from bot.handlers import COMMANDS, channel_command, error_handler, message_handler
from services.classifier import backend_name
from storage.db import db_path, init_db

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def build_app(token: str):
    app = ApplicationBuilder().token(token).build()

    for name, handler in COMMANDS.items():
        app.add_handler(CommandHandler(name, handler))

    # Slash commands posted in channels (CommandHandler only covers messages).
    app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST & filters.COMMAND, channel_command))

    # Watcher: text or media caption, messages and channel posts, no commands.
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.CAPTION)
            & ~filters.COMMAND
            & (filters.UpdateType.MESSAGE | filters.UpdateType.CHANNEL_POST),
            message_handler,
        )
    )
    app.add_error_handler(error_handler)
    return app


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)

    init_db()
    app = build_app(token)

    logger.info("▶  Propaganda Watchdog Bot is running.")
    logger.info("   Classifier backend : %s", backend_name())
    logger.info("   Database           : %s", db_path())
    app.run_polling(allowed_updates=[Update.MESSAGE, Update.CHANNEL_POST])


if __name__ == "__main__":
    main()
