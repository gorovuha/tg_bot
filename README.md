# Propaganda Watchdog Bot

> **DIAL 2026 Hackathon** · Problem 03 "Telegram Narrative Bot" · post-hackathon continuation

A Telegram bot that spots propaganda narratives in groups and channels, shows receipts for every flag,
and groups flagged messages into narrative clusters.

Status: hackathon skeleton fixed up. The classifier is still a
**deterministic keyword mock** — every result is marked as such. Phase 1–2 replace it with embedding
retrieval over the EUvsDisinfo case database.

## Structure

```
bot/main.py            entry point (polling)
bot/handlers.py        commands + watcher; channel-post command dispatcher
bot/formatter.py       Telegram HTML rendering
services/classifier.py classify(text) -> ClassificationResult (backends: mock; retrieval planned)
storage/db.py          SQLite: messages (with origin/link), flagged (unique per message), watch_chats
tests/                 pytest
```

## Quick start

```bash
uv sync                       # Python 3.12 venv + deps (installs uv: brew install uv)
cp .env.example .env          # paste TELEGRAM_BOT_TOKEN from @BotFather
uv run python bot/main.py
```

In @BotFather run `/setprivacy` → your bot → **Disable**, otherwise the bot won't see group messages.
Add the bot to a group or channel **as admin**, then send `/watch`.


## Commands

| Command | Description |
|---|---|
| `/watch` | Toggle real-time monitoring (chat admins only) |
| `/analyze` / `/analyze 20` | Re-analyse last N watched messages |
| `/analyze <text>` | Analyse a specific text |
| `/analyze` as a reply | Analyse the replied-to message |
| `/report [N]` | Last N flagged messages with source and link |
| `/cluster` | Narrative clusters with source channels |

Commands work in private chats, groups and channels (channel posts are dispatched manually since
Telegram's command handler ignores them).

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

Env: `TELEGRAM_BOT_TOKEN`, `CLASSIFIER_BACKEND` (`mock`), `BOT_DB_PATH`.
