# Propaganda Watchdog Bot

> **DIAL 2026 Hackathon** · Problem 03 "Telegram Narrative Bot" · post-hackathon continuation

A Telegram bot that spots propaganda narratives in groups and channels, shows receipts for every flag,
and groups flagged messages into narrative clusters.

How it works: every message is embedded with a multilingual sentence model and compared against
19.7k documented pro-Kremlin disinformation cases from [EUvsDisinfo](https://euvsdisinfo.eu/) (2015–2026).
If the closest case is similar enough, the message is flagged and the bot replies with the matching
cases, their debunks and links (the receipts), plus the top-level narrative it belongs to. No LLM, no
external API — everything runs locally.

Evaluation (held-out EUvsDisinfo claims vs neutral news in EN/RU/UK, `scripts/evaluate.py`):
AUC 0.99, precision 0.93 / recall 0.96 at the calibrated threshold (0.825). Real Telegram posts are noisier
than the curated claims, so expect somewhat lower recall in the wild.

## Structure

```
bot/main.py            entry point (polling)
bot/handlers.py        commands + watcher; channel-post command dispatcher
bot/formatter.py       Telegram HTML rendering
services/classifier.py classify(text) -> ClassificationResult; backend switch (retrieval | mock)
services/retrieval.py  nearest-case search over the EUvsDisinfo index, receipts, narrative label
services/narratives.json  31 top-level narratives (EN/RU labels, prototype sentences)
storage/db.py          SQLite: messages (origin, link, embedding), flagged (unique per message), watch_chats
scripts/               fetch_euvsdisinfo.py (data), build_index.py (index), evaluate.py (metrics + threshold)
tests/                 pytest
```

## Quick start

```bash
uv sync --group data --group ml   # Python 3.12 venv + deps (install uv first: brew install uv)
cp .env.example .env              # paste TELEGRAM_BOT_TOKEN from @BotFather

# Build the narrative base once (~1 h of polite scraping, then ~2 min of embedding):
uv run python scripts/fetch_euvsdisinfo.py listing
uv run python scripts/fetch_euvsdisinfo.py reports
uv run python scripts/fetch_euvsdisinfo.py merge
uv run python scripts/build_index.py
uv run python scripts/evaluate.py    # optional: metrics + calibrated threshold

uv run python bot/main.py
```

The bot refuses to start with `CLASSIFIER_BACKEND=retrieval` (the default) if `data/index/` is missing,
so a broken setup can never silently degrade to the keyword mock.

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

Env: `TELEGRAM_BOT_TOKEN`, `CLASSIFIER_BACKEND` (`retrieval` | `mock`), `RETRIEVAL_THRESHOLD`, `BOT_DB_PATH`.

Data credits: EUvsDisinfo database (EU East StratCom Task Force); Mendeley dump by FloFloB (CC BY 4.0).
