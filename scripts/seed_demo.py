"""
scripts/seed_demo.py — fill a database with synthetic multi-channel posts to exercise /cluster and /channels.

  BOT_DB_PATH=/tmp/demo.db uv run python scripts/seed_demo.py         # separate DB, render offline
  uv run python scripts/seed_demo.py                                   # into data/bot.db → visible in Telegram

Posts are classified with the real retrieval backend; channels are fictional (titles start with "DEMO").
Chat ids are negative fakes, so only `/cluster all` and `/channels` show them (not per-chat `/cluster`).
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.classifier import classify
from storage import db

CHANNELS = {
    -900001: "DEMO Z-Новости",
    -900002: "DEMO Правда о Западе",
    -900003: "DEMO Голос Донбасса",
    -900004: "DEMO Городские новости",
}
POSTS = [
    # (channel id, hours ago, text, forwarded-from username or None)
    (-900001, 2, "США строили биолаборатории в Украине для создания биооружия против славян", None),
    (
        -900002,
        3,
        "Пентагон признал: секретные биолаборатории на Украине разрабатывали патогены",
        "infodefGERMANY",
    ),
    (-900003, 5, "Біолабораторії США в Україні — доказ підготовки біологічної війни", None),
    (-900001, 26, "НАТО годами провоцировало Россию, расширяясь на восток. Война была неизбежна", None),
    (-900002, 27, "NATO expansion left Russia no choice but to defend itself", "RusBotschaft"),
    (-900003, 50, "Киевский режим восемь лет бомбил Донбасс, а Запад молчал о геноциде", None),
    (-900001, 52, "ВСУ обстреливают мирных жителей Донбасса, западные СМИ это скрывают", None),
    (-900002, 75, "Санкции ударили по Европе сильнее, чем по России: немецкие заводы закрываются", None),
    (-900001, 76, "Европа замерзает из-за собственных санкций, а Россия только богатеет", None),
    (-900002, 100, "Зеленский нелегитимен, его срок истёк, Запад держит его ради продолжения войны", None),
    (-900003, 101, "Крым вернулся домой по итогам референдума, это признано международным правом", None),
    (-900004, 4, "В городе открыли новую станцию метро, интервал движения поездов сократят до 3 минут", None),
    (-900004, 30, "Кабмін затвердив бюджет на наступний рік з дефіцитом 20%", None),
    (-900004, 60, "Абрамович утретє програв у Суді ЄС: його активи залишаться під санкціями", None),
]


async def main() -> None:
    db.init_db()
    random.seed(0)
    flagged = 0
    for chat_id, hours_ago, text, fwd in POSTS:
        result = await classify(text)
        msg_id = db.save_message(
            chat_id,
            None,
            None,
            text,
            chat_title=CHANNELS[chat_id],
            sender_chat_id=chat_id,
            sender_chat_title=CHANNELS[chat_id],
            fwd_chat_title=fwd,
            fwd_chat_username=fwd,
            link=f"https://t.me/c/{-chat_id}/{random.randint(100, 999)}",
        )
        ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
        conn = db.get_connection()
        with conn:
            conn.execute("UPDATE messages SET timestamp=? WHERE id=?", (ts, msg_id))
        conn.close()
        db.save_embedding(msg_id, result.embedding)
        if result.is_propaganda:
            flagged += 1
            db.save_flagged(
                msg_id, chat_id, result.narrative_label, result.confidence, result.cluster_id, result.backend
            )
        print(
            f"{'🚨' if result.is_propaganda else '✅'} {CHANNELS[chat_id][:22]:22s} {result.narrative_label[:40]:40s} {text[:50]}"
        )
    print(f"\nseeded {len(POSTS)} posts, {flagged} flagged → {db.db_path()}")


if __name__ == "__main__":
    asyncio.run(main())
