import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_DB_PATH", str(tmp_path / "test.db"))
    from storage import db

    db.init_db()
    return db
