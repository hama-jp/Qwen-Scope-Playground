"""Feature notes — small SQLite-backed CRUD store.

Lives next to corpus.db in `backend/data/notes.db` so a single `data/`
directory holds all v0.2 persistence. Schema is created lazily on first
access; ALTER TABLE migrations can be added later if needed.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import List, Optional

from .schemas import Note, NoteCreate

DB_PATH = Path(__file__).resolve().parent / "data" / "notes.db"


def _open() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            layer INTEGER NOT NULL,
            feature_idx INTEGER NOT NULL,
            label TEXT NOT NULL,
            memo TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_notes_layer_idx "
        "ON notes(layer, feature_idx)"
    )
    return con


def _row_to_note(row) -> Note:
    return Note(
        id=row[0], layer=row[1], feature_idx=row[2],
        label=row[3], memo=row[4], created_at=row[5],
    )


def list_notes(layer: Optional[int] = None) -> List[Note]:
    con = _open()
    try:
        cur = con.cursor()
        if layer is None:
            cur.execute(
                "SELECT id, layer, feature_idx, label, memo, created_at "
                "FROM notes ORDER BY id DESC"
            )
        else:
            cur.execute(
                "SELECT id, layer, feature_idx, label, memo, created_at "
                "FROM notes WHERE layer = ? ORDER BY id DESC",
                (layer,),
            )
        return [_row_to_note(r) for r in cur.fetchall()]
    finally:
        con.close()


def create_note(payload: NoteCreate) -> Note:
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    con = _open()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO notes (layer, feature_idx, label, memo, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (payload.layer, payload.feature_idx, payload.label,
             payload.memo, created_at),
        )
        con.commit()
        new_id = cur.lastrowid
        return Note(
            id=int(new_id), layer=payload.layer, feature_idx=payload.feature_idx,
            label=payload.label, memo=payload.memo, created_at=created_at,
        )
    finally:
        con.close()


def delete_note(note_id: int) -> bool:
    con = _open()
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()
