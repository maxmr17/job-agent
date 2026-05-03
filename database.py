"""
SQLite-backed multi-user data layer for JobAgent AI.

Schema:
  users             — accounts with bcrypt-hashed passwords
  applications      — one row per AI pipeline run, scoped to a user
  user_resumes      — saved resumes, one default per user
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = os.path.join("data", "jobagent.db")


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

APP_STATUSES = ["Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Declined", "Archived"]


def init_db() -> None:
    """Create all tables if they do not exist. Safe to call on every startup."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE NOT NULL,
                name         TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                last_login   TEXT,
                is_active    INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS applications (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                company_name    TEXT,
                job_description TEXT,
                resume_snapshot TEXT,
                result          TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_resumes (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                is_default  INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        # Migration-safe: add status and notes columns if they don't exist yet
        for col, defn in [
            ("status", "TEXT NOT NULL DEFAULT 'Applied'"),
            ("notes",  "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE applications ADD COLUMN {col} {defn}")
            except sqlite3.OperationalError:
                pass  # already exists
        # Performance indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_applications_user_id ON applications(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_applications_user_status ON applications(user_id, status)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

def create_user_record(email: str, name: str, password_hash: str) -> dict:
    """Insert a new user row and return it as a dict."""
    user_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?,?,?,?,?)",
            (user_id, email.lower().strip(), name.strip(), password_hash, now),
        )
    return {"id": user_id, "email": email, "name": name, "created_at": now, "last_login": None}


def get_user_by_email(email: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? AND is_active = 1",
            (email.lower().strip(),),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ? AND is_active = 1",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def update_last_login(user_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id),
        )


def update_user_name(user_id: str, name: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE users SET name = ? WHERE id = ?", (name.strip(), user_id))


def update_password_hash(user_id: str, new_hash: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))


def deactivate_user(user_id: str) -> None:
    """Soft-delete: mark is_active=0. Data is retained for audit/recovery."""
    with _conn() as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))


def get_all_users() -> list[dict]:
    """Admin use only — returns all active users."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, email, name, created_at, last_login FROM users WHERE is_active = 1 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Applications
# ─────────────────────────────────────────────────────────────────────────────

def save_application(user_id: str, job_description: str, resume: str, result: dict) -> str:
    """Persist one pipeline run. Returns the new application id."""
    app_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    company = (result.get("cover_letter_strategy") or {}).get("company_name", "")
    with _conn() as conn:
        conn.execute(
            """INSERT INTO applications
               (id, user_id, timestamp, company_name, job_description, resume_snapshot, result)
               VALUES (?,?,?,?,?,?,?)""",
            (app_id, user_id, now, company, job_description, resume,
             json.dumps(result, ensure_ascii=False)),
        )
    return app_id


def load_applications(user_id: str) -> list[dict]:
    """All applications for a user, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM applications WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
    result_list = []
    for r in rows:
        d = dict(r)
        result_list.append({
            "id": d["id"],
            "timestamp": d["timestamp"],
            "company_name": d["company_name"],
            "job_description": d["job_description"],
            "resume_snapshot": d["resume_snapshot"],
            "result": json.loads(d["result"]),
            "status": d.get("status", "Applied") or "Applied",
            "notes": d.get("notes", "") or "",
        })
    return result_list


def delete_application(app_id: str, user_id: str) -> None:
    """Hard-delete one application (user must own it)."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM applications WHERE id = ? AND user_id = ?",
            (app_id, user_id),
        )


def update_application_status(app_id: str, user_id: str, status: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE applications SET status = ? WHERE id = ? AND user_id = ?",
            (status, app_id, user_id),
        )


def update_application_notes(app_id: str, user_id: str, notes: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE applications SET notes = ? WHERE id = ? AND user_id = ?",
            (notes, app_id, user_id),
        )


def application_count(user_id: str) -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM applications WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Resumes
# ─────────────────────────────────────────────────────────────────────────────

def save_default_resume(user_id: str, text: str, name: str = "My Resume") -> None:
    """Upsert the user's default resume."""
    now = datetime.now().isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT id FROM user_resumes WHERE user_id = ? AND is_default = 1",
            (user_id,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE user_resumes SET name=?, resume_text=?, updated_at=? WHERE id=?",
                (name, text, now, existing["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO user_resumes (id, user_id, name, resume_text, is_default, created_at, updated_at)
                   VALUES (?,?,?,?,1,?,?)""",
                (str(uuid.uuid4()), user_id, name, text, now, now),
            )


def load_default_resume(user_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT name, resume_text FROM user_resumes WHERE user_id = ? AND is_default = 1",
            (user_id,),
        ).fetchone()
        return {"name": row["name"], "text": row["resume_text"]} if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Data export
# ─────────────────────────────────────────────────────────────────────────────

def export_user_data(user_id: str) -> dict:
    """Return all user data as a serializable dict for download."""
    user = get_user_by_id(user_id) or {}
    apps = load_applications(user_id)
    resume = load_default_resume(user_id)
    return {
        "exported_at": datetime.now().isoformat(),
        "user": {k: user.get(k) for k in ("id", "email", "name", "created_at", "last_login")},
        "applications": apps,
        "default_resume": resume,
    }
