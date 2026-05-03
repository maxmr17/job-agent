"""
Authentication logic for JobAgent AI.
All password storage uses bcrypt (cost factor 12).
"""

import re
import bcrypt

from database import (
    create_user_record,
    deactivate_user,
    get_user_by_email,
    get_user_by_id,
    update_last_login,
    update_password_hash,
    update_user_name,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_email(email: str) -> str:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    return email


def _validate_password(password: str, label: str = "Password") -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"{label} must be at least {MIN_PASSWORD_LEN} characters.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_user(email: str, name: str, password: str, confirm_password: str) -> dict:
    """
    Create a new account.

    Raises ValueError with a human-readable message on any validation failure
    so the caller can surface it directly in the UI.
    """
    email = _validate_email(email)
    name = name.strip()
    if len(name) < 2:
        raise ValueError("Name must be at least 2 characters.")
    if password != confirm_password:
        raise ValueError("Passwords do not match.")
    _validate_password(password)
    if get_user_by_email(email):
        raise ValueError("An account with this email already exists.")
    return create_user_record(email, name, hash_password(password))


def login_user(email: str, password: str) -> dict | None:
    """
    Authenticate and return the user dict on success, None on failure.
    Updates last_login on success.
    """
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    update_last_login(user["id"])
    return user


def change_password(user_id: str, old_password: str, new_password: str, confirm_new: str) -> None:
    """
    Change a user's password after verifying the old one.
    Raises ValueError on any failure.
    """
    if new_password != confirm_new:
        raise ValueError("New passwords do not match.")
    _validate_password(new_password, "New password")
    user = get_user_by_id(user_id)
    if not user or not verify_password(old_password, user["password_hash"]):
        raise ValueError("Current password is incorrect.")
    update_password_hash(user_id, hash_password(new_password))


def rename_user(user_id: str, new_name: str) -> None:
    """Update display name. Raises ValueError if name is too short."""
    new_name = new_name.strip()
    if len(new_name) < 2:
        raise ValueError("Name must be at least 2 characters.")
    update_user_name(user_id, new_name)


def delete_account(user_id: str, password: str) -> None:
    """
    Soft-delete a user account after password confirmation.
    Raises ValueError if the password is wrong.
    """
    user = get_user_by_id(user_id)
    if not user or not verify_password(password, user["password_hash"]):
        raise ValueError("Password is incorrect. Account not deleted.")
    deactivate_user(user_id)
