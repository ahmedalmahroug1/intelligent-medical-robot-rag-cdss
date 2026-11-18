"""
app/concierge/accounts.py
-------------------------
Patient accounts for the REMOTE web channel: a username + password identifies a patient online,
the way face recognition identifies them at the robot. It parallels `identity.py:IdentityRegistry`
(file-backed JSON, minimal) but stores a salted password hash and maps each username to a stable
`patient_id`, so a returning patient's history is preserved across logins.

Passwords are hashed with the standard library only (`hashlib.pbkdf2_hmac` + a per-user random
salt) — no external dependency. This is demo-grade auth (see the security note in the plan): fine
for a graduation project, but a real deployment needs TLS, persistent sessions, and lockout.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from pathlib import Path

_DEFAULT_PATH = "data/patient_accounts.json"
_ITERATIONS = 200_000


def _norm(username: str) -> str:
    return username.strip().lower()


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _ITERATIONS).hex()


class PatientAccountRegistry:
    """A file-backed username -> {patient_id, name, salt, hash} store."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._accounts: dict[str, dict] = {}
        if self._path.exists():
            try:
                self._accounts = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt file starts empty, never crashes
                self._accounts = {}

    def register(self, username: str, password: str, name: str = "") -> str:
        """Create a new account and return its patient_id. Raises ValueError if the username is
        empty/taken or the password is empty."""
        key = _norm(username)
        if not key or not password:
            raise ValueError("username and password are required")
        if key in self._accounts:
            raise ValueError("username already taken")
        salt = secrets.token_hex(16)
        patient_id = uuid.uuid4().hex
        self._accounts[key] = {
            "patient_id": patient_id,
            "name": (name or username).strip(),
            "salt": salt,
            "hash": _hash(password, salt),
        }
        self._save()
        return patient_id

    def authenticate(self, username: str, password: str) -> str | None:
        """Return the patient_id for valid credentials, else None (constant-time compare)."""
        rec = self._accounts.get(_norm(username))
        if not rec:
            return None
        if secrets.compare_digest(_hash(password, rec["salt"]), rec["hash"]):
            return rec["patient_id"]
        return None

    def name_for_username(self, username: str) -> str:
        return self._accounts.get(_norm(username), {}).get("name", "")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._accounts, ensure_ascii=False, indent=1), encoding="utf-8")
