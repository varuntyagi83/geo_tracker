# api/user_service.py
"""
User authentication and registration service.
Handles webapp user signup, login, and JWT tokens.
"""
import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
import json
import base64

from db import (
    get_user_by_email, create_user, update_user_last_login,
    get_user_by_id, update_user_profile, get_users_count
)

# Secret key for JWT-like tokens (use env var in production)
USER_SECRET_KEY = os.getenv("USER_SECRET_KEY", "geo-tracker-user-secret-change-in-production")

# Token expiry (7 days for webapp users)
TOKEN_EXPIRY_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = USER_SECRET_KEY[:16]
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


def generate_user_token(user_id: int, email: str) -> str:
    """Generate a JWT-like token for user authentication."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": (datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat(),
        "nonce": secrets.token_hex(8)
    }
    payload_json = json.dumps(payload)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()

    # Create signature
    signature = hashlib.sha256(f"{payload_b64}{USER_SECRET_KEY}".encode()).hexdigest()[:32]

    return f"{payload_b64}.{signature}"


def verify_user_token(token: str) -> Optional[Dict]:
    """Verify and decode a user token. Returns payload if valid, None if invalid."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None

        payload_b64, signature = parts

        # Verify signature
        expected_sig = hashlib.sha256(f"{payload_b64}{USER_SECRET_KEY}".encode()).hexdigest()[:32]
        if signature != expected_sig:
            return None

        # Decode payload
        payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        payload = json.loads(payload_json)

        # Check expiry
        exp = datetime.fromisoformat(payload["exp"])
        if datetime.now(timezone.utc) > exp:
            return None

        return payload
    except Exception:
        return None


def register_user(email: str, password: str, name: str, company: Optional[str] = None) -> Optional[Dict]:
    """
    Register a new user.
    Returns token and user info if successful, None if email already exists.
    """
    # Check password requirements
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    # Hash the password
    password_hash = hash_password(password)

    try:
        # Create user
        user_id = create_user(email, password_hash, name, company)

        # Generate token
        token = generate_user_token(user_id, email)

        return {
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "name": name,
                "company": company,
            },
            "expires_in": TOKEN_EXPIRY_DAYS * 24 * 3600,  # seconds
        }
    except ValueError as e:
        # Email already registered
        raise e


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user.
    Returns token and user info if successful, None if failed.
    """
    user = get_user_by_email(email)
    if not user:
        return None

    if not user.get("is_active", True):
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    # Update last login
    update_user_last_login(user["id"])

    # Generate token
    token = generate_user_token(user["id"], user["email"])

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "company": user["company"],
        },
        "expires_in": TOKEN_EXPIRY_DAYS * 24 * 3600,  # seconds
    }


def get_user_from_token(token: str) -> Optional[Dict]:
    """Get user info from token."""
    payload = verify_user_token(token)
    if not payload:
        return None

    user = get_user_by_id(payload["user_id"])
    if not user or not user.get("is_active", True):
        return None

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "company": user["company"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
    }


def update_user(user_id: int, name: Optional[str] = None, company: Optional[str] = None) -> bool:
    """Update user profile."""
    return update_user_profile(user_id, name, company)


def get_total_users() -> int:
    """Get total registered users count."""
    return get_users_count()


# Initialize demo user on startup
def initialize_demo_user():
    """Initialize a demo user if it doesn't exist."""
    demo_email = "demo@geotracker.io"
    user = get_user_by_email(demo_email)
    if not user:
        try:
            create_user(demo_email, hash_password("demo123"), "Demo User", "Demo Company")
            print("[user_service] Created demo user: demo@geotracker.io")
        except ValueError:
            pass  # Already exists


def initialize_admin_dashboard_user():
    """Initialize an admin user for dashboard access if it doesn't exist."""
    admin_email = "admin@geotracker.io"
    user = get_user_by_email(admin_email)
    if not user:
        try:
            admin_password = os.getenv("ADMIN_PASSWORD", "geotracker2024!")
            create_user(admin_email, hash_password(admin_password), "Admin User", "GEO Tracker")
            print("[user_service] Created admin dashboard user: admin@geotracker.io")
        except ValueError:
            pass  # Already exists
