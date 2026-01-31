# api/admin_service.py
"""
Admin authentication and authorization service.
Handles admin login, JWT tokens, and role-based access.
"""
import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
import json
import base64

from db import (
    get_admin_user, create_admin_user, update_admin_last_login,
    get_all_leads, get_lead_by_id, update_lead_status, get_leads_stats,
    insert_lead
)

# Secret key for JWT-like tokens (use env var in production)
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "geo-tracker-admin-secret-change-in-production")

# Token expiry (24 hours)
TOKEN_EXPIRY_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = ADMIN_SECRET_KEY[:16]
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


def generate_token(username: str, role: str) -> str:
    """Generate a simple token for admin authentication."""
    payload = {
        "username": username,
        "role": role,
        "exp": (datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat(),
        "nonce": secrets.token_hex(8)
    }
    payload_json = json.dumps(payload)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()

    # Create signature
    signature = hashlib.sha256(f"{payload_b64}{ADMIN_SECRET_KEY}".encode()).hexdigest()[:32]

    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode an admin token. Returns payload if valid, None if invalid."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None

        payload_b64, signature = parts

        # Verify signature
        expected_sig = hashlib.sha256(f"{payload_b64}{ADMIN_SECRET_KEY}".encode()).hexdigest()[:32]
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


def authenticate_admin(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate an admin user.
    Returns token and user info if successful, None if failed.
    """
    user = get_admin_user(username)
    if not user:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    # Update last login
    update_admin_last_login(username)

    # Generate token
    token = generate_token(username, user["role"])

    return {
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "expires_in": TOKEN_EXPIRY_HOURS * 3600,  # seconds
    }


def initialize_default_admin():
    """
    Initialize default admin and demo users if they don't exist.
    Called on startup.
    """
    # Check if admin exists
    admin = get_admin_user("admin")
    if not admin:
        # Create admin user with password from env or default
        admin_password = os.getenv("ADMIN_PASSWORD", "geotracker2024!")
        create_admin_user("admin", hash_password(admin_password), "admin")
        print("[admin] Created default admin user")

    # Check if demo user exists
    demo = get_admin_user("demo")
    if not demo:
        # Create demo user with fixed password
        create_admin_user("demo", hash_password("demo123"), "demo")
        print("[admin] Created demo user")


def get_user_permissions(role: str) -> Dict:
    """Get permissions based on user role."""
    if role == "admin":
        return {
            "can_view_leads": True,
            "can_view_emails": True,  # Full email addresses
            "can_update_leads": True,
            "can_delete_leads": True,
            "can_view_stats": True,
            "can_manage_users": True,
        }
    elif role == "demo":
        return {
            "can_view_leads": True,
            "can_view_emails": False,  # Masked emails only
            "can_update_leads": False,
            "can_delete_leads": False,
            "can_view_stats": True,
            "can_manage_users": False,
        }
    else:
        return {
            "can_view_leads": False,
            "can_view_emails": False,
            "can_update_leads": False,
            "can_delete_leads": False,
            "can_view_stats": False,
            "can_manage_users": False,
        }


# Convenience functions for lead management with role checks

def get_leads_for_role(role: str, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> list:
    """Get leads with appropriate data based on role."""
    permissions = get_user_permissions(role)
    if not permissions["can_view_leads"]:
        return []

    return get_all_leads(
        status=status,
        limit=limit,
        offset=offset,
        include_emails=permissions["can_view_emails"]
    )


def can_update_lead(role: str) -> bool:
    """Check if role can update leads."""
    return get_user_permissions(role).get("can_update_leads", False)
