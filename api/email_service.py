# api/email_service.py
"""
Email service for sending emails using SendGrid HTTP API.

SendGrid free tier: 100 emails/day, no recipient restrictions.

Setup instructions:
1. Go to https://sendgrid.com and create a free account
2. Go to Settings > Sender Authentication > Verify a Single Sender
3. Verify your email address (e.g., hi@corevisionailabs.com)
4. Go to Settings > API Keys > Create API Key (Full Access)
5. Set environment variables:
   - SENDGRID_API_KEY=SG.xxxxxxxxxx (your SendGrid API key)
   - SENDGRID_FROM_EMAIL=hi@corevisionailabs.com (your verified sender email)
   - ADMIN_NOTIFICATION_EMAILS=hi@corevisionailabs.com (comma-separated)
"""
import os
import threading
import urllib.request
import urllib.error
import json
from typing import Optional, List
from datetime import datetime


def get_sendgrid_api_key() -> str:
    """Get SendGrid API key from environment variables."""
    return os.getenv("SENDGRID_API_KEY", "")


def is_email_service_configured() -> bool:
    """Check if email service is properly configured."""
    return bool(get_sendgrid_api_key())


def get_admin_emails() -> List[str]:
    """Get list of admin emails to notify for new leads."""
    emails_str = os.getenv("ADMIN_NOTIFICATION_EMAILS", "")
    if not emails_str:
        return []
    return [email.strip() for email in emails_str.split(",") if email.strip()]


def send_email_sendgrid(
    to_emails: List[str],
    subject: str,
    html_content: str,
    reply_to: Optional[str] = None
) -> dict:
    """
    Send an email using SendGrid HTTP API.
    """
    api_key = get_sendgrid_api_key()

    if not api_key:
        return {
            "success": False,
            "error": "SendGrid API key not configured. Set SENDGRID_API_KEY env var."
        }

    try:
        from_email = os.getenv("SENDGRID_FROM_EMAIL", "hi@corevisionailabs.com")

        # SendGrid API format
        data = {
            "personalizations": [
                {
                    "to": [{"email": email} for email in to_emails]
                }
            ],
            "from": {
                "email": from_email,
                "name": "GEO Tracker"
            },
            "subject": subject,
            "content": [
                {
                    "type": "text/html",
                    "value": html_content
                }
            ]
        }

        if reply_to:
            data["reply_to"] = {"email": reply_to}

        json_data = json.dumps(data).encode("utf-8")

        print(f"[email] Sending via SendGrid: from={from_email}, to={to_emails}, subject={subject[:50]}...")

        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=json_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            # SendGrid returns 202 Accepted with empty body on success
            print(f"[email] SendGrid success: status {response.status}")
            return {
                "success": True,
                "message_id": f"sendgrid-{datetime.utcnow().timestamp()}",
                "sent_at": datetime.utcnow().isoformat()
            }

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except:
            error_body = str(e)
        print(f"[email] SendGrid error ({e.code}): {error_body}")
        return {
            "success": False,
            "error": f"SendGrid error ({e.code}): {error_body}"
        }
    except Exception as e:
        print(f"[email] Exception: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def get_onboarding_email_html(
    company_name: str,
    service: str,
    contact_name: Optional[str] = None
) -> str:
    """Generate the onboarding email HTML template."""
    greeting = f"Dear {contact_name}," if contact_name else f"Dear {company_name} Team,"

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f4f5;">
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <tr>
                        <td style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 40px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px;">GEO Tracker</h1>
                            <p style="margin: 8px 0 0; color: rgba(255, 255, 255, 0.9); font-size: 14px;">AI Visibility Intelligence Platform</p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 20px; color: #18181b; font-size: 24px;">Thank You for Your Interest!</h2>
                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">{greeting}</p>
                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                We've received your inquiry about <strong style="color: #4f46e5;">{service}</strong>.
                                Our team will help you understand and optimize your brand's visibility across AI platforms.
                            </p>
                            <p style="margin: 0 0 24px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                A member of our team will be in touch within <strong>24 hours</strong>.
                            </p>
                            <div style="background-color: #f4f4f5; border-radius: 8px; padding: 24px; margin-bottom: 24px;">
                                <h3 style="margin: 0 0 16px; color: #18181b; font-size: 18px;">What Happens Next?</h3>
                                <ol style="margin: 0; padding-left: 20px; color: #3f3f46; font-size: 15px; line-height: 1.8;">
                                    <li>We'll review your company profile</li>
                                    <li>Our team will prepare a consultation</li>
                                    <li>We'll reach out to schedule a call</li>
                                    <li>You'll receive AI visibility insights</li>
                                </ol>
                            </div>
                            <p style="margin: 0; color: #3f3f46; font-size: 16px;">
                                Best regards,<br><strong>The GEO Tracker Team</strong>
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="background-color: #f4f4f5; padding: 24px; text-align: center;">
                            <p style="margin: 0; color: #71717a; font-size: 14px;">GEO Tracker - AI Visibility Intelligence</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def get_admin_notification_html(
    company_name: str,
    email: str,
    service: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    contact_name: Optional[str] = None
) -> str:
    """Generate the admin notification email HTML template."""
    rows = f"""
        <tr><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Company</td><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #111827; font-weight: 600;">{company_name}</td></tr>
        <tr><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Email</td><td style="padding: 12px; border-bottom: 1px solid #e5e7eb;"><a href="mailto:{email}" style="color: #4f46e5;">{email}</a></td></tr>
    """
    if contact_name:
        rows += f'<tr><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Contact</td><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #111827;">{contact_name}</td></tr>'
    if website:
        rows += f'<tr><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Website</td><td style="padding: 12px; border-bottom: 1px solid #e5e7eb;"><a href="{website}" style="color: #4f46e5;">{website}</a></td></tr>'
    if industry:
        rows += f'<tr><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Industry</td><td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #111827;">{industry}</td></tr>'
    rows += f'<tr><td style="padding: 12px; color: #6b7280;">Service</td><td style="padding: 12px; color: #111827; font-weight: 600;">{service}</td></tr>'

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6;">
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden;">
                    <tr>
                        <td style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 24px 40px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 20px;">New Lead Received</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 32px 40px;">
                            <p style="margin: 0 0 24px; color: #374151; font-size: 16px;">A new lead has submitted the contact form:</p>
                            <table style="width: 100%; border-collapse: collapse; background-color: #f9fafb; border-radius: 8px;">
                                {rows}
                            </table>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def send_lead_acknowledgment(
    to_email: str,
    company_name: str,
    service: str,
    contact_name: Optional[str] = None
) -> dict:
    """Send an acknowledgment email to a lead."""
    html_content = get_onboarding_email_html(
        company_name=company_name,
        service=service,
        contact_name=contact_name
    )

    return send_email_sendgrid(
        to_emails=[to_email],
        subject="Welcome to GEO Tracker - We've Received Your Request!",
        html_content=html_content
    )


def send_admin_notification(
    company_name: str,
    email: str,
    service: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    contact_name: Optional[str] = None
) -> dict:
    """Send a notification email to admin(s) when a new lead comes in."""
    admin_emails = get_admin_emails()
    if not admin_emails:
        return {
            "success": False,
            "error": "No admin notification emails configured. Set ADMIN_NOTIFICATION_EMAILS env var."
        }

    html_content = get_admin_notification_html(
        company_name=company_name,
        email=email,
        service=service,
        website=website,
        industry=industry,
        contact_name=contact_name
    )

    result = send_email_sendgrid(
        to_emails=admin_emails,
        subject=f"[New Lead] {company_name} - {service}",
        html_content=html_content,
        reply_to=email
    )

    if result.get("success"):
        result["sent_to"] = admin_emails

    return result


def _send_emails_background(
    company_name: str,
    email: str,
    service: str,
    website: Optional[str],
    industry: Optional[str],
    contact_name: Optional[str]
):
    """Background thread function to send emails without blocking the API."""
    try:
        print(f"[email] Starting background email send for {email}...")

        # Send acknowledgment to lead
        lead_result = send_lead_acknowledgment(
            to_email=email,
            company_name=company_name,
            service=service,
            contact_name=contact_name
        )
        status = "sent" if lead_result.get("success") else lead_result.get("error")
        print(f"[email] Lead acknowledgment to {email}: {status}")

        # Send notification to admin(s)
        admin_result = send_admin_notification(
            company_name=company_name,
            email=email,
            service=service,
            website=website,
            industry=industry,
            contact_name=contact_name
        )
        status = "sent" if admin_result.get("success") else admin_result.get("error")
        print(f"[email] Admin notification: {status}")

    except Exception as e:
        print(f"[email] Background email error: {e}")


def send_lead_emails(
    company_name: str,
    email: str,
    service: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    contact_name: Optional[str] = None
) -> dict:
    """
    Send both lead acknowledgment AND admin notification emails.
    Emails are sent in a background thread to avoid blocking the API.
    """
    if not is_email_service_configured():
        return {
            "lead_email": {"success": False, "error": "SendGrid API key not configured"},
            "admin_email": {"success": False, "error": "SendGrid API key not configured"},
            "success": False
        }

    thread = threading.Thread(
        target=_send_emails_background,
        args=(company_name, email, service, website, industry, contact_name),
        daemon=True
    )
    thread.start()

    return {
        "lead_email": {"success": True, "message": "Email queued for sending"},
        "admin_email": {"success": True, "message": "Email queued for sending"},
        "success": True,
        "async": True
    }
