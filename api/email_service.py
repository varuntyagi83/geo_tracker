# api/email_service.py
"""
Email service for sending emails using Resend HTTP API.

This service handles:
1. Auto-reply emails to leads (acknowledgment)
2. Admin notification emails when new leads come in

Using Resend's HTTP API since Railway blocks all outbound SMTP ports.

Setup instructions:
1. Go to https://resend.com and create a free account
2. Verify your domain in Resend
3. Get your API key from the dashboard
4. Set environment variables:
   - RESEND_API_KEY=re_xxxxxxxxxx (your Resend API key)
   - ADMIN_NOTIFICATION_EMAILS=your-email@example.com (comma-separated for multiple)
   - RESEND_FROM_EMAIL=noreply@yourdomain.com (your verified domain)
"""
import os
import threading
import urllib.request
import urllib.error
import json
from typing import Optional, List
from datetime import datetime


def get_resend_api_key() -> str:
    """Get Resend API key from environment variables."""
    return os.getenv("RESEND_API_KEY", "")


def is_email_service_configured() -> bool:
    """Check if email service is properly configured."""
    return bool(get_resend_api_key())


def get_admin_emails() -> List[str]:
    """Get list of admin emails to notify for new leads."""
    emails_str = os.getenv("ADMIN_NOTIFICATION_EMAILS", "")
    if not emails_str:
        return []
    return [email.strip() for email in emails_str.split(",") if email.strip()]


def send_email_resend(
    to_emails: List[str],
    subject: str,
    html_content: str,
    reply_to: Optional[str] = None
) -> dict:
    """
    Send an email using Resend HTTP API.

    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        html_content: HTML content of the email
        reply_to: Optional reply-to address

    Returns:
        dict with success status and message/error
    """
    api_key = get_resend_api_key()

    if not api_key:
        return {
            "success": False,
            "error": "Resend API key not configured. Set RESEND_API_KEY env var."
        }

    try:
        # Get from email from env - MUST be from verified domain
        from_email = os.getenv("RESEND_FROM_EMAIL", "noreply@corevisionailabs.com")

        # Prepare request data - Resend expects specific format
        data = {
            "from": from_email,
            "to": to_emails,
            "subject": subject,
            "html": html_content
        }

        if reply_to:
            data["reply_to"] = reply_to

        json_data = json.dumps(data).encode("utf-8")

        # Log the request for debugging
        print(f"[email] Sending to Resend API: from={from_email}, to={to_emails}, subject={subject[:50]}...")

        # Make HTTP request to Resend API
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            result = json.loads(response_body)
            print(f"[email] Resend API success: {result}")
            return {
                "success": True,
                "message_id": result.get("id"),
                "sent_at": datetime.utcnow().isoformat()
            }

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except:
            error_body = str(e)
        print(f"[email] Resend API error ({e.code}): {error_body}")
        return {
            "success": False,
            "error": f"Resend API error ({e.code}): {error_body}"
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
    <title>Welcome to GEO Tracker</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f4f4f5;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <tr>
                        <td style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 40px 40px 30px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">GEO Tracker</h1>
                            <p style="margin: 8px 0 0; color: rgba(255, 255, 255, 0.9); font-size: 14px;">AI Visibility Intelligence Platform</p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 20px; color: #18181b; font-size: 24px; font-weight: 600;">Thank You for Your Interest!</h2>
                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">{greeting}</p>
                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                We've received your inquiry about <strong style="color: #4f46e5;">{service}</strong>.
                                Our team is excited to help you understand and optimize your brand's visibility
                                across AI platforms like ChatGPT, Claude, Gemini, and Perplexity.
                            </p>
                            <p style="margin: 0 0 24px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                A member of our team will be in touch within <strong>24 hours</strong> to discuss your specific needs.
                            </p>
                            <div style="background-color: #f4f4f5; border-radius: 8px; padding: 24px; margin-bottom: 24px;">
                                <h3 style="margin: 0 0 16px; color: #18181b; font-size: 18px; font-weight: 600;">What Happens Next?</h3>
                                <ol style="margin: 0; padding-left: 20px; color: #3f3f46; font-size: 15px; line-height: 1.8;">
                                    <li style="margin-bottom: 8px;">We'll review your company profile and industry</li>
                                    <li style="margin-bottom: 8px;">Our team will prepare a personalized consultation</li>
                                    <li style="margin-bottom: 8px;">We'll reach out to schedule a call at your convenience</li>
                                    <li>You'll receive insights about your AI visibility landscape</li>
                                </ol>
                            </div>
                            <p style="margin: 24px 0 0; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                Best regards,<br><strong>The GEO Tracker Team</strong>
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="background-color: #f4f4f5; padding: 24px 40px; text-align: center;">
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
    website_row = f'<tr><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Website</td><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;"><a href="{website}" style="color: #4f46e5;">{website}</a></td></tr>' if website else ""
    industry_row = f'<tr><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Industry</td><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">{industry}</td></tr>' if industry else ""
    contact_row = f'<tr><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Contact</td><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">{contact_name}</td></tr>' if contact_name else ""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>New Lead: {company_name}</title></head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6;">
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden;">
                    <tr>
                        <td style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 24px 40px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 20px; font-weight: 600;">New Lead Received</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 32px 40px;">
                            <p style="margin: 0 0 24px; color: #374151; font-size: 16px;">A new lead has submitted the contact form:</p>
                            <table style="width: 100%; border-collapse: collapse; background-color: #f9fafb; border-radius: 8px; margin-bottom: 24px;">
                                <tr><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Company</td><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px; font-weight: 600;">{company_name}</td></tr>
                                <tr><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Email</td><td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;"><a href="mailto:{email}" style="color: #4f46e5;">{email}</a></td></tr>
                                {contact_row}
                                {website_row}
                                {industry_row}
                                <tr><td style="padding: 12px 16px; color: #6b7280; font-size: 14px;">Service</td><td style="padding: 12px 16px; color: #111827; font-size: 14px; font-weight: 600;">{service}</td></tr>
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

    return send_email_resend(
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

    result = send_email_resend(
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
    # Check if email service is configured
    if not is_email_service_configured():
        return {
            "lead_email": {"success": False, "error": "Resend API key not configured"},
            "admin_email": {"success": False, "error": "Resend API key not configured"},
            "success": False
        }

    # Start background thread to send emails
    thread = threading.Thread(
        target=_send_emails_background,
        args=(company_name, email, service, website, industry, contact_name),
        daemon=True
    )
    thread.start()

    # Return immediately - emails will be sent in background
    return {
        "lead_email": {"success": True, "message": "Email queued for sending"},
        "admin_email": {"success": True, "message": "Email queued for sending"},
        "success": True,
        "async": True
    }
