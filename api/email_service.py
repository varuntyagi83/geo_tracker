# api/email_service.py
"""
Email service for sending emails using Gmail SMTP.

This service handles:
1. Auto-reply emails to leads (acknowledgment)
2. Admin notification emails when new leads come in

Gmail SMTP is free and doesn't require domain verification.
You just need to create an App Password in your Google account.

Setup instructions:
1. Go to Google Account > Security > 2-Step Verification (enable if not already)
2. Go to Google Account > Security > App passwords
3. Create a new app password for "Mail" on "Other (Custom name)" - call it "GEO Tracker"
4. Copy the 16-character password
5. Set environment variables:
   - GMAIL_USER=your-email@gmail.com
   - GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx (the 16-char app password)
   - ADMIN_NOTIFICATION_EMAILS=your-email@gmail.com (comma-separated for multiple)
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from datetime import datetime


# Gmail SMTP configuration
GMAIL_SMTP_SERVER = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


def get_gmail_credentials() -> tuple:
    """Get Gmail credentials from environment variables."""
    user = os.getenv("GMAIL_USER", "")
    password = os.getenv("GMAIL_APP_PASSWORD", "")
    return user, password


def is_email_service_configured() -> bool:
    """Check if email service is properly configured."""
    user, password = get_gmail_credentials()
    return bool(user and password)


def get_admin_emails() -> List[str]:
    """Get list of admin emails to notify for new leads."""
    emails_str = os.getenv("ADMIN_NOTIFICATION_EMAILS", "")
    if not emails_str:
        return []
    return [email.strip() for email in emails_str.split(",") if email.strip()]


def send_email_smtp(
    to_emails: List[str],
    subject: str,
    html_content: str,
    reply_to: Optional[str] = None
) -> dict:
    """
    Send an email using Gmail SMTP.

    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        html_content: HTML content of the email
        reply_to: Optional reply-to address

    Returns:
        dict with success status and message/error
    """
    user, password = get_gmail_credentials()

    if not user or not password:
        return {
            "success": False,
            "error": "Gmail credentials not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD env vars."
        }

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"GEO Tracker <{user}>"
        msg["To"] = ", ".join(to_emails)

        if reply_to:
            msg["Reply-To"] = reply_to

        # Attach HTML content
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)

        # Send via SMTP
        with smtplib.SMTP(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, to_emails, msg.as_string())

        return {
            "success": True,
            "message_id": f"gmail-{datetime.utcnow().timestamp()}",
            "sent_at": datetime.utcnow().isoformat()
        }

    except smtplib.SMTPAuthenticationError as e:
        return {
            "success": False,
            "error": f"Gmail authentication failed. Check GMAIL_USER and GMAIL_APP_PASSWORD. Error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_onboarding_email_html(
    company_name: str,
    service: str,
    contact_name: Optional[str] = None
) -> str:
    """
    Generate the onboarding email HTML template.

    Args:
        company_name: The lead's company name
        service: The service they're interested in
        contact_name: Optional contact name
    """
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
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 40px 40px 30px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">
                                GEO Tracker
                            </h1>
                            <p style="margin: 8px 0 0; color: rgba(255, 255, 255, 0.9); font-size: 14px;">
                                AI Visibility Intelligence Platform
                            </p>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 20px; color: #18181b; font-size: 24px; font-weight: 600;">
                                Thank You for Your Interest!
                            </h2>

                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                {greeting}
                            </p>

                            <p style="margin: 0 0 16px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                We've received your inquiry about <strong style="color: #4f46e5;">{service}</strong>.
                                Our team is excited to help you understand and optimize your brand's visibility
                                across AI platforms like ChatGPT, Claude, Gemini, and Perplexity.
                            </p>

                            <p style="margin: 0 0 24px; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                A member of our team will be in touch within <strong>24 hours</strong> to discuss
                                your specific needs and how we can help.
                            </p>

                            <!-- What's Next Box -->
                            <div style="background-color: #f4f4f5; border-radius: 8px; padding: 24px; margin-bottom: 24px;">
                                <h3 style="margin: 0 0 16px; color: #18181b; font-size: 18px; font-weight: 600;">
                                    What Happens Next?
                                </h3>
                                <ol style="margin: 0; padding-left: 20px; color: #3f3f46; font-size: 15px; line-height: 1.8;">
                                    <li style="margin-bottom: 8px;">We'll review your company profile and industry</li>
                                    <li style="margin-bottom: 8px;">Our team will prepare a personalized consultation</li>
                                    <li style="margin-bottom: 8px;">We'll reach out to schedule a call at your convenience</li>
                                    <li>You'll receive insights about your AI visibility landscape</li>
                                </ol>
                            </div>

                            <!-- Service Info -->
                            <div style="border-left: 4px solid #4f46e5; padding-left: 16px; margin-bottom: 24px;">
                                <p style="margin: 0 0 8px; color: #71717a; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                                    Your Selected Service
                                </p>
                                <p style="margin: 0; color: #18181b; font-size: 18px; font-weight: 600;">
                                    {service}
                                </p>
                            </div>

                            <p style="margin: 0; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                In the meantime, feel free to reply to this email if you have any questions.
                            </p>

                            <p style="margin: 24px 0 0; color: #3f3f46; font-size: 16px; line-height: 1.6;">
                                Best regards,<br>
                                <strong>The GEO Tracker Team</strong>
                            </p>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f4f4f5; padding: 24px 40px; text-align: center;">
                            <p style="margin: 0 0 8px; color: #71717a; font-size: 14px;">
                                GEO Tracker - AI Visibility Intelligence
                            </p>
                            <p style="margin: 0; color: #a1a1aa; font-size: 12px;">
                                Helping brands thrive in the age of AI search
                            </p>
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
    """
    Generate the admin notification email HTML template.

    This email is sent to you/your team when a new lead comes in.
    """
    website_row = f"""
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Website</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">
                                        <a href="{website}" style="color: #4f46e5; text-decoration: none;">{website}</a>
                                    </td>
                                </tr>
    """ if website else ""

    industry_row = f"""
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Industry</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">{industry}</td>
                                </tr>
    """ if industry else ""

    contact_row = f"""
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Contact</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">{contact_name}</td>
                                </tr>
    """ if contact_name else ""

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Lead: {company_name}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 24px 40px;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 20px; font-weight: 600;">
                                New Lead Received
                            </h1>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td style="padding: 32px 40px;">
                            <p style="margin: 0 0 24px; color: #374151; font-size: 16px; line-height: 1.6;">
                                A new lead has submitted the contact form on GEO Tracker. Here are the details:
                            </p>

                            <!-- Lead Details Table -->
                            <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f9fafb; border-radius: 8px; overflow: hidden; margin-bottom: 24px;">
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Company</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px; font-weight: 600;">{company_name}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Email</td>
                                    <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">
                                        <a href="mailto:{email}" style="color: #4f46e5; text-decoration: none;">{email}</a>
                                    </td>
                                </tr>
                                {contact_row}
                                {website_row}
                                {industry_row}
                                <tr>
                                    <td style="padding: 12px 16px; color: #6b7280; font-size: 14px; width: 120px;">Service</td>
                                    <td style="padding: 12px 16px; color: #111827; font-size: 14px; font-weight: 600;">{service}</td>
                                </tr>
                            </table>

                            <!-- Action Buttons -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center; padding: 10px 0;">
                                        <a href="mailto:{email}?subject=Re: Your GEO Tracker Inquiry&body=Hi {contact_name or company_name},%0D%0A%0D%0AThank you for your interest in GEO Tracker!%0D%0A%0D%0A"
                                           style="display: inline-block; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                                                  color: #ffffff; text-decoration: none; padding: 12px 24px;
                                                  border-radius: 8px; font-size: 14px; font-weight: 600;
                                                  margin-right: 12px;">
                                            Reply to Lead
                                        </a>
                                        <a href="https://geo-tracker-frontend.vercel.app/admin"
                                           style="display: inline-block; background-color: #f3f4f6;
                                                  color: #374151; text-decoration: none; padding: 12px 24px;
                                                  border-radius: 8px; font-size: 14px; font-weight: 600;
                                                  border: 1px solid #d1d5db;">
                                            View in Admin
                                        </a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f9fafb; padding: 16px 40px; text-align: center;">
                            <p style="margin: 0; color: #9ca3af; font-size: 12px;">
                                This notification was sent by GEO Tracker Lead Management System
                            </p>
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
    """
    Send an acknowledgment email to a lead.

    Args:
        to_email: Lead's email address
        company_name: Lead's company name
        service: Service they're interested in
        contact_name: Optional contact name

    Returns:
        dict with success status and message/error
    """
    html_content = get_onboarding_email_html(
        company_name=company_name,
        service=service,
        contact_name=contact_name
    )

    result = send_email_smtp(
        to_emails=[to_email],
        subject="Welcome to GEO Tracker - We've Received Your Request!",
        html_content=html_content
    )

    return result


def send_admin_notification(
    company_name: str,
    email: str,
    service: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    contact_name: Optional[str] = None
) -> dict:
    """
    Send a notification email to admin(s) when a new lead comes in.

    Args:
        company_name: Lead's company name
        email: Lead's email address
        service: Service they're interested in
        website: Optional company website
        industry: Optional industry/sector
        contact_name: Optional contact person name

    Returns:
        dict with success status and message/error
    """
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

    result = send_email_smtp(
        to_emails=admin_emails,
        subject=f"[New Lead] {company_name} - {service}",
        html_content=html_content,
        reply_to=email  # Reply goes directly to the lead
    )

    if result.get("success"):
        result["sent_to"] = admin_emails

    return result


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

    This is the main function to call when a new lead comes in.
    It handles both emails and returns combined results.

    Args:
        company_name: Lead's company name
        email: Lead's email address
        service: Service they're interested in
        website: Optional company website
        industry: Optional industry/sector
        contact_name: Optional contact person name

    Returns:
        dict with results for both emails
    """
    results = {
        "lead_email": None,
        "admin_email": None,
        "success": False
    }

    # 1. Send acknowledgment to lead
    lead_result = send_lead_acknowledgment(
        to_email=email,
        company_name=company_name,
        service=service,
        contact_name=contact_name
    )
    results["lead_email"] = lead_result

    # 2. Send notification to admin(s)
    admin_result = send_admin_notification(
        company_name=company_name,
        email=email,
        service=service,
        website=website,
        industry=industry,
        contact_name=contact_name
    )
    results["admin_email"] = admin_result

    # Consider success if at least the lead email was sent
    # (admin notification is nice-to-have)
    results["success"] = lead_result.get("success", False)

    return results
