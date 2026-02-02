# api/email_service.py
"""
Email service for sending auto-reply emails to leads using Resend.

Resend free tier: 100 emails/day, 3,000 emails/month
https://resend.com
"""
import os
from typing import Optional
from datetime import datetime

# Check if resend is available
try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False


def init_resend():
    """Initialize Resend with API key."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return False
    resend.api_key = api_key
    return True


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

                            <!-- CTA Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center; padding: 10px 0 30px;">
                                        <a href="https://geotracker.io"
                                           style="display: inline-block; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                                                  color: #ffffff; text-decoration: none; padding: 14px 32px;
                                                  border-radius: 8px; font-size: 16px; font-weight: 600;
                                                  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);">
                                            Learn More About GEO Optimization
                                        </a>
                                    </td>
                                </tr>
                            </table>

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


def send_lead_acknowledgment(
    to_email: str,
    company_name: str,
    service: str,
    contact_name: Optional[str] = None,
    from_email: str = "GEO Tracker <hello@geotracker.io>"
) -> dict:
    """
    Send an acknowledgment email to a lead.

    Args:
        to_email: Lead's email address
        company_name: Lead's company name
        service: Service they're interested in
        contact_name: Optional contact name
        from_email: From email address (must be verified domain in Resend)

    Returns:
        dict with success status and message/error
    """
    if not RESEND_AVAILABLE:
        return {
            "success": False,
            "error": "Resend library not installed. Run: pip install resend"
        }

    if not init_resend():
        return {
            "success": False,
            "error": "RESEND_API_KEY not configured"
        }

    try:
        html_content = get_onboarding_email_html(
            company_name=company_name,
            service=service,
            contact_name=contact_name
        )

        params = {
            "from": from_email,
            "to": [to_email],
            "subject": f"Welcome to GEO Tracker - We've Received Your Request!",
            "html": html_content,
            "reply_to": "support@geotracker.io"
        }

        response = resend.Emails.send(params)

        return {
            "success": True,
            "message_id": response.get("id"),
            "sent_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def is_email_service_configured() -> bool:
    """Check if email service is properly configured."""
    return RESEND_AVAILABLE and bool(os.getenv("RESEND_API_KEY"))
