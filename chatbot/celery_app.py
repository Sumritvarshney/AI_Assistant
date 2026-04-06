# ─────────────────────────────────────────────────────────────────────────────
# FILE 1: celery_app.py
# Place this in the same directory as agent2.py
# ─────────────────────────────────────────────────────────────────────────────

from celery import Celery
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

# --- CELERY CONFIGURATION ---
# Uses Redis as broker. Install: pip install celery redis
# Run worker: celery -A celery_app worker --loglevel=info
# Run beat:   celery -A celery_app beat --loglevel=info

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "spog_ai",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# --- EMAIL CONFIGURATION ---
EMAIL_HOST     = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM     = os.getenv("EMAIL_FROM")


def _build_email_html(response_text: str, user_name: str, query: str) -> str:
    """Build a clean HTML email body."""
    # Convert markdown bold to HTML
    import re
    html_response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response_text)
    # Convert newlines to <br>
    html_response = html_response.replace('\n', '<br>')

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 0; }}
            .container {{ max-width: 600px; margin: 30px auto; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .header {{ background: #1a1a2e; padding: 24px 32px; }}
            .header h1 {{ color: #fff; margin: 0; font-size: 20px; font-weight: 500; }}
            .header p {{ color: #aaa; margin: 4px 0 0; font-size: 13px; }}
            .query-box {{ background: #f0f4ff; border-left: 4px solid #4a6cf7; margin: 24px 32px 0; padding: 12px 16px; border-radius: 0 6px 6px 0; }}
            .query-box p {{ margin: 0; font-size: 13px; color: #555; }}
            .query-box strong {{ color: #333; }}
            .content {{ padding: 24px 32px; color: #333; font-size: 15px; line-height: 1.7; }}
            .footer {{ background: #f9f9f9; padding: 16px 32px; border-top: 1px solid #eee; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Spog.ai</h1>
                <p>Your response is ready, {user_name.split()[0] if user_name else 'there'}</p>
            </div>
            <div class="query-box">
                <p><strong>Your query:</strong> {query}</p>
            </div>
            <div class="content">
                {html_response}
            </div>
            <div class="footer">
                This email was sent by Spog.ai Assistant. You requested this response to be delivered to your inbox.
            </div>
        </div>
    </body>
    </html>
    """


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="send_response_email")
def send_response_email(self, to_email: str, user_name: str, query: str, response_text: str):
    """
    Celery task: sends the chatbot response to the user's email.
    Retries up to 3 times with 30s delay on failure.
    """
    try:
        print(f"[CELERY] Sending email to {to_email} for query: {query[:50]}...")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Spog.ai — Your requested response"
        msg["From"]    = EMAIL_FROM
        msg["To"]      = to_email

        # Plain text fallback
        plain = f"Hi {user_name},\n\nYour query: {query}\n\nResponse:\n{response_text}\n\n— Spog.ai Assistant"
        msg.attach(MIMEText(plain, "plain"))

        # HTML version
        html = _build_email_html(response_text, user_name, query)
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())

        print(f"[CELERY] Email sent successfully to {to_email}")
        return {"status": "sent", "to": to_email}

    except smtplib.SMTPAuthenticationError as e:
        print(f"[CELERY] SMTP Auth failed: {e}")
        raise self.retry(exc=e)

    except Exception as e:
        print(f"[CELERY] Email failed: {e}")
        raise self.retry(exc=e)
