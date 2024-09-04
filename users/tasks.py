from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string

from config import celery_app

import logging

User = get_user_model()
logger = logging.getLogger(__name__)


@celery_app.task()
def send_password_request_email(guid, reset_url):
    """Send password reset email to user."""
    try:
        user = User.objects.get(guid=guid)
    except User.DoesNotExist:
        logger.error(f"User with guid {guid} not found.")
        return

    try:
        email_body_html = render_to_string('password_reset_request.html', {
            'username': user.username,
            'user_email': user.email,
            'reset_url': reset_url,
        })

        send_mail(
            subject='Password Reset Requested',
            message="",
            html_message=email_body_html,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False
        )

    except Exception as e:
        logger.error(f"Error sending password reset email: {e}")
        return
