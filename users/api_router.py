from django.conf import settings
from django.urls import path, include
from django.contrib.auth import views as auth_views

from rest_framework.routers import DefaultRouter, SimpleRouter

from users.api import UserViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("", UserViewSet)


app_name = "users"
urlpatterns = [
    path("", include(router.urls)),

    # User submits email for reset password
    path(
        "password_reset/",
        auth_views.PasswordResetView.as_view(),
        name="password_reset",
    ),

    # Email sent to user with reset link
    path(
        "password_reset/done/",
        auth_views.PasswordResetDoneView.as_view(),
        name="password_reset_done",
    ),

    # User clicks on reset link
    path(
        "reset/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(),
        name="password_reset_confirm",
    ),

    # User successfully reset password
    path(
        "reset/done/",
        auth_views.PasswordResetCompleteView.as_view(),
        name="password_reset_complete",
    ),
]

urlpatterns += [
]
