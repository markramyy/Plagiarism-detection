from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.decorators import action

from core.base_viewset import BaseViewSet
from users.models import User
from users.serializers import (
    GetUserSerializer, UpdateUserSerializer,
    PasswordResetSerializer, ValidatePasswordSerializer,
    RequestNewPasswordSerializer
)
from rest_framework.permissions import AllowAny

import logging

logger = logging.getLogger(__name__)


class UserViewSet(BaseViewSet):
    queryset = User.objects.all()
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'User'

    def get_serializer_class(self):
        if self.action == 'password_reset':
            return PasswordResetSerializer
        elif self.action == 'validate_password':
            return ValidatePasswordSerializer
        elif self.action == 'me':
            return GetUserSerializer
        elif self.action == 'update':
            return UpdateUserSerializer
        elif self.action == 'request_new_password':
            return RequestNewPasswordSerializer

        return GetUserSerializer

    @action(detail=False, methods=['POST'], url_path='password-reset', url_name='password-reset', permission_classes=[AllowAny])
    def password_reset(self, request, *args, **kwargs):
        """Reset user password. send ?error=1 to simulate error response."""

        if 'error' in str(request.query_params):
            return Response({
                "message": "Invalid Opreation Just for testing purpose.",
                "data": []
            }, status=400)

        serializer = self.get_serializer_class()(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    "message": "Password reset successfully.",
                    "showToast": True,
                    "data": {},
                },
                status=200
            )
        else:
            return Response(
                {
                    "message": "Something went Wrong.",
                    "showToast": True,
                    "data": serializer.errors,
                },
                status=400
            )

    @action(detail=False, methods=['POST'], url_path='validate-password', url_name='validate-password')
    def validate_password(self, request, *args, **kwargs):
        """Validate user password. send ?error=1 to simulate error response."""
        serializer = self.get_serializer_class()(data=request.data, instance=request.user)

        if 'error' in str(request.query_params):
            return Response({
                "message": "Invalid Opreation Just for testing purpose.",
                "data": []
            }, status=400)

        if serializer.is_valid():
            return Response(
                {
                    "message": "Password validated successfully.",
                    "showToast": True,
                },
                status=200
            )
        else:
            return Response(
                {
                    "message": "Something went Wrong.",
                    "showToast": True,
                    "data": serializer.errors,
                },
                status=400
            )

    @action(detail=False, methods=['GET'], url_path='me', url_name='me')
    def me(self, request, *args, **kwargs):
        """Retrieve user details."""
        serializer = self.get_serializer_class()(request.user, context={"request": request})
        return Response(status=200, data=serializer.data)

    def update(self, request, *args, **kwargs):
        """Update user details. send ?error=1 to simulate error response."""
        serializer = self.get_serializer_class()(request.user, data=request.data)

        if 'error' in str(request.query_params):
            return Response({
                "message": "Invalid Opreation Just for testing purpose.",
                "data": []
            }, status=400)

        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    "message": "User details updated successfully.",
                    "showToast": True,
                    "data": serializer.data,
                },
                status=200
            )
        else:
            return Response(
                {
                    "message": "Something went Wrong.",
                    "showToast": True,
                    "data": serializer.errors,
                },
                status=400
            )

    @action(detail=False, methods=['POST'], url_path='request-password-reset', url_name='request-password-reset', permission_classes=[AllowAny])
    def request_new_password(self, request, *args, **kwargs):
        """Request new password. send ?error=1 to simulate error response."""
        if 'error' in str(request.query_params):
            return Response({
                "message": "Invalid Opreation Just for testing purpose.",
                "data": []
            }, status=400)

        serializer = self.get_serializer_class()(data=request.data, context={'request': request})

        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    "message": "New password reset requested successfully, Please check your email.",
                    "showToast": True,
                    "data": serializer.data,
                },
                status=200
            )
        else:
            return Response(
                {
                    "message": "Something went Wrong.",
                    "showToast": True,
                    "data": serializer.errors,
                },
                status=400
            )
