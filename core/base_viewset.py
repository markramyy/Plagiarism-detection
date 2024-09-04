from rest_framework import viewsets
from rest_framework.exceptions import APIException
from rest_framework.filters import OrderingFilter, SearchFilter
from rest_framework.authentication import SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication


from django.conf import settings
from django_filters.rest_framework import DjangoFilterBackend

from core.pagination import (
    StandardResultsSetPagination
)


class MethodNotAllowed(APIException):
    status_code = 405
    default_detail = 'Method not allowed in this app.'
    default_code = 'method_not_allowed'


class BaseViewSet(viewsets.GenericViewSet):
    ...
    lookup_field = 'guid'
    ...

    if settings.DEBUG:
        authentication_classes = [SessionAuthentication, JWTAuthentication]
    else:
        authentication_classes = [JWTAuthentication]

    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, OrderingFilter, SearchFilter]
    search_fields = ['name']
    ordering_fields = ['created_at', 'updated_at']
