from rest_framework.pagination import (
    PageNumberPagination, LimitOffsetPagination,
    CursorPagination
)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 100
    page_size_query_param = 'page_size'
    max_page_size = 100


class StandardLimitOffsetPagination(LimitOffsetPagination):
    default_limit = 1
    max_limit = 100


class StandardCursorPagination(CursorPagination):
    page_size = 10
    ordering = 'name'
    cursor_query_param = 'cursor'
    page_size_query_param = 'page_size'
    max_page_size = 100
