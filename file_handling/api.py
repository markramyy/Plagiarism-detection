from rest_framework.response import Response

from core.base_viewset import BaseViewSet
from file_handling.models import UploadedFile
from file_handling.serializers import (
    UploadedFileListSerializer, UploadedFileDetailSerializer
)

import logging

logger = logging.getLogger(__name__)


class FileUploadViewSet(BaseViewSet):
    queryset = UploadedFile.objects.all()

    def get_serializer_class(self):
        if self.action == 'list':
            return UploadedFileListSerializer
        elif self.action == 'retrieve':
            return UploadedFileDetailSerializer

        return UploadedFileListSerializer

    def list(self, request, *args, **kwargs):
        files = UploadedFile.objects.filter(user=request.user)

        if not files:
            return Response(
                {
                    "message": "No files found.",
                    "data": [],
                },
                status=404
            )

        page = self.paginate_queryset(files)
        if page is not None:
            serializer = self.get_serializer_class()(page, many=True)
            current_page = self.paginator.page.number
            last_page = self.paginator.page.paginator.num_pages

            response_data = {
                "count": self.paginator.page.paginator.count,
                "page": current_page,
                "last_page": last_page,
                "next": self.paginator.get_next_link(),
                "previous": self.paginator.get_previous_link(),
                "message": "Files fetched successfully paginated",
                "data": serializer.data
            }

            return Response(response_data, status=200)

        serializer = self.get_serializer_class()(files, many=True)
        return Response(
            {
                "message": "Files fetched successfully.",
                "data": serializer.data,
            },
            status=200
        )

    def retrieve(self, request, *args, **kwargs):
        file = self.get_object()

        if not file:
            return Response(
                {
                    "message": "File not found.",
                    "data": [],
                },
                status=404
            )

        serializer = self.get_serializer_class()(file)
        return Response(
            {
                "message": "File fetched successfully.",
                "data": serializer.data,
            },
            status=200
        )

    def destroy(self, request, *args, **kwargs):
        file = self.get_object()

        if not file:
            return Response(
                {
                    "message": "File not found.",
                    "data": [],
                },
                status=404
            )

        file.delete()
        return Response(
            {
                "message": "File deleted successfully.",
                "data": [],
            },
            status=204
        )
