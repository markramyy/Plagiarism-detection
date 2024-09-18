from rest_framework.response import Response
from rest_framework.decorators import action

from core.base_viewset import BaseViewSet
from file_handling.models import UploadedFile
from file_handling.serializers import (
    UploadedFileSerializer, ZipFolderSerializer,
    UploadedFileListSerializer, UploadedFileDetailSerializer
)
from file_handling.tasks import process_file

import logging

logger = logging.getLogger(__name__)


class FileUploadViewSet(BaseViewSet):
    queryset = UploadedFile.objects.all()

    def get_serializer_class(self):
        if self.action == 'list':
            return UploadedFileListSerializer
        elif self.action == 'retrieve':
            return UploadedFileDetailSerializer
        elif self.action == 'upload_file':
            return UploadedFileSerializer
        elif self.action == 'upload_zip':
            return ZipFolderSerializer

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

    @action(detail=False, methods=['POST'], url_path='upload-file', url_name='upload-file')
    def upload_file(self, request):
        serializer = self.get_serializer_class()(data=request.data)

        if serializer.is_valid():
            uploaded_file = serializer.save(user=request.user)
            process_file.delay(uploaded_file.guid)

            return Response(
                {
                    "message": "File uploaded and processing started.",
                    "data": serializer.data,
                },
                status=201
            )
        return Response(
            {
                "message": "Something went wrong.",
                "data": serializer.errors,
            },
            status=400
        )

    @action(detail=False, methods=['POST'], url_path='upload-zip', url_name='upload-zip')
    def upload_zip(self, request):
        zip_serializer = self.get_serializer_class()(data=request.data)

        if zip_serializer.is_valid():
            zip_folder = zip_serializer.save(user=request.user)
            files_to_create = []

            for file in request.FILES.getlist('files'):
                if file.name.endswith('.pdf'):
                    files_to_create.append(UploadedFile(
                        user=request.user,
                        file=file,
                        file_type='pdf',
                        zip_folder=zip_folder
                    ))

            UploadedFile.objects.bulk_create(files_to_create)

            # Process files in the background
            for uploaded_file in files_to_create:
                process_file.delay(uploaded_file.guid)

            return Response(
                {
                    "message": "ZIP uploaded and processing started.",
                    "data": zip_serializer.data,
                },
                status=201
            )
        return Response(
            {
                "message": "Something went wrong.",
                "data": zip_serializer.errors,
            },
            status=400
        )
