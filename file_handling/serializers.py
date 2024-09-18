from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field

from core.base_serializer import BaseSerializer
from file_handling.models import UploadedFile, FileContent, ZipFolder


class UploadedFileListSerializer(BaseSerializer):
    class Meta:
        model = UploadedFile
        fields = ['guid', 'file', 'file_type', 'zip_folder']
        read_only_fields = ['guid']


class ZipFolderSerializer(BaseSerializer):
    class Meta:
        model = ZipFolder
        fields = ['guid', 'name', 'zip_file']
        read_only_fields = ['guid']


class UploadedFileSerializer(BaseSerializer):
    class Meta:
        model = UploadedFile
        fields = ['guid', 'file', 'file_type']
        read_only_fields = ['guid']


class FileContentSerializer(BaseSerializer):
    class Meta:
        model = FileContent
        fields = ['guid', 'content']
        read_only_fields = ['guid']


class UploadedFileDetailSerializer(BaseSerializer):
    content = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = ['guid', 'file', 'file_type', 'zip_folder', 'content']
        read_only_fields = ['guid']

    @extend_schema_field(FileContentSerializer(many=True))
    def get_content(self, obj):
        content_qs = FileContent.objects.filter(uploaded_file=obj)
        return FileContentSerializer(content_qs, many=True).data
