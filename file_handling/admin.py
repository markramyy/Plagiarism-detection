from django.contrib import admin
from file_handling.models import FileContent, UploadedFile, ZipFolder


class FileContentAdmin(admin.ModelAdmin):
    list_display = ['uploaded_file', 'content']
    search_fields = ['uploaded_file', 'content']
    list_filter = ['uploaded_file']


admin.site.register(FileContent, FileContentAdmin)


class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['file', 'file_type', 'processed', 'zip_folder']
    search_fields = ['file', 'file_type', 'processed']
    list_filter = ['file_type']


admin.site.register(UploadedFile, UploadedFileAdmin)


class ZipFolderAdmin(admin.ModelAdmin):
    list_display = ['name', 'processed']
    search_fields = ['name']


admin.site.register(ZipFolder, ZipFolderAdmin)
