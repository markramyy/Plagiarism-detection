from django.db import models
from users.models import User
from core.models import DBBase


class ZipFolder(DBBase):
    name = models.CharField(max_length=255)
    processed = models.BooleanField(default=False)

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='zip_folders',
        to_field='guid'
    )

    def __str__(self):
        return self.name


class UploadedFile(DBBase):
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10, choices=(('pdf', 'PDF'),))
    processed = models.BooleanField(default=False)

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='uploaded_files',
        to_field='guid'
    )
    zip_folder = models.ForeignKey(
        ZipFolder, on_delete=models.CASCADE, related_name='files',
        null=True, blank=True, to_field='guid'
    )

    def __str__(self):
        return self.file.name


class FileContent(DBBase):
    content = models.TextField()

    uploaded_file = models.ForeignKey(
        UploadedFile, on_delete=models.CASCADE, related_name='contents',
        to_field='guid'
    )

    def __str__(self):
        return f"Content for {self.uploaded_file.file.name}"
