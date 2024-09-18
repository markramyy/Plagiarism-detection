from django.conf import settings
from rest_framework.routers import DefaultRouter, SimpleRouter

from file_handling.api import FileUploadViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("", FileUploadViewSet)

app_name = "files"
urlpatterns = router.urls
