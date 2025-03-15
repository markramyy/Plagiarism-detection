from django.conf import settings

from rest_framework.routers import DefaultRouter, SimpleRouter

from plagiarism_backend.api import PlagiarismViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("", PlagiarismViewSet, basename="plagiarism")

app_name = "plagiarism_backend"
urlpatterns = router.urls
