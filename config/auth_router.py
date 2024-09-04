from django.urls import path

from rest_framework_simplejwt.views import (
    TokenRefreshView,
)

from users.jwt import CustomTokenObtainPairView

app_name = "auth"
urlpatterns = [

    path('api/token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
