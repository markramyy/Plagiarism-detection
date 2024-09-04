from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import exceptions, serializers


class PasswordField(serializers.CharField):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("style", {})

        kwargs["style"]["input_type"] = "password"
        kwargs["write_only"] = True

        super().__init__(*args, **kwargs)


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fields[self.username_field] = serializers.CharField(
            write_only=True
        )
        self.fields["password"] = PasswordField()

    """
    Custom TokenObtainPairSerializer to add more fields to the token
    """
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims
        token['username'] = user.username
        token['email'] = user.email
        token['id'] = user.id
        token['guid'] = str(user.guid)

        return token

    def validate(self, attrs):
        try:
            data = super().validate(attrs)
        except Exception as e:
            print(e)
            raise exceptions.ValidationError("Wrong Username or Password")

        return data


class CustomTokenObtainPairView(TokenObtainPairView):
    """
    Custom TokenObtainPairView to add more fields to the token
    """
    serializer_class = CustomTokenObtainPairSerializer
