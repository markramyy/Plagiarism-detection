from django.conf import settings
from django.contrib.auth.tokens import default_token_generator

from rest_framework import serializers

from users.models import User
from users.tasks import send_password_request_email


class UserSignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'password_confirm'
        ]

    def validate(self, attrs):
        if User.objects.filter(username=attrs['username']).exists():
            raise serializers.ValidationError({
                "username": "Username already exists."
            })

        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                "password": "Password fields didn't match."
            })

        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class GetUserSerializer(serializers.Serializer):
    email = serializers.EmailField()
    first_name = serializers.CharField()
    last_name = serializers.CharField()
    guid = serializers.CharField()
    created = serializers.DateTimeField()
    modified = serializers.DateTimeField()
    last_login = serializers.DateTimeField()
    is_active = serializers.BooleanField()


class UpdateUserSerializer(serializers.Serializer):
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)

    def update(self, instance, validated_data):
        instance.first_name = validated_data.get(
            'first_name', instance.first_name
        )
        instance.last_name = validated_data.get(
            'last_name', instance.last_name
        )
        instance.save()
        return instance


class PasswordResetSerializer(serializers.Serializer):
    token = serializers.CharField()
    new_password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)
    guid = serializers.CharField()

    def validate(self, data):
        if data.get('new_password') != data.get('confirm_password'):
            raise serializers.ValidationError("Password does not match")

        if not data.get('token'):
            raise serializers.ValidationError("Token is required")

        try:
            self.user = User.objects.get(guid=data.get('guid'))
        except User.DoesNotExist:
            raise serializers.ValidationError(
                "Invalid or expired token for the user extraction."
            )

        if not default_token_generator.check_token(
            self.user, data.get('token')
        ):
            raise serializers.ValidationError("Invalid or expired token.")

        return data

    def save(self):
        self.user.set_password(self.validated_data.get('new_password'))
        self.user.save()
        return self.user


class ValidatePasswordSerializer(serializers.Serializer):
    password = serializers.CharField()

    def validate(self, data):
        if not self.instance.check_password(data.get('password')):
            raise serializers.ValidationError("Invalid Password")
        return data


class RequestNewPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()

    def validate_email(self, value):
        try:
            self.user = User.objects.get(email=value)
        except User.DoesNotExist:
            raise serializers.ValidationError("User does not exist")
        return value

    def save(self):
        token = default_token_generator.make_token(self.user)
        reset_url = (
            f"""
                {settings.FRONTEND_URL}/reset-password/{self.user.guid}/{token}/
            """
        )
        send_password_request_email.delay(self.user.guid, reset_url)
