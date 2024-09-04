from rest_framework_simplejwt.exceptions import AuthenticationFailed

from users.models import User


def custom_user_authentication_rule(user: User):

    if not user.is_active:
        raise AuthenticationFailed('User account is not active')

    return True
