"""
Authentication for the demo API.

PRODUCTION NOTE: this uses a static API-key lookup so wht whole project runs
without standing up an auth service. For real production traffic, replace
`get_current_user` with:
  - OAuth2 password/bearer flow or
  - JWT verification against your identity provider (Auth0, Cognito, etc.)
  - Rate limiting per-identity (e.g. via slowapi or an API gateway)
Th rest of this application (agent context, tool gating) does not change -
it only depends on receiving a `CurrentUser`, not on *how* you authenticated.
"""
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

# api_key -> (user_id, role). "guest" has no user_id - represents an
# unauthenticated visitor who has NOT verified their identity yet.
_DEMO_USERS: dict[str, tuple[str | None, str]] = {
    "guest-key": (None, "guest"),
    "customer-key-1": ("cust-1", "customer"),
    "customer-key-2": ("cust-2", "customer"),
    "admin-key": ("admin-1", "admin")
}


@dataclass(frozen=True)
class CurrentUser:
    user_id: str | None
    role: str # "guest" | "customer" | "admin"


async def get_current_user(x_api_key: str = Header(default="guest-key")) -> CurrentUser:
    if x_api_key not in _DEMO_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Try one of: guest-key, customer-key-1, customer-key-2, admin-key"
        )
    user_id, role = _DEMO_USERS[x_api_key]
    return CurrentUser(user_id=user_id, role=role)