"""
Stand-in for a real deployment system's API. In production this module
would call your actual orchestration platform (Kubernetes API, ECS,
internal deploy service, etc). Isolating it here means the MCP tool
functions in server.py stay thin wrappers, which is the right shape:
MCP tools should orchestrate and format, not contain business logic.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

_SERVICES = {
    "checkout-api": {"version": "v2.3.1", "healthy": True},
    "inventory-service": {"version": "v1.8.0", "healthy": True},
    "notifications-worker": {"version": "v0.9.4", "healthy": False}
}

_RUNBOOKS = {
    "checkout-api": (
        "1. Check p99 latency in the checkout-api dashboard.\n"
        "2. Verify the payments-gateway dependency is healthy.\n"
        "3. If latency spikes correlate with a recent deploy, roll back."
    ),
    "notifications-worker": (
        "1. Check the dead-letter queue depth first - this service backs up\n"
        " when the email provider is rate-limiting us.\n"
        "2. Verify SMTP_PROVIDER_KEY hasn't expired.\n"
        "3. Restart clears transient state but does not fix a queue backup."
    )
}


def list_services() -> list[str]:
    return list(_SERVICES.keys())


def get_status(service_name: str) -> dict:
    if service_name not in _SERVICES:
        raise ValueError(f"Unknown service: {service_name}")
    info = _SERVICES[service_name]
    deployed_at = datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48))
    return {
        "service": service_name,
        "version": info["version"],
        "healthy": info["healthy"],
        "deployed_at": deployed_at.isoformat()
    }


def restart(service_name: str) -> dict:
    if service_name not in _SERVICES:
        raise ValueError(f"Unknown service: {service_name}")
    _SERVICES[service_name]["healthy"] = True
    return {"service": service_name, "restarted": True}


def get_runbook(service_name: str) -> str:
    return _RUNBOOKS.get(service_name, f"No runbook on file for {service_name}.")