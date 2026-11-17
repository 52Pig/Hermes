"""Load broker-specific gateway clients from explicit Hermes configuration."""

from __future__ import annotations

import importlib
from typing import Any, Dict

from .multi_gateway import (
    BrokerAlgoGateway,
    BrokerApiGateway,
    CtpGateway,
    FixGateway,
    HostedStrategyGateway,
    PTradeGateway,
)


_ADAPTERS = {
    "ptrade": PTradeGateway,
    "fix": FixGateway,
    "ctp": CtpGateway,
    "broker_api": BrokerApiGateway,
    "broker_algo": BrokerAlgoGateway,
    "hosted_strategy": HostedStrategyGateway,
}


def load_gateway_plugins(config: Any, gateways: Dict[str, Any]) -> Dict[str, Any]:
    """Load configured vendor clients using ``module:factory`` entry points.

    Each non-built-in channel uses a section named ``gateway:<channel>`` with
    ``factory=package.module:create_client``. All remaining section values are
    passed as keyword arguments to that factory.
    """

    channels = [
        name.strip().lower()
        for name in config.get("execution", "channels", fallback="mini_qmt").split(",")
        if name.strip()
    ]
    for channel in channels:
        if channel in gateways:
            continue
        section_name = f"gateway:{channel}"
        if not config.has_section(section_name):
            raise ValueError(f"missing configuration section [{section_name}]")
        section = dict(config.items(section_name))
        factory_path = section.pop("factory", "")
        if ":" not in factory_path:
            raise ValueError(f"[{section_name}] factory must be module:callable")
        module_name, factory_name = factory_path.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        client = factory(**section)
        try:
            adapter = _ADAPTERS[channel]
        except KeyError as exc:
            raise ValueError(f"unsupported execution channel: {channel}") from exc
        gateway = adapter(client)
        if not callable(getattr(gateway, "submit", None)) or not callable(getattr(gateway, "cancel", None)):
            raise TypeError(f"gateway plugin {channel} does not implement submit/cancel")
        gateways[channel] = gateway
    return gateways
