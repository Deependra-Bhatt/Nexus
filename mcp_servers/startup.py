"""Process-level startup evidence for the MCP servers (spec §47).

Each server, when started as an independent stdio process, emits one SERVER_START event with its
PID, Python executable and registered surface to its log sink (stderr + logs/nexus.jsonl). stdout
is reserved for the MCP protocol, so nothing is printed there. The live demo
(`python run_local_demo.py --live-mcp`) and the logs therefore show four distinct server PIDs.
"""

import os
import sys


def announce_startup(logger, server_name: str, mcp) -> None:
    surface = {}
    for attr in ("tools", "resources", "prompts"):
        registry = getattr(mcp, attr, None)
        if isinstance(registry, dict):
            surface[attr] = len(registry)
    logger.info(
        "SERVER_START",
        extra={
            "payload": {
                "event": "SERVER_START",
                "server": server_name,
                "pid": os.getpid(),
                "python": sys.executable,
                "transport": "stdio",
                **({"surface": surface} if surface else {}),
            }
        },
    )
