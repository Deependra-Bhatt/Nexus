"""MCP client: connects to the four Nexus MCP servers over stdio (spec §5, §47)."""

import asyncio
import atexit
import concurrent.futures
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Protocol-level errors (unknown resource/prompt, invalid params) are not transport failures.
# The class is MCPError in MCP SDK 2.x and was McpError in 1.x.
try:
    from mcp.shared.exceptions import MCPError as McpError
except ImportError:  # pragma: no cover - MCP SDK 1.x
    try:
        from mcp.shared.exceptions import McpError
    except ImportError:

        class McpError(Exception):
            pass


from config import settings
from errors import (
    InvalidIdentifierError,
    MCPServerUnavailableError,
    MCPToolError,
    UnauthorizedFieldAccessError,
)
import progress
from logging_config import get_logger, trace

logger = get_logger("nexus.mcp_client")

BASE_DIR = Path(__file__).resolve().parent.parent

_SERVER_MODULES = {
    "banking": "mcp_servers.banking_server.server",
    "insurance": "mcp_servers.insurance_server.server",
    "wealth": "mcp_servers.wealth_server.server",
    "concierge_ops": "mcp_servers.concierge_ops_server.server",
}


def _server_environment() -> dict[str, str]:
    """Environment for a server subprocess.

    The MCP SDK otherwise replaces the environment with a small allowlist that drops
    DATABASE_PATH/CHROMA_PATH, which would silently point a server at a different database
    than the orchestrator is using. The Anthropic credentials are deliberately not forwarded:
    no MCP server calls the LLM (least privilege).
    """
    from mcp.client.stdio import get_default_environment

    env = dict(get_default_environment())
    for key in (
        "DATABASE_PATH",
        "CHROMA_PATH",
        "CHROMA_COLLECTION",
        "LOG_LEVEL",
        "LOG_DIR",
        "CONFIDENCE_THRESHOLD",
        "MAX_INPUT_LENGTH",
        "RETRIEVAL_TOP_K",
        "EMBEDDING_MODEL",
        "EMBEDDING_BACKEND",
        "HF_HUB_OFFLINE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "MCP_CALL_TIMEOUT_SECONDS",
        "MCP_CONNECT_TIMEOUT_SECONDS",
    ):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    # Explicitly blank so load_dotenv() inside the server (which never overrides an existing
    # variable) cannot re-import the key from .env: no MCP server may call the LLM.
    env["ANTHROPIC_API_KEY"] = ""
    return env


def _typed_tool_error(server_name: str, tool_name: str, detail) -> Exception:
    """Map a tool-side error back to the typed error it had inside the server (spec §36).

    FastMCP serialises any exception raised by a tool into an error result whose text is the
    exception message, so the typed distinction (ownership refusal / invalid identifier vs. a
    genuine tool failure) would otherwise be lost at the transport boundary and the customer
    would see a misleading "system unavailable" message.
    """
    text = str(detail)
    if "not accessible for this customer" in text:
        return UnauthorizedFieldAccessError(f"{server_name}.{tool_name}: {text}")
    if (
        (" not found" in text)
        or ("Invalid " in text and " format" in text)
        or ("Unknown document_id" in text)
    ):
        return InvalidIdentifierError(f"{server_name}.{tool_name}: {text}")
    return MCPToolError(f"{server_name}.{tool_name} returned an error: {text}")


def _is_error(result) -> bool:
    # MCP SDK >=2 exposes snake_case fields; <2 used camelCase.
    value = getattr(result, "is_error", None)
    if value is None:
        value = getattr(result, "isError", None)
    return bool(value)


def _extract_result(result) -> dict | list | str | None:
    # MCP SDK >=2 names the field structured_content; only consult the legacy camelCase name on
    # result objects that predate it (reading it on new objects emits a deprecation warning).
    if hasattr(type(result), "structured_content") or "structured_content" in getattr(
        result, "__dict__", {}
    ):
        structured = getattr(result, "structured_content", None)
    else:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        # FastMCP wraps non-object return values (lists, scalars) as {"result": value} to keep
        # structured content a JSON object. Unwrap so callers see the tool's real return type.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return text
    return None


class NexusMCPClient:
    """One client instance manages lazily-started connections to all four servers.

    Each server has its own exit stack so a failed/hung server can be dropped and restarted on
    the next call without tearing down the healthy ones.
    """

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        # One lock per server so concurrent first calls do not both spawn a server process.
        self._locks: dict[str, asyncio.Lock] = {}
        # Servers an operator (or the live demo's Stress Test 4) has stopped. Their process is
        # terminated and they are not restarted on the next call until start_server() is called.
        self._stopped: set[str] = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()

    async def aclose(self) -> None:
        for server_name in list(self._stacks):
            await self._evict(server_name)
        self._sessions.clear()

    async def _evict(self, server_name: str) -> None:
        """Forget a server's session (and best-effort close its transport) so the next call reconnects."""
        self._sessions.pop(server_name, None)
        stack = self._stacks.pop(server_name, None)
        if stack is None:
            return
        try:
            await stack.aclose()
        except (
            Exception
        ) as exc:  # the subprocess may already be gone; never mask the original error
            logger.debug(
                "MCP transport close failed during eviction",
                extra={
                    "payload": {
                        "event": "LOG",
                        "server": server_name,
                        "error": str(exc),
                    }
                },
            )

    async def stop_server(self, server_name: str) -> bool:
        """Terminate a server's process (by closing its stdio transport) and keep it down.

        Returns True if a running process was stopped. Used by the live demo's Stress Test 4 to
        take a real server offline, not just flip a flag in the in-process dispatcher.
        """
        if server_name not in _SERVER_MODULES:
            raise MCPServerUnavailableError(f"Unknown MCP server: {server_name!r}")
        self._stopped.add(server_name)
        was_running = server_name in self._sessions
        await self._evict(server_name)
        return was_running

    def start_server(self, server_name: str) -> None:
        """Allow a stopped server to be started again (lazily, on its next call)."""
        self._stopped.discard(server_name)

    def running_servers(self) -> list[str]:
        return sorted(self._sessions)

    async def warm_up(self, server_names=None) -> dict[str, str]:
        """Start the given (default: all) servers concurrently. Cold start is ~9-13 s per server;
        in parallel the four cost about one start instead of four in a row."""
        names = list(server_names or _SERVER_MODULES)

        async def _one(name):
            try:
                await self.connect(name)
                return name, "ready"
            except Exception as exc:
                return name, f"unavailable: {type(exc).__name__}"

        return dict(await asyncio.gather(*(_one(n) for n in names)))

    async def connect(self, server_name: str) -> ClientSession:
        if server_name in self._sessions:
            return self._sessions[server_name]
        if server_name not in _SERVER_MODULES:
            raise MCPServerUnavailableError(f"Unknown MCP server: {server_name!r}")
        if server_name in self._stopped:
            raise MCPServerUnavailableError(
                f"MCP server '{server_name}' is stopped (process terminated)"
            )

        lock = self._locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            if server_name in self._sessions:  # another task connected while we waited
                return self._sessions[server_name]

            # One retry, with a fresh process, when the stdio handshake times out: on a loaded
            # Windows host (antivirus scanning a fresh interpreter, four servers cold-starting at
            # once) the first initialize can exceed the budget although the server is healthy --
            # observed as TC-152 failing with TimeoutError in the 3-hour live run. Any other
            # failure (crash, bad module) is not retried.
            attempts = 1 + max(0, settings.MCP_CONNECT_RETRIES)
            for attempt in range(1, attempts + 1):
                try:
                    session, stack = await self._spawn(server_name)
                    break
                except MCPServerUnavailableError as exc:
                    timed_out = isinstance(
                        exc.__cause__, (asyncio.TimeoutError, TimeoutError)
                    )
                    if not timed_out or attempt == attempts:
                        raise
                    logger.warning(
                        "MCP server handshake timed out; retrying with a fresh process",
                        extra={
                            "payload": {
                                "event": "LOG",
                                "server": server_name,
                                "attempt": attempt,
                            }
                        },
                    )

            self._stacks[server_name] = stack
            self._sessions[server_name] = session
            return session

    async def _spawn(self, server_name: str):
        """Start one server subprocess and complete the MCP handshake; returns (session, stack)."""
        module = _SERVER_MODULES[server_name]
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", module],
            cwd=str(BASE_DIR),
            env=_server_environment(),
        )
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(
                session.initialize(), timeout=settings.MCP_CONNECT_TIMEOUT_SECONDS
            )
        except BaseException as exc:
            # Also on cancellation (e.g. the caller's timeout): close the half-built transport so
            # the spawned server process does not leak, then propagate.
            try:
                await stack.aclose()
            except Exception:
                pass
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise MCPServerUnavailableError(
                f"Could not start/connect to MCP server '{server_name}': {type(exc).__name__}: {exc}"
            ) from exc
        return session, stack

    async def _request(self, server_name: str, operation: str, make_call):
        session = await self.connect(server_name)
        try:
            return await asyncio.wait_for(
                make_call(session), timeout=settings.MCP_CALL_TIMEOUT_SECONDS
            )
        except McpError as exc:
            # The server answered with a protocol error (e.g. unknown resource): the session is healthy.
            raise MCPToolError(f"{server_name}.{operation} failed: {exc}") from exc
        except Exception as exc:
            # Broken pipe, server crash, timeout: drop the session so the next call restarts it.
            await self._evict(server_name)
            raise MCPServerUnavailableError(
                f"Connection to '{server_name}' failed during {operation}: {type(exc).__name__}: {exc}"
            ) from exc

    @trace(logger)
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        result = await self._request(
            server_name, "call_tool", lambda s: s.call_tool(tool_name, arguments)
        )
        if _is_error(result):
            raise _typed_tool_error(server_name, tool_name, _extract_result(result))
        return _extract_result(result)

    @trace(logger)
    async def read_resource(self, server_name: str, uri: str):
        result = await self._request(
            server_name, "read_resource", lambda s: s.read_resource(uri)
        )
        contents = getattr(result, "contents", [])
        if not contents:
            return None
        text = getattr(contents[0], "text", None)
        if text is None:
            return None
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return text

    @trace(logger)
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: dict):
        result = await self._request(
            server_name, "get_prompt", lambda s: s.get_prompt(prompt_name, arguments)
        )
        messages = getattr(result, "messages", [])
        texts = []
        for message in messages:
            content = getattr(message, "content", None)
            text = getattr(content, "text", None) if content is not None else None
            if text:
                texts.append(text)
        return "\n".join(texts)


def run_sync(coro):
    """Run an async MCP client coroutine from synchronous orchestrator/agent code."""
    return asyncio.run(coro)


async def call_tool(server_name: str, tool_name: str, arguments: dict):
    """Open a short-lived client, call one tool, close it. Spawns a server process per call."""
    async with NexusMCPClient() as client:
        return await client.call_tool(server_name, tool_name, arguments)


class _PersistentClient:
    """Keeps one client -- and therefore warm server processes -- alive for the process.

    Without this, every synchronous tool call spawned a fresh server subprocess and paid the
    full startup cost (~9s each), making a single cross-product request take ~30s. The client
    and its stdio contexts are driven from one dedicated event loop thread, because an
    AsyncExitStack must be unwound on the same loop that entered it.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: NexusMCPClient | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._loop is not None:
            return
        with self._lock:
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="nexus-mcp-loop", daemon=True
            ).start()
            self._client = NexusMCPClient()
            self._loop = loop
            atexit.register(self.close)

    def _submit(self, make_coro, server_name: str | None = None):
        """Start the loop first, then build the coroutine on the live client.

        The wait is bounded by MCP_CALL_TIMEOUT_SECONDS (which also covers a cold server start)
        so a stalled server never hangs the calling request indefinitely. On timeout the task is
        cancelled, which skips the in-task eviction, so the (possibly hung) server session is
        evicted here instead and the next call reconnects.
        """
        self._ensure_started()
        loop, client = self._loop, self._client
        future = asyncio.run_coroutine_threadsafe(make_coro(client), loop)
        timeout = settings.MCP_CALL_TIMEOUT_SECONDS
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            if server_name is not None and client is not None and loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        client._evict(server_name), loop
                    ).result(timeout=10)
                except Exception:
                    pass
            raise MCPServerUnavailableError(
                f"MCP call to '{server_name or 'client'}' did not complete within {timeout:g}s"
            ) from None

    def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        return self._submit(
            lambda c: c.call_tool(server_name, tool_name, arguments), server_name
        )

    def read_resource(self, server_name: str, uri: str):
        return self._submit(lambda c: c.read_resource(server_name, uri), server_name)

    def get_prompt(self, server_name: str, prompt_name: str, arguments: dict):
        return self._submit(
            lambda c: c.get_prompt(server_name, prompt_name, arguments), server_name
        )

    def warm_up(self, server_names=None) -> dict[str, str]:
        return self._submit(lambda c: c.warm_up(server_names))

    def stop_server(self, server_name: str) -> bool:
        return self._submit(lambda c: c.stop_server(server_name), server_name)

    def start_server(self, server_name: str) -> None:
        self._ensure_started()
        self._client.start_server(server_name)

    def running_servers(self) -> list[str]:
        if self._client is None:
            return []
        return self._client.running_servers()

    def close(self) -> None:
        loop, client = self._loop, self._client
        if loop is None:
            return
        self._loop, self._client = None, None
        try:
            asyncio.run_coroutine_threadsafe(client.aclose(), loop).result(timeout=10)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)


_persistent_client = _PersistentClient()


def call_tool_sync(server_name: str, tool_name: str, arguments: dict):
    """Call an MCP tool from synchronous code, reusing warm server processes."""
    return progress.track(
        server_name,
        "call_tool",
        tool_name,
        lambda: _persistent_client.call_tool(server_name, tool_name, arguments),
    )


def read_resource_sync(server_name: str, uri: str):
    return progress.track(
        server_name,
        "read_resource",
        uri.split("://", 1)[0] + "://",
        lambda: _persistent_client.read_resource(server_name, uri),
    )


def get_prompt_sync(server_name: str, prompt_name: str, arguments: dict):
    return progress.track(
        server_name,
        "get_prompt",
        prompt_name,
        lambda: _persistent_client.get_prompt(server_name, prompt_name, arguments),
    )


def shutdown_mcp_client() -> None:
    """Close the persistent client and stop its warm server processes."""
    _persistent_client.close()


def warm_up_mcp_servers(server_names=None) -> dict[str, str]:
    """Start the MCP servers in parallel ahead of the first request (UI start-up, live demo)."""
    return _persistent_client.warm_up(server_names)


def stop_mcp_server(server_name: str) -> bool:
    """Terminate one server process and keep it offline until start_mcp_server() (Stress Test 4)."""
    return _persistent_client.stop_server(server_name)


def start_mcp_server(server_name: str) -> None:
    _persistent_client.start_server(server_name)


def running_mcp_servers() -> list[str]:
    return _persistent_client.running_servers()
