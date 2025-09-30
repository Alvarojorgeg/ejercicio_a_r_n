"""A lightweight subset of the Flask API used for this project.

The implementation is intentionally minimal and only supports the features
required by ``app/app.py``:

* Routing for GET requests via the ``route`` decorator.
* ``before_first_request`` hooks.
* ``jsonify`` helper that returns JSON responses.
* ``run`` method built on top of ``wsgiref`` for local development.

This module does **not** aim to be a drop-in replacement for the real Flask
framework, but it preserves the public symbols used in the assignment so the
application can be executed in restricted environments without external
dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from wsgiref.simple_server import make_server


ResponseValue = Any
Handler = Callable[[], ResponseValue]


class Response:
    def __init__(self, body: bytes, status: int = 200, headers: Optional[List[Tuple[str, str]]] = None):
        self.body = body
        self.status = status
        self.headers = headers or [("Content-Type", "text/html; charset=utf-8")]

    def __iter__(self) -> Iterable[bytes]:
        yield self.body


class Flask:
    def __init__(self, import_name: str):
        self.import_name = import_name
        self._routes: Dict[Tuple[str, str], Handler] = {}
        self._before_first_request: List[Callable[[], None]] = []
        self._before_executed = False

    # Decorators -----------------------------------------------------------------
    def route(self, rule: str, methods: Optional[List[str]] = None) -> Callable[[Handler], Handler]:
        methods = methods or ["GET"]

        def decorator(func: Handler) -> Handler:
            for method in methods:
                self._routes[(rule, method.upper())] = func
            return func

        return decorator

    def before_first_request(self, func: Callable[[], None]) -> Callable[[], None]:
        self._before_first_request.append(func)
        return func

    # Request handling -----------------------------------------------------------
    def _dispatch_request(self, path: str, method: str) -> Response:
        if not self._before_executed:
            for hook in self._before_first_request:
                hook()
            self._before_executed = True

        handler = self._routes.get((path, method))
        if handler is None:
            return Response(b"Not Found", status=404)

        rv = handler()
        return self.make_response(rv)

    def make_response(self, rv: ResponseValue) -> Response:
        if isinstance(rv, Response):
            return rv
        if isinstance(rv, tuple):
            body, status = rv
            return Response(self._to_bytes(body), status=status)
        return Response(self._to_bytes(rv))

    def _to_bytes(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return str(value).encode("utf-8")

    # WSGI integration -----------------------------------------------------------
    def wsgi_app(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()
        response = self._dispatch_request(path, method)
        reason = {200: 'OK', 404: 'NOT FOUND'}.get(response.status, 'OK')
        status_line = f"{response.status} {reason}"
        start_response(status_line, response.headers)
        return iter(response)

    __call__ = wsgi_app

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        with make_server(host, port, self.wsgi_app) as httpd:
            print(f" * Running on http://{host}:{port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n * Server stopped")


def jsonify(data: Dict[str, Any]) -> Response:
    body = json.dumps(data).encode("utf-8")
    return Response(body, headers=[("Content-Type", "application/json")])


__all__ = ["Flask", "jsonify", "Response"]
