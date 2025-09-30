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
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server


ResponseValue = Any
Handler = Callable[[], ResponseValue]


class Request:
    def __init__(self, environ):
        self.method = environ.get("REQUEST_METHOD", "GET").upper()
        self.path = environ.get("PATH_INFO", "/")
        query_string = environ.get("QUERY_STRING", "")
        parsed_args = parse_qs(query_string)
        self.args = {key: values[0] if len(values) == 1 else values for key, values in parsed_args.items()}
        content_length = environ.get("CONTENT_LENGTH", "0") or "0"
        try:
            length = int(content_length)
        except ValueError:
            length = 0
        body = b""
        if length > 0:
            body = environ["wsgi.input"].read(length)
        self._body = body
        self._json_cache: Any = None

    @property
    def data(self) -> bytes:
        return self._body

    def get_json(self, silent: bool = False) -> Any:
        if not self._body:
            return {} if silent else None
        if self._json_cache is not None:
            return self._json_cache
        try:
            self._json_cache = json.loads(self._body.decode("utf-8"))
        except json.JSONDecodeError:
            if silent:
                return None
            raise
        return self._json_cache


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
    def _dispatch_request(self, environ) -> Response:
        global request
        if not self._before_executed:
            for hook in self._before_first_request:
                hook()
            self._before_executed = True

        req = Request(environ)
        request = req
        handler = self._routes.get((req.path, req.method))
        if handler is None:
            request = None
            return Response(b"Not Found", status=404)

        rv = handler()
        request = None
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
        response = self._dispatch_request(environ)
        reason = {200: 'OK', 404: 'NOT FOUND', 409: 'CONFLICT', 500: 'INTERNAL SERVER ERROR'}.get(
            response.status, 'OK'
        )
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


def jsonify(data: Any, status: int = 200) -> Response:
    body = json.dumps(data).encode("utf-8")
    return Response(body, status=status, headers=[("Content-Type", "application/json")])


request: Optional[Request] = None


__all__ = ["Flask", "jsonify", "Response", "request", "Request"]
