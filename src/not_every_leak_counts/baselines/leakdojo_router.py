"""Small local OpenAI-compatible router separating target and attacker models."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request


class Handler(BaseHTTPRequestHandler):
    endpoints = {}

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"object": "list", "data": [
                {"id": "generator", "object": "model"},
                {"id": "attacker", "object": "model"},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        model = payload.get("model")
        if model not in self.endpoints:
            self.send_error(400, f"unknown model {model}")
            return
        payload["model"] = model
        payload.setdefault("max_tokens", 1536)
        if model == "generator":
            payload.setdefault("reasoning_effort", "none")
        target = self.endpoints[model] + self.path.removeprefix("/v1")
        request = urllib.request.Request(target, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as error:
            body = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--attacker", required=True)
    args = parser.parse_args()
    Handler.endpoints = {"generator": args.generator, "attacker": args.attacker}
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
