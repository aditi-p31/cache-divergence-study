"""Tests for cachecheck against a fake OpenAI-compatible server.

Run: python -m unittest test_cachecheck   (from this folder)
"""

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

import cachecheck

P1, C1, P2 = 10, 5, 25
FULL_HIT = P1 + C1 - 1


class FakeServer:
    """Answers deterministically; the options below change its behaviour.

    diverge:           cache-hit turn-2 answers differ from recomputed ones
    diverge_reasoning: as diverge, but only in the reasoning text
    flaky_cache:       cache-path answers change on every call
    telemetry:         report cached tokens
    style:             "timings" (llama.cpp) or "usage" (vLLM)
    short_hit:         the cache hit stops at the turn-1 prompt
    leak:              the recompute path reports cached tokens
    reasoning_only:    answers have reasoning text and empty content
    fail:              every chat request returns HTTP 500
    slots:             total_slots reported by /props
    salted:            a hit needs a cache_salt seen earlier (two-servers)
    shortfall:         tokens a hit falls short by (block-sized caches)
    """

    def __init__(self, diverge=False, diverge_reasoning=False,
                 flaky_cache=False, telemetry=True, style="timings",
                 short_hit=False, leak=False, reasoning_only=False,
                 fail=False, slots=1, salted=False, shortfall=0):
        self.__dict__.update(locals())
        self.calls, self.counter, self.salts = [], 0, set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, payload):
                out = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self):
                if outer.slots is None:
                    self.reply(404, {"error": "no props"})
                else:
                    self.reply(200, {"total_slots": outer.slots,
                                     "build_info": "b-fake"})

            def do_POST(self):
                body = json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])))
                outer.calls.append(body)
                if outer.fail:
                    self.reply(500, {"error": "exceeds context size"})
                    return
                self.reply(200, outer.answer(body))

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def answer(self, body):
        msgs = body["messages"]
        turn2 = len(msgs) > 1
        salt = body.get("cache_salt")
        if self.salted:
            hit = turn2 and salt is not None and salt in self.salts
            if salt is not None:
                self.salts.add(salt)
        else:
            hit = turn2 and body.get("cache_prompt") is True
        text = "answer to: " + msgs[-1]["content"]
        reasoning = ""
        if hit and self.diverge:
            text += " (cache path)"
        if hit and self.diverge_reasoning:
            reasoning = "cache thoughts"
        if hit and self.flaky_cache:
            self.counter += 1
            text += " #%d" % self.counter
        if self.reasoning_only:
            reasoning, text = "thinking about " + text, ""
        cached = 0
        if hit:
            cached = P1 if self.short_hit else FULL_HIT - self.shortfall
        elif turn2 and self.leak:
            cached = 7
        data = {"choices": [{"message": {"content": text,
                                         "reasoning_content": reasoning},
                             "finish_reason": "stop"}]}
        usage = {"prompt_tokens": P2 if turn2 else P1, "completion_tokens": C1}
        if self.telemetry and self.style == "usage":
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        if self.telemetry and self.style == "timings":
            data["timings"] = {"cache_n": cached}
        data["usage"] = usage
        return data

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class CacheCheckTests(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for s in self.servers:
            s.close()
        self.tmp.cleanup()

    def server(self, **kw):
        s = FakeServer(**kw)
        self.servers.append(s)
        return s

    def run_tool(self, server, *extra, n="3"):
        out = os.path.join(self.tmp.name, "sub", "r.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cachecheck.main(["--base-url", server.url, "--n", n,
                                    "--out", out, "--timeout", "10"]
                                   + list(extra))
        with open(out) as fh:
            report = json.load(fh)
        return code, report, buf.getvalue()

    def test_no_divergence(self):
        code, report, out = self.run_tool(self.server())
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["probes_valid"], 3)
        self.assertEqual(report["summary"]["probes_diverged"], 0)
        self.assertIn("0 of 3 valid probes", out)
        self.assertEqual(report["server_props"]["build_info"], "b-fake")

    def test_divergence_detected(self):
        code, report, out = self.run_tool(self.server(diverge=True))
        self.assertEqual(code, 1)
        s = report["summary"]
        self.assertEqual((s["probes_valid"], s["probes_diverged"]), (3, 3))
        p = report["probes"][0]
        self.assertEqual(p["first_difference_char"],
                         len(p["recompute_path"][0]["content"]))
        self.assertIn("DIFFERENT (from character", out)
        self.assertIn("3 of 3 valid probes gave a different", out)

    def test_divergence_in_reasoning_only(self):
        code, report, out = self.run_tool(self.server(diverge_reasoning=True))
        self.assertEqual(code, 1)
        self.assertTrue(report["probes"][0]["reasoning_differs"])
        self.assertIn("in the reasoning text", out)

    def test_usage_style_telemetry(self):
        code, report, _ = self.run_tool(self.server(diverge=True,
                                                    style="usage"))
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["probes_valid"], 3)

    def test_unrepeatable_path_is_inconclusive(self):
        code, report, out = self.run_tool(self.server(diverge=True,
                                                      flaky_cache=True))
        self.assertEqual(code, 2)
        s = report["summary"]
        self.assertEqual((s["probes_valid"], s["probes_diverged"]), (0, 0))
        self.assertEqual(s["inconclusive_reasons"]["cache path did not repeat"],
                         3)
        self.assertIn("--cache-ram 0", out)

    def test_short_cache_hit_cannot_pass_as_same(self):
        # A hit that stops at the turn-1 prompt says nothing about the KV of
        # the generated answer, so "same" must not be reported as a pass.
        code, report, out = self.run_tool(self.server(short_hit=True))
        self.assertEqual(code, 2)
        self.assertEqual(report["summary"]["probes_valid"], 0)
        self.assertIn("cache hit too short: 3", out)

    def test_no_cache_hit_at_all_cannot_pass_as_same(self):
        # A server that ignores cache_prompt looks identical on both paths.
        srv = self.server(salted=True)
        code, report, _ = self.run_tool(srv)
        self.assertEqual(code, 2)
        self.assertEqual(report["summary"]["probes_valid"], 0)

    def test_recompute_leak_is_inconclusive(self):
        code, report, _ = self.run_tool(self.server(leak=True))
        self.assertEqual(code, 2)
        self.assertIn("recompute path reused 7 cached tokens",
                      report["probes"][0]["problems"])

    def test_no_telemetry_is_inconclusive(self):
        code, report, out = self.run_tool(self.server(telemetry=False))
        self.assertEqual(code, 2)
        self.assertEqual(report["summary"]["probes_valid"], 0)
        self.assertIn("cache hit not confirmed", out)

    def test_skip_hit_check(self):
        code, report, out = self.run_tool(self.server(telemetry=False),
                                          "--skip-hit-check")
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["hit_check"], "skipped")
        self.assertIn("assumed, not confirmed", out)

    def test_reasoning_only_answers_are_inconclusive(self):
        code, report, _ = self.run_tool(self.server(reasoning_only=True))
        self.assertEqual(code, 2)
        self.assertTrue(any("empty answer" in p
                            for p in report["probes"][0]["problems"]))

    def test_request_shapes_llamacpp(self):
        srv = self.server()
        self.run_tool(srv, n="1")
        flags = [(len(c["messages"]), c.get("cache_prompt"))
                 for c in srv.calls]
        # turn 1 fresh, turn 2 cached, twice; then turn 2 recomputed twice
        self.assertEqual(flags, [(1, False), (3, True), (1, False), (3, True),
                                 (3, False), (3, False)])
        for c in srv.calls:
            self.assertEqual(c["temperature"], 0)
            self.assertNotIn("cache_salt", c)
        self.assertEqual(srv.calls[1]["messages"][1]["role"], "assistant")

    def test_two_servers_mode(self):
        cached = self.server(diverge=True, salted=True, style="usage")
        plain = self.server(salted=True, style="usage")
        code, report, _ = self.run_tool(
            cached, "--mode", "two-servers", "--nocache-url", plain.url,
            "--model", "m", n="1")
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["probes_valid"], 1)
        self.assertTrue(all("cache_prompt" not in c
                            for c in cached.calls + plain.calls))
        # each iteration has its own salt, shared by its two turns
        salts = [c["cache_salt"] for c in cached.calls]
        self.assertEqual(salts[0], salts[1])
        self.assertEqual(salts[2], salts[3])
        self.assertNotEqual(salts[0], salts[2])
        self.assertTrue(all("cache_salt" not in c for c in plain.calls))
        self.assertEqual(len(plain.calls), 2)
        self.assertTrue(all(c["model"] == "m" for c in plain.calls))

    def test_block_sized_hit_passes_in_two_servers_mode(self):
        # Block-sized caches stop short of the last full block. The default
        # tolerance of 32 accepts that; a tolerance of 0 does not.
        cached = self.server(salted=True, style="usage", shortfall=10)
        plain = self.server(salted=True, style="usage")
        args = ["--mode", "two-servers", "--nocache-url", plain.url]
        self.assertEqual(self.run_tool(cached, *args, n="1")[0], 0)
        strict = args + ["--hit-tolerance", "0"]
        self.assertEqual(self.run_tool(cached, *strict, n="1")[0], 2)

    def test_slot_warning(self):
        _, _, out = self.run_tool(self.server(slots=4), n="1")
        self.assertIn("server has 4 slots", out)
        _, _, out = self.run_tool(self.server(slots=1), n="1")
        self.assertNotIn("slots.", out)

    def test_server_error_is_inconclusive_and_shows_body(self):
        code, report, out = self.run_tool(self.server(fail=True), n="2")
        self.assertEqual(code, 2)
        self.assertEqual(report["summary"]["probes_failed"], 2)
        self.assertIn("exceeds context size", out)

    def test_divergence_wins_over_a_failed_request(self):
        srv = self.server(diverge=True)
        real = cachecheck.post_chat
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 7:  # first request of probe 2
                raise cachecheck.RequestError("boom")
            return real(*a, **kw)

        cachecheck.post_chat = flaky
        try:
            code, report, _ = self.run_tool(srv, n="3")
        finally:
            cachecheck.post_chat = real
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["probes_failed"], 1)

    def test_too_few_valid_probes_is_inconclusive(self):
        srv = self.server()
        real = cachecheck.run_probe

        def partly_bad(probe, index, args, nonce):
            r = real(probe, index, args, nonce)
            if index > 1:
                r["valid"], r["problems"] = False, ["cache path did not repeat"]
            return r

        cachecheck.run_probe = partly_bad
        try:
            code, _, _ = self.run_tool(srv, n="3")
        finally:
            cachecheck.run_probe = real
        self.assertEqual(code, 2)

    def test_api_key_not_taken_from_openai_env(self):
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "secret"
        seen = []
        real = cachecheck.http_json

        def spy(url, body=None, api_key=None, timeout=300):
            seen.append(api_key)
            return real(url, body, api_key, timeout)

        cachecheck.http_json = spy
        try:
            self.run_tool(self.server(), n="1")
        finally:
            cachecheck.http_json = real
            if old is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = old
        self.assertNotIn("secret", seen)

    def test_first_difference(self):
        f = cachecheck.first_difference
        self.assertIsNone(f("abc", "abc"))
        self.assertEqual(f("abc", "abd"), 2)
        self.assertEqual(f("ab", "abc"), 2)

    def test_argument_errors(self):
        for argv in (["--base-url", "http://x", "--mode", "two-servers"],
                     ["--base-url", "http://x", "--n", "0"],
                     ["--base-url", "http://x", "--n", "999"]):
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()), \
                        redirect_stderr(io.StringIO()):
                    cachecheck.main(argv)


if __name__ == "__main__":
    unittest.main()
