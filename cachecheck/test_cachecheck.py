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

# Realistic sizes: the hit check of the non-llamacpp modes allows a
# shortfall of 32 tokens, so smaller numbers would let it pass vacuously.
P1, C1, P2 = 60, 40, 110


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
    ignore_salt:       with salted, every turn-2 request hits, whatever
                       its salt (a server that drops cache_salt)
    never_hit:         no request ever hits (a server with caching off that
                       still accepts cache_salt), cached tokens 0
    completion:        completion tokens reported for every answer
    shortfall:         tokens a hit falls short by (block-sized caches)
    """

    def __init__(self, diverge=False, diverge_reasoning=False,
                 flaky_cache=False, telemetry=True, style="timings",
                 short_hit=False, leak=False, reasoning_only=False,
                 fail=False, slots=1, salted=False, ignore_salt=False,
                 never_hit=False, completion=C1, shortfall=0):
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

    # Answers any fake server has given. A request is a turn 2 when it
    # extends one of them, so a turn 1 may carry its own earlier turns.
    produced = set()

    def answer(self, body):
        msgs = body["messages"]
        turn2 = (len(msgs) > 2 and msgs[-2]["role"] == "assistant"
                 and msgs[-2]["content"] in FakeServer.produced)
        salt = body.get("cache_salt")
        if self.never_hit:
            hit = False
        elif self.salted and self.ignore_salt:
            hit = turn2
        elif self.salted:
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
        if not turn2:
            FakeServer.produced.add(text)
        cached = 0
        if hit:
            full = P1 + self.completion - 1
            cached = P1 if self.short_hit else full - self.shortfall
        elif turn2 and self.leak:
            cached = 7
        data = {"choices": [{"message": {"content": text,
                                         "reasoning_content": reasoning},
                             "finish_reason": "stop"}]}
        usage = {"prompt_tokens": P2 if turn2 else P1,
                 "completion_tokens": self.completion}
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

    def test_result_block_is_printed_and_safe_to_share(self):
        server = self.server(diverge=True)
        code, report, out = self.run_tool(server)
        self.assertIn("---- llm-cachecheck result ----", out)
        self.assertIn("engine build: b-fake", out)
        self.assertIn("probes: 3 planned, 3 valid, 3 differed", out)
        self.assertIn(cachecheck.SHARE_URL, out)
        block = out.split("---- llm-cachecheck result ----")[1]
        self.assertNotIn(server.url, block)

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

    def test_one_server_mode(self):
        srv = self.server(diverge=True, salted=True, style="usage")
        code, report, out = self.run_tool(
            srv, "--mode", "one-server", "--model", "m", n="1")
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["probes_valid"], 1)
        self.assertEqual(report["mode"], "one-server")
        self.assertIn("mode: one-server", out)
        # 4 cache-path requests, then 2 recompute requests, all to one
        # server, none with cache_prompt
        self.assertEqual([len(c["messages"]) for c in srv.calls],
                         [1, 3, 1, 3, 3, 3])
        self.assertTrue(all("cache_prompt" not in c for c in srv.calls))
        self.assertTrue(all(c["model"] == "m" for c in srv.calls))
        salts = [c["cache_salt"] for c in srv.calls]
        self.assertEqual(salts[0], salts[1])
        self.assertEqual(salts[2], salts[3])
        # each recompute request has a salt no earlier request used
        self.assertNotIn(salts[4], salts[:4])
        self.assertNotIn(salts[5], salts[:5])
        self.assertEqual(srv.calls[4]["messages"], srv.calls[1]["messages"])
        self.assertEqual(report["probes"][0]["recompute_cached_tokens"],
                         [0, 0])

    def test_one_server_mode_no_divergence(self):
        srv = self.server(salted=True, style="usage")
        code, report, _ = self.run_tool(srv, "--mode", "one-server")
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["probes_valid"], 3)

    def test_one_server_salts_are_fresh_across_runs(self):
        srv = self.server(salted=True, style="usage")
        self.run_tool(srv, "--mode", "one-server", n="1")
        first = [c["cache_salt"] for c in srv.calls]
        self.run_tool(srv, "--mode", "one-server", n="1")
        second = [c["cache_salt"] for c in srv.calls[len(first):]]
        self.assertFalse(set(first) & set(second))
        # the second run found nothing left by the first
        self.assertEqual(self.run_tool(srv, "--mode", "one-server",
                                       n="1")[0], 0)

    def test_one_server_mode_server_that_ignores_salt(self):
        # A server that drops cache_salt serves the recompute request from
        # the cache, so it must not pass as a valid "same".
        srv = self.server(salted=True, ignore_salt=True, style="usage")
        code, report, out = self.run_tool(srv, "--mode", "one-server")
        self.assertEqual(code, 2)
        self.assertEqual(report["summary"]["probes_valid"], 0)
        self.assertIn("seems to ignore cache_salt", out)

    def test_one_server_mode_needs_telemetry(self):
        srv = self.server(salted=True, style="usage", telemetry=False)
        code, report, _ = self.run_tool(srv, "--mode", "one-server")
        self.assertEqual(code, 2)
        self.assertTrue(any("recompute path not confirmed" in p
                            for p in report["probes"][0]["problems"]))

    def write(self, name, text):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_probe_file_short_form(self):
        path = self.write("p.jsonl",
                          '{"id": "a1", "turn1": "Q one", "turn2": "F one"}\n'
                          '\n'
                          '{"turn1": "Q two", "turn2": "F two"}\n')
        srv = self.server(diverge=True)
        out = os.path.join(self.tmp.name, "r.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cachecheck.main(["--base-url", srv.url, "--probes", path,
                                    "--out", out, "--timeout", "10"])
        with open(out) as fh:
            report = json.load(fh)
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["probes_planned"], 2)
        self.assertEqual(len(srv.calls), 12)
        self.assertEqual(srv.calls[0]["messages"],
                         [{"role": "user", "content": "Q one"}])
        self.assertEqual(srv.calls[1]["messages"][2],
                         {"role": "user", "content": "F one"})
        self.assertEqual([p["probe_id"] for p in report["probes"]],
                         ["a1", None])
        self.assertEqual(report["probe_set"], "own file")
        with open(path, "rb") as fh:
            import hashlib
            self.assertEqual(report["probe_file_sha256"],
                             hashlib.sha256(fh.read()).hexdigest())
        printed = buf.getvalue()
        self.assertIn("keep it private", printed)
        block = printed.split("---- llm-cachecheck result ----")[1]
        self.assertIn("probe set: own file", block)
        for text in ("Q one", "F one", "Q two", "a1"):
            self.assertNotIn(text, block)

    def test_probe_file_full_form_keeps_system_prompt(self):
        path = self.write("p.jsonl", json.dumps({
            "messages": [{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": "Hi"},
                         {"role": "assistant", "content": "Hello."},
                         {"role": "user", "content": "Q"}],
            "followup": "F"}) + "\n")
        srv = self.server()
        code, report, _ = self.run_tool(srv, "--probes", path, n="1")
        self.assertEqual(code, 0)
        self.assertEqual([m["role"] for m in srv.calls[0]["messages"]],
                         ["system", "user", "assistant", "user"])
        turn2 = srv.calls[1]["messages"]
        self.assertEqual([m["role"] for m in turn2],
                         ["system", "user", "assistant", "user",
                          "assistant", "user"])
        self.assertEqual(turn2[-1]["content"], "F")
        self.assertEqual(srv.calls[4]["messages"], turn2)

    def test_probe_file_json_array_and_bom(self):
        path = self.write("p.json", "﻿" + json.dumps(
            [{"turn1": "A", "turn2": "B"}, {"turn1": "C", "turn2": "D"}]))
        probes, _ = cachecheck.load_probes(path)
        self.assertEqual([p["followup"] for p in probes], ["B", "D"])

    def test_probe_file_n_limit(self):
        path = self.write("p.jsonl", '{"turn1": "A", "turn2": "B"}\n')
        srv = self.server()
        self.assertEqual(self.run_tool(srv, "--probes", path, n="1")[0], 0)
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()), \
                    redirect_stderr(io.StringIO()):
                cachecheck.main(["--base-url", srv.url, "--probes", path,
                                 "--n", "2"])

    def test_probe_file_errors(self):
        good = '{"turn1": "A", "turn2": "B"}\n'
        bad = {
            "line 2: unknown key": good + '{"turn1": "A", "follow_up": "B"}',
            "line 1: give either": '{"turn1": "A", "followup": "B"}',
            "last message must": json.dumps({
                "messages": [{"role": "user", "content": "A"},
                             {"role": "assistant", "content": "B"}],
                "followup": "C"}),
            "text \"content\"": json.dumps({
                "messages": [{"role": "user",
                              "content": [{"type": "text", "text": "A"}]}],
                "followup": "C"}),
            "\"turn2\" must be": '{"turn1": "A", "turn2": "  "}',
            "line 2: not valid JSON": good + "{oops",
            "holds no conversations": "\n\n",
            "\"id\" must be": '{"turn1": "A", "turn2": "B", "id": [1]}',
            "a whole number": '{"turn1": "A", "turn2": "B", "id": 1.5}',
            "or a whole number": '{"turn1": "A", "turn2": "B", "id": true}',
            "non-empty \"content\"": json.dumps({
                "messages": [{"role": "user", "content": " "}],
                "followup": "C"}),
            "expected a JSON object": "[1, 2]",
        }
        for expected, text in bad.items():
            path = self.write("bad.jsonl", text)
            err = io.StringIO()
            with self.assertRaises(SystemExit, msg=expected):
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    cachecheck.main(["--base-url", "http://x",
                                     "--probes", path])
            self.assertIn(expected, err.getvalue())
        err = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                cachecheck.main(["--base-url", "http://x", "--probes",
                                 os.path.join(self.tmp.name, "missing")])
        self.assertIn("cannot read --probes file", err.getvalue())

    def test_probe_file_unicode_separators_and_crlf(self):
        # U+2028 and U+0085 may sit unescaped inside a JSON string; only
        # "\n" ends a line. CRLF files work too.
        line = json.dumps({"turn1": "A B\u0085C", "turn2": "D"},
                          ensure_ascii=False)
        path = self.write("u.jsonl", line + "\r\n" + line + "\r\n")
        probes, _ = cachecheck.load_probes(path)
        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["messages"][0]["content"], "A B\u0085C")
        # old Mac line endings (CR only)
        with open(path, "wb") as fh:
            fh.write(b'{"turn1": "A", "turn2": "B"}\r{"turn1": "C", '
                     b'"turn2": "D"}\r')
        self.assertEqual(len(cachecheck.load_probes(path)[0]), 2)

    def test_example_file_parses(self):
        # The README links this file, so a release must ship it.
        here = os.path.dirname(os.path.abspath(__file__))
        probes, _ = cachecheck.load_probes(
            os.path.join(here, "examples", "conversations_example.jsonl"))
        self.assertEqual([p["id"] for p in probes], ["returns-1", "sql-1"])
        self.assertEqual(probes[1]["messages"][0]["role"], "system")

    def test_failed_request_with_own_file_warns_before_sharing(self):
        path = self.write("p.jsonl", '{"turn1": "secret", "turn2": "B"}\n')
        code, _, out = self.run_tool(self.server(fail=True), "--probes",
                                     path, n="1")
        self.assertEqual(code, 2)
        self.assertIn("share only the result block", out)

    def test_server_with_caching_off_cannot_pass(self):
        # A server that accepts cache_salt but never hits (caching off):
        # the cache path must not pass as a cache hit, in either mode.
        srv = self.server(salted=True, never_hit=True, style="usage")
        code, report, out = self.run_tool(srv, "--mode", "one-server")
        self.assertEqual((code, report["summary"]["probes_valid"]), (2, 0))
        self.assertIn("cache hit too short: 3", out)
        plain = self.server(salted=True, style="usage")
        code, report, _ = self.run_tool(
            srv, "--mode", "two-servers", "--nocache-url", plain.url)
        self.assertEqual((code, report["summary"]["probes_valid"]), (2, 0))

    def test_hit_must_reach_past_the_turn1_prompt(self):
        # A short answer with a full hit is fine: the hit reaches into it.
        srv = self.server(salted=True, style="usage", completion=20)
        self.assertEqual(self.run_tool(srv, "--mode", "one-server")[0], 0)
        # A short answer whose hit stops at the prompt cannot be confirmed.
        srv = self.server(salted=True, style="usage", completion=20,
                          short_hit=True)
        code, report, out = self.run_tool(srv, "--mode", "one-server")
        self.assertEqual((code, report["summary"]["probes_valid"]), (2, 0))
        self.assertIn("turn-1 answer too short to confirm the cache hit: 3",
                      out)
        self.assertIn("Ask for longer turn-1 answers", out)

    def test_hit_check_boundaries(self):
        # Hits of p1 + c1 - 1 - shortfall tokens, tolerance 32 (one-server).
        def code(completion, shortfall):
            srv = self.server(salted=True, style="usage",
                              completion=completion, shortfall=shortfall)
            return self.run_tool(srv, "--mode", "one-server", n="1")[0]
        self.assertEqual(code(34, 32), 0)   # hit ends 1 token into answer
        self.assertEqual(code(34, 33), 2)   # hit stops at the prompt
        self.assertEqual(code(33, 32), 2)   # hit stops at the prompt
        self.assertEqual(code(80, 32), 0)   # long answer, shortfall 32 ok
        self.assertEqual(code(80, 33), 2)   # shortfall beyond tolerance
        # llama.cpp mode, tolerance 1: a 2-token answer needs a full hit
        self.assertEqual(self.run_tool(self.server(completion=2), n="1")[0], 0)
        self.assertEqual(self.run_tool(self.server(completion=2,
                                                   shortfall=1), n="1")[0], 2)
        h = cachecheck.hit_check
        t1 = {"prompt_tokens": 12, "completion_tokens": 9}
        self.assertFalse(h(t1, {"cached_tokens": 0}, 32)[0])
        self.assertFalse(h(t1, {"cached_tokens": 12}, 32)[0])
        self.assertTrue(h(t1, {"cached_tokens": 13}, 32)[0])

    def test_builtin_probes_unchanged(self):
        probes = cachecheck.builtin_probes()
        self.assertEqual(len(probes), 20)
        self.assertEqual(probes[0]["messages"],
                         [{"role": "user", "content": cachecheck.PROBES[0][0]}])
        self.assertEqual(probes[0]["followup"], cachecheck.PROBES[0][1])

    def test_release_has_no_unfilled_placeholders(self):
        # The one-server mode is described only after its GPU validation.
        # Until then the docs carry <<FILL markers, and this test keeps a
        # release tag from publishing them.
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("cachecheck.py", "README.md"):
            path = os.path.join(here, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    found = "<<" + "FILL" in fh.read()
                self.assertFalse(found, "unfilled <<" + "FILL marker in "
                                 + name)

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
                     ["--base-url", "http://x", "--n", "999"],
                     ["--base-url", "http://x", "--mode", "one-server",
                      "--nocache-url", "http://y"],
                     ["--base-url", "http://x", "--mode", "one-server",
                      "--skip-hit-check"],
                     ["--base-url", "http://x", "--nocache-url", "http://y"]):
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()), \
                        redirect_stderr(io.StringIO()):
                    cachecheck.main(argv)


if __name__ == "__main__":
    unittest.main()
