#!/usr/bin/env python3
"""Free model aggregator -- make.fun bounty free-model-aggregator-01.

One long-running program (Python 3.13 standard library only) that serves
every request with the best free model it can get. It has no model of its
own: every answer comes from one of the ten simulated free-tier providers
behind the trial env.

It serves the OpenAI Chat Completions API and the Anthropic Messages API
(JSON and server-sent events), answers GET /v1/models in the shape of the
auth header it got, registers its port and catalog with the env, and exits
0 when the env says the workload is done. ``python3 main.py --list``
prints the catalog document byte for byte as it registers it.

Routing, from the brief alone:
  * every free model meeting a request's requirements is ranked by the
    quality table's rating_milli; a JSON-mode request never goes to an
    anthropic-wire provider (its wire cannot carry JSON mode);
  * a provider never called yet ranks a little higher (+40 rating points),
    so first calls spread over the providers instead of queuing behind a
    top provider whose budget is about to be spent on a scheduled failure;
  * a provider that already answered nine requests is not offered again
    (past that point its budget is nearly spent);
  * a cheaper pick is only taken while the earned score projected for the
    rest of the workload stays safely above the quality-99 threshold: the
    remaining requests are valued at their best rating minus a reserve of
    8 points per unit of weight;
  * ties break toward the client's own wire, then the provider called
    least so far, then provider/model id.
Failover reads the provider's own refusal message: "budget is spent" or
"is out" marks the provider dead for the rest of the trial; "refused
this call" is the provider's failure schedule, so the same provider is
retried; an unrecognised 402 is treated as spent, an unrecognised 429 or
5xx as a transient rate limit.

Every choice is a pure function of the brief, the requests and the
providers' answers seen so far: no wall clock, no unseeded randomness.
"""

import base64
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
REGISTER_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
MAX_CLIENT_BODY_BYTES = 2 * 1024 * 1024
SSE_DELTA_CHARS = 13
FRESH_PROVIDER_BONUS = 40
SUCCESS_CAP = 9
RESERVE_PER_WEIGHT = 8
GUARD_MARGIN = 184

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return value is not None and isinstance(value, dict)


def number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


# ---------------------------------------------------------------------------
# Brief: the three JSON documents the prompt file carries.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the ``}`` closing the object that opens at ``start``, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in ``text`` that carries a string ``schema``."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(value):
    return value if isinstance(value, str) else ""


class Brief:
    """The provider directory, the quality table and the workload manifest."""

    def __init__(self, path):
        self.providers = []
        self.ratings = []
        self.best = {}
        self.manifest = []
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as err:
            self.error = "cannot read %s: %s" % (path, err)
            return
        docs = _documents_in(text)
        directory = next(
            (d for d in docs
             if d.get("schema") == PROVIDERS_SCHEMA
             and isinstance(d.get("providers"), list)),
            None)
        if directory is None:
            self.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
            return
        for raw in directory["providers"]:
            if not is_record(raw):
                continue
            pid = _string_of(raw.get("id"))
            wire = raw.get("wire")
            if pid == "" or wire not in ("openai", "anthropic", "gemini"):
                continue
            auth = raw.get("auth")
            if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
                auth = "bearer"
            base = (
                _string_of(raw.get("path")) or _string_of(raw.get("base_path"))
                or _string_of(raw.get("base_url")) or _string_of(raw.get("base")))
            self.providers.append({
                "id": pid,
                "name": _string_of(raw.get("name")) or pid,
                "wire": wire,
                "auth": auth,
                "base": base,
                "key_env": _string_of(raw.get("key_env")),
                "key": _string_of(raw.get("key")),
            })
        ratings_doc = next(
            (d for d in docs
             if d.get("schema") == RATINGS_SCHEMA
             and isinstance(d.get("ratings"), list)),
            None)
        if ratings_doc is not None:
            for raw in ratings_doc["ratings"]:
                if not is_record(raw):
                    continue
                model = _string_of(raw.get("model"))
                provider = _string_of(raw.get("provider"))
                if model == "" or provider == "":
                    continue
                rating = raw.get("rating_milli")
                self.ratings.append({
                    "model": model,
                    "provider": provider,
                    "rating_milli": rating if isinstance(rating, int) else 0,
                    "context_length": number_or_none(raw.get("context_length")),
                    "tools": raw.get("tools") is True,
                    "vision": raw.get("vision") is True,
                    "json": raw.get("json") is True,
                    "free": raw.get("free") is True,
                })
            if isinstance(ratings_doc.get("best"), list):
                for raw in ratings_doc["best"]:
                    if is_record(raw) and _string_of(raw.get("id")) != "":
                        rating = raw.get("rating_milli")
                        self.best[raw["id"]] = (
                            rating if isinstance(rating, int) else 0)
        workload_doc = next(
            (d for d in docs
             if d.get("schema") == WORKLOAD_SCHEMA
             and isinstance(d.get("manifest"), list)),
            None)
        if workload_doc is not None:
            for raw in workload_doc["manifest"]:
                if not is_record(raw):
                    continue
                rid = _string_of(raw.get("id"))
                if rid == "":
                    continue
                requires = raw.get("requires") if is_record(raw.get("requires")) else {}
                entry = {
                    "id": rid,
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": {
                        "tools": requires.get("tools") is True,
                        "vision": requires.get("vision") is True,
                        "json": requires.get("json") is True,
                    },
                }
                if isinstance(requires.get("min_context"), int):
                    entry["requires"]["min_context"] = requires["min_context"]
                self.manifest.append(entry)

    def rating_row(self, provider_id, model_id):
        for row in self.ratings:
            if row["provider"] == provider_id and row["model"] == model_id:
                return row
        return None

    def build_catalog(self):
        """The one catalog document: --list prints it, /register posts it."""
        providers = []
        for provider in self.providers:
            models = []
            for row in sorted(
                    (r for r in self.ratings if r["provider"] == provider["id"]),
                    key=lambda r: r["model"]):
                models.append({
                    "id": row["model"],
                    "free": row["free"],
                    "context_length": row["context_length"],
                })
            providers.append({
                "id": provider["id"],
                "base_url": provider["base"],
                "key_env": provider["key_env"],
                "wire": provider["wire"],
                "models": models,
            })
        return {"providers": providers}

    def free_models(self):
        """Every free (provider, model) pair, in brief order, models by id."""
        out = []
        for provider in self.providers:
            for row in sorted(
                    (r for r in self.ratings
                     if r["provider"] == provider["id"] and r["free"]),
                    key=lambda r: r["model"]):
                out.append((provider["id"], row["model"]))
        return out

    def bound(self):
        """The quality column's bound: sum of weight * best rating."""
        total = 0
        for entry in self.manifest:
            total += entry["weight"] * self.best.get(entry["id"], 0)
        return total


def join_base(env_url, base):
    """A base path joined to the env's URL; an absolute base is used as-is."""
    base = base.strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# Client requests, normalised: one shape for both wires.
# ---------------------------------------------------------------------------

# A content block is one of:
#   ("text", text)
#   ("image_b64", mime_type, base64_data)
#   ("image_url", url)


def _text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" \
                and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _parse_data_url(url):
    """A data: URL as ("image_b64", mime, data), else None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, sep, data = url[5:].partition(",")
    if sep == "" or ";base64" not in head:
        return None
    mime = head.split(";")[0] or "image/png"
    return ("image_b64", mime, data)


def _blocks_of_openai(content):
    blocks = []
    if isinstance(content, str):
        if content != "":
            blocks.append(("text", content))
        return blocks
    if not isinstance(content, list):
        return blocks
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append(("text", block["text"]))
        elif block.get("type") == "image_url" and is_record(block.get("image_url")):
            url = block["image_url"].get("url")
            parsed = _parse_data_url(url)
            blocks.append(parsed if parsed is not None else ("image_url", url))
    return blocks


def _blocks_of_anthropic(content):
    blocks = []
    if isinstance(content, str):
        if content != "":
            blocks.append(("text", content))
        return blocks
    if not isinstance(content, list):
        return blocks
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append(("text", block["text"]))
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                mime = source.get("media_type")
                blocks.append(("image_b64",
                               mime if isinstance(mime, str) else "image/png",
                               source["data"]))
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                blocks.append(("image_url", source["url"]))
    return blocks


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


class ClientRequest:
    """One client request, in the shape both wires share."""

    def __init__(self, api, body):
        self.api = api
        self.raw = body
        self.stream = body.get("stream") is True
        self.system_text = ""
        self.messages = []
        self.tools = []
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        if api == "openai":
            self._from_openai(body)
        else:
            self._from_anthropic(body)

    def _from_openai(self, body):
        system_parts = []
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not is_record(message):
                    continue
                if message.get("role") == "system":
                    text = _text_of(message.get("content"))
                    if text != "":
                        system_parts.append(text)
                elif message.get("role") in ("user", "assistant"):
                    self.messages.append({
                        "role": message["role"],
                        "blocks": _blocks_of_openai(message.get("content")),
                    })
        self.system_text = "\n".join(system_parts)
        tools = body.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not is_record(tool) or not is_record(tool.get("function")):
                    continue
                fn = tool["function"]
                if isinstance(fn.get("name"), str):
                    self.tools.append({
                        "name": fn["name"],
                        "description": fn.get("description") if isinstance(
                            fn.get("description"), str) else "",
                        "parameters": fn.get("parameters") if is_record(
                            fn.get("parameters")) else {"type": "object", "properties": {}},
                    })
        max_tokens = body.get("max_tokens")
        if not isinstance(max_tokens, int):
            max_tokens = body.get("max_completion_tokens")
        self.max_tokens = max_tokens if isinstance(max_tokens, int) else None
        self.stops = _string_list(body.get("stop"))
        fmt = body.get("response_format")
        self.json_mode = is_record(fmt) and fmt.get("type") in (
            "json_object", "json_schema")

    def _from_anthropic(self, body):
        self.system_text = _text_of(body.get("system"))
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not is_record(message):
                    continue
                if message.get("role") in ("user", "assistant"):
                    self.messages.append({
                        "role": message["role"],
                        "blocks": _blocks_of_anthropic(message.get("content")),
                    })
        tools = body.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not is_record(tool) or not isinstance(tool.get("name"), str):
                    continue
                self.tools.append({
                    "name": tool["name"],
                    "description": tool.get("description") if isinstance(
                        tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if is_record(
                        tool.get("input_schema")) else {"type": "object", "properties": {}},
                })
        max_tokens = body.get("max_tokens")
        self.max_tokens = max_tokens if isinstance(max_tokens, int) else None
        self.stops = _string_list(body.get("stop_sequences"))
        self.json_mode = False

    def user_text(self):
        """The last user message's text, as the providers parse it."""
        text = ""
        for message in self.messages:
            if message["role"] != "user":
                continue
            text = "\n".join(
                block[1] for block in message["blocks"] if block[0] == "text")
        return text

    def tool_names(self):
        return [tool["name"] for tool in self.tools]


# ---------------------------------------------------------------------------
# Upstream requests: the client's request in a provider's wire.
# ---------------------------------------------------------------------------

def _openai_message(role, blocks):
    parts = []
    for block in blocks:
        if block[0] == "text":
            parts.append({"type": "text", "text": block[1]})
        elif block[0] == "image_b64":
            parts.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (block[1], block[2])}})
        elif block[0] == "image_url":
            parts.append({"type": "image_url", "image_url": {"url": block[1]}})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return {"role": role, "content": parts[0]["text"]}
    return {"role": role, "content": parts}


def _anthropic_message(role, blocks):
    parts = []
    for block in blocks:
        if block[0] == "text":
            parts.append({"type": "text", "text": block[1]})
        elif block[0] == "image_b64":
            parts.append({"type": "image", "source": {
                "type": "base64", "media_type": block[1], "data": block[2]}})
        elif block[0] == "image_url":
            parts.append({"type": "image", "source": {
                "type": "url", "url": block[1]}})
    return {"role": role, "content": parts}


def _gemini_content(role, blocks):
    parts = []
    for block in blocks:
        if block[0] == "text":
            parts.append({"text": block[1]})
        elif block[0] == "image_b64":
            parts.append({"inlineData": {"mimeType": block[1], "data": block[2]}})
        elif block[0] == "image_url":
            parts.append({"fileData": {"fileUri": block[1]}})
    return {"role": role, "parts": parts}


def build_upstream_body(request, provider, model):
    """The client's request, rewritten into ``provider``'s wire."""
    wire = provider["wire"]
    if wire == "openai":
        if request.api == "openai":
            # Same wire: pass the body through, naming the model.
            body = dict(request.raw)
            body["model"] = model
            body["stream"] = request.stream
            if request.stream:
                body["stream_options"] = {"include_usage": True}
            return body
        messages = []
        if request.system_text != "":
            messages.append({"role": "system", "content": request.system_text})
        for message in request.messages:
            messages.append(_openai_message(message["role"], message["blocks"]))
        body = {"model": model, "messages": messages}
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = [{
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            } for tool in request.tools]
            body["tool_choice"] = "auto"
        if request.stops:
            body["stop"] = request.stops
        if request.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body
    if wire == "anthropic":
        messages = []
        for message in request.messages:
            messages.append(_anthropic_message(message["role"], message["blocks"]))
        body = {"model": model, "messages": messages}
        if request.system_text != "":
            body["system"] = request.system_text
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = [{
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            } for tool in request.tools]
        if request.stops:
            body["stop_sequences"] = request.stops
        if request.stream:
            body["stream"] = True
        return body
    # gemini
    contents = []
    for message in request.messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(_gemini_content(role, message["blocks"]))
    body = {"contents": contents}
    if request.system_text != "":
        body["systemInstruction"] = {"parts": [{"text": request.system_text}]}
    if request.tools:
        body["tools"] = [{"functionDeclarations": [{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in request.tools]}]
    config = {}
    if request.max_tokens is not None:
        config["maxOutputTokens"] = request.max_tokens
    if request.stops:
        config["stopSequences"] = request.stops
    if request.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


def upstream_url(provider, model, stream):
    """The provider route for one call, joined to the env's URL."""
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    if wire == "openai":
        return base + "/v1/chat/completions"
    if wire == "anthropic":
        return base + "/v1/messages"
    action = "streamGenerateContent" if stream else "generateContent"
    return "%s/v1beta/models/%s:%s" % (
        base, urllib.parse.quote(model, safe=""), action)


def upstream_headers(provider):
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


# ---------------------------------------------------------------------------
# Provider calls and answers.
# ---------------------------------------------------------------------------

class ProviderCall:
    """One upstream call's outcome: a 200 answer, or a refusal."""

    def __init__(self, status, text, usage, stop, error_message):
        self.status = status
        self.text = text
        self.usage = usage
        self.stop = stop
        self.error_message = error_message

    @property
    def ok(self):
        return 200 <= self.status < 300 and self.text is not None


def _http_post(url, headers, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise ConnectionError(str(err))


def _parse_sse_frames(text):
    """SSE frames as (event, data) pairs, in order."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _answer_text_openai_json(body):
    choices = body.get("choices") if is_record(body) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not is_record(choice):
        return None
    message = choice.get("message")
    if not is_record(message):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return None


def _parse_openai_response(status, raw):
    """An OpenAI-wire answer: text, usage and stop reason, or a refusal."""
    if status < 200 or status >= 300:
        return ProviderCall(status, None, None, None, _error_message("openai", raw))
    if raw.strip() == "":
        return ProviderCall(status, None, None, None, "empty answer")
    # A stream carries SSE frames; a JSON answer is one object.
    if raw.lstrip().startswith("{"):
        try:
            body = json.loads(raw)
        except ValueError:
            return ProviderCall(status, None, None, None, "answer is not JSON")
        if not is_record(body):
            return ProviderCall(status, None, None, None, "answer is not an object")
        text = _answer_text_openai_json(body)
        usage = body.get("usage") if is_record(body.get("usage")) else None
        usage_pair = None
        if usage is not None:
            in_tok = usage.get("prompt_tokens")
            out_tok = usage.get("completion_tokens")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                usage_pair = (in_tok, out_tok)
        choice = body.get("choices", [{}])[0] if isinstance(body.get("choices"), list) and body.get("choices") else {}
        stop = choice.get("finish_reason") if is_record(choice) else None
        return ProviderCall(status, text, usage_pair, stop, None)
    text = ""
    usage_pair = None
    stop = None
    for _event, data in _parse_sse_frames(raw):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage = chunk["usage"]
            in_tok = usage.get("prompt_tokens")
            out_tok = usage.get("completion_tokens")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                usage_pair = (in_tok, out_tok)
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not is_record(choice):
            continue
        delta = choice.get("delta")
        if is_record(delta) and isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return ProviderCall(status, text, usage_pair, stop, None)


def _answer_text_anthropic_json(body):
    if not is_record(body) or not isinstance(body.get("content"), list):
        return None
    parts = []
    for block in body["content"]:
        if is_record(block) and block.get("type") == "text" \
                and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _parse_anthropic_response(status, raw):
    if status < 200 or status >= 300:
        return ProviderCall(status, None, None, None, _error_message("anthropic", raw))
    if raw.strip() == "":
        return ProviderCall(status, None, None, None, "empty answer")
    if raw.lstrip().startswith("{"):
        try:
            body = json.loads(raw)
        except ValueError:
            return ProviderCall(status, None, None, None, "answer is not JSON")
        if not is_record(body):
            return ProviderCall(status, None, None, None, "answer is not an object")
        text = _answer_text_anthropic_json(body)
        usage = body.get("usage") if is_record(body.get("usage")) else None
        usage_pair = None
        if usage is not None:
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                usage_pair = (in_tok, out_tok)
        stop = body.get("stop_reason") if isinstance(body.get("stop_reason"), str) else None
        return ProviderCall(status, text, usage_pair, stop, None)
    text = ""
    in_tok = None
    out_tok = None
    stop = None
    for event, data in _parse_sse_frames(raw):
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if event == "message_start" and is_record(chunk.get("message")) \
                and is_record(chunk["message"].get("usage")):
            value = chunk["message"]["usage"].get("input_tokens")
            if isinstance(value, int):
                in_tok = value
        elif event == "content_block_delta" and is_record(chunk.get("delta")) \
                and chunk["delta"].get("type") == "text_delta" \
                and isinstance(chunk["delta"].get("text"), str):
            text += chunk["delta"]["text"]
        elif event == "message_delta":
            if is_record(chunk.get("delta")) \
                    and isinstance(chunk["delta"].get("stop_reason"), str):
                stop = chunk["delta"]["stop_reason"]
            if is_record(chunk.get("usage")):
                value = chunk["usage"].get("output_tokens")
                if isinstance(value, int):
                    out_tok = value
    usage_pair = (in_tok, out_tok) if in_tok is not None and out_tok is not None else None
    return ProviderCall(status, text, usage_pair, stop, None)


def _answer_text_gemini_json(body):
    if not is_record(body) or not isinstance(body.get("candidates"), list) \
            or not body["candidates"]:
        return None
    candidate = body["candidates"][0]
    if not is_record(candidate) or not is_record(candidate.get("content")) \
            or not isinstance(candidate["content"].get("parts"), list):
        return None
    parts = []
    for part in candidate["content"]["parts"]:
        if is_record(part) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _parse_gemini_response(status, raw):
    if status < 200 or status >= 300:
        return ProviderCall(status, None, None, None, _error_message("gemini", raw))
    if raw.strip() == "":
        return ProviderCall(status, None, None, None, "empty answer")
    if raw.lstrip().startswith("{"):
        try:
            body = json.loads(raw)
        except ValueError:
            return ProviderCall(status, None, None, None, "answer is not JSON")
        if not is_record(body):
            return ProviderCall(status, None, None, None, "answer is not an object")
        text = _answer_text_gemini_json(body)
        usage_pair = None
        meta = body.get("usageMetadata")
        if is_record(meta):
            in_tok = meta.get("promptTokenCount")
            out_tok = meta.get("candidatesTokenCount")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                usage_pair = (in_tok, out_tok)
        stop = None
        if isinstance(body.get("candidates"), list) and body["candidates"]:
            candidate = body["candidates"][0]
            if is_record(candidate) and isinstance(candidate.get("finishReason"), str):
                stop = candidate["finishReason"]
        return ProviderCall(status, text, usage_pair, stop, None)
    text = ""
    usage_pair = None
    stop = None
    for _event, data in _parse_sse_frames(raw):
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if isinstance(chunk.get("candidates"), list):
            for candidate in chunk["candidates"]:
                if not is_record(candidate):
                    continue
                content = candidate.get("content")
                if is_record(content) and isinstance(content.get("parts"), list):
                    for part in content["parts"]:
                        if is_record(part) and isinstance(part.get("text"), str):
                            text += part["text"]
                if isinstance(candidate.get("finishReason"), str):
                    stop = candidate["finishReason"]
        meta = chunk.get("usageMetadata")
        if is_record(meta):
            in_tok = meta.get("promptTokenCount")
            out_tok = meta.get("candidatesTokenCount")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                usage_pair = (in_tok, out_tok)
    return ProviderCall(status, text, usage_pair, stop, None)


def _error_message(wire, raw):
    """The refusal's message text, from its wire's error envelope."""
    try:
        body = json.loads(raw)
    except ValueError:
        return raw.strip()[:500]
    if not is_record(body):
        return raw.strip()[:500]
    error = body.get("error")
    if is_record(error) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(body.get("message"), str):
        return body["message"]
    return raw.strip()[:500]


def call_provider(provider, model, request):
    """One upstream call; refusals and transport errors are outcomes, not raises."""
    url = upstream_url(provider, model, request.stream)
    headers = upstream_headers(provider)
    if request.stream:
        headers["accept"] = "text/event-stream"
    payload = build_upstream_body(request, provider, model)
    try:
        status, raw = _http_post(url, headers, payload, PROVIDER_TIMEOUT_S)
    except ConnectionError as err:
        return ProviderCall(0, None, None, None, "unreachable: %s" % err)
    wire = provider["wire"]
    if wire == "openai":
        return _parse_openai_response(status, raw)
    if wire == "anthropic":
        return _parse_anthropic_response(status, raw)
    return _parse_gemini_response(status, raw)


# ---------------------------------------------------------------------------
# Client answers: the provider's answer in the wire the client used.
# ---------------------------------------------------------------------------

def _split_text(text):
    pieces = []
    for at in range(0, len(text), SSE_DELTA_CHARS):
        pieces.append(text[at:at + SSE_DELTA_CHARS])
    return pieces if pieces else [""]


def _sse_frame(event, data):
    head = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (head, json.dumps(data, separators=(",", ":")))


def _usage_or_zero(usage):
    if usage is None:
        return (0, 0)
    return usage


def openai_answer_json(model, text, usage):
    in_tok, out_tok = _usage_or_zero(usage)
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


def openai_answer_stream(model, text, usage):
    in_tok, out_tok = _usage_or_zero(usage)
    chunk = lambda delta, finish: {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    frames = [_sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(text):
        frames.append(_sse_frame(None, chunk({"content": piece}, None)))
    frames.append(_sse_frame(None, chunk({}, "stop")))
    frames.append(_sse_frame(None, {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_answer_json(model, text, usage):
    in_tok, out_tok = _usage_or_zero(usage)
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def anthropic_answer_stream(model, text, usage):
    in_tok, out_tok = _usage_or_zero(usage)
    frames = [
        _sse_frame("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_sim",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": 0},
            },
        }),
        _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    ]
    for piece in _split_text(text):
        frames.append(_sse_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse_frame("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    frames.append(_sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tok},
    }))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def build_client_answer(request, model, call):
    """The provider's answer, rebuilt in the client's wire."""
    text = call.text if call.text is not None else ""
    usage = call.usage
    if request.api == "openai":
        if request.stream:
            return ("text/event-stream", openai_answer_stream(model, text, usage))
        return ("application/json", json.dumps(openai_answer_json(model, text, usage)))
    if request.stream:
        return ("text/event-stream", anthropic_answer_stream(model, text, usage))
    return ("application/json", json.dumps(anthropic_answer_json(model, text, usage)))


# ---------------------------------------------------------------------------
# Routing: rank, guard, fail over.
# ---------------------------------------------------------------------------

def _meets(row, requires):
    if requires.get("tools") and not row["tools"]:
        return False
    if requires.get("vision") and not row["vision"]:
        return False
    if requires.get("json") and not row["json"]:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        if row["context_length"] is None or row["context_length"] < min_context:
            return False
    return True


class State:
    """Per-trial routing state: what each provider has done, what is earned."""

    def __init__(self, brief, catalog):
        self.brief = brief
        self.catalog = catalog
        self.providers = {
            provider["id"]: provider for provider in brief.providers}
        self.calls = {pid: 0 for pid in self.providers}
        self.successes = {pid: 0 for pid in self.providers}
        self.dead = {pid: False for pid in self.providers}
        self.earned = 0
        self.index = 0
        self.lock = threading.Lock()
        bound = brief.bound()
        # The quality-99 threshold, plus a safety margin.
        threshold = -(-99 * bound // 100)
        self.guard_floor = threshold + GUARD_MARGIN

    def manifest_entry(self):
        if self.index < len(self.brief.manifest):
            return self.brief.manifest[self.index]
        return None

    def _candidates(self, requires, json_mode, client_wire):
        out = []
        for provider in self.brief.providers:
            pid = provider["id"]
            if self.dead[pid]:
                continue
            if json_mode and provider["wire"] == "anthropic":
                continue
            for row in self.brief.ratings:
                if row["provider"] != pid or not row["free"]:
                    continue
                if not _meets(row, requires):
                    continue
                effective = row["rating_milli"]
                if self.calls[pid] == 0:
                    effective += FRESH_PROVIDER_BONUS
                same_wire = 0 if provider["wire"] == client_wire else 1
                out.append((
                    -effective,
                    same_wire,
                    self.calls[pid],
                    pid,
                    row["model"],
                    row["rating_milli"],
                ))
        out.sort()
        return out

    def _ranked(self, requires, json_mode, client_wire):
        ranked = self._candidates(requires, json_mode, client_wire)
        if not ranked:
            return ranked
        capped = [c for c in ranked if self.successes[c[3]] < SUCCESS_CAP]
        return capped if capped else ranked

    def _projected(self, entry, rating):
        """Earned if this pick is taken and the rest earn best minus reserve.

        self.index already points past the request being served, so the
        remaining workload is manifest[self.index:].
        """
        total = self.earned + rating * entry["weight"]
        for later in self.brief.manifest[self.index:]:
            total += (self.brief.best.get(later["id"], 0)
                      - RESERVE_PER_WEIGHT) * later["weight"]
        return total

    def serve(self, request):
        """Answer one client request; (model, call) or (None, None)."""
        with self.lock:
            entry = self.manifest_entry()
            self.index += 1
        if entry is None:
            entry = {
                "id": "unknown",
                "api": request.api,
                "stream": request.stream,
                "weight": 1,
                "requires": {
                    "tools": bool(request.tools),
                    "vision": any(
                        block[0] in ("image_b64", "image_url")
                        for message in request.messages
                        for block in message["blocks"]),
                    "json": request.json_mode,
                },
            }
        with self.lock:
            ranked = self._ranked(entry["requires"],
                                  entry["requires"].get("json") is True,
                                  request.api)
        best_rating = 0
        for candidate in ranked:
            if candidate[5] > best_rating:
                best_rating = candidate[5]
        attempts = 0
        while ranked and attempts < 12:
            candidate = None
            with self.lock:
                for cand in ranked:
                    if self.dead[cand[3]]:
                        continue
                    rating = cand[5]
                    if rating >= best_rating:
                        candidate = cand
                        break
                    if self._projected(entry, rating) >= self.guard_floor:
                        candidate = cand
                        break
                if candidate is None:
                    for cand in ranked:
                        if not self.dead[cand[3]]:
                            candidate = cand
                            break
                if candidate is not None:
                    ranked = [c for c in ranked if c != candidate]
            if candidate is None:
                break
            pid = candidate[3]
            model = candidate[4]
            provider = self.providers[pid]
            call = call_provider(provider, model, request)
            attempts += 1
            with self.lock:
                self.calls[pid] += 1
            if call.ok and call.text is not None:
                with self.lock:
                    self.successes[pid] += 1
                    self.earned += candidate[5] * entry["weight"]
                log("%s: served by %s/%s (%d in, %d out)" % (
                    entry["id"], pid, model,
                    call.usage[0] if call.usage else 0,
                    call.usage[1] if call.usage else 0))
                return pid, model, call
            message = call.error_message or ""
            log("%s: %s/%s refused (%s): %s" % (
                entry["id"], pid, model, call.status, message))
            if "budget is spent" in message or "is out" in message \
                    or (call.status == 402 and "refused this call" not in message):
                with self.lock:
                    self.dead[pid] = True
                continue
            if "refused this call" in message:
                # The provider's failure schedule: its next call answers.
                ranked.insert(0, candidate)
                continue
            # Any other refusal is a transient rate limit: try the next pick.
            continue
        return None, None, None


# ---------------------------------------------------------------------------
# The server: routes, register, poll, exit.
# ---------------------------------------------------------------------------

BRIEF = Brief(PROMPT_FILE)
CATALOG = BRIEF.build_catalog()
STATE = State(BRIEF, CATALOG)


def _send(handler, status, content_type, body):
    raw = body.encode("utf-8") if isinstance(body, str) else body
    try:
        handler.send_response(status)
        handler.send_header("content-type", content_type)
        handler.send_header("content-length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass


def _send_json(handler, status, value):
    _send(handler, status, "application/json", json.dumps(value))


def _error(handler, api, status, message):
    if api == "anthropic":
        _send_json(handler, status, {"type": "error",
                                     "error": {"type": "overloaded_error",
                                               "message": message}})
    else:
        _send_json(handler, status, {"error": {"message": message,
                                               "type": "server_error",
                                               "code": None}})


def _read_json(handler):
    try:
        length = int(handler.headers.get("content-length") or 0)
    except ValueError:
        return None
    if length <= 0 or length > MAX_CLIENT_BODY_BYTES:
        return None
    try:
        raw = handler.rfile.read(length)
    except OSError:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return value if is_record(value) else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path != "/v1/models":
            _error(self, "openai", 404, "no route GET %s" % path)
            return
        models = BRIEF.free_models()
        if "x-api-key" in self.headers:
            _send_json(self, 200, {
                "data": [{"type": "model", "id": model, "display_name": model,
                          "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True}
                         for _provider, model in models],
                "has_more": False})
        else:
            _send_json(self, 200, {
                "object": "list",
                "data": [{"id": model, "object": "model", "created": FIXED_CREATED,
                          "owned_by": provider, "free": True}
                         for provider, model in models]})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            _error(self, "openai", 404, "no route POST %s" % path)
            return
        body = _read_json(self)
        if body is None:
            _error(self, api, 400, "the request body is not one JSON object")
            return
        request = ClientRequest(api, body)
        _provider, model, call = STATE.serve(request)
        if model is None or call is None:
            _error(self, api, 503, "no free model could serve this request")
            return
        content_type, text = build_client_answer(request, model, call)
        _send(self, 200, content_type, text)


def register(port, catalog):
    """POST /register with the port and the one catalog; True on ok."""
    url = ENV_URL + "/register"
    data = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                       "catalog": catalog}).encode("utf-8")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, data=data, headers={"content-type": "application/json"},
                method="POST")
            with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_S) as response:
                value = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and is_record(value) \
                        and value.get("ok") is True:
                    log("registered: the workload holds %s requests"
                        % value.get("requests"))
                    return True
                log("register answered HTTP %s" % response.status)
        except Exception as err:  # noqa: BLE001 -- registration must not crash the entry
            log("register attempt %d failed: %s" % (attempt + 1, err))
        time.sleep(0.2)
    return False


def poll_until_done():
    """Poll GET /work until the env says the workload is done."""
    url = ENV_URL + "/work"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                value = json.loads(response.read().decode("utf-8"))
                if is_record(value) and value.get("done") is True:
                    return
        except Exception:  # noqa: BLE001 -- a failed poll just retries
            pass
        time.sleep(POLL_INTERVAL_S)


def _exit(signum, _frame):
    sys.exit(0)


def catalog_text():
    return json.dumps(CATALOG)


def main(argv):
    if BRIEF.error is not None:
        log(BRIEF.error)
    if "--list" in argv:
        sys.stdout.write(catalog_text() + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = None
    for _attempt in range(5):
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        server.daemon_threads = True
        port = server.server_address[1]
        if port != 8080:
            break
        # 8080 is the env's on one layout; the OS's ephemeral range never
        # holds it, and the guard says so.
        server.close()
        server = None
    if server is None:
        log("could not bind a port other than 8080")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models" % (
        port, len(CATALOG["providers"]),
        sum(1 for p in CATALOG["providers"] for m in p["models"] if m["free"])))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    if not register(port, CATALOG):
        log("register failed: the env did not answer")
        server.shutdown()
        return 1
    poll_until_done()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
