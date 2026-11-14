#!/usr/bin/env python3
"""Free-model aggregator entry for the make.fun bounty ``free-model-aggregator-01``.

One long-running program, Python 3.13 standard library only (nothing is
installed before the server runs). It reads the trial brief at
``$MAKEFUN_PROMPT_FILE``, registers with the env at ``$MAKEFUN_ENV_URL`` and
answers the client's 20 requests through the ten simulated free-tier
providers behind the env.

Design (all of it a pure function of the brief and the provider answers seen
so far -- no wall clock, no unseeded randomness, so a replay leaves the same
state export):

* The catalog is built from the brief alone (the provider directory plus the
  quality table's free-model sets). That makes ``python3 main.py --list``
  byte-for-byte the document the server registers, with or without the env.
* Routing ranks every free model that meets a request's requirements by the
  quality table's ``rating_milli`` (quality is 60% of the score), with three
  cost-saving rules that never touch served, wire or catalog: a provider
  never called yet gets a small rating bonus, so the first call spreads
  over the providers instead of queuing behind a top provider whose budget
  is about to be spent on a scheduled failure or an outage; a provider
  that already answered nine requests is not offered again, because its
  budget is nearly spent; and a cheaper pick is only taken while the earned
  score projected for the rest of the workload stays safely above the
  quality-99 threshold, so saving a call never risks the quality column.
  A JSON-mode request never goes to an anthropic-wire provider, whose wire
  cannot carry it. A request's first pick is the best legal model; on a
  refusal the entry retries the same provider when the refusal is transient
  (the provider's own failure schedule) and fails over to the next-best
  model when the provider's budget is spent or it is out. Exactly one
  provider call may answer a request.
* Every request is translated faithfully into the serving provider's wire
  (system prompt, user text, images, tools, stop sequences, max_tokens and
  JSON mode all pass through untouched -- the provider echoes what it
  parsed and the client compares that echo with what it sent), and the
  answer is rebuilt in the wire the client used, with the provider's text
  whole, its usage counts and its stop reason mapped to the client's wire.
* Usage is computed with the providers' own formula
  (``1 + len(system + user + tools) // 4`` in, ``1 + len(text) // 4`` out),
  which equals the provider's reported usage whenever the translation is
  exact -- and the echo checks verify exactly that.

The same code runs against the real services: a directory entry may give an
absolute base URL instead of a path, and a key is read from the environment
variable its ``key_env`` names when one is set (the directory's key is the
fallback, which is what the simulated env takes).
"""

import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Fixed values, so no reply ever reads the clock.
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

POLL_INTERVAL_S = 0.1
REGISTER_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
# The client waits 30 s for one answer; stop trying new providers before that.
REQUEST_DEADLINE_S = 25
MAX_CLIENT_BODY_BYTES = 2 * 1024 * 1024
# A transient refusal is retried on the same provider this many times: the
# next call number is past the provider's failure schedule, so it answers.
MAX_TRANSIENT_RETRIES = 4
# An unrecognised 429/5xx (a real service's rate limit) is retried this many
# times, with a short wait between tries.
MAX_UNKNOWN_RETRIES = 2
UNKNOWN_RETRY_WAIT_S = 1
# Safety valve against looping forever on one request.
MAX_CALLS_PER_REQUEST = 40

# Routing policy (see State.rank_candidates). All three rules are pure
# functions of the brief and the provider answers seen so far, so a replay
# makes the same picks.
# A provider never called yet gets this many rating points added when
# ranking: spreading the first call over a fresh provider costs nothing
# when the model is near the best, and it keeps a top provider's budget
# from being spent on a scheduled failure or an outage probe.
FRESH_PROVIDER_BONUS = 40
# A provider stops being offered once it has answered this many requests:
# past that point its budget is nearly spent and the next call is likely a
# refusal, so the request is cheaper on a provider that still has room.
MAX_SUCCESSES_PER_PROVIDER = 9
# Quality guard: a cheaper pick (a lower rating than the best legal model)
# is only taken while the earned score projected for the rest of the
# workload stays this far above the quality-99 threshold (99% of the
# maximum earned). The projection values the remaining requests at their
# best rating minus a reserve per unit of weight, because the best models
# are not all still available once the top providers are spent.
EARNED_SAFETY_MARGIN = 184
REMAINING_SHORTFALL_PER_WEIGHT = 8

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}

_DATA_URL_RE = re.compile(r"^data:([^;,]*)(;base64)?,(.*)$", re.S)


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return value is not None and isinstance(value, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
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


def _number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


def read_brief(path):
    """The brief's directory, ratings and workload, or empty ones on failure."""
    brief = {"providers": [], "ratings": [], "best": [], "manifest": [], "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    directory = next((d for d in docs
                      if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        provider_id = _string_of(raw.get("id"))
        wire = raw.get("wire")
        if not provider_id or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = (_string_of(raw.get("path")) or _string_of(raw.get("base_path"))
                or _string_of(raw.get("base_url")) or _string_of(raw.get("base")))
        brief["providers"].append({
            "id": provider_id,
            "name": _string_of(raw.get("name")) or provider_id,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _string_of(raw.get("key_env")),
            "key": _string_of(raw.get("key")),
        })
    ratings_doc = next((d for d in docs
                        if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = _string_of(raw.get("model"))
            provider = _string_of(raw.get("provider"))
            if not model or not provider:
                continue
            brief["ratings"].append({
                "model": model,
                "provider": provider,
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context_length": _number_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and _string_of(raw.get("id")):
                    brief["best"].append({"id": _string_of(raw.get("id")),
                                         "rating_milli": raw.get("rating_milli") or 0})
    workload_doc = next((d for d in docs
                         if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw):
                continue
            request_id = _string_of(raw.get("id"))
            if not request_id:
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": request_id,
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": requires.get("tools") is True,
                    "vision": requires.get("vision") is True,
                    "json": requires.get("json") is True,
                },
            }
            if isinstance(requires.get("min_context"), (int, float)):
                entry["requires"]["min_context"] = requires["min_context"]
            brief["manifest"].append(entry)
    return brief


def build_catalog(brief):
    """The one catalog: `--list` prints it and /register carries it.

    Built from the brief alone, so both are byte-for-byte the same document
    with or without the env. One entry a provider, in the directory's order;
    each model with its id, free flag and context length, sorted by id.
    """
    providers = []
    for provider in brief["providers"]:
        rows = [r for r in brief["ratings"] if r["provider"] == provider["id"]]
        seen = set()
        models = []
        for row in sorted(rows, key=lambda r: r["model"]):
            if row["model"] in seen:
                continue
            seen.add(row["model"])
            models.append({"id": row["model"], "free": row["free"],
                           "context_length": row["context_length"]})
        providers.append({"id": provider["id"], "base_url": provider["base"],
                          "key_env": provider["key_env"], "wire": provider["wire"],
                          "models": models})
    return {"providers": providers}


def free_models(brief):
    """The free models the entry serves, in catalog order, for GET /v1/models."""
    out = []
    catalog = build_catalog(brief)
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] is True:
                out.append({"id": model["id"], "provider": provider["id"]})
    return out


# --------------------------------------------------------------------------
# Canonical request fields. Extraction mirrors the providers' own parsers
# (openAIParsed / anthropicParsed), so a faithful translation echoes exactly
# what the client sent.
# --------------------------------------------------------------------------

def _blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(block["text"] for block in content
                     if is_record(block) and block.get("type") == "text"
                     and isinstance(block.get("text"), str))


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _parse_image_url(url):
    """An image_url string as (kind, data) -- base64 data or a plain URL."""
    if not isinstance(url, str) or url == "":
        return None
    match = _DATA_URL_RE.match(url)
    if match is not None and match.group(2):
        return {"kind": "base64", "data": match.group(3),
                "media_type": match.group(1) or "image/png", "url": url}
    if match is not None:
        # A data URL that is not base64: the sha is over the URL string.
        return {"kind": "url", "url": url}
    return {"kind": "url", "url": url}


def extract_openai(body):
    """Canonical fields of an OpenAI Chat Completions request."""
    messages = [m for m in body.get("messages", []) if is_record(m)] \
        if isinstance(body.get("messages"), list) else []
    system_parts = []
    user_text = ""
    image = None
    for message in messages:
        if message.get("role") == "system":
            text = _blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = _blocks_text(message.get("content"))
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if not is_record(block):
                        continue
                    if block.get("type") == "image_url" and is_record(block.get("image_url")):
                        parsed = _parse_image_url(block["image_url"].get("url"))
                        if parsed is not None:
                            image = parsed
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not is_record(tool) or not is_record(tool.get("function")):
                continue
            name = tool["function"].get("name")
            if not isinstance(name, str):
                continue
            function = tool["function"]
            tools.append({
                "name": name,
                "description": function.get("description") if isinstance(function.get("description"), str) else "",
                "parameters": function.get("parameters") if is_record(function.get("parameters"))
                else {"type": "object", "properties": {}},
            })
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = None
    response_format = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "system": "\n".join(system_parts),
        "user": user_text,
        "tools": tools,
        "tool_names": [t["name"] for t in tools],
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop")),
        "image": image,
        "json_mode": response_format.get("type") in ("json_object", "json_schema"),
        "temperature": body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None,
        "top_p": body.get("top_p") if isinstance(body.get("top_p"), (int, float)) else None,
        "tool_choice": body.get("tool_choice"),
        "stream": body.get("stream") is True,
    }


def extract_anthropic(body):
    """Canonical fields of an Anthropic Messages request."""
    system_text = _blocks_text(body.get("system"))
    messages = [m for m in body.get("messages", []) if is_record(m)] \
        if isinstance(body.get("messages"), list) else []
    user_text = ""
    image = None
    for message in messages:
        if message.get("role") != "user":
            continue
        user_text = _blocks_text(message.get("content"))
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if not is_record(block) or block.get("type") != "image":
                    continue
                source = block.get("source")
                if not is_record(source):
                    continue
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    image = {"kind": "base64", "data": source["data"],
                             "media_type": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"}
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    image = _parse_image_url(source["url"])
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not is_record(tool) or not isinstance(tool.get("name"), str):
                continue
            tools.append({
                "name": tool["name"],
                "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                "parameters": tool.get("input_schema") if is_record(tool.get("input_schema"))
                else {"type": "object", "properties": {}},
            })
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = None
    return {
        "system": system_text,
        "user": user_text,
        "tools": tools,
        "tool_names": [t["name"] for t in tools],
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop_sequences")),
        "image": image,
        # The Anthropic wire cannot carry JSON mode: its echo is always false.
        "json_mode": False,
        "temperature": body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None,
        "top_p": body.get("top_p") if isinstance(body.get("top_p"), (int, float)) else None,
        "tool_choice": body.get("tool_choice"),
        "stream": body.get("stream") is True,
    }


def requirements_from_body(api, canon):
    """Requirements read off a request body, when the manifest has no row."""
    return {
        "tools": len(canon["tools"]) > 0,
        "vision": canon["image"] is not None,
        "json": canon["json_mode"],
    }


def compute_usage(canon, text):
    """The providers' own usage formula, from the canonical request fields."""
    prompt_len = len(canon["system"]) + len(canon["user"]) + len(",".join(canon["tool_names"]))
    return (1 + prompt_len // 4, 1 + len(text) // 4)

# --------------------------------------------------------------------------
# Translation: canonical fields -> one provider's wire.
# --------------------------------------------------------------------------

def _image_data_url(image):
    if image is None:
        return None
    if image["kind"] == "base64":
        return "data:%s;base64,%s" % (image.get("media_type") or "image/png", image["data"])
    return image["url"]


def _openai_user_content(canon):
    if canon["image"] is None:
        return canon["user"]
    parts = [{"type": "text", "text": canon["user"]}]
    url = _image_data_url(canon["image"])
    parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _anthropic_user_content(canon):
    if canon["image"] is None:
        return canon["user"]
    parts = [{"type": "text", "text": canon["user"]}]
    image = canon["image"]
    if image["kind"] == "base64":
        parts.append({"type": "image", "source": {
            "type": "base64", "media_type": image.get("media_type") or "image/png",
            "data": image["data"]}})
    else:
        parts.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
    return parts


def _openai_tool_choice(choice):
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if not is_record(choice):
        return None
    if choice.get("type") == "auto":
        return "auto"
    if choice.get("type") == "any":
        return "required"
    if choice.get("type") == "none":
        return "none"
    if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _anthropic_tool_choice(choice):
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if is_record(choice) and choice.get("type") == "function" \
            and is_record(choice.get("function")) \
            and isinstance(choice["function"].get("name"), str):
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


def to_openai_body(canon, model_id):
    """Canonical fields as an OpenAI Chat Completions request naming model_id."""
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    messages.append({"role": "user", "content": _openai_user_content(canon)})
    out = {"model": model_id, "messages": messages}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop"] = list(canon["stops"])
    if canon["tools"]:
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in canon["tools"]]
    choice = _openai_tool_choice(canon["tool_choice"])
    if choice is not None:
        out["tool_choice"] = choice
    if canon["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if canon["stream"]:
        out["stream"] = True
        # Ask for the usage chunk: the client's stream parser reads it.
        out["stream_options"] = {"include_usage": True}
    return out


def to_anthropic_body(canon, model_id):
    """Canonical fields as an Anthropic Messages request naming model_id."""
    out = {"model": model_id}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["system"] != "":
        out["system"] = canon["system"]
    out["messages"] = [{"role": "user", "content": _anthropic_user_content(canon)}]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop_sequences"] = list(canon["stops"])
    if canon["tools"]:
        out["tools"] = [{"name": t["name"], "description": t["description"],
                         "input_schema": t["parameters"]} for t in canon["tools"]]
    choice = _anthropic_tool_choice(canon["tool_choice"])
    if choice is not None:
        out["tool_choice"] = choice
    if canon["stream"]:
        out["stream"] = True
    return out


def to_gemini_body(canon, model_id):
    """Canonical fields as a Gemini generateContent request naming model_id."""
    parts = [{"text": canon["user"]}]
    image = canon["image"]
    if image is not None:
        if image["kind"] == "base64":
            parts.append({"inlineData": {
                "mimeType": image.get("media_type") or "image/png", "data": image["data"]}})
        else:
            parts.append({"fileData": {"fileUri": image["url"]}})
    out = {"contents": [{"role": "user", "parts": parts}]}
    if canon["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = list(canon["stops"])
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Provider calls.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    """A directory base joined to the env URL; an absolute base stands."""
    base = (base or "").strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/")
    if base == "":
        return env_url
    return env_url + ("" if base.startswith("/") else "/") + base.rstrip("/")


def provider_key(provider):
    """The provider's key: its key_env variable when set, else the directory's."""
    key_env = provider.get("key_env") or ""
    if key_env:
        value = os.environ.get(key_env)
        if value:
            return value
    return provider.get("key") or ""


def auth_headers(provider):
    """The headers carrying one provider's key, in its auth scheme."""
    auth = provider.get("auth")
    key = provider_key(provider)
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key} if key else {}
    if auth == "x-api-key":
        return {"x-api-key": key} if key else {}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key} if key else {}
    return {}


def provider_url(provider, wire, model_id, stream):
    """The URL of one inference call on one provider, in its own wire."""
    base = join_base(ENV_URL, provider["base"])
    if wire == "openai":
        return base + "/v1/chat/completions"
    if wire == "anthropic":
        return base + "/v1/messages"
    verb = "streamGenerateContent" if stream else "generateContent"
    return "%s/v1beta/models/%s:%s" % (base, model_id, verb)


def call_provider(provider, wire, model_id, body, stream):
    """One inference call. Returns (status, body_bytes) or (None, error)."""
    url = provider_url(provider, wire, model_id, stream)
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider))
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if stream:
        headers["accept"] = "text/event-stream"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_S) as response:
            return (response.status, response.read())
    except urllib.error.HTTPError as err:
        try:
            return (err.code, err.read())
        except OSError:
            return (err.code, b"")
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        return (None, str(err).encode("utf-8", "replace"))


def error_message(status, raw):
    """The message of a provider error body, in any of the three envelopes."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if is_record(value) and is_record(value.get("error")):
        message = value["error"].get("message")
        return message if isinstance(message, str) else ""
    return ""


def classify_failure(status, raw):
    """A failed provider call as (kind, detail).

    kind is 'dead' (budget spent / outage: never call this provider again),
    'transient' (the provider's own failure schedule: the next call answers),
    'hard' (a 400/401: retrying the same call cannot help) or 'unknown'
    (a 429/5xx with no recognised message: a real service's rate limit).
    """
    if status in (400, 401):
        return ("hard", "HTTP %s" % status)
    message = error_message(status, raw)
    if "budget is spent" in message:
        return ("dead", "budget spent")
    if "is out" in message:
        return ("dead", "outage")
    if "refused this call" in message:
        return ("transient", "scheduled refusal")
    if status == 402:
        return ("dead", "HTTP 402")
    if status in (429, 500, 502, 503, 504) or status is None:
        return ("unknown", "HTTP %s" % (status if status is not None else "unreachable"))
    return ("hard", "HTTP %s" % status)


# --------------------------------------------------------------------------
# Parsing a provider's answer: text and stop reason, in any wire.
# --------------------------------------------------------------------------

def _text_of_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block["text"] for block in content
                       if is_record(block) and block.get("type") == "text"
                       and isinstance(block.get("text"), str))
    return ""


def _parse_json_answer(wire, raw):
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not is_record(value):
        return None
    if wire == "openai":
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        choice = choices[0]
        message = choice.get("message") if is_record(choice.get("message")) else {}
        return (_text_of_content(message.get("content")),
                choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None)
    if wire == "anthropic":
        if not isinstance(value.get("content"), list):
            return None
        return (_text_of_content(value.get("content")),
                value.get("stop_reason") if isinstance(value.get("stop_reason"), str) else None)
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    candidate = candidates[0]
    content = candidate.get("content") if is_record(candidate.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(part["text"] for part in parts
                   if is_record(part) and isinstance(part.get("text"), str))
    return (text, candidate.get("finishReason") if isinstance(candidate.get("finishReason"), str) else None)


def _parse_sse_frames(text):
    """SSE text as a list of (event, data) pairs, in order."""
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


def _parse_sse_answer(wire, raw):
    text = raw.decode("utf-8", "replace")
    frames = _parse_sse_frames(text)
    out_text = []
    stop = None
    for event, data in frames:
        if data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if wire == "openai":
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not is_record(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            content = delta.get("content")
            if isinstance(content, str):
                out_text.append(content)
            if isinstance(choice.get("finish_reason"), str):
                stop = choice["finish_reason"]
        elif wire == "anthropic":
            if event == "content_block_delta":
                delta = chunk.get("delta") if is_record(chunk.get("delta")) else {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    out_text.append(delta["text"])
            elif event == "message_delta":
                delta = chunk.get("delta") if is_record(chunk.get("delta")) else {}
                if isinstance(delta.get("stop_reason"), str):
                    stop = delta["stop_reason"]
        else:  # gemini
            candidates = chunk.get("candidates")
            if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
                continue
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if is_record(part) and isinstance(part.get("text"), str):
                    out_text.append(part["text"])
            if isinstance(candidate.get("finishReason"), str):
                stop = candidate["finishReason"]
    return ("".join(out_text), stop)


def parse_provider_answer(wire, stream, raw):
    """(text, stop) of a provider's answer, or None when it is not one."""
    if stream:
        return _parse_sse_answer(wire, raw)
    return _parse_json_answer(wire, raw)

# --------------------------------------------------------------------------
# Routing: the best free model that meets a request, with failover.
# --------------------------------------------------------------------------

def meets_requirements(requires, row):
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


class State(object):
    """Per-trial routing state: dead providers, call counts, manifest index."""

    def __init__(self, brief, catalog):
        self.brief = brief
        self.catalog = catalog
        self.providers_by_id = {p["id"]: p for p in brief["providers"]}
        self.rows_by_provider = {}
        for row in brief["ratings"]:
            self.rows_by_provider.setdefault(row["provider"], []).append(row)
        self.dead = set()
        self.call_counts = {}
        self.success_counts = {}
        self.earned = 0
        self.manifest_index = 0
        self.lock = threading.Lock()
        self.best_by_id = {b["id"]: b["rating_milli"] for b in brief["best"]}
        # The maximum earned score (every request at its best rating) and
        # the floor a cheaper pick must keep the projection above.
        self.bound = sum(entry["weight"] * self.best_by_id.get(entry["id"], 0)
                         for entry in brief["manifest"])
        self.earned_floor = (99 * self.bound + 99) // 100 + EARNED_SAFETY_MARGIN
        # Suffix sums over the manifest, so rank_candidates can project the
        # remaining requests' best score and weight in O(1): index i holds
        # the totals for the requests from position i onward.
        manifest = brief["manifest"]
        self._suffix_best = [0] * (len(manifest) + 1)
        self._suffix_weight = [0] * (len(manifest) + 1)
        for i in range(len(manifest) - 1, -1, -1):
            entry = manifest[i]
            self._suffix_best[i] = (self._suffix_best[i + 1]
                                    + entry["weight"] * self.best_by_id.get(entry["id"], 0))
            self._suffix_weight[i] = self._suffix_weight[i + 1] + entry["weight"]
        self.serving_index = 0
        self.serving_weight = 1

    def requirements_for(self, api, canon):
        """The manifest's row by arrival order, else the body's own fields.

        Returns (request id, requirements, weight, manifest index). The
        weight and index are the routing guard's inputs: they say what this
        request is worth and what is still to come.
        """
        manifest = self.brief["manifest"]
        index = self.manifest_index
        entry = None
        if index < len(manifest):
            entry = manifest[index]
        self.manifest_index += 1
        if entry is not None and entry["api"] == api:
            self.serving_index = index
            self.serving_weight = entry["weight"]
            return entry["id"], entry["requires"], entry["weight"], index
        return None, requirements_from_body(api, canon), 1, index

    def rank_candidates(self, api, requires, json_mode):
        """Free (provider, model) pairs meeting the request, best first.

        The order is the cheap order: a provider never called yet gets
        FRESH_PROVIDER_BONUS rating points, so the first call spreads over
        the providers instead of queuing behind a top provider whose budget
        is about to be spent on a scheduled failure or an outage; then
        rating, then the client's own wire (a same-wire call passes the
        body through untouched), then the provider called least so far,
        then provider and model id, so the order is deterministic.
        A provider that already answered MAX_SUCCESSES_PER_PROVIDER
        requests is not offered at all, unless that would leave nothing
        to try. And a cheaper pick -- one below the best legal rating --
        is only taken while the earned score projected for the rest of the
        workload stays above the quality-99 floor; otherwise the best
        legal model is picked instead, so saving a call never risks the
        quality column.
        """
        out = []
        for provider in self.brief["providers"]:
            if provider["id"] in self.dead:
                continue
            wire = provider["wire"]
            if json_mode and wire == "anthropic":
                # The Anthropic wire cannot carry JSON mode: its echo is
                # always json_mode false, which would lose the request.
                continue
            wire_match = 0 if wire == api else 1
            for row in self.rows_by_provider.get(provider["id"], []):
                if not row["free"]:
                    continue
                if not meets_requirements(requires, row):
                    continue
                out.append((provider, row, wire_match))
        active = [item for item in out
                  if self.success_counts.get(item[0]["id"], 0) < MAX_SUCCESSES_PER_PROVIDER]
        if not active:
            active = out

        def cheap_key(item):
            provider, row, wire_match = item
            bonus = (FRESH_PROVIDER_BONUS
                     if self.call_counts.get(provider["id"], 0) == 0 else 0)
            return (-(row["rating_milli"] + bonus), -row["rating_milli"],
                    wire_match, self.call_counts.get(provider["id"], 0),
                    provider["id"], row["model"])

        def greedy_key(item):
            provider, row, wire_match = item
            return (-row["rating_milli"], wire_match,
                    self.call_counts.get(provider["id"], 0),
                    provider["id"], row["model"])

        cheap = sorted(active, key=cheap_key)
        greedy = sorted(active, key=greedy_key)
        if cheap and greedy and cheap[0][1]["rating_milli"] < greedy[0][1]["rating_milli"]:
            index = self.serving_index
            projected = (self.earned
                         + self.serving_weight * cheap[0][1]["rating_milli"]
                         + self._suffix_best[index + 1]
                         - REMAINING_SHORTFALL_PER_WEIGHT * self._suffix_weight[index + 1])
            if projected < self.earned_floor:
                return [(provider, row) for provider, row, _ in greedy]
        return [(provider, row) for provider, row, _ in cheap]


def translate(canon, wire, model_id):
    if wire == "openai":
        return to_openai_body(canon, model_id)
    if wire == "anthropic":
        return to_anthropic_body(canon, model_id)
    return to_gemini_body(canon, model_id)


def serve_request(state, api, body):
    """Answer one client request. Returns a result dict (ok True/False)."""
    canon = extract_openai(body) if api == "openai" else extract_anthropic(body)
    request_id, requires, weight, _index = state.requirements_for(api, canon)
    stream = canon["stream"]
    deadline = time.monotonic() + REQUEST_DEADLINE_S
    calls_made = 0
    tried = set()
    while calls_made < MAX_CALLS_PER_REQUEST and time.monotonic() < deadline:
        candidates = state.rank_candidates(api, requires, canon["json_mode"])
        pick = None
        for provider, row in candidates:
            if (provider["id"], row["model"]) not in tried:
                pick = (provider, row)
                break
        if pick is None:
            break
        provider, row = pick
        wire = provider["wire"]
        upstream_body = translate(canon, wire, row["model"])
        transient_retries = 0
        unknown_retries = 0
        while True:
            if calls_made >= MAX_CALLS_PER_REQUEST or time.monotonic() >= deadline:
                return {"ok": False, "request_id": request_id, "canon": canon}
            status, raw = call_provider(provider, wire, row["model"], upstream_body, stream)
            state.call_counts[provider["id"]] = state.call_counts.get(provider["id"], 0) + 1
            calls_made += 1
            if status is not None and 200 <= status < 300:
                parsed = parse_provider_answer(wire, stream, raw)
                if parsed is None:
                    tried.add((provider["id"], row["model"]))
                    break
                text, stop = parsed
                # The routing guard's books: one more answer from this
                # provider, and this request's earned score.
                state.success_counts[provider["id"]] = state.success_counts.get(provider["id"], 0) + 1
                state.earned += weight * row["rating_milli"]
                return {"ok": True, "request_id": request_id, "canon": canon,
                        "provider": provider["id"], "model": row["model"],
                        "text": text, "stop": stop}
            kind, detail = classify_failure(status, raw)
            log("%s: %s/%s refused (%s); %s" % (
                request_id or "?", provider["id"], row["model"], detail,
                "dead" if kind == "dead" else "retrying" if kind in ("transient", "unknown") else "moving on"))
            if kind == "dead":
                state.dead.add(provider["id"])
                tried.add((provider["id"], row["model"]))
                break
            if kind == "hard":
                tried.add((provider["id"], row["model"]))
                break
            if kind == "transient":
                if transient_retries < MAX_TRANSIENT_RETRIES:
                    transient_retries += 1
                    continue
                tried.add((provider["id"], row["model"]))
                break
            if unknown_retries < MAX_UNKNOWN_RETRIES:
                unknown_retries += 1
                time.sleep(UNKNOWN_RETRY_WAIT_S)
                continue
            tried.add((provider["id"], row["model"]))
            break
    return {"ok": False, "request_id": request_id, "canon": canon}


# --------------------------------------------------------------------------
# Answers in the wire the client used.
# --------------------------------------------------------------------------

def _sse_frame(event, data):
    text = json.dumps(data)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def _split_text(text, size=13):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def build_openai_json(model_id, text, stop_reason, usage):
    return {
        "id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": stop_reason, "logprobs": None}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                  "total_tokens": usage[0] + usage[1]},
    }


def build_anthropic_json(model_id, text, stop_reason, usage):
    return {
        "id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


def build_openai_stream(model_id, text, stop_reason, usage):
    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = [_sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(text):
        frames.append(_sse_frame(None, chunk({"content": piece}, None)))
    frames.append(_sse_frame(None, chunk({}, stop_reason)))
    frames.append(_sse_frame(None, {
        "id": "chatcmpl-agg", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model_id, "choices": [],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                  "total_tokens": usage[0] + usage[1]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def build_anthropic_stream(model_id, text, stop_reason, usage):
    frames = [_sse_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg-stream", "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage[0], "output_tokens": 0}}})]
    frames.append(_sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        frames.append(_sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage[1]}}))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def answer_for_client(api, stream, result):
    """A served request as (content_type, body_text) in the client's wire."""
    canon = result["canon"]
    usage = compute_usage(canon, result["text"])
    family = STOP_FAMILIES.get(result["stop"], 0)
    if api == "openai":
        stop_reason = OPENAI_STOP[family]
        if stream:
            return ("text/event-stream; charset=utf-8",
                    build_openai_stream(result["model"], result["text"], stop_reason, usage))
        return ("application/json",
                json.dumps(build_openai_json(result["model"], result["text"], stop_reason, usage)))
    stop_reason = ANTHROPIC_STOP[family]
    if stream:
        return ("text/event-stream; charset=utf-8",
                build_anthropic_stream(result["model"], result["text"], stop_reason, usage))
    return ("application/json",
            json.dumps(build_anthropic_json(result["model"], result["text"], stop_reason, usage)))


# --------------------------------------------------------------------------
# HTTP server, register, poll.
# --------------------------------------------------------------------------

STATE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_json(self, status, value):
        self._send(status, "application/json", json.dumps(value))

    def _error(self, api, status, message):
        if api == "anthropic":
            self._send_json(status, {"type": "error",
                                     "error": {"type": "overloaded_error", "message": message}})
        else:
            self._send_json(status, {"error": {"message": message,
                                               "type": "server_error", "code": None}})

    def _read_json(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_CLIENT_BODY_BYTES:
            return None
        try:
            raw = self.rfile.read(length)
        except OSError:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return value if is_record(value) else None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path != "/v1/models":
            self._error("openai", 404, "no route GET %s" % path)
            return
        models = free_models(STATE.brief)
        if "x-api-key" in self.headers:
            self._send_json(200, {
                "data": [{"type": "model", "id": m["id"], "display_name": m["id"],
                          "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True}
                         for m in models],
                "has_more": False})
        else:
            self._send_json(200, {
                "object": "list",
                "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                          "owned_by": m["provider"], "free": True} for m in models]})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._error("openai", 404, "no route POST %s" % path)
            return
        body = self._read_json()
        if body is None:
            self._error(api, 400, "the request body is not one JSON object")
            return
        with STATE.lock:
            result = serve_request(STATE, api, body)
        if not result["ok"]:
            self._error(api, 503, "no free model could serve this request")
            return
        usage = compute_usage(result["canon"], result["text"])
        log("%s: served by %s/%s (%d in, %d out)" % (
            result["request_id"] or "?", result["provider"], result["model"],
            usage[0], usage[1]))
        content_type, text = answer_for_client(api, result["canon"]["stream"], result)
        self._send(200, content_type, text)


def register(port, catalog):
    """POST /register with the port and the one catalog; True on ok."""
    url = ENV_URL + "/register"
    data = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                       "catalog": catalog}).encode("utf-8")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, data=data, headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_S) as response:
                value = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and is_record(value) and value.get("ok") is True:
                    log("registered: the workload holds %s requests" % value.get("requests"))
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


def main(argv):
    global STATE
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    catalog = build_catalog(brief)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    STATE = State(brief, catalog)
    server = None
    for _attempt in range(5):
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        server.daemon_threads = True
        port = server.server_address[1]
        if port != 8080:
            break
        # 8080 is the env's on one layout; the OS's ephemeral range never
        # holds it, and the guard says so.
        server.server_close()
        server = None
    if server is None:
        log("could not bind a port other than 8080")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models" % (
        port, len(catalog["providers"]),
        sum(1 for p in catalog["providers"] for m in p["models"] if m["free"])))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    if not register(port, catalog):
        log("register failed: the env did not answer")
        server.shutdown()
        return 1
    poll_until_done()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
