"""One small interface over three ways of running an open-weight model locally.

    ollama        Ollama's HTTP API (easiest on Windows and Linux: install Ollama, `ollama pull qwen3:8b`)
    openai        any OpenAI-compatible server (LM Studio, vLLM, llama.cpp server, Ollama's /v1)
    transformers  load the model in this Python process (needs a GPU with ~20 GB for an 8B model)

Every backend exposes: chat(system, user) -> str, stream(system, user) -> iterator of str,
and batch(list_of_(system, user)) -> list of str (fast path for building the graph).
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config


class LLM:
    def __init__(self, backend=None, model=None, url=None):
        self.backend = backend or config.LLM_BACKEND
        self.model_name = model or config.LLM_MODEL
        self.url = (url or config.LLM_URL).rstrip("/")
        self._hf = None
        if self.backend == "transformers":
            self._load_hf()

    # ------------------------------------------------------------------ public
    def chat(self, system, user, max_tokens=600, json_mode=False):
        return "".join(self.stream(system, user, max_tokens, json_mode)).strip()

    def stream(self, system, user, max_tokens=600, json_mode=False):
        if self.backend == "transformers":
            yield from self._hf_stream(system, user, max_tokens)
        elif self.backend == "ollama":
            yield from self._ollama(system, user, max_tokens, json_mode)
        else:
            yield from self._openai(system, user, max_tokens, json_mode)

    def batch(self, prompts, max_tokens=900, json_mode=False, batch_size=16, workers=4):
        if self.backend == "transformers":
            out = []
            for i in range(0, len(prompts), batch_size):
                out += self._hf_batch(prompts[i:i + batch_size], max_tokens)
            return out
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda p: self.chat(p[0], p[1], max_tokens, json_mode), prompts))

    def check(self):
        """Return (ok, message) - used by the app's status panel and the installer."""
        try:
            if self.backend == "transformers":
                return True, f"{self.model_name} loaded in-process"
            if self.backend == "ollama":
                tags = requests.get(self.url + "/api/tags", timeout=5).json().get("models", [])
                names = [t["name"] for t in tags]
                if not any(n == self.model_name or n.split(":")[0] == self.model_name for n in names):
                    return False, f"Ollama is running but '{self.model_name}' is not pulled. Run: ollama pull {self.model_name}"
                return True, f"Ollama: {self.model_name}"
            requests.get(self.url + "/models", timeout=5, headers=self._auth())
            return True, f"OpenAI-compatible server at {self.url}: {self.model_name}"
        except requests.RequestException:
            return False, f"Cannot reach the {self.backend} server at {self.url}. Is it running?"

    # ------------------------------------------------------------------ HTTP backends
    def _ollama(self, system, user, max_tokens, json_mode):
        body = {"model": self.model_name, "stream": True, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384}}
        if json_mode:
            body["format"] = "json"
        with requests.post(self.url + "/api/chat", json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    piece = json.loads(line).get("message", {}).get("content", "")
                    if piece:
                        yield piece

    def _auth(self):
        return {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}

    def _openai(self, system, user, max_tokens, json_mode):
        body = {"model": self.model_name, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        with requests.post(self.url + "/chat/completions", json=body, stream=True, timeout=600,
                           headers=self._auth()) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    # ------------------------------------------------------------------ in-process (transformers)
    def _load_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self._hf = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.bfloat16, device_map=config.device())
        self._hf.eval()

    def _prompt(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"enable_thinking": False} if "qwen3" in self.model_name.lower() else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _gen_kwargs(self, max_tokens):
        return dict(max_new_tokens=max_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)

    def _hf_stream(self, system, user, max_tokens):
        from transformers import TextIteratorStreamer
        inputs = self.tok(self._prompt(system, user), return_tensors="pt").to(self._hf.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        th = threading.Thread(target=self._hf.generate, kwargs=dict(**inputs, streamer=streamer,
                                                                    **self._gen_kwargs(max_tokens)))
        th.start()
        yield from streamer
        th.join()

    def _hf_batch(self, prompts, max_tokens):
        texts = [self._prompt(s, u) for s, u in prompts]
        inputs = self.tok(texts, return_tensors="pt", padding=True).to(self._hf.device)
        with self._torch.no_grad():
            out = self._hf.generate(**inputs, **self._gen_kwargs(max_tokens))
        new = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]


def parse_json(text):
    """Best-effort JSON from a model reply (strips code fences and trailing chatter)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start = t.find("{")
    if start < 0:
        return None
    depth, end = 0, None
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return json.loads(t[start:end] if end else t[start:])
    except json.JSONDecodeError:
        return _repair(t[start:])


def _repair(t):
    """A reply cut off by the token limit: keep everything up to the last complete object and close the brackets."""
    for cut in [i for i, ch in enumerate(t) if ch == "}"][::-1][:40]:
        prefix, stack, in_str, esc = t[:cut + 1], [], False, False
        for ch in prefix:
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            continue
        try:
            return json.loads(prefix + "".join(reversed(stack)))
        except json.JSONDecodeError:
            continue
    return None
