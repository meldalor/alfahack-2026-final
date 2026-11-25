"""Клиент LLM (OpenAI-совместимый прокси платформы).

Адрес и ключ берутся из OPENAI_BASE_URL / OPENAI_API_KEY. Любая ошибка обращения не роняет агента:
метод возвращает None, а после серии сбоев LLM отключается и агент работает по встроенной стратегии.
"""

import ast
import json
import os
import re
import time

from ag_utils import elapsed, log, short

DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LLM:
    def __init__(self, max_calls: int = 40, call_timeout: float = 240.0, time_limit: float = 2400.0,
                 reasoning_effort: str | None = "medium"):
        self.model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.max_calls = max_calls
        self.call_timeout = call_timeout
        self.time_limit = time_limit
        # без ограничения модель рассуждает до 12 тыс. токенов и не успевает ответить; "medium" — 2–3 тыс.
        self.reasoning_effort = reasoning_effort
        self.calls = 0
        self.failures = 0
        self.time_spent = 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.disabled_reason = None
        self.client = None
        if not os.environ.get("OPENAI_API_KEY"):
            self.disabled_reason = "нет OPENAI_API_KEY"
        else:
            try:
                from openai import OpenAI

                self.client = OpenAI(timeout=call_timeout, max_retries=0)
            except Exception as e:
                self.disabled_reason = f"не удалось создать клиент: {e!r}"
        log(f"LLM: модель={self.model}", "включена" if self.enabled else f"выключена ({self.disabled_reason})")

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.disabled_reason is None

    def _disable(self, reason: str):
        if self.disabled_reason is None:
            self.disabled_reason = reason
            log(f"LLM отключена: {reason}")

    def chat(self, messages, max_tokens: int = 6000, temperature: float = 0.4, timeout: float | None = None):
        """Один запрос с повтором. Возвращает текст ответа (без блока рассуждений) или None."""
        if not self.enabled:
            return None
        if self.calls >= self.max_calls:
            self._disable("исчерпан лимит запросов")
            return None
        if self.time_spent > self.time_limit:
            self._disable("исчерпан лимит времени на LLM")
            return None
        timeout = min(timeout or self.call_timeout, self.call_timeout)
        effort = self.reasoning_effort
        for attempt in range(2):
            self.calls += 1
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **({"reasoning_effort": effort} if effort else {}),
                )
                dt = time.time() - t0
                self.time_spent += dt
                text, finish = self._extract_text(resp)
                info = f"finish={finish}"
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                    self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
                    info += f", токенов {getattr(usage, 'completion_tokens', None)}, из них рассуждения {reasoning}"
                if text:
                    self.failures = 0
                    log(f"LLM ответ за {dt:.1f}s, {len(text)} симв., {info}")
                    return text
                log(f"LLM вернула пустой ответ за {dt:.1f}s, {info}")
                # пустой ответ почти всегда означает, что все токены ушли в рассуждения: просим ответить сразу
                messages = list(messages) + [
                    {"role": "user", "content": "Answer now, briefly, strictly in the requested format. /no_think"}
                ]
                effort = "low" if effort else None
            except Exception as e:
                dt = time.time() - t0
                self.time_spent += dt
                err = repr(e)
                low = err.lower()
                if effort and "reasoning_effort" in low:
                    # прокси не знает параметр: дальше без него, сбоем это не считаем
                    log(f"LLM: прокси не принимает reasoning_effort, запросы без него ({short(err, 200)})")
                    self.reasoning_effort = effort = None
                    continue
                self.failures += 1
                log(f"LLM ошибка ({dt:.1f}s): {short(err, 300)}")
                if any(s in low for s in ("budget", "authentication", "401", "403", "invalid api key", "insufficient")):
                    self._disable("ошибка доступа/бюджета: " + short(err, 120))
                    return None
                if self.failures >= 4:
                    self._disable("слишком много ошибок подряд")
                    return None
                time.sleep(min(20, 4 * (attempt + 1)))
        return None

    @staticmethod
    def _extract_text(resp) -> tuple:
        """(текст ответа без рассуждений, finish_reason)."""
        choices = getattr(resp, "choices", None)
        if not choices:
            return "", None
        finish = getattr(choices[0], "finish_reason", None)
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return "", finish
        content = getattr(msg, "content", None) or ""
        if isinstance(content, list):
            content = "".join(str(getattr(c, "text", c)) for c in content)
        content = re.sub(r"<think>.*?</think>", "", str(content), flags=re.S).strip()
        if content.startswith("<think>"):
            content = ""
        if not content and finish != "length":
            # некоторые прокси кладут ответ в поле reasoning. При обрезке (length) там только черновик рассуждений:
            # из него разбирается случайный JSON, поэтому тогда пустой ответ и повтор запроса
            for attr in ("reasoning_content", "reasoning"):
                extra = getattr(msg, attr, None)
                if isinstance(extra, str) and "{" in extra:
                    return extra, finish
        return content, finish

    def ask_json(self, system: str, user: str, max_tokens: int = 6000, temperature: float = 0.4, timeout=None):
        text = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if text is None:
            return None
        obj = extract_json(text)
        if obj is None:
            log("LLM: не удалось разобрать JSON:", short(text, 400))
        return obj

    def stats(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "time_spent": round(self.time_spent, 1),
            "usage": self.usage,
            "reasoning_effort": self.reasoning_effort,
            "disabled_reason": self.disabled_reason,
            "elapsed": round(elapsed(), 1),
        }


def _balanced_objects(text: str):
    """Все подстроки вида {...} со сбалансированными скобками (с учётом строк)."""
    out = []
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
        if len(out) > 20:
            break
    return out


def _try_load(s: str):
    s = s.strip()
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            # strict=False: модели часто вставляют переводы строк прямо внутрь строк (например, код)
            return json.loads(candidate, strict=False)
        except Exception:
            pass
        try:
            py = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", candidate)))
            val = ast.literal_eval(py)
            if isinstance(val, (dict, list)):
                return val
        except Exception:
            pass
    return None


def extract_json(text: str):
    """Достаёт JSON-объект из ответа модели: блок ```json```, затем самый длинный {...}."""
    if not text:
        return None
    for block in re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S):
        val = _try_load(block)
        if isinstance(val, dict):
            return val
    candidates = sorted(_balanced_objects(text), key=len, reverse=True)
    for c in candidates:
        val = _try_load(c)
        if isinstance(val, dict):
            return val
    return None


def extract_code_blocks(text: str) -> list:
    """Все блоки ```python ...``` с определением функции, по порядку."""
    blocks = re.findall(r"```(?:python|py|Python)\s*\n(.*?)```", text or "", flags=re.S)
    return [b for b in blocks if "def " in b]
