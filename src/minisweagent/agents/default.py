"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

        # ── Memory primitive accumulators (written to MSWEA_TOKEN_LOG_PATH) ──
        self._mem_prompt_tokens      = 0
        self._mem_completion_tokens  = 0
        self._mem_total_latency      = 0.0
        self._mem_call_latencies: list[float] = []
        self._mem_compression_events        = 0
        self._mem_tokens_saved              = 0
        self._mem_compression_ratios: list[float] = []
        self._mem_summarization_prompt_tokens = 0
        self._mem_summarization_latency_s     = 0.0
        # per-step and per-compression-event detail
        self._mem_step_prompt_tokens: list[int] = []
        self._mem_step_completion_tokens: list[int] = []
        self._mem_compression_event_steps: list[int] = []
        self._mem_context_tokens_at_compression: list[int] = []
        self._mem_context_tokens_after_compression: list[int] = []
        self._mem_trc_fallback_events               = 0
        # online TRC accumulators
        self._mem_online_trc_flags: list[str] = []
        self._mem_online_trc_tokens_saved: int = 0

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages.

        Memory primitive hook
        ---------------------
        Reads two environment variables before every LLM call:
          MSWEA_PRIMITIVE    : "truncation" | "summarization"
          MSWEA_TOKEN_BUDGET : int  — fires when estimated prompt tokens exceed this

        When the budget is hit, the chosen primitive compresses self.messages down
        to budget * 0.5 tokens (compression ratio r = 0.5), keeping the system
        prompt and first user message (task statement) protected.

        Token usage and compression stats are accumulated on self._mem_* and written
        to MSWEA_TOKEN_LOG_PATH (if set) after every call.
        """
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        # ── Memory primitive hook ────────────────────────────────────────────
        #
        # WHAT IS self.messages?
        #   The full conversation history accumulated so far:
        #     messages[0]  — system prompt (never changes)
        #     messages[1]  — user message containing the task (never changes)
        #     messages[2+] — alternating assistant/user messages from each step
        #   On every LLM call (model.query below), the ENTIRE history is sent.
        #   So the history IS the context window.
        #
        # WHAT IS THE BUDGET?
        #   MSWEA_TOKEN_BUDGET is a token count threshold for the context window.
        #   It is set by run_experiment.py as:
        #     budget = max(step_prompt_tokens from baseline run) * budget_pct
        #   e.g. if the baseline peak context was 40K and budget_pct=0.60,
        #   budget = 24K.  Compression fires when the history exceeds 24K tokens.
        #   At p100 (baseline) the budget is set to 999999999 so it never fires.
        #
        # TRIGGER: context window size > budget
        #   We measure the current history size with count_tokens(self.messages)
        #   using tiktoken (cl100k_base, accurate to ~5% across models).
        #   This is checked BEFORE each LLM call, so compression always happens
        #   before the model sees an oversized context.
        #   No reset is needed: after compression the history shrinks below budget,
        #   so the check naturally won't fire again until the history grows back.
        #
        # WHAT COMPRESSION DOES:
        #   Both primitives protect messages[0:2] (system + task) — never touched.
        #   They operate only on messages[2:] (the agent's working history).
        #   target = current_size * COMPRESSION_RATIO (0.5) — compress to 50%.
        #
        #   truncation  — drops the oldest messages from the front of messages[2:]
        #                 until size <= target.  No extra LLM call.
        #   summarization — one extra LLM call produces a structured summary of
        #                 messages[2:], which replaces the entire compressible
        #                 window with a single summary message.
        #
        _primitive = os.environ.get("MSWEA_PRIMITIVE", "")
        _budget    = int(os.environ.get("MSWEA_TOKEN_BUDGET", "0") or "0")

        # ── Online TRC hook ──────────────────────────────────────────────────
        # Fires every step (no budget needed) when primitive == "online_trc".
        # The model emits NEED_RESULT: <flag> in its response.  At call n+2 we
        # apply that flag to the tool result from call n (messages[-3]).
        #
        # Warmup: skip the first 5 calls so early exploration is never cleared.
        # Guard:  need ≥4 compressible messages (2 full steps) for [-4]/[-3].
        # Graceful degradation: if [-4] has no valid flag, default = "full" (no-op).
        if _primitive == "online_trc" and self.n_calls >= 5 and len(self.messages) >= 6:
            import re as _re
            import memory as _mem_otrc
            _asst_msg    = self.messages[-4]  # assistant from call n (has NEED_RESULT flag)
            _result_msg  = self.messages[-3]  # tool result from call n  (target for clearing)
            _asst_content = _asst_msg.get("content") or ""
            if isinstance(_asst_content, list):
                _asst_content = " ".join(
                    b.get("text", "") for b in _asst_content if isinstance(b, dict)
                )
            _flag_match = _re.search(
                r"NEED_RESULT:\s*(none|first_half|second_half|full)",
                str(_asst_content),
                _re.IGNORECASE,
            )
            _flag = _flag_match.group(1).lower() if _flag_match else "full"

            _orig_content = _result_msg.get("content") or ""
            _orig_tokens  = _mem_otrc.count_tokens([_result_msg])
            _new_content  = _orig_content  # default: no change

            if _flag == "none":
                _new_content = f"[TOOL OUTPUT CLEARED — online-trc — {_orig_tokens} tokens — step {self.n_calls - 2}]"
            elif _flag == "first_half":
                _mid = max(1, len(str(_orig_content)) // 2)
                _new_content = str(_orig_content)[:_mid] + "\n[...truncated by online-trc (first_half)...]"
            elif _flag == "second_half":
                _mid = max(1, len(str(_orig_content)) // 2)
                _new_content = "[...truncated by online-trc (second_half)...]\n" + str(_orig_content)[_mid:]
            # "full" → no change

            if _flag != "full":
                self.messages[-3] = {**_result_msg, "content": _new_content}

            _tokens_saved_otrc = max(0, _orig_tokens - _mem_otrc.count_tokens([self.messages[-3]]))
            self._mem_online_trc_flags.append({
                "step":           self.n_calls,        # call number when clearing happens
                "flag_from_step": self.n_calls - 2,    # call that emitted the flag
                "flag":           _flag,
                "flag_found":     _flag_match is not None,
                "tokens_cleared": _tokens_saved_otrc,
            })
            self._mem_online_trc_tokens_saved += _tokens_saved_otrc
            _mem_otrc.write_token_log(self)
        # ── End online TRC hook ──────────────────────────────────────────────

        if _primitive and _budget > 0:
            import memory as _mem   # agentCtx root must be on PYTHONPATH
            # Measure the current context window size (= full history size).
            # This is what the model would receive on the next call.
            _current = _mem.count_tokens(self.messages)
            if _current > _budget:
                # History has grown past the budget — compress it now.
                # Target: reduce to 50% of current size.
                _target = max(1, int(_current * _mem.COMPRESSION_RATIO))
                if _primitive == "summarization":
                    # LLM call produces a structured summary replacing messages[2:].
                    # Tokens used by that summary call are tracked separately so we
                    # can distinguish them from the main agent's token usage.
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.summarize(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens              += _pt
                    self._mem_completion_tokens          += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s    += _sum_lat
                elif _primitive == "structured_summarize":
                    # LLM call produces a schema-guided summary (Task / Files Modified /
                    # Files Examined / Execution Anchors / Current State).
                    self.messages, _saved, _pt, _ct, _sum_lat = _mem.structured_summarize(
                        self.messages, self.model, _target
                    )
                    self._mem_prompt_tokens               += _pt
                    self._mem_completion_tokens           += _ct
                    self._mem_summarization_prompt_tokens += _pt
                    self._mem_summarization_latency_s     += _sum_lat
                elif _primitive == "tool_result_clear":
                    # Stubs out bash output bodies oldest-first; falls back to
                    # truncate() if clearing alone is insufficient.
                    self.messages, _saved, _trc_fallback = _mem.tool_result_clear(self.messages, _target)
                    if _trc_fallback:
                        self._mem_trc_fallback_events += 1
                elif _primitive == "scored_tool_result_clear":
                    # Ranked clearing: stubs out bash output bodies lowest-score first.
                    # Score = type_weight × size + citation_boost (ACT-R inspired).
                    # Falls back to truncate() if scored clearing is insufficient.
                    self.messages, _saved, _trc_fallback = _mem.scored_tool_result_clear(self.messages, _target)
                    if _trc_fallback:
                        self._mem_trc_fallback_events += 1
                else:  # truncation
                    # Drop oldest messages from messages[2:] until size <= target.
                    self.messages, _saved = _mem.truncate(self.messages, _target)

                # Record event metadata for the token log.
                _after = _mem.count_tokens(self.messages)
                if _current > 0:
                    self._mem_compression_ratios.append(_after / _current)
                self._mem_compression_events += 1
                self._mem_tokens_saved       += _saved
                self._mem_compression_event_steps.append(self.n_calls)
                self._mem_context_tokens_at_compression.append(_current)
                self._mem_context_tokens_after_compression.append(_after)
                # No reset of _mem_prompt_tokens needed: the trigger now checks
                # current context size directly, which is already small after
                # compression.  It will not fire again until history grows back.
        # ────────────────────────────────────────────────────────────────────

        self.n_calls += 1
        _t0      = time.time()
        message  = self.model.query(self.messages)
        _latency = time.time() - _t0

        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)

        # ── Accumulate token usage and write log ─────────────────────────────
        _extra = message.get("extra", {})
        _resp  = _extra.get("response", {})
        _usage = _resp.get("usage", {}) if isinstance(_resp, dict) else {}
        _step_pt = _usage.get("prompt_tokens", 0) or 0
        _step_ct = _usage.get("completion_tokens", 0) or 0
        self._mem_prompt_tokens     += _step_pt
        self._mem_completion_tokens += _step_ct
        self._mem_total_latency     += _latency
        self._mem_call_latencies.append(_latency)
        self._mem_step_prompt_tokens.append(_step_pt)
        self._mem_step_completion_tokens.append(_step_ct)
        if _primitive and _budget > 0:
            _mem.write_token_log(self)
        # ────────────────────────────────────────────────────────────────────

        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
