# ================================================
# updated agent.py with filler word and interruption control
# ================================================

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    UserInputTranscribedEvent,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ============================================================
# 1. CONFIG & TEXT HELPERS
# ============================================================

load_dotenv(".env.local")

@dataclass
class InterruptConfig:
    """
    Holds all configuration used by the interruption / filler logic.
    Values can be overridden via environment variables.
    """

    ignored_words: List[str]
    stop_phrases: List[str]
    min_words_for_interrupt: int
    max_tts_speaking_before_any_interrupt: float

    @classmethod
    def from_env(cls) -> "InterruptConfig":
        # Filler / hesitation words (Hindi + English flavored)
        default_fillers = (
            "uh, umm, ummm, hmm, hmm, haan, hn, haann, mhm, eh, er, ah, uhh, ok, okay, Okay, okayy, yeah, ya, yaa, acha, accha, arre,"
            "right, basically, literally, like, you know, so, well, actually, just"
        )
        fillers_raw = os.getenv("IGNORED_WORDS", default_fillers)
        ignored_words = [w.strip().lower() for w in fillers_raw.split(",") if w.strip()]

        # Stop / interruption phrases
        default_stops = (
            "stop, wait, hold on, pause, cancel, one second, hang on, listen,"
            "start over, repeat, bas, ruk, ruko, band karo"
        )
        stops_raw = os.getenv("INTERRUPT_WORDS", default_stops)
        stop_phrases = [p.strip().lower() for p in stops_raw.split(",") if p.strip()]

        min_words = int(os.getenv("MIN_WORDS_FOR_INTERRUPT", "3"))
        max_tts_age = float(os.getenv("INTERRUPT_DURATION_S", "4.0"))

        return cls(
            ignored_words=ignored_words,
            stop_phrases=stop_phrases,
            min_words_for_interrupt=min_words,
            max_tts_speaking_before_any_interrupt=max_tts_age,
        )


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for simple word-level checks."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _split_words(text: str) -> list[str]:
    norm = _normalize(text)
    return norm.split() if norm else []


# ============================================================
# 2. LOGGING SETUP
# ============================================================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# Only add handlers once to avoid duplicate logs on reload
if not logger.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Log to file with daily-ish appends
    logfile = logging.FileHandler("voice_agent_interrupts.log", mode="a", encoding="utf-8")
    logfile.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

# ============================================================
# 3. VOICE BEHAVIOR CONTROLLER
# ============================================================

class ConversationGuard:
    """
    ConversationGuard attaches to a LiveKit AgentSession and listens for:
      - TTS start/stop events (to know when the assistant is speaking)
      - Final user transcripts (user_input_transcribed)

    It then classifies overlapping user speech into:
      - pure filler -> ignored
      - explicit stop / control phrases -> interrupt
      - meaningful speech -> interrupt

    When the assistant is *not* speaking, it mostly lets the normal flow happen,
    only ignoring filler lines.
    """

    def __init__(self, session: AgentSession, config: InterruptConfig | None = None) -> None:
        self._session = session
        self._config = config or InterruptConfig.from_env()

        # Convert config lists to faster lookup structures
        self._filler_set = set(self._config.ignored_words)
        self._stop_phrases = list(self._config.stop_phrases)

        # Runtime state
        self._assistant_tts_active: bool = False
        self._tts_started_at: float | None = None

        # Register event listeners on the session
        self._register_event_hooks()

        logger.info(
            "ConversationGuard initialized | fillers=%d | stop_phrases=%d",
            len(self._filler_set),
            len(self._stop_phrases),
        )

    # --- Event hook registration ---

    def _register_event_hooks(self) -> None:
        @self._session.on("tts_started")
        def _on_tts_started(_ev) -> None:
            self._assistant_tts_active = True
            self._tts_started_at = time.time()
            logger.info("ConversationGuard: TTS started (assistant is speaking)")

        @self._session.on("tts_stopped")
        def _on_tts_stopped(_ev) -> None:
            self._assistant_tts_active = False
            self._tts_started_at = None
            logger.info("ConversationGuard: TTS stopped (assistant is silent)")

        @self._session.on("user_input_transcribed")
        def _on_user_input(ev: UserInputTranscribedEvent) -> None:
            # We only care about final ASR output to avoid jittery partials
            if not getattr(ev, "is_final", True):
                return

            raw_text = (ev.transcript or "").strip()
            if not raw_text:
                return

            self._process_user_text(raw_text)

    # --- Classification helpers ---

    def _is_pure_filler(self, text: str) -> bool:
        words = _split_words(text)
        return bool(words) and all(w in self._filler_set for w in words)

    def _contains_stop_phrase(self, text: str) -> bool:
        norm = text.lower()
        for phrase in self._stop_phrases:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, norm):
                return True
        return False

    def _elapsed_tts_time(self) -> float:
        if self._tts_started_at is None:
            return 0.0
        return time.time() - self._tts_started_at

    # --- Core decision engine ---

    def _process_user_text(self, text: str) -> None:
        """Route user speech depending on whether TTS is active."""
        if self._assistant_tts_active:
            self._handle_overlap_with_tts(text)
        else:
            self._handle_while_silent(text)

    def _handle_overlap_with_tts(self, text: str) -> None:
        """
        Logic when user speech happens while the assistant is talking.
        We decide whether:
          - to ignore (pure filler / very short),
          - or to actively interrupt the assistant.
        """
        words = _split_words(text)
        word_count = len(words)
        elapsed = self._elapsed_tts_time()

        # Pure filler -> ignore & optionally clean up the current user turn
        if self._is_pure_filler(text):
            logger.info(
                "ConversationGuard: filler DURING TTS ignored | text=%r | elapsed=%.2fs",
                text,
                elapsed,
            )
            try:
                # clear any accumulated user turn so this doesn't feed the LLM
                self._session.clear_user_turn()
            except RuntimeError:
                pass
            return

        # Explicit stop / correction phrase -> immediate interruption
        if self._contains_stop_phrase(text):
            logger.info(
                "ConversationGuard: STOP phrase DURING TTS -> interrupt | text=%r | elapsed=%.2fs",
                text,
                elapsed,
            )
            self._session.interrupt(force=True)
            return

        # Long enough or late enough -> treat as a real interruption
        if (
            word_count >= self._config.min_words_for_interrupt
            or elapsed >= self._config.max_tts_speaking_before_any_interrupt
        ):
            logger.info(
                (
                    "ConversationGuard: REAL INTERRUPTION DURING TTS -> interrupt | "
                    "text=%r | words=%d | elapsed=%.2fs"
                ),
                text,
                word_count,
                elapsed,
            )
            self._session.interrupt(force=True)
            return

        # Short non-filler utterance, early in assistant speech:
        logger.info(
            "ConversationGuard: short overlap DURING TTS ignored | text=%r | words=%d | elapsed=%.2fs",
            text,
            word_count,
            elapsed,
        )

    def _handle_while_silent(self, text: str) -> None:
        """
        Logic when the assistant is NOT speaking.
        Mostly let the normal agent flow handle it, but:
          - ignore standalone filler lines
          - optionally react to stop phrases (e.g. queued speech)
        """
        if self._is_pure_filler(text):
            logger.info("ConversationGuard: filler while silent ignored | text=%r", text)
            return

        if self._contains_stop_phrase(text):
            logger.info("ConversationGuard: stop phrase while silent | text=%r", text)
            # In most cases nothing to interrupt, but we try for consistency
            try:
                self._session.interrupt(force=True)
            except RuntimeError:
                pass
            return

        # Normal user input, nothing special from our side
        logger.info("ConversationGuard: normal user input | text=%r", text)


# ============================================================
# 4. ASSISTANT CLASS & PREWARM
# ============================================================

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a natural, friendly voice AI assistant. "
                "Users talk to you by voice, so reply in clear, simple sentences, "
                "without emojis or fancy formatting. Be concise but warm."
            ),
        )


def prewarm(proc: JobProcess) -> None:
    """
    Load any heavy models *once* per worker process so that
    subsequent jobs start faster.
    """
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm: Silero VAD loaded and cached in proc.userdata")


# ============================================================
# 5. JOB ENTRYPOINT
# ============================================================

async def entrypoint(ctx: JobContext) -> None:
    # Add room name into all log records from this job, if desired
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting new voice session for room=%s", ctx.room.name)

    # Assemble the voice pipeline: STT + LLM + TTS + turn detection + VAD
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        # you can tweak these AgentSession-level params if needed
        min_interruption_words=1,
        false_interruption_timeout=0.4,
    )

    # Attach our custom conversation guard to the session
    guard_config = InterruptConfig.from_env()
    ConversationGuard(session, config=guard_config)

    # Usage metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def _log_usage_summary() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(_log_usage_summary)

    # Start agent session and connect to the room
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()
    logger.info("Agent session fully started and connected for room=%s", ctx.room.name)


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))