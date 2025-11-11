# =====================================================
# check1.py — Voice Agent with Filler & Interrupt Logging
# =====================================================

import logging
import os
import re
import time
import datetime
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


# =========================================================================
# 1️⃣ LOGGING CONFIGURATION
# =========================================================================
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("agent_filler_logs.log", mode="a")

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)


# =========================================================================
# 2️⃣ CONFIGURATION & UTILITIES
# =========================================================================
load_dotenv(".env.local")

DEFAULT_FILLERS = (
    "uh, um, umm, hmm, haan, mhm, hmmmmm, eh, er, ah, uh-huh, "
    "you know, sort of, kind of, i guess, yeah sure, like, right, basically, literally, "
    "haina, matlab, accha, acha, arre, arey, waise, to kya, ha, ji, theek hai, "
    "hmm yeah, okay, ok, okk, okey, okeyy, ya, yaa, yup"
)
FILLER_WORDS_STR = os.environ.get("IGNORED_WORDS", DEFAULT_FILLERS).lower()

DEFAULT_STOPS = (
    "stop, wait, hold on, pause, one sec, one second, cancel, "
    "hang on, listen, repeat, start over, quiet, mute, enough, bas, ruk, ruko, band karo"
)
STOP_WORDS_STR = os.environ.get("INTERRUPT_WORDS", DEFAULT_STOPS).lower()

ASR_MIN_CONFIDENCE = float(os.environ.get("ASR_MIN_CONFIDENCE", "0.60"))
INTERRUPT_DURATION_S = float(os.environ.get("INTERRUPT_DURATION_S", "5.0"))
MIN_WORDS_FOR_INTERRUPT = int(os.environ.get("MIN_WORDS_FOR_INTERRUPT", "3"))

FILLER_SET = set(w.strip() for w in FILLER_WORDS_STR.split(",") if w.strip())
STOP_PHRASES = [p.strip() for p in STOP_WORDS_STR.split(",") if p.strip()]
STOP_TOKENS = {p for p in STOP_PHRASES if " " not in p}


# --- Utilities ---
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def tokenize(text: str):
    return clean_text(text).split()


def is_pure_filler(text: str) -> bool:
    """True if all tokens are fillers (or text empty)."""
    if not text or not text.strip():
        return True
    words = tokenize(text)
    return bool(words) and all(word in FILLER_SET for word in words)


def contains_interrupt_word(text: str) -> bool:
    """True if text contains any configured stop/interrupt signal (whole words)."""
    if not text or not text.strip():
        return False
    low = text.lower()

    # Phrase-level (handles "hold on", "one second")
    for phrase in STOP_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, low):
            return True

    toks = set(tokenize(low))
    return any(tok in STOP_TOKENS for tok in toks)


# =========================================================================
# 3️⃣ ASSISTANT CLASS
# =========================================================================
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful, friendly, and conversational voice AI assistant. "
                "The user interacts with you via voice, so keep responses natural, concise, and free from complex symbols."
            ),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# =========================================================================
# 4️⃣ ENTRYPOINT & EVENT HANDLERS
# =========================================================================
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"🚀 Starting agent in room: {ctx.room.name}")

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        min_interruption_words=1,
        false_interruption_timeout=0.4,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📊 Usage Summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # =====================================================
    # 🔹 MONITOR ASSISTANT SPEECH (TTS STATE)
    # =====================================================
    assistant_speaking = False
    speak_start_time = None

    @session.on("tts_started")
    def _on_tts_start(_):
        nonlocal assistant_speaking, speak_start_time
        assistant_speaking = True
        speak_start_time = datetime.datetime.now()
        logger.info("🗣️ Assistant started speaking — ASR monitoring active.")

    @session.on("tts_stopped")
    def _on_tts_end(_):
        nonlocal assistant_speaking
        assistant_speaking = False
        logger.info("🔇 Assistant finished speaking — ASR monitoring paused.")

    # =====================================================
    # 🔹 USER SPEECH PROCESSING
    # =====================================================
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        if not ev.is_final:
            return

        text = (ev.transcript or "").strip()
        if not text:
            return

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        conf = getattr(ev, "confidence", None)
        clean = clean_text(text)
        word_count = len(clean.split())
        duration = (
            (datetime.datetime.now() - speak_start_time).total_seconds()
            if speak_start_time
            else 0
        )

        # While assistant speaking
        if assistant_speaking:
            # Filler words
            if is_pure_filler(clean):
                logger.info(f"[{ts}] 💤 ASR filler ignored during TTS: '{text}' (elapsed={duration:.2f}s)")
                return

            # Stopwords (immediate interrupt)
            if contains_interrupt_word(clean):
                logger.info(f"[{ts}] ⛔ ASR stopword interrupt during TTS: '{text}' (elapsed={duration:.2f}s)")
                session.interrupt()
                return

            # Real user speech (time/length based)
            if word_count >= MIN_WORDS_FOR_INTERRUPT or duration > INTERRUPT_DURATION_S:
                logger.info(
                    f"[{ts}] ⚡ REAL interrupt (while speaking): '{text}' "
                    f"(words={word_count}, elapsed={duration:.2f}s, conf={conf})"
                )
                session.interrupt()
                return

            # Short bursts (optional interrupt)
            logger.info(f"[{ts}] ⚡ SHORT burst ignored during TTS: '{text}' (words={word_count})")
            return

        # When agent is quiet
        if is_pure_filler(clean):
            logger.info(f"[{ts}] 💤 FILLER (agent quiet): '{text}'")
            return

        if contains_interrupt_word(clean):
            logger.info(f"[{ts}] ⛔ STOPWORD (agent quiet): '{text}'")
            session.interrupt()
            return

        logger.info(f"[{ts}] 📝 REGISTER normal input: '{text}'")

    # =====================================================
    # 🔹 START SESSION
    # =====================================================

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()
    logger.info("✅ Agent session initialized successfully and listening...")


# =========================================================================
# 5️⃣ MAIN ENTRYPOINT
# =========================================================================
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))