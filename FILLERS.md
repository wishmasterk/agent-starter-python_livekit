# 🎙️ Voice Agent — Previous vs Current Flow

> Compact markdown overview comparing the base LiveKit agent vs. the upgraded version with filler, stopword, ASR, and time-based interruption logic.

---

## 🚀 1. What Changed
| Feature | Previous | Current |
|----------|-----------|----------|
| Filler Detection | ❌ None | ✅ Ignores filler-only speech during TTS |
| Stopword Handling | ❌ None | ✅ Interrupts immediately on key words (“stop”, “wait”, etc.) |
| ASR Confidence | ❌ Not used | ✅ Low-confidence ASR ignored while agent speaks |
| Time-Based Interrupt | ❌ None | ✅ Interrupts if speech lasts > 5s or ≥ 3 words |
| TTS Awareness | ❌ None | ✅ Tracks TTS state (`tts_started`, `tts_stopped`) |
| Logging | Minimal | ✅ Structured logs with icons & timestamps |

---

## 🧠 2. Previous Flow (Base Framework)

**Behavior**
- Simple pipeline: **STT → LLM → TTS**
- No check for filler, no interrupt control.
- Every final transcript sent directly to the LLM.

**Pseudo-flow**
```text
on user_input_transcribed(final):
    if text:
        forward to LLM
on tts_started → assistant_speaking = True
on tts_stopped → assistant_speaking = False

on user_input_transcribed(final):
    if stopword → ⛔ interrupt immediately

    if assistant_speaking:
        if conf < ASR_MIN_CONF → 🤫 ignore low-confidence
        elif is_pure_filler(text) → 💤 ignore filler (auto-resume)
        elif words ≥ MIN_WORDS or elapsed ≥ INTERRUPT_DURATION_S → ⚡ real interrupt
        else → ⚡ short burst ignored/logged
    else:
        if is_pure_filler → 💤 log filler
        elif stopword → ⛔ interrupt
        else → 📝 register normally