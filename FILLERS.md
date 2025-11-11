# 🎙️ Voice Agent — Context-Aware Filler & Interrupt Guard

Enhanced conversational AI built on LiveKit Agents SDK, featuring real-time filler suppression, controlled interruptions, and detailed logging through a modular ConversationGuard class.

---

## 🚀 1. What Changed

| Feature                   | Base Version    | This Version                                                        |
| ------------------------- | --------------- | ------------------------------------------------------------------- |
| **Filler Handling**       | ❌ None          | ✅ Ignores fillers (“uh”, “hmm”, “ok”) during agent speech           |
| **Stopword Detection**    | ❌ None          | ✅ Interrupts instantly on words like “stop”, “ruko”, “wait”         |
| **Context Awareness**     | ❌ None          | ✅ Distinguishes when agent is speaking vs. silent                   |
| **Time-based Interrupts** | ❌ None          | ✅ Ends TTS if user talks for > 5 s or ≥ 3 words                     |
| **Environment Config**    | Hardcoded       | ✅ Controlled through `.env` variables via `InterruptConfig`         |
| **Logging**               | Minimal         | ✅ Timestamped structured logs with reason + transcript              |
| **Architecture**          | Flat procedural | ✅ Modular, OOP: `ConversationGuard`, `InterruptConfig`, `Assistant` |
| **Extensibility**         | None            | ✅ Easily tunable for multilingual or noise-aware sessions           |

---

## 🧩 2. System Architecture

| Module              | Purpose                                                        |
| ------------------- | -------------------------------------------------------------- |
| `InterruptConfig`   | Loads filler & stopword lists, thresholds from `.env`          |
| `ConversationGuard` | Attaches to `AgentSession`; classifies and acts on user speech |
| `Assistant`         | Simple conversational LLM prompt wrapper                       |
| `prewarm()`         | Loads Silero VAD early for faster startup                      |
| `entrypoint()`      | Initializes agent session, attaches guard, handles metrics     |

---

## 🧠 3. Runtime Workflow

1️⃣  AgentSession initializes:
       → Loads STT, LLM, TTS, VAD
       → Binds ConversationGuard

2️⃣  ConversationGuard hooks into events:
       - tts_started / tts_stopped
       - user_input_transcribed

3️⃣  When agent speaks:
       - Pure filler → ignored
       - Stopword → immediate interrupt
       - ≥3 words or >5 s → real interrupt
       - Short bursts → ignored + logged

4️⃣  When agent silent:
       - Fillers → ignored
       - Stopword → interrupt
       - Else → forward to LLM

5️⃣  Every event logged with timestamp, confidence, and duration.

---

## ✅ 4. What Works (Tested)

| Scenario                                     | Expected                     | Result |
| -------------------------------------------- | ---------------------------- | ------ |
| Say “umm hmm ok” while agent speaks          | Agent continues              | 🟢     |
| Say “stop” or “wait” mid-TTS                 | Immediate halt               | 🟢     |
| Speak sentence (“tell me something”) mid-TTS | Agent interrupts and listens | 🟢     |
| Filler while silent                          | Ignored                      | 🟢     |
| Normal prompt while silent                   | Processed normally           | 🟢     |

---

## ⚠️ 5. Known Issues
| Issue                    | Description                                          | Severity                |
| ------------------------ | ---------------------------------------------------- | ----------------------- |
| ASR Delay                | Occasional 200–400 ms delay before detecting fillers | ⚙️ Medium               |
| Accent / filler variants | Missed if not in `IGNORED_WORDS` list                | 🔧 Low                  |
| Simultaneous overlap     | Long dual speech may cause STT drift                 | ⚙️ Medium               |

---

## 🧪 5. How to run
uv run src/agent.py console
