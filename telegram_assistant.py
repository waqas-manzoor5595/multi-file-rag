"""
telegram_assistant.py

Turns incoming Telegram messages into commands the dashboard can answer:

  /ask <question>   -> answered using the RAG document index PLUS current
                        app state (todos, timetable, important messages)
  ?<question>       -> shortcut for /ask
  /todos            -> current to-do list
  /timetable        -> today's schedule
  /important        -> recent important messages
  /summary          -> condensed digest of all of the above
  /help             -> list of commands

Anything that isn't a recognized command falls through to the existing
inbox/important keyword classification, unchanged.
"""

COMMAND_WORDS = ("/ask", "/todos", "/timetable", "/important", "/summary", "/help")


def is_command(text):
    stripped = text.strip()
    if stripped.startswith("?"):
        return True
    first_word = stripped.split(maxsplit=1)[0].lower() if stripped else ""
    return first_word in COMMAND_WORDS


def parse_command(text):
    """Returns (command, argument)."""
    stripped = text.strip()
    if stripped.startswith("?"):
        return "ask", stripped[1:].strip()

    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/todos", "/timetable", "/important", "/summary", "/help"):
        return cmd[1:], arg
    return "ask", arg  # "/ask ..." and any unrecognized slash command


def format_todos(todos):
    if not todos:
        return "No to-dos right now."
    lines = [f"{'✅' if t['done'] else '⬜'} {t['text']}" for t in todos]
    return "\n".join(lines)


def format_timetable(timetable):
    if not timetable:
        return "Nothing scheduled."
    return "\n".join(f"{e['time']} — {e['activity']}" for e in timetable)


def format_important(messages, limit=5):
    if not messages:
        return "No important messages flagged."
    recent = messages[-limit:]
    return "\n".join(f"• {m['text']}" for m in reversed(recent))


HELP_TEXT = (
    "Here's what I can do:\n\n"
    "/ask <question> — ask about your documents, todos, timetable, or important messages\n"
    "(or just start your message with ? as a shortcut)\n"
    "/todos — show your current to-do list\n"
    "/timetable — show today's schedule\n"
    "/important — show recent important messages\n"
    "/summary — a quick digest of everything above\n"
    "/help — this message"
)


def build_state_context(session_state):
    """Plain-text snapshot of the app's current state, fed to the LLM
    alongside document context so it can answer questions about todos,
    timetable, and important messages too — not just documents."""
    todos = session_state.get("todos", [])
    timetable = session_state.get("timetable", [])
    important = session_state.get("telegram_important", [])

    return (
        "TO-DO LIST:\n" + format_todos(todos) + "\n\n"
        "TIMETABLE:\n" + format_timetable(timetable) + "\n\n"
        "RECENT IMPORTANT MESSAGES:\n" + format_important(important)
    )


def detect_dashboard_intent(question):
    """Best-effort detection of whether a free-form /ask question is really
    just asking for dashboard state (todos/timetable/important/summary), so
    it can be answered deterministically from session state instead of
    relying on the LLM to correctly prioritize which context to use.
    Returns one of 'todos', 'timetable', 'important', 'summary', or None
    (None means: treat it as a real document/general question, and route
    it through the full RAG + LLM pipeline as usual).

    Important: this is a fast-path for *obvious* dashboard questions only.
    Anything that references documents/files explicitly always falls through
    to the LLM/RAG path, even if it also contains a dashboard-ish word —
    e.g. "is there anything important in the contract" should still search
    documents, not just dump the Important Messages list.
    """
    q = question.lower().strip()

    # If the question clearly references documents/files, never shortcut —
    # let the full RAG pipeline handle it, since that's the actual answer source.
    document_hint_words = (
        "document", "file", "pdf", "doc ", "report", "contract", "text",
        "uploaded", "mentioned", "according to", "says about", "paper",
        "article", "content", "attachment", "clause", "section",
    )
    if any(w in q for w in document_hint_words):
        return None

    summary_words = ("summary", "digest", "overview", "what's going on", "whats going on", "anything going on")
    important_words = ("important", "urgent", "priority", "flagged")
    todo_words = ("to-do", "to do", "todo", "task")
    timetable_words = ("timetable", "schedule", "calendar", "agenda")

    if any(w in q for w in summary_words):
        return "summary"
    if any(w in q for w in important_words):
        return "important"
    if any(w in q for w in todo_words):
        return "todos"
    if any(w in q for w in timetable_words):
        return "timetable"
    return None


def build_digest(session_state):
    """A condensed, Telegram-friendly digest of everything for /summary
    and for the manual 'Send Digest' notification button."""
    todos = session_state.get("todos", [])
    timetable = session_state.get("timetable", [])
    important = session_state.get("telegram_important", [])

    pending = [t for t in todos if not t["done"]]
    return (
        "📋 Dashboard Summary\n\n"
        f"To-dos: {len(pending)} pending / {len(todos)} total\n"
        + format_todos(todos[:5])
        + ("\n...\n" if len(todos) > 5 else "\n")
        + f"\n🗓️ Timetable:\n"
        + format_timetable(timetable)
        + f"\n\n⭐ Important ({len(important)} total):\n"
        + format_important(important, limit=3)
    )
