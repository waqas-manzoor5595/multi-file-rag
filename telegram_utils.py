"""
telegram_utils.py

Small helper layer around the raw Telegram Bot HTTP API. No extra SDK needed —
just `requests`. Covers:

- sending a message to a chat
- polling for new incoming messages (long-polling via getUpdates)
- classifying a message as "important" or "inbox" based on keywords
"""

import requests

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _url(token, method):
    return TELEGRAM_API_BASE.format(token=token, method=method)


def send_telegram_message(token, chat_id, text, parse_mode=None):
    """Send a text message to a chat via the bot.

    Returns (success: bool, info: str) — info is the message id on success,
    or an error string on failure.
    """
    if not token or not chat_id:
        return False, "Missing bot token or chat ID."
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = requests.post(_url(token, "sendMessage"), json=payload, timeout=10)
        data = resp.json()
    except Exception as e:
        return False, f"Request failed: {e}"

    if not data.get("ok"):
        return False, data.get("description", "Unknown Telegram API error.")
    return True, str(data["result"]["message_id"])


def get_telegram_updates(token, offset=None, timeout=0):
    """Fetch new updates (incoming messages) since `offset`.

    `offset` should be the last seen update_id + 1 so Telegram doesn't
    resend messages you've already processed. Returns (updates: list, error: str|None).
    """
    if not token:
        return [], "Missing bot token."
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(_url(token, "getUpdates"), params=params, timeout=timeout + 10)
        data = resp.json()
    except Exception as e:
        return [], f"Request failed: {e}"

    if not data.get("ok"):
        return [], data.get("description", "Unknown Telegram API error.")
    return data.get("result", []), None


def extract_messages(updates):
    """Pull out plain-text message info from a list of raw Telegram updates.

    Returns a list of dicts: {update_id, message_id, chat_id, from, text, date}
    Non-text updates (photos, stickers, etc.) are skipped.
    """
    messages = []
    for upd in updates:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg or "text" not in msg:
            continue
        sender = msg.get("from", {})
        name = sender.get("username") or sender.get("first_name") or "unknown"
        messages.append(
            {
                "update_id": upd["update_id"],
                "message_id": msg.get("message_id"),
                "chat_id": msg.get("chat", {}).get("id"),
                "from": name,
                "text": msg["text"],
                "date": msg.get("date"),
            }
        )
    return messages


def classify_message(text, keywords):
    """Return 'important' if any keyword appears in the message (case-insensitive),
    otherwise 'inbox'. `keywords` is a list of strings."""
    if not keywords:
        return "inbox"
    lowered = text.lower()
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and kw in lowered:
            return "important"
    return "inbox"
