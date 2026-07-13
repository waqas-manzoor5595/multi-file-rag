"""
dashboard_utils.py

Small in-memory CRUD helpers for the dashboard's Timetable and To-Do List
widgets. Everything lives in st.session_state, so it resets on refresh /
app restart — there's no database backing this yet.
"""

import uuid


def new_id():
    return uuid.uuid4().hex[:8]


# --- Timetable -------------------------------------------------------------

def add_timetable_entry(session_state, time_str, activity):
    if not activity.strip():
        return
    session_state.timetable.append(
        {"id": new_id(), "time": time_str, "activity": activity.strip()}
    )
    session_state.timetable.sort(key=lambda e: e["time"])


def remove_timetable_entry(session_state, entry_id):
    session_state.timetable = [
        e for e in session_state.timetable if e["id"] != entry_id
    ]


# --- To-Do list --------------------------------------------------------------

def add_todo(session_state, text):
    if not text.strip():
        return
    session_state.todos.append({"id": new_id(), "text": text.strip(), "done": False})


def toggle_todo(session_state, todo_id, done):
    for t in session_state.todos:
        if t["id"] == todo_id:
            t["done"] = done
            break


def remove_todo(session_state, todo_id):
    session_state.todos = [t for t in session_state.todos if t["id"] != todo_id]
