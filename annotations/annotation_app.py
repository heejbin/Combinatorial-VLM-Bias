#!/usr/bin/env python3
"""Four-hour fully-crossed annotation server.

Four annotators see the same young-adult images in the same batch order.
Block A is blind. Location, and Block B on 100 images, are verification.
Decoy flags and the prompt stay on the server.

The previous prompt-matching app is ``annotation_app_legacy.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import lead_examples
import protocol_4h as proto

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

_PROTOCOL = None
_WRITE = threading.Lock()


def protocol() -> dict:
    global _PROTOCOL
    if _PROTOCOL is None:
        if not proto.ASSIGNMENT_PATH.exists():
            proto.save_assignment(proto.build_protocol())
        _PROTOCOL = proto.load_assignment()
    return _PROTOCOL


def items_by_key() -> dict[str, dict]:
    data = protocol()
    found = {item["item_key"]: item for item in data["main"] + data["calibration"]}
    for row in lead_examples.practice_rows():
        item = lead_examples.practice_item(row)
        found[item["item_key"]] = item
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_json(path: Path, payload) -> None:
    """Write atomically and keep the previous file as name.bak."""
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _read_json(path: Path, default):
    data = _read_json_resilient(path)
    return default if data is None else data


def _read_json_resilient(path: Path):
    """Prefer the main file, then its .bak. A torn write does not become the only copy."""
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.exists():
            continue
        try:
            with candidate.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if candidate != path:
            try:
                _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
            except OSError:
                pass
        return data
    return None


def journal_paths(name: str) -> list[Path]:
    return [
        proto.RESPONSE_DIR / f"{name}.journal.jsonl",
        proto.PROTOCOL_DIR / "backup" / f"{name}.journal.jsonl",
    ]


def append_journal(name: str, event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    wrote = False
    for path in journal_paths(name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        wrote = True
    if not wrote:
        raise OSError("journal path missing")


def replay_journal(name: str) -> dict:
    state: dict = {}
    existing = [path for path in journal_paths(name) if path.exists()]
    if not existing:
        return state
    # A torn primary file is shorter. The other copy still has the last line.
    path = max(existing, key=lambda candidate: candidate.stat().st_size)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "answer":
                continue
            key = event.get("item_key")
            if not key:
                continue
            prev = state.get(key) or {}
            if prev.get("timestamp", "") > event.get("timestamp", ""):
                continue
            incoming = event.get("answers") or {}
            if prev.get("submitted") and not event.get("submitted"):
                answers = dict(prev.get("answers") or {})
                answers.update(incoming)
            else:
                answers = incoming or dict(prev.get("answers") or {})
            elapsed = prev.get("elapsed_ms")
            if elapsed is None:
                elapsed = event.get("elapsed_ms")
            draft = event.get("draft_elapsed_ms")
            if prev.get("draft_elapsed_ms") is not None:
                draft = max(int(draft or 0), int(prev["draft_elapsed_ms"]))
            memo = event.get("memo")
            if not isinstance(memo, str):
                memo = prev.get("memo") or ""
            state[key] = {
                "item_key": key,
                "answers": answers,
                "memo": memo,
                "elapsed_ms": elapsed,
                "draft_elapsed_ms": draft,
                "submitted": bool(event.get("submitted") or prev.get("submitted")),
                "revised": bool(event.get("revised") or prev.get("revised")),
                "timestamp": event.get("timestamp"),
            }
    return state


def _merge_records(disk: dict, journal: dict) -> dict:
    merged = dict(disk)
    for key, rec in journal.items():
        prev = merged.get(key)
        if prev is None or (rec.get("timestamp") or "") >= (prev.get("timestamp") or ""):
            rec = dict(rec)
            if prev and prev.get("submitted") and not rec.get("submitted"):
                answers = dict(prev.get("answers") or {})
                answers.update(rec.get("answers") or {})
                rec["answers"] = answers
                rec["submitted"] = True
            if prev and prev.get("elapsed_ms") is not None:
                rec["elapsed_ms"] = prev["elapsed_ms"]
            if prev and prev.get("draft_elapsed_ms") is not None:
                rec["draft_elapsed_ms"] = max(
                    int(rec.get("draft_elapsed_ms") or 0),
                    int(prev["draft_elapsed_ms"]),
                )
            if "memo" not in rec:
                rec["memo"] = prev.get("memo") or "" if prev else ""
            merged[key] = rec
    return merged


def snapshot_responses(name: str, responses: dict, tag: str) -> None:
    for folder in (
        proto.RESPONSE_DIR / "milestones" / name,
        proto.PROTOCOL_DIR / "backup" / "milestones" / name,
    ):
        _atomic_write(
            folder / f"{tag}.json",
            json.dumps(responses, ensure_ascii=False, indent=2),
        )


_AUTOSAVE_COUNT: dict[str, int] = {}


def store_responses(name: str, responses: dict, *, snapshot: str | None) -> None:
    _write_json(response_path(name), responses)
    _write_json(proto.PROTOCOL_DIR / "backup" / f"{name}.json", responses)
    if snapshot:
        snapshot_responses(name, responses, snapshot)


def load_responses(name: str) -> dict:
    disk = _read_json_resilient(response_path(name)) or {}
    if not isinstance(disk, dict):
        disk = {}
    journal = replay_journal(name)
    merged = _merge_records(disk, journal)
    if merged != disk:
        try:
            store_responses(name, merged, snapshot=None)
        except OSError:
            pass
    return merged


def progress_path(name: str) -> Path:
    return proto.PROTOCOL_DIR / f"progress_{name}.json"


def response_path(name: str) -> Path:
    return proto.RESPONSE_DIR / f"{name}.json"


def empty_progress(name: str) -> dict:
    return {
        "annotator": name,
        "guidelines_done": False,
        "cal_index": 0,
        "batch": 1,
        "index": 0,
        "on_break": False,
        "done": False,
        "practice_index": 0,
        "practice_reveal": None,
    }


def cursor_from_answers(responses: dict) -> dict:
    """Put the annotator back on the first image they have not submitted."""
    data = protocol()
    n_batches = active_batches(load_runtime())
    progress = empty_progress("")
    progress["guidelines_done"] = True
    rows = lead_examples.practice_rows()
    study_started = any(
        (responses.get(item["item_key"]) or {}).get("submitted")
        for item in data["calibration"] + data["main"]
    )
    if rows and not study_started:
        progress["cal_index"] = proto.N_CALIBRATION
        for index, row in enumerate(rows):
            if not (responses.get(row["id"]) or {}).get("submitted"):
                progress["practice_index"] = index
                progress["cal_index"] = 0
                return progress
        progress["practice_index"] = len(rows)
    else:
        progress["cal_index"] = proto.N_CALIBRATION
        for item in data["calibration"]:
            if not (responses.get(item["item_key"]) or {}).get("submitted"):
                progress["cal_index"] = item["order"]
                return progress
    for batch in range(1, n_batches + 1):
        for item in proto.batch_items(data, batch):
            if not (responses.get(item["item_key"]) or {}).get("submitted"):
                progress["batch"] = batch
                progress["index"] = item["order"]
                return progress
    progress["batch"] = n_batches
    progress["index"] = proto.N_BATCH
    progress["done"] = True
    return progress


def load_progress(name: str) -> dict:
    path = progress_path(name)
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and "cal_index" in data:
            data.setdefault("annotator", name)
            data.setdefault("practice_index", 0)
            data.setdefault("practice_reveal", None)
            return data
    # The cursor file is missing or torn. Saved answers still say where to resume.
    # A stale progress.bak is not used here, so one lost save cannot rewind the session.
    responses = load_responses(name)
    if responses:
        data = cursor_from_answers(responses)
        data["annotator"] = name
        try:
            _write_json(path, data)
        except OSError:
            pass
        return data
    bak = _read_json_resilient(path.with_name(path.name + ".bak"))
    if isinstance(bak, dict) and "cal_index" in bak:
        return bak
    return empty_progress(name)


def load_runtime() -> dict:
    return _read_json(proto.RUNTIME_PATH, {
        "tail_medians_ms": {},
        "n_batches": proto.N_BATCHES,
        "lock_reason": None,
    })


def save_runtime(runtime: dict) -> None:
    _write_json(proto.RUNTIME_PATH, runtime)


def annotator_name() -> str:
    payload = request.get_json(silent=True) or {}
    name = payload.get("annotator") or request.args.get("annotator") or ""
    if name not in proto.ANNOTATORS:
        payload_error = jsonify({"error": "annotator를 먼저 고르세요"})
        payload_error.status_code = 400
        raise AnnotatorRequired(payload_error)
    return name


class AnnotatorRequired(Exception):
    def __init__(self, response):
        self.response = response


@app.errorhandler(AnnotatorRequired)
def _annotator_required(exc: AnnotatorRequired):
    return exc.response


def active_batches(runtime: dict) -> int:
    n = int(runtime.get("n_batches") or proto.N_BATCHES)
    return 2 if n <= 2 else 3


def refresh_lock(responses: dict, name: str) -> dict:
    runtime = load_runtime()
    if runtime.get("lead_batch_lock"):
        return runtime
    times = []
    for item in protocol()["calibration"]:
        saved = responses.get(item["item_key"])
        if not saved or saved.get("elapsed_ms") is None:
            return runtime
        times.append(saved["elapsed_ms"])
    medians = dict(runtime.get("tail_medians_ms") or {})
    medians[name] = proto.median(times[-proto.CALIBRATION_TAIL:])
    decision = proto.lock_from_calibration(medians)
    if runtime.get("n_batches") == 2 and decision["n_batches"] == 2:
        decision["lock_reason"] = runtime.get("lock_reason") or decision["lock_reason"]
    runtime["tail_medians_ms"] = medians
    runtime["n_batches"] = decision["n_batches"]
    runtime["lock_reason"] = decision["lock_reason"]
    save_runtime(runtime)
    return runtime


def _practice_view(progress: dict, n_batches: int) -> dict | None:
    rows = lead_examples.practice_rows()
    if not rows:
        return None
    idx = int(progress.get("practice_index") or 0)
    reveal = progress.get("practice_reveal")
    started_study = bool(
        (progress.get("cal_index") or 0) > 0
        or (progress.get("index") or 0) > 0
        or (progress.get("batch") or 1) > 1
        or progress.get("done")
        or progress.get("on_break")
    )
    if started_study and not reveal and idx == 0:
        return None
    if reveal:
        row = next((item for item in rows if item["id"] == reveal), None)
        if row is not None:
            position = min(idx, len(rows) - 1) + 1
            return {
                "phase": "practice_feedback",
                "item": lead_examples.practice_item(row),
                "row": row,
                "position": position,
                "length": len(rows),
                "n_batches": n_batches,
            }
    if idx < len(rows):
        return {
            "phase": "practice",
            "item": lead_examples.practice_item(rows[idx]),
            "row": rows[idx],
            "position": idx + 1,
            "length": len(rows),
            "n_batches": n_batches,
        }
    return None


def current_view(progress: dict, runtime: dict) -> dict:
    n_batches = active_batches(runtime)
    if not progress["guidelines_done"]:
        return {"phase": "guidelines", "n_batches": n_batches}
    practice = _practice_view(progress, n_batches)
    if practice is not None:
        return practice
    if progress["cal_index"] < proto.N_CALIBRATION:
        item = protocol()["calibration"][progress["cal_index"]]
        return {
            "phase": "calibration",
            "item": item,
            "position": progress["cal_index"] + 1,
            "length": proto.N_CALIBRATION,
            "n_batches": n_batches,
        }
    if progress["on_break"]:
        return {
            "phase": "break",
            "after_batch": progress["batch"],
            "next_batch": progress["batch"] + 1,
            "n_batches": n_batches,
        }
    if progress["done"] or progress["batch"] > n_batches or (
        progress["index"] >= proto.N_BATCH and progress["batch"] >= n_batches
    ):
        return {"phase": "done", "n_batches": n_batches}
    rows = proto.batch_items(protocol(), progress["batch"])
    return {
        "phase": "batch",
        "item": rows[progress["index"]],
        "position": progress["index"] + 1,
        "length": proto.N_BATCH,
        "n_batches": n_batches,
    }


def advance(progress: dict, item: dict, runtime: dict) -> None:
    if item["batch"] is None:
        progress["cal_index"] = max(progress["cal_index"], item["order"] + 1)
        progress["done"] = False
        return
    progress["batch"] = item["batch"]
    progress["index"] = item["order"] + 1
    progress["on_break"] = False
    progress["done"] = False
    if progress["index"] >= proto.N_BATCH:
        if progress["batch"] < active_batches(runtime):
            progress["on_break"] = True
        else:
            progress["done"] = True


def _expected(item: dict) -> dict[str, set[str]]:
    return {q["id"]: {opt["id"] for opt in q["options"]} for q in item["questions"]}


def check_partial(item: dict, answers: dict) -> str | None:
    if not isinstance(answers, dict) or not answers:
        return "answers가 없습니다"
    expected = _expected(item)
    for qid, value in answers.items():
        if qid not in expected or value not in expected[qid]:
            return f"고를 수 없는 답입니다: {qid}"
    return None


def check_answers(item: dict, answers: dict) -> str | None:
    error = check_partial(item, answers)
    if error:
        return error
    if set(answers) != set(_expected(item)):
        return "이 이미지의 문항과 답이 맞지 않습니다"
    return None


def _memo(payload: dict) -> str | None:
    if "memo" not in payload or payload.get("memo") is None:
        return None
    memo = payload.get("memo")
    if not isinstance(memo, str):
        return ""
    return memo.strip()[:500]


def record_answers(
    name: str,
    item: dict,
    answers: dict,
    submitted: bool,
    elapsed_ms: int | None,
    draft_elapsed_ms: int,
    memo: str | None = None,
) -> dict:
    responses = load_responses(name)
    previous = responses.get(item["item_key"]) or {}
    memo_value = previous.get("memo") or "" if memo is None else memo
    timestamp = _now()
    event = {
        "type": "answer",
        "item_key": item["item_key"],
        "answers": answers,
        "memo": memo_value,
        "submitted": bool(submitted),
        "elapsed_ms": elapsed_ms if submitted else None,
        "draft_elapsed_ms": int(draft_elapsed_ms),
        "revised": item["item_key"] in responses,
        "timestamp": timestamp,
    }
    append_journal(name, event)
    rec = {
        "item_key": item["item_key"],
        "answers": answers,
        "memo": memo_value,
        "elapsed_ms": event["elapsed_ms"],
        "draft_elapsed_ms": event["draft_elapsed_ms"],
        "submitted": bool(submitted),
        "revised": event["revised"],
        "timestamp": timestamp,
    }
    responses = _merge_records(responses, {item["item_key"]: rec})
    _AUTOSAVE_COUNT[name] = _AUTOSAVE_COUNT.get(name, 0) + 1
    snap = None
    if submitted:
        snap = timestamp.replace(":", "").replace("+", "") + "_" + item["item_key"]
    elif _AUTOSAVE_COUNT[name] % 10 == 0:
        snap = f"auto_{_AUTOSAVE_COUNT[name]:05d}"
    store_responses(name, responses, snapshot=snap)
    return responses[item["item_key"]]


def task_payload(name: str) -> dict:
    progress = load_progress(name)
    runtime = load_runtime()
    view = current_view(progress, runtime)
    saved = load_responses(name)
    body = {
        "annotator": name,
        "phase": view["phase"],
        "n_batches": view["n_batches"],
        "lock_reason": runtime.get("lock_reason"),
        "protocol": protocol().get("protocol"),
        "position": view.get("position"),
        "length": view.get("length"),
        "batch": None,
        "after_batch": view.get("after_batch"),
        "next_batch": view.get("next_batch"),
    }
    catalog = lead_examples.public_catalog()
    body["reference_examples"] = catalog["reference"]
    body["lead_note"] = catalog["note"]
    body["n_practice"] = catalog["n_practice"]
    body["examples_frozen"] = catalog["frozen"]
    body["is_lead"] = name == lead_examples.LEAD
    body["labels"] = list(proto.LABEL_CATALOG)
    item = view.get("item")
    if item is not None:
        if item.get("kind") == "practice":
            public = lead_examples.public_practice(
                view["row"], view["position"] - 1, view["length"],
            )
        else:
            public = proto.public_item(item)
        rec = saved.get(item["item_key"]) or {}
        public["saved"] = rec.get("answers") or None
        public["saved_memo"] = rec.get("memo") or ""
        public["draft_elapsed_ms"] = rec.get("draft_elapsed_ms") or 0
        public["saved_at"] = rec.get("timestamp")
        body["item"] = public
        body["batch"] = item.get("batch")
        if view["phase"] == "practice_feedback":
            body["feedback"] = lead_examples.feedback_for(
                item["item_key"], rec.get("answers") or {},
            )
    return body


@app.route("/")
def index():
    return render_template("protocol_4h.html")


@app.route("/api/annotators")
def get_annotators():
    return jsonify({"annotators": proto.ANNOTATORS, "protocol": protocol().get("protocol")})


@app.route("/api/task")
def get_task():
    return jsonify(task_payload(annotator_name()))


@app.route("/api/guidelines", methods=["POST"])
def ack_guidelines():
    name = annotator_name()
    with _WRITE:
        progress = load_progress(name)
        progress["guidelines_done"] = True
        _write_json(progress_path(name), progress)
        return jsonify(task_payload(name))


def _snapshot_manual(name: str, responses: dict) -> str:
    saved_at = _now()
    store_responses(
        name,
        responses,
        snapshot="manual_" + saved_at.replace(":", "").replace("+", ""),
    )
    return saved_at


@app.route("/api/save", methods=["POST"])
def manual_save():
    name = annotator_name()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        saved_at = _now()
        item_key = payload.get("item_key")
        if item_key and payload.get("answers"):
            item = items_by_key().get(item_key)
            if item is None:
                return jsonify({"error": "unknown item"}), 404
            error = check_partial(item, payload.get("answers"))
            if error:
                return jsonify({"error": error}), 400
            rec = record_answers(
                name, item, payload["answers"], False, None, _draft_ms(payload), _memo(payload),
            )
            saved_at = rec.get("timestamp") or saved_at
        responses = load_responses(name)
        saved_at = _snapshot_manual(name, responses)
        return jsonify({
            "ok": True,
            "saved_at": saved_at,
            "n_items": len(responses),
            "path": f"responses/{name}.json",
        })


@app.route("/api/export")
def export_responses():
    name = annotator_name()
    body = {
        "annotator": name,
        "exported_at": _now(),
        "protocol": protocol().get("protocol"),
        "progress": load_progress(name),
        "responses": load_responses(name),
    }
    payload = json.dumps(body, ensure_ascii=False, indent=2)
    response = app.response_class(payload, mimetype="application/json; charset=utf-8")
    filename = f"annotation_{name}.json"
    response.headers["Content-Disposition"] = (
        "attachment; filename=\"annotation_export.json\"; "
        f"filename*=UTF-8''{quote(filename)}"
    )
    return response


@app.route("/api/autosave", methods=["POST"])
def autosave():
    name = annotator_name()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        item = items_by_key().get(payload.get("item_key"))
        if item is None:
            return jsonify({"error": "unknown item"}), 404
        error = check_partial(item, payload.get("answers"))
        if error:
            return jsonify({"error": error}), 400
        rec = record_answers(
            name, item, payload["answers"], False, None, _draft_ms(payload), _memo(payload),
        )
        return jsonify({"ok": True, "saved_at": rec.get("timestamp")})


@app.route("/api/answer", methods=["POST"])
def save_answer():
    name = annotator_name()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        return _save_answer(name, payload)


def _draft_ms(payload: dict) -> int:
    raw = payload.get("draft_elapsed_ms")
    if raw is None:
        raw = payload.get("elapsed_ms") or 0
    try:
        return max(0, min(int(raw), 30 * 60 * 1000))
    except (TypeError, ValueError):
        return 0


def _save_answer(name: str, payload: dict):
    item = items_by_key().get(payload.get("item_key"))
    if item is None:
        return jsonify({"error": "unknown item"}), 404
    error = check_answers(item, payload.get("answers"))
    if error:
        return jsonify({"error": error}), 400
    elapsed = payload.get("elapsed_ms")
    try:
        elapsed = int(elapsed)
    except (TypeError, ValueError):
        return jsonify({"error": "elapsed_ms가 필요합니다"}), 400
    elapsed = max(0, min(elapsed, 30 * 60 * 1000))
    record_answers(
        name, item, payload["answers"], True, elapsed,
        _draft_ms(payload) or elapsed, _memo(payload),
    )
    if item.get("kind") == "practice":
        progress = load_progress(name)
        progress["practice_reveal"] = item["item_key"]
        _write_json(progress_path(name), progress)
        return jsonify(task_payload(name))
    responses = load_responses(name)
    runtime = refresh_lock(responses, name) if item["batch"] is None else load_runtime()
    progress = load_progress(name)
    view = current_view(progress, runtime)
    shown = view.get("item")
    if shown and shown["item_key"] == item["item_key"]:
        advance(progress, item, runtime)
        _write_json(progress_path(name), progress)
    return jsonify(task_payload(name))


@app.route("/api/back", methods=["POST"])
def go_back():
    name = annotator_name()
    with _WRITE:
        return _go_back(name)


def _go_back(name: str):
    progress = load_progress(name)
    runtime = load_runtime()
    view = current_view(progress, runtime)
    phase = view["phase"]
    if phase == "practice_feedback":
        progress["practice_reveal"] = None
    elif phase == "practice" and int(progress.get("practice_index") or 0) > 0:
        progress["practice_index"] -= 1
        progress["practice_reveal"] = None
    elif phase == "calibration" and progress["cal_index"] > 0:
        progress["cal_index"] -= 1
    elif phase == "batch" and progress["index"] > 0:
        progress["index"] -= 1
    elif phase == "break":
        progress["on_break"] = False
        progress["index"] = proto.N_BATCH - 1
    elif phase == "done":
        progress["done"] = False
        progress["on_break"] = False
        progress["index"] = proto.N_BATCH - 1
        progress["batch"] = min(progress["batch"], active_batches(runtime))
    _write_json(progress_path(name), progress)
    return jsonify(task_payload(name))


@app.route("/api/continue", methods=["POST"])
def continue_after_break():
    name = annotator_name()
    with _WRITE:
        return _continue_after_break(name)


def _continue_after_break(name: str):
    progress = load_progress(name)
    runtime = load_runtime()
    if not progress["on_break"]:
        return jsonify(task_payload(name))
    nxt = progress["batch"] + 1
    if nxt > active_batches(runtime):
        progress["on_break"] = False
        progress["done"] = True
    else:
        progress["batch"] = nxt
        progress["index"] = 0
        progress["on_break"] = False
        progress["done"] = False
    _write_json(progress_path(name), progress)
    return jsonify(task_payload(name))


@app.route("/api/image/<item_key>")
def serve_image(item_key: str):
    item = items_by_key().get(item_key)
    if item is None or item.get("kind") == "practice" or "prompt_id" not in item:
        abort(404)
    path = proto.image_path(item["prompt_id"], item["sample"])
    if not path.is_file():
        abort(404)
    return send_from_directory(path.parent, path.name)


def _require_lead() -> str:
    name = annotator_name()
    if name != lead_examples.LEAD:
        payload_error = jsonify({"error": "예시는 리드(희진)만 준비합니다"})
        payload_error.status_code = 403
        raise AnnotatorRequired(payload_error)
    return name


def _lead_call(fn):
    try:
        return fn()
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/examples")
def examples_catalog():
    annotator_name()
    return jsonify(lead_examples.public_catalog())


@app.route("/api/example-image/<example_id>")
def serve_example(example_id: str):
    path = lead_examples.image_file(example_id)
    if path is None:
        abort(404)
    return send_from_directory(path.parent, path.name)


@app.route("/api/lead/pool/<int:prompt_id>/<int:sample>")
def lead_pool_image(prompt_id: int, sample: int):
    _require_lead()
    if prompt_id in lead_examples.main_prompt_ids():
        abort(404)
    if not 0 <= sample < proto.SAMPLES_PER_PROMPT:
        abort(404)
    path = proto.image_path(prompt_id, sample)
    if not path.is_file():
        abort(404)
    return send_from_directory(path.parent, path.name)


@app.route("/api/lead/studio")
def lead_studio():
    _require_lead()
    lead_examples.ensure_split()
    payload = lead_examples.studio_payload()
    payload["labels"] = list(proto.LABEL_CATALOG)
    runtime = load_runtime()
    payload["n_batches"] = active_batches(runtime)
    payload["lock_reason"] = runtime.get("lock_reason")
    return jsonify(payload)


@app.route("/api/lead/candidates")
def lead_candidates():
    _require_lead()
    page = request.args.get("page") or 0
    boundary = request.args.get("boundary") or "teen_young"
    result = _lead_call(lambda: lead_examples.candidate_page(boundary, page))
    if isinstance(result, tuple):
        return result
    return jsonify(result)


@app.route("/api/lead/preview")
def lead_preview():
    _require_lead()
    raw = request.args.get("decoy") or ""
    decoy = {part for part in raw.split(",") if part}

    def run():
        return lead_examples.preview(
            int(request.args["prompt_id"]),
            int(request.args["sample"]),
            decoy,
        )

    result = _lead_call(run)
    if isinstance(result, tuple):
        return result
    return jsonify(result)


@app.route("/api/lead/reference", methods=["POST"])
def lead_save_reference():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        result = _lead_call(lambda: lead_examples.save_reference(payload))
    if isinstance(result, tuple):
        return result
    return jsonify({"ok": True, "example": {
        "id": result["id"],
        "axis": result["axis"],
        "caption": result["caption"],
    }})


@app.route("/api/lead/practice", methods=["POST"])
def lead_save_practice():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        result = _lead_call(lambda: lead_examples.save_practice(payload))
    if isinstance(result, tuple):
        return result
    return jsonify({"ok": True, "example": result})


@app.route("/api/lead/reference/swap", methods=["POST"])
def lead_swap_reference():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        result = _lead_call(lambda: lead_examples.swap_reference_image(payload.get("id") or ""))
    if isinstance(result, tuple):
        return result
    return jsonify({"ok": True, "example": {
        "id": result["id"],
        "axis": result["axis"],
        "caption": result["caption"],
        "prompt_id": result["prompt_id"],
        "sample": result["sample"],
        "image_url": f"/api/example-image/{result['id']}",
    }})


@app.route("/api/lead/delete", methods=["POST"])
def lead_delete_example():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        result = _lead_call(lambda: lead_examples.delete_example(
            payload.get("kind") or "", payload.get("id") or "",
        ))
    if isinstance(result, tuple):
        return result
    return jsonify({"ok": True})


@app.route("/api/lead/note", methods=["POST"])
def lead_note():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        result = _lead_call(lambda: lead_examples.set_note(payload.get("note") or ""))
    if isinstance(result, tuple):
        return result
    return jsonify({"ok": True, "note": result})


@app.route("/api/lead/freeze", methods=["POST"])
def lead_freeze():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    with _WRITE:
        frozen = lead_examples.set_frozen(bool(payload.get("frozen", True)))
    return jsonify({"ok": True, "frozen": frozen})


@app.route("/api/lead/batches", methods=["POST"])
def lead_batches():
    _require_lead()
    payload = request.get_json(silent=True) or {}
    try:
        n = int(payload.get("n"))
    except (TypeError, ValueError):
        return jsonify({"error": "2 또는 3 batch"}), 400
    if n not in (2, 3):
        return jsonify({"error": "2 또는 3 batch"}), 400
    with _WRITE:
        runtime = load_runtime()
        runtime["n_batches"] = n
        runtime["lead_batch_lock"] = True
        runtime["lock_reason"] = None if n == 3 else "리드가 200장(2 batch)으로 정했습니다."
        save_runtime(runtime)
    return jsonify({"ok": True, "n_batches": n, "lock_reason": runtime["lock_reason"]})


@app.route("/api/practice/next", methods=["POST"])
def practice_next():
    name = annotator_name()
    with _WRITE:
        progress = load_progress(name)
        rows = lead_examples.practice_rows()
        reveal = progress.get("practice_reveal")
        if reveal:
            progress["practice_reveal"] = None
            idx = int(progress.get("practice_index") or 0)
            if idx < len(rows) and rows[idx]["id"] == reveal:
                progress["practice_index"] = idx + 1
            if rows and progress["practice_index"] >= len(rows):
                progress["cal_index"] = proto.N_CALIBRATION
            _write_json(progress_path(name), progress)
        return jsonify(task_payload(name))


@app.route("/api/stats")
def stats():
    name = annotator_name()
    responses = load_responses(name)
    data = protocol()
    def done(rows):
        return sum(1 for item in rows if item["item_key"] in responses)
    runtime = load_runtime()
    return jsonify({
        "annotator": name,
        "calibration_done": done(data["calibration"]),
        "batches": {str(b): done(proto.batch_items(data, b)) for b in (1, 2, 3)},
        "n_batches": active_batches(runtime),
        "lock_reason": runtime.get("lock_reason"),
    })


if __name__ == "__main__":
    info = proto.summary(protocol())
    print("4-hour young-adult annotation")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("open http://localhost:5003")
    app.run(host="0.0.0.0", port=5003, debug=False)
