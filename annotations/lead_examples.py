#!/usr/bin/env python3
"""Examples the lead (희진) prepares for the other annotators.

Reference examples are age, body, and location only. Race, faith, and SES
never become an example gallery. Practice answers stay on the server until
that annotator submits the image. The race question has no answer key.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import shutil
import threading
from datetime import datetime, timezone
from urllib.parse import quote

import protocol_4h as proto

LEAD = "희진"
REFERENCE_AXES = ("age", "body", "location")
MAX_REFERENCE_PER_AXIS = 12
MAX_PRACTICE = 20
PAGE_SIZE = 8
NOTE_MAX = 2000
CAPTION_MAX = 280

BOUNDARIES = (
    ("teen_young", "Teenager vs young adult"),
    ("infant_gender", "Infant gender"),
    ("ambiguous_body", "ambiguous body"),
    ("suburb_city", "Suburban area vs Urban area"),
    ("decoy", "decoy"),
)
BOUNDARY_IDS = frozenset(key for key, _label in BOUNDARIES)
DECOY_AXES = frozenset({
    "location",
    "faith",
    "socioeconomic_status",
    "disability_status",
    "appearance_body_status",
})
AXIS_LABEL = {"age": "age", "body": "body", "location": "location"}
RACE_RULE = (
    "race_ethnicity는 정답을 맞히는 문항이 아닙니다. "
    "확신이 없으면 cannot judge를 고릅니다."
)
FAITH_RULE = "faith는 복식이나 상징이 그림에 보일 때만 aligned입니다."
SES_RULE = "socioeconomic_status는 옷, 장소, 물건이 그림에 보일 때만 aligned입니다."

_LOCK = threading.Lock()
_PROMPTS: dict[int, dict] | None = None


def examples_path():
    return proto.PROTOCOL_DIR / "lead_examples.json"


def image_dir():
    return proto.PROTOCOL_DIR / "examples" / "images"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def empty_store() -> dict:
    return {"lead": LEAD, "frozen": False, "note": "", "reference": [], "practice": []}


def load_store() -> dict:
    path = examples_path()
    if not path.exists():
        return empty_store()
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("note", "")
    data.setdefault("frozen", False)
    data.setdefault("reference", [])
    data.setdefault("practice", [])
    return data


def _save_store(store: dict) -> None:
    store["updated_at"] = _now()
    _atomic_write(examples_path(), json.dumps(store, ensure_ascii=False, indent=2))


def prompts() -> dict[int, dict]:
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = proto.load_prompts()
    return _PROMPTS


def main_prompt_ids() -> set[int]:
    data = proto.load_assignment()
    return {int(item["prompt_id"]) for item in data["main"]}


def _used_pairs(store: dict) -> set[tuple[int, int]]:
    used = set()
    for row in store["reference"] + store["practice"]:
        used.add((int(row["prompt_id"]), int(row["sample"])))
    return used


def _boundary_label(boundary: str) -> str:
    for key, label in BOUNDARIES:
        if key == boundary:
            return label
    return boundary


def _age_tag(age: str) -> str:
    if age == "Adult":
        return "young adult"
    return age


def _card_tags(prompt: dict) -> dict:
    return {
        "age": _age_tag(prompt["age"]),
        "gender": prompt["gender"],
        "body": prompt["appearance_body_status"],
        "location": prompt["location"],
    }


def _matches(boundary: str, prompt: dict) -> bool:
    if boundary == "teen_young":
        return prompt["age"] in ("Teenager", "Adult")
    if boundary == "infant_gender":
        return prompt["age"] == "Infant"
    if boundary == "ambiguous_body":
        return prompt["appearance_body_status"] not in ("Fat", "Thin")
    if boundary == "suburb_city":
        return prompt["location"] in ("Suburban area", "Urban area", "City", "Neighborhood")
    if boundary == "decoy":
        return True
    raise ValueError(boundary)


def _question_rng(prompt_id: int, sample: int) -> random.Random:
    digest = hashlib.sha256(f"lead-example-{prompt_id}-{sample}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_questions(prompt: dict, sample: int, decoy_axes: set[str]) -> list[dict]:
    rng = _question_rng(prompt["prompt_id"], sample)
    questions = proto._blind_questions(prompt)
    questions.append(proto._location_question(rng, prompt, "location" in decoy_axes))
    questions.extend(proto._block_b_questions(rng, prompt, decoy_axes))
    return questions


def _require_editable(store: dict) -> None:
    if store.get("frozen"):
        raise ValueError("예시가 동결되어 있습니다. 동결을 푼 뒤에 고치세요.")


def _require_outside_main(prompt_id: int, sample: int) -> dict:
    if prompt_id in main_prompt_ids():
        raise ValueError("본 표본에 들어 있는 프롬프트는 예시로 쓸 수 없습니다.")
    if not 0 <= sample < proto.SAMPLES_PER_PROMPT:
        raise ValueError("sample 번호가 범위를 벗어났습니다.")
    try:
        prompt = prompts()[prompt_id]
    except KeyError as exc:
        raise ValueError("그 프롬프트는 풀에 없습니다.") from exc
    if not proto.image_path(prompt_id, sample).is_file():
        raise ValueError("그 그림 파일이 없습니다.")
    return prompt


def _clean_text(value, limit: int, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}이 비어 있습니다.")
    text = " ".join(value.split())
    if not text:
        raise ValueError(f"{label}이 비어 있습니다.")
    if len(text) > limit:
        raise ValueError(f"{label}은 {limit}자까지입니다.")
    return text


def _copy_image(example_id: str, prompt_id: int, sample: int) -> None:
    src = proto.image_path(prompt_id, sample)
    dest = image_dir() / f"{example_id}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def image_file(example_id: str):
    store = load_store()
    known = {row["id"] for row in store["reference"] + store["practice"]}
    if example_id not in known:
        return None
    path = image_dir() / f"{example_id}.jpg"
    return path if path.is_file() else None


def _public_questions(questions: list[dict]) -> list[dict]:
    public = []
    for question in questions:
        public.append({
            "id": question["id"],
            "block": question["block"],
            "kind": question["kind"],
            "text": question["text"],
            "options": question["options"],
        })
    return public


def candidate_page(boundary: str, page: int) -> dict:
    if boundary not in BOUNDARY_IDS:
        raise ValueError("고를 수 없는 경계 사례입니다.")
    page = max(0, int(page))
    store = load_store()
    used = _used_pairs(store)
    blocked = main_prompt_ids()
    rows = [
        prompt for prompt in prompts().values()
        if prompt["prompt_id"] not in blocked and _matches(boundary, prompt)
    ]
    rows.sort(key=lambda prompt: prompt["prompt_id"])
    random.Random(proto.SEED).shuffle(rows)
    found = []
    for prompt in rows:
        for sample in (7, 0, 1, 2, 3, 4, 5):
            pair = (prompt["prompt_id"], sample)
            if pair in used:
                continue
            if proto.image_path(prompt["prompt_id"], sample).is_file():
                found.append((prompt, sample))
                break
    pages = max(1, (len(found) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, pages - 1)
    chunk = found[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    items = []
    for prompt, sample in chunk:
        items.append({
            "prompt_id": prompt["prompt_id"],
            "sample": sample,
            "image_url": f"/api/lead/pool/{prompt['prompt_id']}/{sample}?annotator={quote(LEAD)}",
            **_card_tags(prompt),
        })
    return {"boundary": boundary, "page": page, "pages": pages, "items": items}


def preview(prompt_id: int, sample: int, decoy_axes: set[str]) -> dict:
    unknown = decoy_axes - DECOY_AXES
    if unknown:
        raise ValueError("고를 수 없는 decoy 축입니다.")
    prompt = _require_outside_main(prompt_id, sample)
    questions = build_questions(prompt, sample, decoy_axes)
    return {
        "prompt_id": prompt_id,
        "sample": sample,
        "image_url": f"/api/lead/pool/{prompt_id}/{sample}?annotator={quote(LEAD)}",
        "tags": _card_tags(prompt),
        "decoy_axes": sorted(decoy_axes),
        "questions": _public_questions(questions),
    }


def _option_ok(question: dict, value: str) -> bool:
    return any(opt["id"] == value for opt in question["options"])


def _validate_key(questions: list[dict], answers: dict) -> dict:
    if not isinstance(answers, dict):
        raise ValueError("연습 정답이 없습니다.")
    if "race" in answers:
        raise ValueError("인종 문항은 정답을 지정하지 않습니다. 규칙만 확인합니다.")
    expected = {q["id"] for q in questions if q["id"] != "race"}
    if set(answers) != expected:
        missing = sorted(expected - set(answers))
        raise ValueError("연습 정답이 문항에 맞지 않습니다: " + ", ".join(missing))
    by_id = {q["id"]: q for q in questions}
    for qid, value in answers.items():
        if not _option_ok(by_id[qid], value):
            raise ValueError(f"고를 수 없는 정답입니다: {qid}")
    return dict(answers)


def save_reference(payload: dict) -> dict:
    axis = payload.get("axis")
    boundary = payload.get("boundary")
    if axis not in REFERENCE_AXES:
        raise ValueError("기준 예시는 나이, 체형, 장소만 만들 수 있습니다.")
    if boundary not in BOUNDARY_IDS:
        raise ValueError("경계 사례를 고르세요.")
    prompt_id = int(payload["prompt_id"])
    sample = int(payload["sample"])
    caption = _clean_text(payload.get("caption"), CAPTION_MAX, "설명")
    _require_outside_main(prompt_id, sample)
    with _LOCK:
        store = load_store()
        _require_editable(store)
        example_id = payload.get("id") or ("r" + secrets.token_hex(8))
        others = [row for row in store["reference"] if row["id"] != example_id]
        if any((int(row["prompt_id"]), int(row["sample"])) == (prompt_id, sample) for row in others):
            raise ValueError("이 그림은 이미 기준 예시에 있습니다.")
        if any(
            (int(row["prompt_id"]), int(row["sample"])) == (prompt_id, sample)
            for row in store["practice"]
        ):
            raise ValueError("이 그림은 이미 연습 세트에 있습니다.")
        if sum(1 for row in others if row["axis"] == axis) >= MAX_REFERENCE_PER_AXIS:
            raise ValueError(f"{AXIS_LABEL[axis]} 예시는 {MAX_REFERENCE_PER_AXIS}장까지입니다.")
        _copy_image(example_id, prompt_id, sample)
        row = {
            "id": example_id,
            "axis": axis,
            "boundary": boundary,
            "prompt_id": prompt_id,
            "sample": sample,
            "caption": caption,
        }
        store["reference"] = others + [row]
        _save_store(store)
        return row


def save_practice(payload: dict) -> dict:
    boundary = payload.get("boundary")
    if boundary not in BOUNDARY_IDS:
        raise ValueError("경계 사례를 고르세요.")
    prompt_id = int(payload["prompt_id"])
    sample = int(payload["sample"])
    reason = _clean_text(payload.get("reason"), CAPTION_MAX, "한 줄 이유")
    decoy_axes = payload.get("decoy_axes") or []
    if not isinstance(decoy_axes, list) or any(axis not in DECOY_AXES for axis in decoy_axes):
        raise ValueError("고를 수 없는 decoy 축입니다.")
    decoy_set = set(decoy_axes)
    prompt = _require_outside_main(prompt_id, sample)
    questions = build_questions(prompt, sample, decoy_set)
    answers = _validate_key(questions, payload.get("answers") or {})
    public_questions = _public_questions(questions)
    with _LOCK:
        store = load_store()
        _require_editable(store)
        example_id = payload.get("id") or ("p" + secrets.token_hex(8))
        others = [row for row in store["practice"] if row["id"] != example_id]
        if len(others) >= MAX_PRACTICE and not any(row["id"] == example_id for row in store["practice"]):
            raise ValueError(f"연습 세트는 {MAX_PRACTICE}장까지입니다.")
        if any((int(row["prompt_id"]), int(row["sample"])) == (prompt_id, sample) for row in others):
            raise ValueError("이 그림은 이미 연습 세트에 있습니다.")
        if any(
            (int(row["prompt_id"]), int(row["sample"])) == (prompt_id, sample)
            for row in store["reference"]
        ):
            raise ValueError("이 그림은 이미 기준 예시에 있습니다.")
        _copy_image(example_id, prompt_id, sample)
        row = {
            "id": example_id,
            "boundary": boundary,
            "prompt_id": prompt_id,
            "sample": sample,
            "reason": reason,
            "decoy_axes": sorted(decoy_set),
            "answers": answers,
            "questions": public_questions,
        }
        store["practice"] = others + [row]
        _save_store(store)
        return {"id": row["id"], "boundary": row["boundary"], "reason": row["reason"]}


def _same_reference_label(axis: str, current: dict, candidate: dict) -> bool:
    if axis == "age":
        return candidate["age"] == current["age"]
    if axis == "body":
        current_body = current["appearance_body_status"]
        next_body = candidate["appearance_body_status"]
        if current_body in ("Fat", "Thin"):
            return next_body == current_body
        return next_body not in ("Fat", "Thin")
    if axis == "location":
        return candidate["location"] == current["location"]
    return False


def swap_reference_image(example_id: str) -> dict:
    """Replace one reference picture. The caption and the label stay."""
    with _LOCK:
        store = load_store()
        _require_editable(store)
        index = next((i for i, row in enumerate(store["reference"]) if row["id"] == example_id), None)
        if index is None:
            raise ValueError("그 예시가 없습니다.")
        row = store["reference"][index]
        current = prompts()[int(row["prompt_id"])]
        blocked = main_prompt_ids()
        used = _used_pairs(store)
        used.discard((int(row["prompt_id"]), int(row["sample"])))
        candidates = [
            prompt for prompt in prompts().values()
            if prompt["prompt_id"] not in blocked
            and prompt["prompt_id"] != current["prompt_id"]
            and _same_reference_label(row["axis"], current, prompt)
        ]
        rng = random.Random(secrets.randbits(64))
        rng.shuffle(candidates)
        found: list[tuple[int, int]] = []
        for prompt in candidates:
            if len(found) >= 40:
                break
            for sample in (0, 1, 2, 3, 4, 5, 7):
                pair = (prompt["prompt_id"], sample)
                if pair in used:
                    continue
                if proto.image_path(prompt["prompt_id"], sample).is_file():
                    found.append(pair)
                    break
        if not found:
            raise ValueError("같은 라벨의 다른 그림이 없습니다.")
        prompt_id, sample = rng.choice(found)
        new_id = "r" + secrets.token_hex(8)
        _copy_image(new_id, prompt_id, sample)
        old_id = row["id"]
        row = {
            **row,
            "id": new_id,
            "prompt_id": prompt_id,
            "sample": sample,
        }
        store["reference"][index] = row
        _save_store(store)
    (image_dir() / f"{old_id}.jpg").unlink(missing_ok=True)
    return row


def delete_example(kind: str, example_id: str) -> None:
    if kind not in ("reference", "practice"):
        raise ValueError("지울 종류가 없습니다.")
    with _LOCK:
        store = load_store()
        _require_editable(store)
        rows = [row for row in store[kind] if row["id"] != example_id]
        if len(rows) == len(store[kind]):
            raise ValueError("그 예시가 없습니다.")
        store[kind] = rows
        _save_store(store)
    image = image_dir() / f"{example_id}.jpg"
    image.unlink(missing_ok=True)


def set_note(note: str) -> str:
    if not isinstance(note, str):
        raise ValueError("덧붙인 설명이 없습니다.")
    text = note.strip()
    if len(text) > NOTE_MAX:
        raise ValueError(f"덧붙인 설명은 {NOTE_MAX}자까지입니다.")
    with _LOCK:
        store = load_store()
        _require_editable(store)
        store["note"] = text
        _save_store(store)
    return text


def set_frozen(frozen: bool) -> bool:
    with _LOCK:
        store = load_store()
        store["frozen"] = bool(frozen)
        _save_store(store)
    return bool(frozen)


def _starter_answers(questions: list[dict]) -> dict:
    """A guideline-shaped key the lead can edit.

    Faith, SES, and the excluded attributes stay on 판단 불가, because those
    count as aligned only when the picture itself shows the cue. Location and
    the blind age, gender, body, and wheelchair questions take the drawn label.
    Race is never keyed.
    """
    answers = {}
    for question in questions:
        qid = question["id"]
        if qid == "race":
            continue
        if qid in ("faith", "ses", "disability_excluded", "appearance_excluded"):
            answers[qid] = proto.CANNOT
            continue
        if question["kind"] == "verification":
            answers[qid] = proto.NOT_ALIGNED if question.get("is_decoy") else proto.ALIGNED
            continue
        match = question.get("match_value")
        if match:
            answers[qid] = match
        elif qid == "wheelchair":
            answers[qid] = "no"
        elif qid == "body":
            answers[qid] = "neither"
        else:
            answers[qid] = proto.CANNOT
    return answers


_REF_PLAN = (
    ("age", "teen_young", lambda prompt: prompt["age"] == "Teenager", 2,
     "Teenager(만 10–19세)와 young adult(만 20–39세)를 나눕니다. 이 그림은 Teenager입니다."),
    ("age", "teen_young", lambda prompt: prompt["age"] == "Adult", 1,
     "young adult는 만 20–39세입니다. 더 어리면 Teenager(만 10–19세), 더 들면 Middle-aged(만 40–59세)입니다."),
    ("body", "ambiguous_body", None, 3,
     "Fat도 Thin도 아니면 neither를 고릅니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Suburban area", 1,
     "이 장면의 location은 Suburban area입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] in ("Urban area", "City"), 1,
     "이 장면의 location은 Urban area 또는 City입니다. Suburban area와 다릅니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Neighborhood", 1,
     "이 장면의 location은 Neighborhood입니다. City와 다릅니다."),
)
_PRACTICE_PLAN = (
    ("teen_young", 4, [], "Teenager(만 10–19세)와 young adult(만 20–39세)의 경계입니다."),
    ("infant_gender", 4, [], "Infant는 gender가 어려우면 cannot judge입니다."),
    ("ambiguous_body", 4, [], "Fat도 Thin도 아니면 neither입니다."),
    ("suburb_city", 4, [], "Suburban area와 Urban area를 장면으로 구분합니다."),
    ("decoy", 4, ["location"], "확인 문장 중 일부는 그림과 다릅니다. 맞을 때만 aligned입니다."),
)


def _take(boundary: str, pred, n: int, blocked: set[int], used: set[tuple[int, int]], rng: random.Random):
    rows = [
        prompt for prompt in prompts().values()
        if prompt["prompt_id"] not in blocked and _matches(boundary, prompt)
        and (pred is None or pred(prompt))
    ]
    rows.sort(key=lambda prompt: prompt["prompt_id"])
    rng.shuffle(rows)
    picked = []
    for prompt in rows:
        if len(picked) >= n:
            break
        for sample in (7, 0, 1, 2, 3, 4, 5):
            pair = (prompt["prompt_id"], sample)
            if pair in used:
                continue
            if proto.image_path(prompt["prompt_id"], sample).is_file():
                picked.append((prompt, sample))
                used.add(pair)
                break
    if len(picked) < n:
        raise RuntimeError(f"{boundary} 그림이 {n}장보다 적습니다. 찾은 수: {len(picked)}")
    return picked


def ensure_split() -> dict:
    """Fill an empty example file: 3 references per age, body, and location, plus 20 practice items.

    A file the lead already edited is left as it is.
    """
    with _LOCK:
        store = load_store()
        if store.get("frozen") or store["reference"] or store["practice"]:
            return store
        blocked = main_prompt_ids()
        used: set[tuple[int, int]] = set()
        rng = random.Random(proto.SEED + 17)
        reference = []
        for axis, boundary, pred, n, caption in _REF_PLAN:
            for prompt, sample in _take(boundary, pred, n, blocked, used, rng):
                example_id = "r" + secrets.token_hex(8)
                _copy_image(example_id, prompt["prompt_id"], sample)
                reference.append({
                    "id": example_id,
                    "axis": axis,
                    "boundary": boundary,
                    "prompt_id": prompt["prompt_id"],
                    "sample": sample,
                    "caption": caption,
                })
        practice = []
        for boundary, n, decoy_axes, reason in _PRACTICE_PLAN:
            decoy_set = set(decoy_axes)
            for prompt, sample in _take(boundary, None, n, blocked, used, rng):
                questions = build_questions(prompt, sample, decoy_set)
                example_id = "p" + secrets.token_hex(8)
                _copy_image(example_id, prompt["prompt_id"], sample)
                practice.append({
                    "id": example_id,
                    "boundary": boundary,
                    "prompt_id": prompt["prompt_id"],
                    "sample": sample,
                    "reason": reason,
                    "decoy_axes": sorted(decoy_set),
                    "answers": _starter_answers(questions),
                    "questions": _public_questions(questions),
                })
        store["reference"] = reference
        store["practice"] = practice
        store["split"] = "auto-4h"
        _save_store(store)
        return store


_ANCHORS = (
    ("age", "infant_gender", lambda prompt: prompt["age"] == "Infant", "이 그림은 Infant(만 0–2세)입니다."),
    ("age", "teen_young", lambda prompt: prompt["age"] == "Child", "이 그림은 Child(만 3–9세)입니다. Infant(만 0–2세)보다 크고 Teenager(만 10–19세)보다 어립니다."),
    ("age", "teen_young", lambda prompt: prompt["age"] == "Middle-aged", "이 그림은 Middle-aged(만 40–59세)입니다. young adult(만 20–39세)보다 나이가 들어 보입니다."),
    ("age", "teen_young", lambda prompt: prompt["age"] == "Elderly", "이 그림은 Elderly(만 60세 이상)입니다."),
    ("body", "ambiguous_body", lambda prompt: prompt["appearance_body_status"] == "Fat", "이 그림은 Fat입니다."),
    ("body", "ambiguous_body", lambda prompt: prompt["appearance_body_status"] == "Thin", "이 그림은 Thin입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Urban area", "이 장면의 location은 Urban area입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "City", "이 장면의 location은 City입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Rural area", "이 장면의 location은 Rural area입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Remote area", "이 장면의 location은 Remote area입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "Workplace", "이 장면의 location은 Workplace입니다."),
    ("location", "suburb_city", lambda prompt: prompt["location"] == "School", "이 장면의 location은 School입니다."),
)


def add_label_anchors() -> int:
    """Add one reference image for each age, body, and location label that has none yet."""
    with _LOCK:
        store = load_store()
        if store.get("frozen"):
            return 0
        blocked = main_prompt_ids()
        used = _used_pairs(store)
        known_prompts = {
            (row["axis"], int(row["prompt_id"]))
            for row in store["reference"]
        }
        captions = {row["caption"] for row in store["reference"]}
        rng = random.Random(proto.SEED + 19)
        added = 0
        for axis, boundary, pred, caption in _ANCHORS:
            if caption in captions:
                continue
            if sum(1 for row in store["reference"] if row["axis"] == axis) >= MAX_REFERENCE_PER_AXIS:
                continue
            rows = [
                prompt for prompt in prompts().values()
                if prompt["prompt_id"] not in blocked and pred(prompt)
                and (axis, prompt["prompt_id"]) not in known_prompts
            ]
            rows.sort(key=lambda prompt: prompt["prompt_id"])
            rng.shuffle(rows)
            chosen = None
            for prompt in rows:
                for sample in (7, 0, 1, 2, 3, 4, 5):
                    pair = (prompt["prompt_id"], sample)
                    if pair in used:
                        continue
                    if proto.image_path(prompt["prompt_id"], sample).is_file():
                        chosen = (prompt, sample)
                        break
                if chosen:
                    break
            if chosen is None:
                continue
            prompt, sample = chosen
            example_id = "r" + secrets.token_hex(8)
            _copy_image(example_id, prompt["prompt_id"], sample)
            store["reference"].append({
                "id": example_id,
                "axis": axis,
                "boundary": boundary,
                "prompt_id": prompt["prompt_id"],
                "sample": sample,
                "caption": caption,
            })
            used.add((prompt["prompt_id"], sample))
            known_prompts.add((axis, prompt["prompt_id"]))
            captions.add(caption)
            added += 1
        if added:
            _save_store(store)
        return added


def practice_rows() -> list[dict]:
    return list(load_store()["practice"])


def practice_item(row: dict) -> dict:
    questions = row.get("questions")
    if not questions:
        prompt = prompts()[int(row["prompt_id"])]
        decoy = set(row.get("decoy_axes") or [])
        questions = _public_questions(build_questions(prompt, int(row["sample"]), decoy))
    return {
        "item_key": row["id"],
        "kind": "practice",
        "batch": None,
        "order": 0,
        "reason": row["reason"],
        "answers_key": dict(row["answers"]),
        "questions": questions,
    }


def public_reference() -> list[dict]:
    items = []
    for row in load_store()["reference"]:
        items.append({
            "id": row["id"],
            "axis": row["axis"],
            "axis_label": AXIS_LABEL[row["axis"]],
            "boundary": row["boundary"],
            "boundary_label": _boundary_label(row["boundary"]),
            "caption": row["caption"],
            "image_url": f"/api/example-image/{row['id']}",
        })
    return items


def public_catalog() -> dict:
    store = load_store()
    return {
        "frozen": bool(store.get("frozen")),
        "note": store.get("note") or "",
        "reference": public_reference(),
        "n_practice": len(store["practice"]),
    }


def studio_payload() -> dict:
    store = load_store()
    reference = []
    for row in store["reference"]:
        reference.append({
            **row,
            "axis_label": AXIS_LABEL[row["axis"]],
            "boundary_label": _boundary_label(row["boundary"]),
            "image_url": f"/api/example-image/{row['id']}",
        })
    practice = []
    for index, row in enumerate(store["practice"], start=1):
        practice.append({
            "id": row["id"],
            "order": index,
            "boundary": row["boundary"],
            "boundary_label": _boundary_label(row["boundary"]),
            "prompt_id": row["prompt_id"],
            "sample": row["sample"],
            "reason": row["reason"],
            "decoy_axes": row["decoy_axes"],
            "answers": row["answers"],
            "image_url": f"/api/example-image/{row['id']}",
        })
    return {
        "frozen": bool(store.get("frozen")),
        "note": store.get("note") or "",
        "reference": reference,
        "practice": practice,
        "boundaries": [{"id": key, "label": label} for key, label in BOUNDARIES],
        "axes": [{"id": key, "label": AXIS_LABEL[key]} for key in REFERENCE_AXES],
        "limits": {
            "reference_per_axis": MAX_REFERENCE_PER_AXIS,
            "practice": MAX_PRACTICE,
        },
    }


def _label(question: dict, value: str) -> str:
    for opt in question["options"]:
        if opt["id"] == value:
            return opt["label"]
    return value


def feedback_for(example_id: str, answers: dict) -> dict:
    store = load_store()
    row = next((item for item in store["practice"] if item["id"] == example_id), None)
    if row is None:
        raise KeyError(example_id)
    item = practice_item(row)
    key = item["answers_key"]
    rows = []
    for question in item["questions"]:
        qid = question["id"]
        yours = (answers or {}).get(qid)
        if qid == "race":
            rows.append({
                "id": qid,
                "text": question["text"],
                "yours": _label(question, yours) if yours else "",
                "rule_only": True,
                "rule": RACE_RULE,
                "followed_rule": yours == proto.CANNOT,
            })
            continue
        expected = key.get(qid)
        rule = None
        if qid == "faith":
            rule = FAITH_RULE
        elif qid == "ses":
            rule = SES_RULE
        rows.append({
            "id": qid,
            "text": question["text"],
            "yours": _label(question, yours) if yours else "",
            "key": _label(question, expected) if expected else "",
            "ok": yours == expected,
            "rule_only": False,
            "rule": rule,
        })
    return {"reason": row["reason"], "rows": rows}


def public_practice(row: dict, index: int, total: int) -> dict:
    item = practice_item(row)
    return {
        "item_key": row["id"],
        "kind": "practice",
        "batch": None,
        "order": index,
        "n_in_batch": total,
        "has_block_b": any(q["block"] == "B" for q in item["questions"]),
        "image_url": f"/api/example-image/{row['id']}",
        "questions": _public_questions(item["questions"]),
    }
