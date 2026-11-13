#!/usr/bin/env python3
"""Four-hour fully-crossed annotation protocol on the young-adult pool.

Young-adult regeneration: Adult prompt folders in
``data/ct_diffusion_4wise_ya_pool`` are the regenerated renders (real
directories). Every other prompt is a symlink to the original 224px image.
Young-adult audit: the age bin stored as ``Adult`` is asked and matched as
``young adult``. The prompt text is never sent to the annotator.

The draw is frozen by ``SEED``. Re-running ``build_protocol`` with the same
seed returns the same images, decoys, and question text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tarfile
from collections import Counter
from pathlib import Path

SEED = 20261113
PROTOCOL_VERSION = "4h-ya-1"

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE / "protocol_4h"
ASSIGNMENT_PATH = PROTOCOL_DIR / "assignment.json"
RUNTIME_PATH = PROTOCOL_DIR / "runtime.json"
RESPONSE_DIR = PROTOCOL_DIR / "responses"

# Full pool, only for rebuilding the draw. The running study does not need it.
EXTERNAL_POOL = Path(os.environ.get(
    "ANNOTATION_POOL",
    "/root/workspace/data/ct_diffusion_4wise_ya_pool",
))
MANIFEST = EXTERNAL_POOL / "meta" / "manifest.csv"
EXTERNAL_IMAGE_ROOT = EXTERNAL_POOL / "images_224" / "by_id"
# Images for the frozen 320, unpacked from images.tar.gz by run_annotation.sh.
PACKAGE_IMAGE_ROOT = PROTOCOL_DIR / "images" / "by_id"
ARCHIVE_PATH = PROTOCOL_DIR / "images.tar.gz"

# Covering-array rows. t=3 / t=4 files prepend the lower-t prompts, so a
# prompt id is new at the first t that contains it.
T2_MAX = 69          # ids 0..68, the previously annotated 69 prompts
T3_MAX = 469         # ids 69..468 are the t=3 rows that t=2 does not cover
N_PROMPTS = 2847     # ids 469..2846 are the t=4 rows

# Previously annotated t=2 images: 69 prompts x the first 10 samples.
T2_SAMPLE_MAX = 10
SAMPLES_PER_PROMPT = 50

N_MAIN = 300
N_BATCH = 100
N_BATCHES = 3
N_CALIBRATION = 20
# 23+38+39 = 100. 115 and 116 are not divisible by 3, so batch 3 swaps the
# extra t=3 image with one t=4 image: 69 / 115 / 116 overall.
BATCH_PLAN = (
    {"t2": 23, "t3": 38, "t4": 39},
    {"t2": 23, "t3": 38, "t4": 39},
    {"t2": 23, "t3": 39, "t4": 38},
)
BLOCK_B_PER_BATCH = (33, 33, 34)  # 100 images; batch 1 carries 33
LOCATION_DECOY_RATE = 0.25
CALIBRATION_SLOW_MS = 30_000
CALIBRATION_TAIL = 10

ANNOTATORS = ["장현", "희진", "승재", "준영"]
# Original single-rater split, used only to join the old Table 2 labels.
LEGACY_PROMPT_OWNERS = {}
for _name, _ids in {
    "장현": list(range(1, 18)),
    "희진": list(range(18, 35)),
    "승재": list(range(35, 52)),
    "준영": list(range(52, 69)),
}.items():
    for _pid in _ids:
        LEGACY_PROMPT_OWNERS[_pid] = _name

QUOTA = {
    "who uses a wheelchair": 50,
    "without a disability": 50,
    "Fat": 50,
    "Thin": 50,
}
KEPT_DISABILITY = frozenset({"who uses a wheelchair", "without a disability"})
KEPT_APPEARANCE = frozenset({"Fat", "Thin"})
EXCLUDED_DISABILITY = (
    "with a visual impairment",
    "with autism",
    "with cerebral palsy",
    "with a hearing impairment",
    "with a prosthetic limb",
)
EXCLUDED_APPEARANCE = ("Attractive", "Unsightliness", "strong", "weak")
ALL_DISABILITY = EXCLUDED_DISABILITY + (
    "who uses a wheelchair",
    "without a disability",
)
ALL_APPEARANCE = EXCLUDED_APPEARANCE + ("Fat", "Thin")
FAITHS = ("Christian", "Muslim", "Hindu", "Buddhist", "Jew", "Unreligious")
SES = ("Upper Class", "Middle Class", "Lower Class")
RACES = ("European", "African", "East-Asian", "South-Asian", "Latino")
AGES = ("Infant", "Child", "Teenager", "Adult", "Middle-aged", "Elderly")
GENDERS = ("Male", "Female")

# Decoys come from a different super-group, so near pairs (city/urban,
# neighborhood/suburban) are not used as each other's foil.
LOCATION_GROUPS = {
    "urban": ("Urban area", "City", "Suburban area", "Neighborhood"),
    "rural": ("Rural area", "Remote area"),
    "institutional": ("Workplace", "School"),
}
LOCATION_GROUP_OF = {
    label: group for group, labels in LOCATION_GROUPS.items() for label in labels
}

# Frozen before labeling. Point estimate >= criterion is pass, a 95% CI
# upper bound below the criterion is fail, and the interval in between is
# inconclusive. alpha_sample_reference is the .800 sample-size formula, not
# the decision threshold.
CRITERIA = {
    "blind_validity": 0.85,
    "alpha": 0.667,
    "verification_j": 0.70,
    "location_decoy_aligned_max": 0.20,
    "consensus_of": 4,
    "consensus_min": 3,
    "alpha_sample_reference": 0.800,
}

AGE_ANSWER = {
    "Infant": "infant",
    "Child": "child",
    "Teenager": "teenager",
    "Adult": "young_adult",
    "Middle-aged": "middle_aged",
    "Elderly": "elderly",
}

CANNOT = "cannot_judge"
ALIGNED = "aligned"
NOT_ALIGNED = "not_aligned"

LOCATION_KO = {
    "Urban area": "도시 지역(urban area)",
    "Suburban area": "교외(suburban area)",
    "Rural area": "농촌(rural area)",
    "Remote area": "외딴 지역(remote area)",
    "Neighborhood": "주거 동네(neighborhood)",
    "City": "도시(city)",
    "Workplace": "일터(workplace)",
    "School": "학교(school)",
}
FAITH_KO = {
    "Christian": "기독교(Christian)",
    "Muslim": "이슬람(Muslim)",
    "Hindu": "힌두교(Hindu)",
    "Buddhist": "불교(Buddhist)",
    "Jew": "유대교(Jewish)",
    "Unreligious": "무종교(unreligious)",
}
SES_KO = {
    "Upper Class": "상류층(upper class)",
    "Middle Class": "중산층(middle class)",
    "Lower Class": "하류층(lower class)",
}
RACE_KO = {
    "European": "유럽계(European)",
    "African": "아프리카계(African)",
    "East-Asian": "동아시아계(East Asian)",
    "South-Asian": "남아시아계(South Asian)",
    "Latino": "라틴계(Latino)",
}
DISABILITY_KO = {
    "with a visual impairment": "시각장애가 있다",
    "with autism": "자폐가 있다",
    "with cerebral palsy": "뇌성마비가 있다",
    "with a hearing impairment": "청각장애가 있다",
    "with a prosthetic limb": "의지를 착용하고 있다",
    "who uses a wheelchair": "휠체어를 사용한다",
    "without a disability": "장애가 없다",
}
APPEARANCE_KO = {
    "Fat": "살이 쪘다(fat)",
    "Thin": "말랐다(thin)",
    "Attractive": "매력적으로 보인다(attractive)",
    "Unsightliness": "볼품없어 보인다(unsightly)",
    "strong": "튼튼해 보인다(strong)",
    "weak": "약해 보인다(weak)",
}

VERIFY_OPTIONS = (
    {"id": ALIGNED, "label": "aligned"},
    {"id": NOT_ALIGNED, "label": "not aligned"},
    {"id": CANNOT, "label": "cannot judge"},
)
BLIND_GENDER = (
    {"id": "male", "label": "Male"},
    {"id": "female", "label": "Female"},
    {"id": CANNOT, "label": "cannot judge"},
)
# FairFace bins: 0-2 / 3-9 / 10-19 / 20-39 / 40-59 / 60+.
# Adult is asked as young adult.
AGE_WITH_YEARS = (
    ("infant", "Infant", "만 0–2세"),
    ("child", "Child", "만 3–9세"),
    ("teenager", "Teenager", "만 10–19세"),
    ("young_adult", "young adult", "만 20–39세"),
    ("middle_aged", "Middle-aged", "만 40–59세"),
    ("elderly", "Elderly", "만 60세 이상"),
)
BLIND_AGE = tuple(
    {"id": key, "label": f"{name} ({years})"} for key, name, years in AGE_WITH_YEARS
) + (
    {"id": CANNOT, "label": "cannot judge"},
)
BLIND_RACE = tuple({"id": r.lower().replace("-", "_"), "label": r} for r in RACES) + (
    {"id": CANNOT, "label": "cannot judge"},
)
RACE_ANSWER = {r: r.lower().replace("-", "_") for r in RACES}
BLIND_WHEELCHAIR = (
    {"id": "yes", "label": "who uses a wheelchair"},
    {"id": "no", "label": "not who uses a wheelchair"},
    {"id": CANNOT, "label": "cannot judge"},
)
BLIND_BODY = (
    {"id": "fat", "label": "Fat"},
    {"id": "thin", "label": "Thin"},
    {"id": "neither", "label": "neither"},
    {"id": CANNOT, "label": "cannot judge"},
)

# Shown before labeling. Values stay in the original label text.
LABEL_CATALOG = (
    {"axis": "gender", "labels": list(GENDERS)},
    {
        "axis": "age",
        "labels": [f"{name} ({years})" for _key, name, years in AGE_WITH_YEARS],
        "note": "Adult는 young adult(만 20–39세)로 고릅니다.",
    },
    {"axis": "race_ethnicity", "labels": list(RACES)},
    {"axis": "appearance_body_status", "labels": list(ALL_APPEARANCE)},
    {"axis": "disability_status", "labels": list(ALL_DISABILITY)},
    {"axis": "faith", "labels": list(FAITHS)},
    {"axis": "socioeconomic_status", "labels": list(SES)},
    {
        "axis": "location",
        "labels": list(LOCATION_GROUP_OF),
        "glosses": [
            {"label": "Urban area", "meaning": "도시 지역입니다. 건물과 도로가 있습니다."},
            {"label": "City", "meaning": "도시 중심입니다. 높은 건물과 번화한 거리가 있습니다."},
            {"label": "Suburban area", "meaning": "도시 밖 주택가입니다. 집과 마당이 보이고 한산합니다."},
            {"label": "Neighborhood", "meaning": "집이 모여 있는 동네입니다. 번화가인 City와 다릅니다."},
            {"label": "Rural area", "meaning": "농촌입니다. 밭, 논, 시골집이 보입니다."},
            {"label": "Remote area", "meaning": "외딴 곳입니다. 집과 사람이 거의 없습니다."},
            {"label": "Workplace", "meaning": "일하는 장소입니다. 사무실, 작업실, 가게가 보입니다."},
            {"label": "School", "meaning": "학교입니다. 교실, 복도, 교정이 보입니다."},
        ],
    },
)

# Fields an annotator must never receive.
SECRET_FIELDS = frozenset({
    "prompt",
    "generation_prompt",
    "gt",
    "is_decoy",
    "match_value",
    "asked_label",
    "stratum",
    "regenerated",
    "legacy_filename",
    "legacy_owner",
    "prompt_id",
    "sample",
    "block_b",
})


def stratum_of(prompt_id: int) -> str:
    if prompt_id < T2_MAX:
        return "t2"
    if prompt_id < T3_MAX:
        return "t3"
    return "t4"


def image_path(prompt_id: int, sample: int) -> Path:
    """Packaged study image, or the external pool when the draw is rebuilt."""
    name = f"{prompt_id:04d}/{sample:02d}.jpg"
    packaged = PACKAGE_IMAGE_ROOT / name
    if packaged.is_file():
        return packaged
    return EXTERNAL_IMAGE_ROOT / name


def regenerated(prompt_id: int) -> bool:
    """Adult prompt folders in the pool are real directories of new renders."""
    folder = EXTERNAL_IMAGE_ROOT / f"{prompt_id:04d}"
    return folder.exists() and not folder.is_symlink()


def export_image_archive(protocol: dict | None = None) -> Path:
    """Pack exactly the frozen study images so the repo does not need the pool."""
    protocol = protocol or load_assignment()
    items = list(protocol["main"]) + list(protocol["calibration"])
    missing = []
    tmp = ARCHIVE_PATH.with_name(ARCHIVE_PATH.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        seen: set[str] = set()
        for item in items:
            rel = str(item["image"])
            if rel in seen:
                continue
            seen.add(rel)
            src = EXTERNAL_IMAGE_ROOT / rel
            if not src.is_file():
                src = PACKAGE_IMAGE_ROOT / rel
            if not src.is_file():
                missing.append(rel)
                continue
            tar.add(src, arcname=f"by_id/{rel}", recursive=False)
    if missing:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"missing {len(missing)} images, e.g. {missing[:3]}")
    tmp.replace(ARCHIVE_PATH)
    return ARCHIVE_PATH


def generation_prompt(prompt: str, gender: str, is_regen: bool) -> str:
    """Prompt actually rendered. Adult rows were rewritten to young adult."""
    if not is_regen:
        return prompt
    if gender == "Male":
        old, new = " Man ", " young adult man "
    elif gender == "Female":
        old, new = " Woman ", " young adult woman "
    else:
        raise ValueError(f"unexpected gender {gender}")
    if prompt.count(old) != 1:
        raise ValueError(f"expected one {old!r} in {prompt!r}")
    return prompt.replace(old, new, 1)


def load_prompts() -> dict[int, dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"young-adult pool manifest missing: {MANIFEST}")
    prompts: dict[int, dict] = {}
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pid = int(row["prompt_id"])
            if pid in prompts:
                continue
            prompts[pid] = {
                "prompt_id": pid,
                "prompt": row["prompt"],
                "race_ethnicity": row["race_ethnicity"],
                "gender": row["gender"],
                "age": row["age"],
                "faith": row["faith"],
                "socioeconomic_status": row["socioeconomic_status"],
                "appearance_body_status": row["appearance_body_status"],
                "disability_status": row["disability_status"],
                "location": row["location"],
            }
    if len(prompts) != N_PROMPTS:
        raise RuntimeError(f"expected {N_PROMPTS} prompts, found {len(prompts)}")
    return prompts


def _quota_key_hits(prompt: dict) -> list[str]:
    hits = []
    if prompt["disability_status"] in QUOTA:
        hits.append(prompt["disability_status"])
    if prompt["appearance_body_status"] in QUOTA:
        hits.append(prompt["appearance_body_status"])
    return hits


def _sample_higher(
    rng: random.Random,
    pool: list[dict],
    n: int,
    deficit: Counter,
) -> list[dict]:
    """Pick ``n`` prompts, preferring whatever still fills a quota deficit."""
    remaining = list(pool)
    rng.shuffle(remaining)
    picked: list[dict] = []
    while len(picked) < n:
        if not remaining:
            raise RuntimeError("ran out of prompts before the quota was met")
        best = -1
        choice = None
        for prompt in remaining:
            score = sum(1 for hit in _quota_key_hits(prompt) if deficit[hit] > 0)
            if score > best:
                best = score
                choice = prompt
        assert choice is not None
        remaining.remove(choice)
        picked.append(choice)
        for hit in _quota_key_hits(choice):
            if deficit[hit] > 0:
                deficit[hit] -= 1
    return picked


def _choose_sample(rng: random.Random, prompt_id: int, lo: int, hi: int) -> int:
    options = list(range(lo, hi))
    rng.shuffle(options)
    for sample in options:
        if image_path(prompt_id, sample).is_file():
            return sample
    raise FileNotFoundError(f"no image for prompt {prompt_id} in samples {lo}..{hi - 1}")


def _other_label(rng: random.Random, options: tuple[str, ...] | list[str], truth: str) -> str:
    choices = [label for label in options if label != truth]
    return rng.choice(choices)


def _location_decoy(rng: random.Random, truth: str) -> str:
    group = LOCATION_GROUP_OF[truth]
    choices = [
        label
        for name, labels in LOCATION_GROUPS.items()
        if name != group
        for label in labels
    ]
    return rng.choice(choices)


def _blind_questions(prompt: dict) -> list[dict]:
    wheelchair_match = None
    if prompt["disability_status"] == "who uses a wheelchair":
        wheelchair_match = "yes"
    elif prompt["disability_status"] == "without a disability":
        wheelchair_match = "no"
    body_match = {"Fat": "fat", "Thin": "thin"}.get(prompt["appearance_body_status"])
    return [
        {
            "id": "gender",
            "block": "A",
            "kind": "blind",
            "axis": "gender",
            "text": "이 사람의 gender는 무엇입니까?",
            "options": list(BLIND_GENDER),
            "match_value": prompt["gender"].lower(),
            "scope": "all",
        },
        {
            "id": "age",
            "block": "A",
            "kind": "blind",
            "axis": "age",
            "text": "이 사람의 age는 어디입니까? Adult는 young adult(만 20–39세)로 고릅니다.",
            "options": list(BLIND_AGE),
            "match_value": AGE_ANSWER[prompt["age"]],
            "scope": "all",
        },
        {
            "id": "race",
            "block": "A",
            "kind": "blind",
            "axis": "race_ethnicity",
            "text": "이 사람의 race_ethnicity는 무엇입니까?",
            "options": list(BLIND_RACE),
            "match_value": RACE_ANSWER[prompt["race_ethnicity"]],
            "scope": "all",
        },
        {
            "id": "wheelchair",
            "block": "A",
            "kind": "blind",
            "axis": "disability_status",
            "text": "이 사람은 who uses a wheelchair입니까?",
            "options": list(BLIND_WHEELCHAIR),
            "match_value": wheelchair_match,
            "scope": "kept_disability",
        },
        {
            "id": "body",
            "block": "A",
            "kind": "blind",
            "axis": "appearance_body_status",
            "text": "이 사람의 appearance_body_status는 Fat, Thin, neither 중 어디입니까?",
            "options": list(BLIND_BODY),
            "match_value": body_match,
            "scope": "kept_appearance",
        },
    ]


def _verify(qid: str, axis: str, text: str, asked: str, is_decoy: bool) -> dict:
    return {
        "id": qid,
        "block": "B" if axis != "location" else "A",
        "kind": "verification",
        "axis": axis,
        "text": text,
        "options": list(VERIFY_OPTIONS),
        "asked_label": asked,
        "is_decoy": is_decoy,
        "match_value": None,
    }


def _location_question(rng: random.Random, prompt: dict, is_decoy: bool) -> dict:
    truth = prompt["location"]
    asked = _location_decoy(rng, truth) if is_decoy else truth
    return _verify(
        "location",
        "location",
        f"이 장면의 location은 {asked}입니까?",
        asked,
        is_decoy,
    )


def _block_b_questions(rng: random.Random, prompt: dict, decoy_axes: set[str]) -> list[dict]:
    questions = []
    faith_decoy = "faith" in decoy_axes
    faith = _other_label(rng, FAITHS, prompt["faith"]) if faith_decoy else prompt["faith"]
    questions.append(_verify(
        "faith",
        "faith",
        f"이 사람의 faith는 {faith}입니까?",
        faith,
        faith_decoy,
    ))
    ses_decoy = "socioeconomic_status" in decoy_axes
    ses = _other_label(rng, SES, prompt["socioeconomic_status"]) if ses_decoy else prompt["socioeconomic_status"]
    questions.append(_verify(
        "ses",
        "socioeconomic_status",
        f"이 사람의 socioeconomic_status는 {ses}입니까?",
        ses,
        ses_decoy,
    ))
    disability = prompt["disability_status"]
    if disability not in KEPT_DISABILITY:
        decoy = "disability_status" in decoy_axes
        asked = _other_label(rng, ALL_DISABILITY, disability) if decoy else disability
        questions.append(_verify(
            "disability_excluded",
            "disability_status",
            f"이 사람의 disability_status는 {asked}입니까?",
            asked,
            decoy,
        ))
    appearance = prompt["appearance_body_status"]
    if appearance not in KEPT_APPEARANCE:
        decoy = "appearance_body_status" in decoy_axes
        asked = _other_label(rng, ALL_APPEARANCE, appearance) if decoy else appearance
        questions.append(_verify(
            "appearance_excluded",
            "appearance_body_status",
            f"이 사람의 appearance_body_status는 {asked}입니까?",
            asked,
            decoy,
        ))
    return questions


def _half_decoy(rng: random.Random, keys: list[str]) -> set[str]:
    # Sort first so a set's hash order cannot change which half is the decoy.
    ordered = sorted(keys)
    rng.shuffle(ordered)
    return set(ordered[: len(ordered) // 2])


def _choose_block_b(rng: random.Random, batches: list[list[tuple[dict, int]]]) -> list[set[int]]:
    """Pick the 100-image Block B so excluded labels are spread across it.

    Kept disability and appearance are already scored on every image in Block A.
    Block B is where the excluded labels are asked, so a pure random 100 can
    leave one of those labels with only a handful of images.
    """
    chosen = [set() for _ in batches]
    need = list(BLOCK_B_PER_BATCH)
    dis_counts: Counter = Counter()
    app_counts: Counter = Counter()
    while any(n > 0 for n in need):
        best = None
        best_key = None
        for batch_i, group in enumerate(batches):
            if need[batch_i] <= 0:
                continue
            for order, (prompt, _sample) in enumerate(group):
                if order in chosen[batch_i]:
                    continue
                dis = prompt["disability_status"]
                app = prompt["appearance_body_status"]
                dis_score = dis_counts[dis] if dis not in KEPT_DISABILITY else 10_000
                app_score = app_counts[app] if app not in KEPT_APPEARANCE else 10_000
                key = (min(dis_score, app_score), dis_score + app_score, rng.random())
                if best_key is None or key < best_key:
                    best_key = key
                    best = (batch_i, order, prompt)
        if best is None:
            raise RuntimeError("could not fill Block B")
        batch_i, order, prompt = best
        chosen[batch_i].add(order)
        need[batch_i] -= 1
        dis_counts[prompt["disability_status"]] += 1
        app_counts[prompt["appearance_body_status"]] += 1
    return chosen


def _make_item(
    rng: random.Random,
    prompt: dict,
    sample: int,
    *,
    item_key: str,
    batch: int | None,
    order: int,
    block_b: bool,
    location_decoy: bool,
    block_b_decoy_axes: set[str],
) -> dict:
    pid = prompt["prompt_id"]
    is_regen = regenerated(pid)
    if (prompt["age"] == "Adult") != is_regen:
        raise RuntimeError(
            f"prompt {pid} age {prompt['age']} regenerated={is_regen}; "
            "young-adult pool and Adult rows are out of sync"
        )
    path = image_path(pid, sample)
    if not path.is_file():
        raise FileNotFoundError(path)
    questions = _blind_questions(prompt)
    questions.append(_location_question(rng, prompt, location_decoy))
    if block_b:
        questions.extend(_block_b_questions(rng, prompt, block_b_decoy_axes))
    return {
        "item_key": item_key,
        "batch": batch,
        "order": order,
        "block_b": block_b,
        "stratum": stratum_of(pid),
        "prompt_id": pid,
        "sample": sample,
        "image": f"{pid:04d}/{sample:02d}.jpg",
        "regenerated": is_regen,
        "legacy_filename": f"{pid}_{sample}.png",
        "legacy_owner": LEGACY_PROMPT_OWNERS.get(pid),
        "prompt": prompt["prompt"],
        "generation_prompt": generation_prompt(prompt["prompt"], prompt["gender"], is_regen),
        "gt": {k: prompt[k] for k in (
            "race_ethnicity", "gender", "age", "faith", "socioeconomic_status",
            "appearance_body_status", "disability_status", "location",
        )},
        "questions": questions,
    }


def build_protocol(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    prompts = load_prompts()
    by_stratum = {"t2": [], "t3": [], "t4": []}
    for pid, prompt in prompts.items():
        by_stratum[stratum_of(pid)].append(prompt)
    for name, rows in by_stratum.items():
        rows.sort(key=lambda p: p["prompt_id"])

    if len(by_stratum["t2"]) != 69 or len(by_stratum["t3"]) != 400:
        raise RuntimeError("unexpected covering-array split")

    t2_counts = Counter()
    for prompt in by_stratum["t2"]:
        for hit in _quota_key_hits(prompt):
            t2_counts[hit] += 1
    deficit = Counter({k: QUOTA[k] - t2_counts[k] for k in QUOTA})
    if any(v < 0 for v in deficit.values()):
        raise RuntimeError(f"t=2 alone exceeds a quota: {deficit}")

    picked = {
        "t2": list(by_stratum["t2"]),
        "t3": _sample_higher(rng, by_stratum["t3"], BATCH_PLAN[0]["t3"] + BATCH_PLAN[1]["t3"] + BATCH_PLAN[2]["t3"], deficit),
        "t4": _sample_higher(rng, by_stratum["t4"], BATCH_PLAN[0]["t4"] + BATCH_PLAN[1]["t4"] + BATCH_PLAN[2]["t4"], deficit),
    }
    if any(deficit[k] > 0 for k in QUOTA):
        raise RuntimeError(f"quota still short after sampling: {dict(deficit)}")

    used_pairs: set[tuple[int, int]] = set()
    batches: list[list[dict]] = []
    cursor = {name: 0 for name in ("t2", "t3", "t4")}
    for batch_i, plan in enumerate(BATCH_PLAN):
        group = []
        for name in ("t2", "t3", "t4"):
            take = picked[name][cursor[name]:cursor[name] + plan[name]]
            cursor[name] += plan[name]
            for prompt in take:
                lo, hi = (0, T2_SAMPLE_MAX) if name == "t2" else (0, SAMPLES_PER_PROMPT)
                sample = _choose_sample(rng, prompt["prompt_id"], lo, hi)
                group.append((prompt, sample))
                used_pairs.add((prompt["prompt_id"], sample))
        rng.shuffle(group)
        batches.append(group)

    # Location decoys: exactly 25 per batch. Block B decoys are half of the
    # 100-image subset as a whole, not half of each batch (33 does not split evenly).
    block_b_sets = _choose_block_b(rng, batches)
    plans = []
    for batch_i, group in enumerate(batches):
        plans.append({
            "group": group,
            "location_decoy": set(rng.sample(range(len(group)), k=round(len(group) * LOCATION_DECOY_RATE))),
            "block_b": block_b_sets[batch_i],
        })
    block_b_keys = [
        f"{batch_i}:{order}"
        for batch_i, plan in enumerate(plans)
        for order in plan["block_b"]
    ]
    faith_decoy = _half_decoy(rng, [f"f:{key}" for key in block_b_keys])
    ses_decoy = _half_decoy(rng, [f"s:{key}" for key in block_b_keys])
    dis_decoy = _half_decoy(rng, [
        f"d:{batch_i}:{order}"
        for batch_i, plan in enumerate(plans)
        for order in plan["block_b"]
        if plan["group"][order][0]["disability_status"] not in KEPT_DISABILITY
    ])
    app_decoy = _half_decoy(rng, [
        f"a:{batch_i}:{order}"
        for batch_i, plan in enumerate(plans)
        for order in plan["block_b"]
        if plan["group"][order][0]["appearance_body_status"] not in KEPT_APPEARANCE
    ])
    main_items: list[dict] = []
    for batch_i, plan in enumerate(plans):
        for order, (prompt, sample) in enumerate(plan["group"]):
            key = f"{batch_i}:{order}"
            axes = set()
            if f"f:{key}" in faith_decoy:
                axes.add("faith")
            if f"s:{key}" in ses_decoy:
                axes.add("socioeconomic_status")
            if f"d:{key}" in dis_decoy:
                axes.add("disability_status")
            if f"a:{key}" in app_decoy:
                axes.add("appearance_body_status")
            main_items.append(_make_item(
                rng, prompt, sample,
                item_key=f"b{batch_i + 1}_{order:03d}",
                batch=batch_i + 1,
                order=order,
                block_b=order in plan["block_b"],
                location_decoy=order in plan["location_decoy"],
                block_b_decoy_axes=axes,
            ))

    # Calibration is a separate 20. t=2 reuses a prompt but not the same sample.
    # t=3 and t=4 use prompts that are not in the 300.
    main_prompts = {item["prompt_id"] for item in main_items}
    cal_t2 = rng.sample(by_stratum["t2"], 4)
    cal_t3 = rng.sample([p for p in by_stratum["t3"] if p["prompt_id"] not in main_prompts], 8)
    cal_t4 = rng.sample([p for p in by_stratum["t4"] if p["prompt_id"] not in main_prompts], 8)
    cal_rows = []
    for prompt in cal_t2:
        taken = {s for pid, s in used_pairs if pid == prompt["prompt_id"]}
        options = [s for s in range(T2_SAMPLE_MAX) if s not in taken and image_path(prompt["prompt_id"], s).is_file()]
        cal_rows.append((prompt, rng.choice(options)))
    for prompt in cal_t3 + cal_t4:
        cal_rows.append((prompt, _choose_sample(rng, prompt["prompt_id"], 0, SAMPLES_PER_PROMPT)))
    rng.shuffle(cal_rows)
    cal_decoy = set(rng.sample(range(len(cal_rows)), k=round(len(cal_rows) * LOCATION_DECOY_RATE)))
    cal_block = set(rng.sample(range(len(cal_rows)), k=7))
    faith_decoy = _half_decoy(rng, [f"f{i}" for i in cal_block])
    ses_decoy = _half_decoy(rng, [f"s{i}" for i in cal_block])
    dis_decoy = _half_decoy(rng, [
        f"d{i}" for i in cal_block
        if cal_rows[i][0]["disability_status"] not in KEPT_DISABILITY
    ])
    app_decoy = _half_decoy(rng, [
        f"a{i}" for i in cal_block
        if cal_rows[i][0]["appearance_body_status"] not in KEPT_APPEARANCE
    ])
    calibration = []
    for order, (prompt, sample) in enumerate(cal_rows):
        axes = set()
        if f"f{order}" in faith_decoy:
            axes.add("faith")
        if f"s{order}" in ses_decoy:
            axes.add("socioeconomic_status")
        if f"d{order}" in dis_decoy:
            axes.add("disability_status")
        if f"a{order}" in app_decoy:
            axes.add("appearance_body_status")
        calibration.append(_make_item(
            rng, prompt, sample,
            item_key=f"c{order:02d}",
            batch=None,
            order=order,
            block_b=order in cal_block,
            location_decoy=order in cal_decoy,
            block_b_decoy_axes=axes,
        ))

    _validate(main_items, calibration)
    return {
        "protocol": PROTOCOL_VERSION,
        "seed": seed,
        "young_adult": {
            "images": "ct_diffusion_4wise_ya_pool",
            "adult_pixels": "regenerated young-adult renders",
            "adult_question": "young adult (GT age remains Adult)",
            "non_adult_pixels": "original images, via pool symlink",
        },
        "criteria": CRITERIA,
        "design": {
            "annotators": ANNOTATORS,
            "n_main": N_MAIN,
            "n_batches": N_BATCHES,
            "batch_plan": [dict(p) for p in BATCH_PLAN],
            "block_b_per_batch": list(BLOCK_B_PER_BATCH),
            "n_calibration": N_CALIBRATION,
            "location_decoy_rate": LOCATION_DECOY_RATE,
            "location_groups": {k: list(v) for k, v in LOCATION_GROUPS.items()},
            "calibration_slow_ms": CALIBRATION_SLOW_MS,
            "calibration_tail": CALIBRATION_TAIL,
            "t2_from_first_n_samples": T2_SAMPLE_MAX,
            "quota": QUOTA,
            "same_order_for_all_annotators": True,
        },
        "schedule": [
            {"phase": "guidelines", "minutes": 15},
            {"phase": "calibration", "images": N_CALIBRATION, "minutes": 25, "discuss": "disagreements only"},
            {"phase": "batch", "batch": 1, "images": 100, "minutes": 55},
            {"phase": "break", "minutes": 10},
            {"phase": "batch", "batch": 2, "images": 100, "minutes": 55},
            {"phase": "break", "minutes": 10},
            {"phase": "batch", "batch": 3, "images": 100, "minutes": 55},
        ],
        "main": main_items,
        "calibration": calibration,
    }


def _count_questions(items: list[dict], qid: str, pred) -> int:
    return sum(1 for item in items for q in item["questions"] if q["id"] == qid and pred(q))


def _validate(main: list[dict], calibration: list[dict]) -> None:
    if len(main) != N_MAIN:
        raise RuntimeError(f"main size {len(main)}")
    keys = [item["item_key"] for item in main + calibration]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate item key")
    pairs = [(item["prompt_id"], item["sample"]) for item in main + calibration]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate image")
    main_prompts = [item["prompt_id"] for item in main]
    if len(main_prompts) != len(set(main_prompts)):
        raise RuntimeError("two images of one prompt in the 300")
    counts = Counter(item["stratum"] for item in main)
    if counts != {"t2": 69, "t3": 115, "t4": 116}:
        raise RuntimeError(f"stratum counts {counts}")
    for batch in (1, 2, 3):
        rows = [item for item in main if item["batch"] == batch]
        if len(rows) != 100:
            raise RuntimeError(f"batch {batch} has {len(rows)}")
        got = Counter(item["stratum"] for item in rows)
        plan = BATCH_PLAN[batch - 1]
        if got != plan:
            raise RuntimeError(f"batch {batch} mix {got} != {plan}")
        orders = [item["order"] for item in rows]
        if orders != list(range(100)):
            raise RuntimeError(f"batch {batch} order is not a permutation")
        if sum(1 for item in rows if item["block_b"]) != BLOCK_B_PER_BATCH[batch - 1]:
            raise RuntimeError(f"batch {batch} block B size")
        n_decoy = _count_questions(rows, "location", lambda q: q["is_decoy"])
        if n_decoy != 25:
            raise RuntimeError(f"batch {batch} location decoys {n_decoy}")
    quota = Counter()
    for item in main:
        if item["gt"]["disability_status"] in QUOTA:
            quota[item["gt"]["disability_status"]] += 1
        if item["gt"]["appearance_body_status"] in QUOTA:
            quota[item["gt"]["appearance_body_status"]] += 1
    for key, minimum in QUOTA.items():
        if quota[key] < minimum:
            raise RuntimeError(f"quota {key} = {quota[key]} < {minimum}")
    for item in main:
        if item["stratum"] == "t2" and not 0 <= item["sample"] < T2_SAMPLE_MAX:
            raise RuntimeError(f"t=2 sample outside the annotated 690: {item['item_key']}")
    if len(calibration) != N_CALIBRATION:
        raise RuntimeError("calibration size")
    if _count_questions(calibration, "location", lambda q: q["is_decoy"]) != 5:
        raise RuntimeError("calibration location decoys")
    # Block B decoys are half of the questions that exist, per family.
    block_b = [item for item in main if item["block_b"]]
    for qid in ("faith", "ses", "disability_excluded", "appearance_excluded"):
        rows = [q for item in block_b for q in item["questions"] if q["id"] == qid]
        if not rows:
            continue
        n_decoy = sum(1 for q in rows if q["is_decoy"])
        if n_decoy != len(rows) // 2:
            raise RuntimeError(f"{qid} decoys {n_decoy} of {len(rows)}")
    # Every question id is stable, and decoy flags are part of the item.
    for item in main + calibration:
        ids = [q["id"] for q in item["questions"]]
        if ids[:6] != ["gender", "age", "race", "wheelchair", "body", "location"]:
            raise RuntimeError(f"block A order changed on {item['item_key']}")
        if any("prompt" in q["text"].lower() for q in item["questions"]):
            raise RuntimeError("question text mentions the prompt")


def public_item(item: dict) -> dict:
    """Drop ground truth, decoy flags, and the prompt."""
    questions = []
    for q in item["questions"]:
        questions.append({
            "id": q["id"],
            "block": q["block"],
            "kind": q["kind"],
            "text": q["text"],
            "options": q["options"],
        })
    return {
        "item_key": item["item_key"],
        "batch": item["batch"],
        "order": item["order"],
        "n_in_batch": N_BATCH if item["batch"] else N_CALIBRATION,
        "has_block_b": any(q["block"] == "B" for q in questions),
        "image_url": f"/api/image/{item['item_key']}",
        "questions": questions,
    }


def load_assignment(path: Path | None = None) -> dict:
    path = path or ASSIGNMENT_PATH
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_assignment(protocol: dict, path: Path | None = None) -> Path:
    path = path or ASSIGNMENT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(protocol, ensure_ascii=False, indent=2)
    # The hash pins the draw. A later edit of the file is visible in the summary.
    protocol = dict(protocol)
    protocol["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def item_index(protocol: dict) -> dict[str, dict]:
    return {item["item_key"]: item for item in protocol["main"] + protocol["calibration"]}


def batch_items(protocol: dict, batch: int) -> list[dict]:
    return [item for item in protocol["main"] if item["batch"] == batch]


def summary(protocol: dict) -> dict:
    main = protocol["main"]
    quota = Counter()
    ages = Counter()
    regen = 0
    for item in main:
        ages[item["gt"]["age"]] += 1
        regen += int(item["regenerated"])
        if item["gt"]["disability_status"] in QUOTA:
            quota[item["gt"]["disability_status"]] += 1
        if item["gt"]["appearance_body_status"] in QUOTA:
            quota[item["gt"]["appearance_body_status"]] += 1
    block_b = [item for item in main if item["block_b"]]
    excluded_dis = Counter(
        item["gt"]["disability_status"] for item in block_b
        if item["gt"]["disability_status"] not in KEPT_DISABILITY
    )
    excluded_app = Counter(
        item["gt"]["appearance_body_status"] for item in block_b
        if item["gt"]["appearance_body_status"] not in KEPT_APPEARANCE
    )
    return {
        "n_main": len(main),
        "n_calibration": len(protocol["calibration"]),
        "strata": dict(Counter(item["stratum"] for item in main)),
        "batches": {
            str(b): dict(Counter(item["stratum"] for item in main if item["batch"] == b))
            for b in (1, 2, 3)
        },
        "quota": dict(quota),
        "age": dict(ages),
        "infant": ages["Infant"],
        "regenerated_adult_images": regen,
        "block_b": len(block_b),
        "excluded_disability_in_block_b": dict(excluded_dis),
        "excluded_appearance_in_block_b": dict(excluded_app),
        "location_decoys": _count_questions(main, "location", lambda q: q["is_decoy"]),
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of an empty list")
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def lock_from_calibration(tail_medians_ms: dict[str, float]) -> dict:
    """Freeze 200 images if any annotator's last calibration images run long.

    The lock only tightens. A later fast annotator does not restore batch 3,
    because dropping the last batch has to happen for all four people.
    """
    slow = {
        name: ms for name, ms in tail_medians_ms.items()
        if ms > CALIBRATION_SLOW_MS
    }
    if slow:
        name = min(slow)
        return {
            "n_batches": 2,
            "lock_reason": (
                f"{name}의 calibration 마지막 {CALIBRATION_TAIL}장 중앙값이 "
                f"{slow[name] / 1000:.1f}초라 200장(2 batch)으로 확정"
            ),
        }
    if tail_medians_ms:
        return {"n_batches": N_BATCHES, "lock_reason": None}
    return {"n_batches": N_BATCHES, "lock_reason": None}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Freeze the 4-hour young-adult annotation draw")
    parser.add_argument("--force", action="store_true", help="replace an existing assignment")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--export-images", action="store_true", help="write protocol_4h/images.tar.gz")
    args = parser.parse_args()
    if args.export_images:
        path = export_image_archive()
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        return
    if ASSIGNMENT_PATH.exists() and not args.force:
        protocol = load_assignment()
        print(f"keeping {ASSIGNMENT_PATH}")
    else:
        protocol = build_protocol(args.seed)
        save_assignment(protocol)
        print(f"wrote {ASSIGNMENT_PATH}")
    print(json.dumps(summary(load_assignment()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
