#!/usr/bin/env python3
"""Consensus, J, alpha, and the two R3-Q2 checks for the 4-hour study.

Primary consensus is 3 of 4. No consensus counts as a validity failure.
A second pass uses a unique mode of at least 2. Verification axes are scored
with J = hit − false alarm. Decisions use the frozen criteria in the
assignment: point estimate at or above the criterion passes, a 95% CI upper
bound below it fails, and the gap is inconclusive.

Run this on the calibration answers before the main batches:

    python analyze_protocol_4h.py --split calibration --skip-vlm
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import protocol_4h as proto

VLM_ROOT = Path("/root/workspace/result/json_results")
VLM_SOURCES = {
    "qwen": {
        "ya": [
            VLM_ROOT / "young_adult_generated" / "qwen.jsonl",
            VLM_ROOT / "young_adult_generated_rest" / "qwen.jsonl",
        ],
        "base": [VLM_ROOT / "qwen_ours_result.jsonl"],
    },
    "vilt": {
        "ya": [
            VLM_ROOT / "young_adult_generated" / "vilt.jsonl",
            VLM_ROOT / "young_adult_generated_rest" / "vilt.jsonl",
        ],
        "base": [VLM_ROOT / "vilt_ours_result.jsonl"],
    },
    "paligemma2": {
        "ya": [
            VLM_ROOT / "young_adult_generated" / "paligemma.jsonl",
            VLM_ROOT / "young_adult_generated_rest" / "paligemma.jsonl",
        ],
        "base": [VLM_ROOT / "paligemma2_ours_result.jsonl"],
    },
    "clip": {
        "ya": [
            VLM_ROOT / "young_adult_generated" / "clip.json",
            VLM_ROOT / "young_adult_generated_rest" / "clip.json",
        ],
        "base": [VLM_ROOT / "clip_small_ours_result.json"],
        "age_jsonl": VLM_ROOT / "young_adult_reprobe" / "clip_age.jsonl",
    },
}
BLIND_AXES = (
    ("gender", "gender", "all"),
    ("age", "age", "all"),
    ("race", "race_ethnicity", "all"),
    ("wheelchair", "disability_status", "kept"),
    ("body", "appearance_body_status", "kept"),
)
VERIFY_AXES = (
    ("location", "location"),
    ("faith", "faith"),
    ("ses", "socioeconomic_status"),
    ("disability_excluded", "disability_excluded"),
    ("appearance_excluded", "appearance_excluded"),
)


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Nominal alpha. ``units`` are the rater labels of one item, missing dropped."""
    labels = sorted({lab for unit in units for lab in unit})
    if not any(len(unit) >= 2 for unit in units):
        return None
    if len(labels) < 2:
        return 1.0
    index = {lab: i for i, lab in enumerate(labels)}
    coinc = np.zeros((len(labels), len(labels)), dtype=float)
    for unit in units:
        m = len(unit)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coinc[index[unit[i]], index[unit[j]]] += 1.0 / (m - 1)
    total = float(coinc.sum())
    if total <= 1:
        return None
    observed = (total - float(np.trace(coinc))) / total
    marg = coinc.sum(axis=1)
    expected = 0.0
    for c in range(len(labels)):
        for k in range(len(labels)):
            if c != k:
                expected += marg[c] * marg[k]
    expected /= total * (total - 1)
    if expected == 0:
        return 1.0 if observed == 0 else None
    return float(1.0 - observed / expected)


def bootstrap_mean(flags: list[int], rng: np.random.Generator, n_boot: int) -> dict:
    arr = np.asarray(flags, dtype=float)
    if arr.size == 0:
        return {"n": 0, "point": None, "ci95": None}
    if arr.size == 1 or n_boot <= 0:
        point = float(arr.mean())
        return {"n": int(arr.size), "point": point, "ci95": [point, point]}
    draws = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n": int(arr.size), "point": float(arr.mean()), "ci95": [float(lo), float(hi)]}


def bootstrap_j(hits: list[int], alarms: list[int], rng: np.random.Generator, n_boot: int) -> dict:
    h = np.asarray(hits, dtype=float)
    f = np.asarray(alarms, dtype=float)
    if h.size == 0 or f.size == 0:
        return {"n_hit": int(h.size), "n_false_alarm": int(f.size), "hit": None, "false_alarm": None, "J": None, "ci95": None}
    point = float(h.mean() - f.mean())
    if n_boot <= 0:
        return {
            "n_hit": int(h.size), "n_false_alarm": int(f.size),
            "hit": float(h.mean()), "false_alarm": float(f.mean()),
            "J": point, "ci95": [point, point],
        }
    hs = h[rng.integers(0, h.size, size=(n_boot, h.size))].mean(axis=1)
    fs = f[rng.integers(0, f.size, size=(n_boot, f.size))].mean(axis=1)
    js = hs - fs
    lo, hi = np.percentile(js, [2.5, 97.5])
    return {
        "n_hit": int(h.size),
        "n_false_alarm": int(f.size),
        "hit": float(h.mean()),
        "false_alarm": float(f.mean()),
        "J": point,
        "ci95": [float(lo), float(hi)],
    }


def bootstrap_alpha(units: list[list[str]], rng: np.random.Generator, n_boot: int) -> dict:
    point = krippendorff_alpha(units)
    if point is None or len(units) < 2 or n_boot <= 0:
        return {"n": len(units), "point": point, "ci95": None if point is None else [point, point]}
    draws = []
    n = len(units)
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        draws.append(krippendorff_alpha([units[i] for i in take]))
    clean = [v for v in draws if v is not None]
    if not clean:
        return {"n": n, "point": point, "ci95": None}
    lo, hi = np.percentile(clean, [2.5, 97.5])
    return {"n": n, "point": point, "ci95": [float(lo), float(hi)]}


def zone(point, upper, criterion: float) -> str:
    if point is None:
        return "unavailable"
    if point >= criterion:
        return "pass"
    if upper is not None and upper < criterion:
        return "fail"
    return "inconclusive"


def combine_zones(zones: list[str]) -> str:
    if any(z == "fail" for z in zones):
        return "fail"
    if any(z in ("inconclusive", "unavailable") for z in zones):
        return "inconclusive"
    if zones and all(z == "pass" for z in zones):
        return "pass"
    return "unavailable"


def consensus_label(labels: list[str], minimum: int) -> str | None:
    if len(labels) < minimum:
        return None
    counts = Counter(labels)
    top, n = counts.most_common(1)[0]
    tied = sum(1 for value in counts.values() if value == n)
    if n >= minimum and tied == 1:
        return top
    return None


def question_map(item: dict) -> dict[str, dict]:
    return {q["id"]: q for q in item["questions"]}


def select_items(protocol: dict, split: str, n_batches: int) -> list[dict]:
    if split == "calibration":
        return list(protocol["calibration"])
    rows = [item for item in protocol["main"] if item["batch"] <= n_batches]
    return rows


def load_response_files(names: list[str]) -> dict[str, dict]:
    found = {}
    for name in names:
        path = proto.RESPONSE_DIR / f"{name}.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                found[name] = json.load(handle)
    return found


def ratings_for(item: dict, responses: dict[str, dict], raters: list[str], qid: str) -> list[str] | None:
    labels = []
    for name in raters:
        saved = responses.get(name, {}).get(item["item_key"])
        if not saved:
            return None
        value = (saved.get("answers") or {}).get(qid)
        if value is None:
            return None
        labels.append(value)
    return labels


def metric_record(est: dict, criterion: float) -> dict:
    upper = None if not est.get("ci95") else est["ci95"][1]
    point = est.get("point", est.get("J"))
    return {**est, "point": point, "criterion": criterion, "zone": zone(point, upper, criterion)}


def evaluate(items, responses, raters, criteria, rng, n_boot, minimum) -> dict:
    blind = {}
    for qid, axis, scope in BLIND_AXES:
        validity = []
        units = []
        for item in items:
            q = question_map(item).get(qid)
            if q is None:
                continue
            labels = ratings_for(item, responses, raters, qid)
            if labels is None:
                continue
            units.append(labels)
            if scope == "kept" and q.get("match_value") is None:
                continue
            label = consensus_label(labels, minimum)
            validity.append(0 if label is None else int(label == q["match_value"]))
        alpha = bootstrap_alpha(units, rng, n_boot)
        valid = bootstrap_mean(validity, rng, n_boot)
        blind[axis] = {
            "question": qid,
            "validity": metric_record(valid, criteria["blind_validity"]),
            "alpha": metric_record(alpha, criteria["alpha"]),
            "decision": combine_zones([
                metric_record(valid, criteria["blind_validity"])["zone"],
                metric_record(alpha, criteria["alpha"])["zone"],
            ]),
        }
    verify = {}
    for qid, name in VERIFY_AXES:
        hits, alarms, units = [], [], []
        for item in items:
            q = question_map(item).get(qid)
            if q is None:
                continue
            labels = ratings_for(item, responses, raters, qid)
            if labels is None:
                continue
            units.append(labels)
            label = consensus_label(labels, minimum)
            aligned = int(label == proto.ALIGNED)
            if q["is_decoy"]:
                alarms.append(aligned)
            else:
                hits.append(aligned)
        alpha = bootstrap_alpha(units, rng, n_boot)
        jay = bootstrap_j(hits, alarms, rng, n_boot)
        verify[name] = {
            "question": qid,
            "J": metric_record(jay, criteria["verification_j"]),
            "alpha": metric_record(alpha, criteria["alpha"]),
            "decision": combine_zones([
                metric_record(jay, criteria["verification_j"])["zone"],
                metric_record(alpha, criteria["alpha"])["zone"],
            ]),
            "pooled": name in ("disability_excluded", "appearance_excluded"),
        }
    return {"blind": blind, "verification": verify}


def location_decoy_screen(items, responses, raters, limit: float) -> dict:
    per = {}
    flagged = []
    for name in raters:
        flags = []
        for item in items:
            q = question_map(item).get("location")
            if q is None or not q["is_decoy"]:
                continue
            saved = responses.get(name, {}).get(item["item_key"])
            if not saved:
                continue
            flags.append(int(saved["answers"].get("location") == proto.ALIGNED))
        rate = None if not flags else sum(flags) / len(flags)
        per[name] = {"n_decoy": len(flags), "aligned_rate": rate, "flagged": rate is not None and rate > limit}
        if per[name]["flagged"]:
            flagged.append(name)
    return {"max_aligned_rate": limit, "annotators": per, "flagged": flagged}


def infant_gender(items, responses, raters, rng, n_boot) -> dict:
    by_age = defaultdict(list)
    for item in items:
        labels = ratings_for(item, responses, raters, "gender")
        if labels is None:
            continue
        by_age[item["gt"]["age"]].append(labels)
    out = {}
    for age, units in sorted(by_age.items()):
        flat = [lab for unit in units for lab in unit]
        consensus_cannot = 0
        for unit in units:
            # Report the pre-registered 3-of-4 rule when four raters are present.
            minimum = 3 if len(unit) >= 4 else max(1, len(unit) - 1)
            label = consensus_label(unit, minimum)
            consensus_cannot += int(label == proto.CANNOT)
        out[age] = {
            "n_images": len(units),
            "cannot_judge_rate": (sum(lab == proto.CANNOT for lab in flat) / len(flat)) if flat else None,
            "consensus_cannot_judge_rate": consensus_cannot / len(units),
            "alpha": bootstrap_alpha(units, rng, n_boot),
        }
    return out


def legacy_table2(items, responses, raters) -> dict:
    """Compare the new 3-of-4 label with the original single rater on t=2.

    Adult rows are regenerated, so that comparison is a different picture.
    The same-pixel comparison is the non-adult t=2 rows.
    """
    files = {}
    for name in proto.ANNOTATORS:
        path = proto.HERE / f"annotations_{name}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        files[name] = {row.get("image_filename"): row for row in rows}
    compared = {"same_pixels": Counter(), "regenerated_adult": Counter()}
    for item in items:
        if item["stratum"] != "t2" or not item.get("legacy_owner"):
            continue
        old = files.get(item["legacy_owner"], {}).get(item["legacy_filename"])
        if not old:
            continue
        labels = {qid: ratings_for(item, responses, raters, qid) for qid, _, _ in BLIND_AXES}
        bucket = "regenerated_adult" if item["regenerated"] else "same_pixels"
        factors = old.get("factor_annotations") or {}
        for qid, axis, scope in BLIND_AXES:
            q = question_map(item)[qid]
            if scope == "kept" and q.get("match_value") is None:
                continue
            old_value = ((factors.get(axis) or {}).get("prompt_image_match") or "").lower()
            if old_value not in ("true", "false"):
                continue
            if labels[qid] is None:
                continue
            new_hit = consensus_label(labels[qid], 3) == q["match_value"]
            old_hit = old_value == "true"
            compared[bucket]["n"] += 1
            compared[bucket]["agree"] += int(new_hit == old_hit)
        loc = question_map(item)["location"]
        if loc["is_decoy"]:
            continue
        old_value = ((factors.get("location") or {}).get("prompt_image_match") or "").lower()
        loc_labels = ratings_for(item, responses, raters, "location")
        if old_value in ("true", "false") and loc_labels is not None:
            new_hit = consensus_label(loc_labels, 3) == proto.ALIGNED
            compared[bucket]["n"] += 1
            compared[bucket]["agree"] += int(new_hit == (old_value == "true"))
    report = {}
    for key, counts in compared.items():
        n = counts["n"]
        report[key] = {
            "n_comparisons": n,
            "agreement": None if n == 0 else counts["agree"] / n,
        }
    report["note"] = (
        "same_pixels는 재생성이 아닌 t=2입니다. regenerated_adult는 예전 판정과 다른 그림이라 "
        "Table 2 검증에 쓰지 않습니다."
    )
    return report


def _pair_from_record(rec: dict) -> tuple[int, int] | None:
    filename = rec.get("image_filename") or ""
    stem = filename.rsplit(".", 1)[0]
    if "_" not in stem:
        return None
    pid, sample = stem.split("_", 1)
    try:
        return int(pid), int(sample)
    except ValueError:
        return None


def _iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buf = ""
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk and not buf:
                break
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] in "[],":
                    buf = buf[1:]
                    continue
                if buf[0] != "{":
                    break
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                yield obj
                buf = buf[idx:]
            if not chunk:
                break


def _iter_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    yield from _iter_json_array(path)


def _extract_correct(rec: dict) -> dict[str, bool]:
    out = {}
    for axis, ent in (rec.get("vqa_results") or {}).items():
        if isinstance(ent, dict) and "is_correct" in ent:
            out[axis] = bool(ent["is_correct"])
    return out


def load_vlm(items: list[dict]) -> dict[str, dict[tuple[int, int], dict[str, bool]]]:
    needed = {(item["prompt_id"], item["sample"]) for item in items}
    adult = {(item["prompt_id"], item["sample"]) for item in items if item["regenerated"]}
    found = {name: {} for name in VLM_SOURCES}
    for name, spec in VLM_SOURCES.items():
        for kind, paths in (("ya", spec["ya"]), ("base", spec["base"])):
            for path in paths:
                if not path.exists():
                    continue
                for rec in _iter_records(path):
                    pair = _pair_from_record(rec)
                    if pair is None or pair not in needed:
                        continue
                    if kind == "ya" and pair not in adult:
                        continue
                    if kind == "base" and pair in adult:
                        continue
                    found[name][pair] = _extract_correct(rec)
        age_path = spec.get("age_jsonl")
        if age_path and age_path.exists():
            # clip_age.jsonl follows the manifest order, 50 samples per prompt.
            wanted_line = {pid * 50 + sample: (pid, sample) for pid, sample in needed}
            with age_path.open(encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    pair = wanted_line.get(i)
                    if pair is None:
                        continue
                    rec = json.loads(line)
                    found[name].setdefault(pair, {})
                    found[name][pair]["age"] = bool(rec["correct"])
    return found


def vlm_error_share(items, responses, raters, vlm) -> dict:
    """Share of VLM errors that fall on images humans agreed match the prompt."""
    report = {}
    axis_of = {qid: axis for qid, axis, _ in BLIND_AXES}
    for model, table in vlm.items():
        per_axis = {}
        for qid, axis, scope in BLIND_AXES:
            errors = 0
            verified_errors = 0
            for item in items:
                q = question_map(item).get(qid)
                if q is None or (scope == "kept" and q.get("match_value") is None):
                    continue
                correct = table.get((item["prompt_id"], item["sample"]), {}).get(axis_of[qid])
                if correct is None or correct:
                    continue
                labels = ratings_for(item, responses, raters, qid)
                if labels is None:
                    continue
                errors += 1
                label = consensus_label(labels, 3 if len(raters) >= 4 else max(1, len(raters) - 1))
                verified_errors += int(label is not None and label == q["match_value"])
            per_axis[axis] = {
                "vlm_errors": errors,
                "errors_on_human_verified": verified_errors,
                "share": None if errors == 0 else verified_errors / errors,
            }
        report[model] = per_axis
    return report


def coverage(items, responses, raters) -> dict:
    complete = 0
    for item in items:
        if all(item["item_key"] in responses.get(name, {}) for name in raters):
            complete += 1
    return {"items": len(items), "fully_answered": complete, "raters": raters}


def analyze(protocol, responses, split, n_batches, n_boot, with_vlm, seed) -> dict:
    criteria = protocol["criteria"]
    items = select_items(protocol, split, n_batches)
    raters = [name for name in proto.ANNOTATORS if name in responses]
    cov = coverage(items, responses, raters)
    rng = np.random.default_rng(seed)
    minimum = criteria["consensus_min"] if len(raters) >= criteria["consensus_of"] else max(1, len(raters) - 1)
    primary = evaluate(items, responses, raters, criteria, rng, n_boot, minimum)
    screen = location_decoy_screen(items, responses, raters, criteria["location_decoy_aligned_max"])
    kept = [name for name in raters if name not in screen["flagged"]]
    without = None
    if screen["flagged"] and len(kept) >= 2:
        without = evaluate(items, responses, kept, criteria, rng, n_boot, max(1, len(kept) - 1))
    sensitivity = evaluate(items, responses, raters, criteria, rng, n_boot, 2 if len(raters) >= 2 else 1)
    return {
        "protocol": protocol.get("protocol"),
        "split": split,
        "n_batches": n_batches if split == "main" else None,
        "criteria": criteria,
        "coverage": cov,
        "consensus_rule": f"unique mode of at least {minimum} of {len(raters)}",
        "primary": primary,
        "sensitivity_at_least_2": sensitivity,
        "location_decoy_screen": screen,
        "without_flagged_annotators": without,
        "infant_gender": infant_gender(items, responses, raters, rng, n_boot),
        "legacy_table2": legacy_table2(items, responses, raters) if split == "main" else None,
        "vlm_error_share": vlm_error_share(items, responses, raters, load_vlm(items)) if with_vlm else None,
        "notes": [
            "합의가 없으면 validity 실패로 분모에 넣습니다.",
            "J = hit − false alarm. hit은 decoy가 아닌 문항에서 합의가 aligned인 비율입니다.",
            "제외 disability·appearance는 속성별로 나누지 않고 묶어서 판정합니다.",
            "셀 단위 상관과 부분집합 MPD는 이 표본에서 계산하지 않습니다.",
            "Location decoy에서 aligned가 20%를 넘는 사람을 빼면, 남은 인원에서는 '반대가 최대 1명'을 합의로 봅니다.",
        ],
    }


def _fmt(est: dict) -> str:
    point = est.get("point", est.get("J"))
    if point is None:
        return "n/a"
    ci = est.get("ci95")
    ci_text = "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{point:.3f}{ci_text} {est.get('zone', '')}"


def print_report(report: dict) -> None:
    cov = report["coverage"]
    print(f"{report['split']}: {cov['fully_answered']}/{cov['items']} fully crossed, raters={cov['raters']}")
    if report.get("n_batches"):
        print(f"batches included: {report['n_batches']}")
    print(f"consensus: {report['consensus_rule']}")
    for group in ("blind", "verification"):
        print(f"[{group}]")
        for axis, row in report["primary"][group].items():
            if group == "blind":
                print(f"  {axis}: validity {_fmt(row['validity'])}  alpha {_fmt(row['alpha'])}  -> {row['decision']}")
            else:
                print(f"  {axis}: J {_fmt(row['J'])}  alpha {_fmt(row['alpha'])}  -> {row['decision']}")
    screen = report["location_decoy_screen"]
    if screen["flagged"]:
        print("location decoy aligned > 20%:", ", ".join(screen["flagged"]))
    infant = report["infant_gender"].get("Infant")
    if infant:
        print(
            f"infant gender: n={infant['n_images']} "
            f"cannot_judge={infant['cannot_judge_rate']} alpha={_fmt(infant['alpha'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("main", "calibration"), default="main")
    parser.add_argument("--batches", type=int, default=0, help="0 = use the calibration lock")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=proto.SEED)
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()
    protocol = proto.load_assignment()
    runtime = {}
    if proto.RUNTIME_PATH.exists():
        runtime = json.loads(proto.RUNTIME_PATH.read_text(encoding="utf-8"))
    n_batches = args.batches or int(runtime.get("n_batches") or proto.N_BATCHES)
    responses = load_response_files(proto.ANNOTATORS)
    if not responses:
        raise SystemExit(f"no responses in {proto.RESPONSE_DIR}")
    report = analyze(
        protocol, responses, args.split, n_batches, args.boot, not args.skip_vlm, args.seed,
    )
    out = proto.PROTOCOL_DIR / f"analysis_{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
