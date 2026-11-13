#!/usr/bin/env python3
"""Checks the frozen draw, the blind payload, and a perfect-rater analysis."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import protocol_4h as proto
from analyze_protocol_4h import analyze, krippendorff_alpha


def _perfect(item: dict) -> dict:
    answers = {}
    for q in item["questions"]:
        if q["kind"] == "blind":
            if q["match_value"] is not None:
                answers[q["id"]] = q["match_value"]
            elif q["id"] == "wheelchair":
                answers[q["id"]] = "no"
            elif q["id"] == "body":
                answers[q["id"]] = "neither"
            else:
                answers[q["id"]] = proto.CANNOT
        else:
            answers[q["id"]] = proto.NOT_ALIGNED if q["is_decoy"] else proto.ALIGNED
    return answers


def _walk_public(obj) -> None:
    if isinstance(obj, dict):
        leaked = proto.SECRET_FIELDS.intersection(obj)
        if leaked:
            raise AssertionError(f"public payload contains {sorted(leaked)}")
        for value in obj.values():
            _walk_public(value)
    elif isinstance(obj, list):
        for value in obj:
            _walk_public(value)


def main() -> None:
    if krippendorff_alpha([["a", "a", "a", "a"], ["b", "b", "b", "b"]]) != 1.0:
        raise SystemExit("alpha of perfect agreement is not 1")
    disagree = krippendorff_alpha([["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]])
    if disagree is None or disagree >= 0:
        raise SystemExit(f"expected negative alpha, got {disagree}")
    locked = proto.lock_from_calibration({"장현": 31_000, "희진": 10_000})
    if locked["n_batches"] != 2:
        raise SystemExit("slow calibration did not lock 200 images")
    fast = proto.lock_from_calibration({"장현": 12_000})
    if fast["n_batches"] != 3:
        raise SystemExit("fast calibration locked the short form")

    protocol = proto.build_protocol()
    info = proto.summary(protocol)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    for key, minimum in proto.QUOTA.items():
        if info["quota"][key] < minimum:
            raise SystemExit(f"quota {key}")
    if info["strata"] != {"t2": 69, "t3": 115, "t4": 116}:
        raise SystemExit(info["strata"])
    if info["regenerated_adult_images"] != info["age"].get("Adult", 0):
        raise SystemExit("Adult images are not exactly the regenerated renders")
    for item in protocol["main"] + protocol["calibration"]:
        _walk_public(proto.public_item(item))
        text = " ".join(q["text"] for q in proto.public_item(item)["questions"])
        if item["prompt"] in text or item["generation_prompt"] in text:
            raise SystemExit("prompt text leaked into a question")

    import tarfile
    if not proto.ARCHIVE_PATH.is_file():
        raise SystemExit("images.tar.gz missing; run python protocol_4h.py --export-images")
    with tarfile.open(proto.ARCHIVE_PATH, "r:gz") as tar:
        names = set(tar.getnames())
    for item in protocol["main"] + protocol["calibration"]:
        if f"by_id/{item['image']}" not in names:
            raise SystemExit(f"archive missing {item['image']}")
    if len(names) != 320:
        raise SystemExit(f"archive has {len(names)} files")

    tmp = Path(tempfile.mkdtemp(prefix="protocol4h-"))
    saved = (proto.PROTOCOL_DIR, proto.ASSIGNMENT_PATH, proto.RUNTIME_PATH, proto.RESPONSE_DIR)
    try:
        proto.PROTOCOL_DIR = tmp
        proto.ASSIGNMENT_PATH = tmp / "assignment.json"
        proto.RUNTIME_PATH = tmp / "runtime.json"
        proto.RESPONSE_DIR = tmp / "responses"
        proto.RESPONSE_DIR.mkdir()
        proto.save_assignment(protocol, proto.ASSIGNMENT_PATH)
        responses = {
            name: {item["item_key"]: {"answers": _perfect(item), "elapsed_ms": 1000} for item in protocol["main"]}
            for name in proto.ANNOTATORS
        }
        report = analyze(protocol, responses, "main", 3, n_boot=20, with_vlm=False, seed=0)
        for axis, row in report["primary"]["blind"].items():
            if row["decision"] != "pass":
                raise SystemExit(f"perfect raters failed blind {axis}: {row}")
        for axis, row in report["primary"]["verification"].items():
            if row["J"]["point"] != 1.0 or row["decision"] != "pass":
                raise SystemExit(f"perfect raters failed verification {axis}: {row['J']}")
        import annotation_app as appmod
        appmod._PROTOCOL = None
        client = appmod.app.test_client()
        listed = client.get("/api/annotators").get_json()
        if listed["annotators"] != proto.ANNOTATORS:
            raise SystemExit("annotator list changed")
        name = proto.ANNOTATORS[0]
        task = client.get("/api/task", query_string={"annotator": name}).get_json()
        if task["phase"] != "guidelines":
            raise SystemExit(task["phase"])
        task = client.post("/api/guidelines", json={"annotator": name}).get_json()
        item = task["item"]
        _walk_public(item)
        answers = _perfect(proto.item_index(protocol)[item["item_key"]])
        image = client.get(item["image_url"])
        if image.status_code != 200 or not image.data:
            raise SystemExit(f"image {image.status_code}")
        other = proto.ANNOTATORS[1]
        other_task = client.post("/api/guidelines", json={"annotator": other}).get_json()
        if other_task["item"]["item_key"] != item["item_key"]:
            raise SystemExit("annotators do not share the first image")
        if other_task["item"]["questions"] != item["questions"]:
            raise SystemExit("annotators do not share the same questions")
        nxt = client.post("/api/answer", json={
            "annotator": name,
            "item_key": item["item_key"],
            "answers": answers,
            "elapsed_ms": 1200,
        }).get_json()
        if nxt["item"]["item_key"] == item["item_key"]:
            raise SystemExit("answer did not advance")
        stayed = client.get("/api/task", query_string={"annotator": other}).get_json()
        if stayed["item"]["item_key"] != item["item_key"]:
            raise SystemExit("the other annotator moved when the first answered")
        second = nxt["item"]
        qid = second["questions"][0]["id"]
        value = second["questions"][0]["options"][0]["id"]
        partial = client.post("/api/autosave", json={
            "annotator": name,
            "item_key": second["item_key"],
            "answers": {qid: value},
            "draft_elapsed_ms": 800,
        })
        if partial.status_code != 200 or not partial.get_json().get("ok"):
            raise SystemExit(f"autosave failed: {partial.status_code} {partial.get_json()}")
        kept = client.get("/api/task", query_string={"annotator": name}).get_json()
        if kept["item"]["item_key"] != second["item_key"] or kept["item"]["saved"].get(qid) != value:
            raise SystemExit("partial answer was not restored on the same image")
        manual = client.post("/api/save", json={"annotator": name})
        if manual.status_code != 200 or manual.get_json().get("n_items", 0) < 1:
            raise SystemExit(f"manual save failed: {manual.status_code} {manual.get_json()}")
        exported = client.get("/api/export", query_string={"annotator": name})
        if exported.status_code != 200 or "attachment" not in exported.headers.get("Content-Disposition", ""):
            raise SystemExit("export did not download")
        exported_body = exported.get_json()
        if exported_body["annotator"] != name or second["item_key"] not in exported_body["responses"]:
            raise SystemExit("export is missing this annotator's answers")
        if any(other in exported.get_data(as_text=True) for other in proto.ANNOTATORS if other != name):
            raise SystemExit("export included another annotator")
        journal = (proto.RESPONSE_DIR / f"{name}.journal.jsonl").read_text(encoding="utf-8")
        copy = (proto.PROTOCOL_DIR / "backup" / f"{name}.journal.jsonl").read_text(encoding="utf-8")
        if journal != copy or journal.count("\n") < 2:
            raise SystemExit("journal was not copied")
        if not (proto.RESPONSE_DIR / f"{name}.json.bak").exists():
            raise SystemExit("previous response file was not kept")
        milestones = list((proto.RESPONSE_DIR / "milestones" / name).glob("*.json"))
        if not milestones:
            raise SystemExit("no milestone snapshot")
        (proto.RESPONSE_DIR / f"{name}.json").write_text("{not json", encoding="utf-8")
        restored = client.get("/api/task", query_string={"annotator": name}).get_json()
        if restored["item"]["saved"].get(qid) != value:
            raise SystemExit("corrupt response file was not recovered")
        (proto.PROTOCOL_DIR / f"progress_{name}.json").unlink()
        resumed = client.get("/api/task", query_string={"annotator": name}).get_json()
        if resumed["item"]["item_key"] != second["item_key"]:
            raise SystemExit("lost progress did not resume from saved answers")
        hidden = client.get("/protocol_4h/assignment.json")
        if hidden.status_code == 200:
            raise SystemExit("assignment json is being served")

        main_ids = {item["prompt_id"] for item in protocol["main"]}
        outside = next(
            pid for pid in range(proto.T3_MAX, proto.N_PROMPTS) if pid not in main_ids
        )
        lead = "희진"
        blocked = client.post("/api/lead/reference", json={
            "annotator": "장현",
            "prompt_id": outside,
            "sample": 0,
            "axis": "age",
            "boundary": "teen_young",
            "caption": "막혀야 함",
        })
        if blocked.status_code != 403:
            raise SystemExit("non-lead saved an example")
        raced = client.post("/api/lead/reference", json={
            "annotator": lead,
            "prompt_id": outside,
            "sample": 0,
            "axis": "race",
            "boundary": "teen_young",
            "caption": "인종 예시",
        })
        if raced.status_code != 400:
            raise SystemExit("race reference was accepted")
        inside = client.post("/api/lead/reference", json={
            "annotator": lead,
            "prompt_id": protocol["main"][0]["prompt_id"],
            "sample": protocol["main"][0]["sample"],
            "axis": "age",
            "boundary": "teen_young",
            "caption": "본 표본",
        })
        if inside.status_code != 400:
            raise SystemExit("main-sample image was accepted as an example")
        made = client.post("/api/lead/reference", json={
            "annotator": lead,
            "prompt_id": outside,
            "sample": 0,
            "axis": "age",
            "boundary": "teen_young",
            "caption": "청소년과 청년의 경계",
        })
        if made.status_code != 200:
            raise SystemExit(f"reference save failed: {made.status_code} {made.get_json()}")
        viewer = "승재"
        seen = client.get("/api/task", query_string={"annotator": viewer}).get_json()
        _walk_public(seen)
        captions = [row["caption"] for row in seen["reference_examples"]]
        if "청소년과 청년의 경계" not in captions:
            raise SystemExit("saved example is not on the other annotator's guidelines")
        image = client.get(seen["reference_examples"][0]["image_url"])
        if image.status_code != 200 or not image.data:
            raise SystemExit("example image was not served")
        example_id = made.get_json()["example"]["id"]
        denied = client.post("/api/lead/reference/swap", json={"annotator": "장현", "id": example_id})
        if denied.status_code != 403:
            raise SystemExit("non-lead swapped an example")
        swapped = client.post("/api/lead/reference/swap", json={"annotator": lead, "id": example_id})
        if swapped.status_code != 200:
            raise SystemExit(f"swap failed: {swapped.status_code} {swapped.get_json()}")
        changed = swapped.get_json()["example"]
        if changed["caption"] != "청소년과 청년의 경계" or changed["id"] == example_id:
            raise SystemExit("swap did not keep the caption on a new picture")
        if client.get(f"/api/example-image/{example_id}").status_code != 404:
            raise SystemExit("old example image still served")
        if client.get(changed["image_url"]).status_code != 200:
            raise SystemExit("swapped example image was not served")
        pool = client.get(f"/api/lead/pool/{outside}/0", query_string={"annotator": "장현"})
        if pool.status_code != 403:
            raise SystemExit("pool image was served to a non-lead")
        preview = client.get("/api/lead/preview", query_string={
            "annotator": lead,
            "prompt_id": outside,
            "sample": 1,
        }).get_json()
        answers = {
            q["id"]: q["options"][0]["id"]
            for q in preview["questions"] if q["id"] != "race"
        }
        practiced = client.post("/api/lead/practice", json={
            "annotator": lead,
            "prompt_id": outside,
            "sample": 1,
            "boundary": "infant_gender",
            "reason": "영아 성별은 판단 불가로 둘 수 있다",
            "decoy_axes": ["location"],
            "answers": answers,
        })
        if practiced.status_code != 200:
            raise SystemExit(f"practice save failed: {practiced.status_code} {practiced.get_json()}")
        started = client.post("/api/guidelines", json={"annotator": viewer}).get_json()
        if started["phase"] != "practice":
            raise SystemExit(f"practice did not replace calibration, phase={started['phase']}")
        _walk_public(started)
        practice_answers = {
            q["id"]: q["options"][-1]["id"] for q in started["item"]["questions"]
        }
        revealed = client.post("/api/answer", json={
            "annotator": viewer,
            "item_key": started["item"]["item_key"],
            "answers": practice_answers,
            "elapsed_ms": 1000,
        }).get_json()
        if revealed["phase"] != "practice_feedback":
            raise SystemExit(revealed["phase"])
        _walk_public(revealed)
        race_rows = [row for row in revealed["feedback"]["rows"] if row["id"] == "race"]
        if len(race_rows) != 1 or "key" in race_rows[0]:
            raise SystemExit(f"race feedback leaked a key: {race_rows}")
        if revealed["feedback"]["reason"] != "영아 성별은 판단 불가로 둘 수 있다":
            raise SystemExit("practice reason missing")
        nxt = client.post("/api/practice/next", json={"annotator": viewer}).get_json()
        if nxt["phase"] != "batch":
            raise SystemExit(f"finished practice did not skip calibration: {nxt['phase']}")
    finally:
        proto.PROTOCOL_DIR, proto.ASSIGNMENT_PATH, proto.RUNTIME_PATH, proto.RESPONSE_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok")


if __name__ == "__main__":
    main()
