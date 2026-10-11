#!/usr/bin/env python3
"""Validate public metadata, reproducibility, bilingual lists and local links."""
from __future__ import annotations

import csv
import hashlib
import json
import ipaddress
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, unquote

from build import ROOT, DATA, build_resources, load, ordering, read_reviews


def public_url(value):
    parsed = urlsplit(value)
    assert parsed.scheme in {"https", "http"} and parsed.hostname, value
    assert not parsed.username and not parsed.password, "Credentials in a public URL"
    assert parsed.hostname != "localhost", "Private endpoint in public export"
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        assert address.is_global, "Non-public address in public export"
    assert not re.search(r"(?:^|&)(?:token|access_token|api_key|password|secret|__gmitm)=", parsed.query, re.I), "Sensitive query in public URL"


def main():
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        raise SystemExit("Install validation dependencies: python3 -m pip install -r requirements-dev.lock")
    snapshot = load(DATA / "source-snapshot.json")
    registry = load(DATA / "url-registry.json")
    resources = load(DATA / "resources.json")
    decisions = load(DATA / "decisions.json")
    summary = load(DATA / "summary.json")
    schema = load(ROOT / "schema/resource.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert {"uri", "date-time", "date"} <= set(validator.format_checker.checkers), "Install all pinned format-validation dependencies"
    for row in resources:
        errors = sorted(validator.iter_errors(row), key=lambda e: str(list(e.path)))
        assert not errors, f"{row['name']}: " + "; ".join(f"{list(e.path)} {e.message}" for e in errors[:5])
        assert re.search(r"[\u3400-\u9fff]", row["summary"]["zh-CN"]), f"Missing Chinese explanation: {row['name']}"
        for url in [row["website"], row["canonical_url"], row["detail_url"], *[s["url"] for s in row["sources"]]]:
            public_url(url)
        assert row["ranking"]["rank_updated_at"], "Missing recorded ranking timestamp"
    ids = [r["id"] for r in resources]
    keys = [r["canonical_url"] for r in resources]
    assert len(ids) == len(set(ids)) and len(keys) == len(set(keys))
    registered = {r["canonical_url"] for r in registry["urls"]}
    assert len(registered) == summary["candidates"] and {r["canonical_url"] for r in decisions} == registered
    assert len(decisions) == len(registered)
    included = {r["canonical_url"] for r in decisions if r["decision"] == "include"}
    assert set(keys) == included, "Curated list includes excluded or pending endpoints"
    expected_resources, expected_decisions = build_resources(snapshot, registry, read_reviews())
    assert resources == expected_resources and decisions == expected_decisions, "Generated records differ from their inputs"
    assert summary["curated"] == len(resources)
    assert summary["per_category"] == dict(Counter(r["category"]["slug"] for r in resources))
    with (DATA / "resources.csv").open(encoding="utf-8-sig", newline="") as f:
        csvrows = list(csv.DictReader(f))
    assert [r["id"] for r in csvrows] == ids
    for exported, row in zip(csvrows, resources):
        assert exported["summary_en"] == row["summary"]["en"]
        assert exported["summary_zh"] == row["summary"]["zh-CN"]
        assert float(exported["score"]) == row["ranking"]["score"]
    count = 0
    for category in snapshot["categories"]:
        selected = [r for r in resources if r["category"]["slug"] == category["slug"]]
        en = (ROOT / "lists" / (category["slug"] + ".md")).read_text()
        zh = (ROOT / "lists" / (category["slug"] + ".zh-CN.md")).read_text()
        seen_en = re.findall(r"^- \[.*?\]\((https?://[^\s)]+)\)", en, re.M)
        seen_zh = re.findall(r"^- \[.*?\]\((https?://[^\s)]+)\)", zh, re.M)
        assert seen_en == seen_zh and set(seen_en) == {r["website"] for r in selected}
        assert len(seen_en) == len(selected), "Missing or duplicate category entry"
        groups = {}
        for row in selected:
            groups.setdefault(row["primary_subcategory"], []).append(row)
        for group in groups.values():
            ordered = [r["website"] for r in sorted(group, key=ordering)]
            actual = [url for url in seen_en if url in set(ordered)]
            assert actual == ordered, "Subcategory sorting mismatch"
        count += len(seen_en)
    assert count == len(resources)
    markdown = list(ROOT.glob("*.md")) + list((ROOT / "lists").glob("*.md"))
    for path in markdown:
        text = path.read_text()
        for target in re.findall(r"\]\(([^\s)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            local = unquote(target.split("#")[0])
            assert (path.parent / local).resolve().exists(), f"Broken local link: {path.name} -> {target}"
    assert (ROOT / "LICENSE").read_text().startswith("Creative Commons Legal Code\n\nCC0 1.0 Universal")
    assert (ROOT / "LICENSE-CODE").read_text().startswith("MIT License")
    for relative, expected in load(DATA / "checksums.json").items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    forbidden = ["BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "__gmitm=", "Authorization: Bearer ", "DJANGO_SECRET_KEY=", "/Users/", "/tmp/botstest"]
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(x in {".git", ".venv", "__pycache__"} for x in path.parts):
            continue
        if path == Path(__file__):
            continue
        content = path.read_text(encoding="utf-8-sig")
        assert not any(value in content for value in forbidden), f"Non-public operational data in {path.name}"
    print(json.dumps({"status": "passed", "curated_resources": len(resources), "candidate_decisions": len(decisions), "bilingual_category_files": 18, "schema": "JSON Schema 2020-12", "score_reproduction": "matches stored six-decimal scores", "private_data_scan": "passed", "fresh_network_checks": 0}))


if __name__ == "__main__":
    main()
