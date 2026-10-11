#!/usr/bin/env python3
"""Build the English list, then its Chinese edition, from public snapshots.

No Django installation, database, API key or network access is required.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATE = "2026-10-11"
FEATURED_PER_CATEGORY = 8
CATEGORY_ZH = {
    "components": "核心元器件", "hardware": "硬件平台", "software": "软件工具",
    "models": "模型", "datasets": "数据集", "benchmarks": "基准与评测",
    "companies": "公司", "products": "产品", "research": "研究与学习",
}
SUBCATEGORY_ZH = dict(zip(
    "actuators-motors reducers-transmissions motor-drives force-tactile vision-range position-inertial edge-compute power-electronics robot-arms hands-grippers teleoperation-rigs research-humanoids mobile-bases diy-education simulation-physics robot-learning navigation-slam planning-control middleware perception-geometry robot-sdks teleoperation visualization-data fleet-autonomy vla robot-policies world-models perception grasping language-planning robot-demonstrations motion-interaction scene-assets manipulation navigation-interaction whole-body policy-evaluation humanoid-developers arm-makers mobile-legged industrial-automation service-logistics ai-software component-suppliers humanoids quadrupeds mobile-manipulators mobile-rovers consumer-education aerial-marine industrial-arms academic-labs research-institutes learning-docs standards-collections".split(),
    "执行器与电机|减速器与传动|电机驱动与控制器|力觉与触觉传感|视觉与距离传感|位置与惯性传感|边缘计算|电源与电子器件|机械臂|灵巧手与夹爪|遥操作设备|研究型人形平台|移动底盘与探测车|DIY与教育硬件|仿真与物理引擎|机器人学习框架|导航与SLAM|运动规划与控制|中间件与通信|感知与几何|机器人SDK|遥操作软件|可视化与数据工具|集群与自主系统|视觉语言动作模型|机器人策略|世界模型|感知模型|抓取模型|语言引导规划|机器人示范数据|动作与交互数据|场景与物体资源|操作任务|导航与具身交互|运动与全身控制|策略评估|人形机器人开发商|机械臂与操作设备厂商|移动与腿式机器人厂商|工业自动化|服务与物流机器人|AI与机器人软件企业|元器件供应商|人形与双足机器人|四足机器人|移动操作机器人|移动机器人与探测车|消费与教育机器人|空中与水下机器人|工业机械臂系统|高校实验室|研究机构|课程与文档|标准与参考合集".split("|")
))
KINDS = {k: v for k, v in zip(CATEGORY_ZH, ["component", "hardware", "software", "model", "dataset", "benchmark", "company", "product", "research"])}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pagerank(nodes, edges, prior, damping=0.85, tolerance=1e-10, max_iterations=100):
    """BotsTest's original deterministic, quality-weighted PageRank."""
    nodes = sorted(set(nodes))
    if not nodes:
        return {}
    adjacency = {n: set() for n in nodes}
    for source, target in edges:
        if source in adjacency and target in adjacency and source != target:
            adjacency[source].add(target)
    raw = {n: max(0.01, prior.get(n, 1.0)) for n in nodes}
    total = sum(raw.values())
    teleport = {n: raw[n] / total for n in nodes}
    rank = teleport.copy()
    for _ in range(max_iterations):
        sink = sum(rank[n] for n in nodes if not adjacency[n])
        updated = {n: (1 - damping) * teleport[n] + damping * sink * teleport[n] for n in nodes}
        for source in nodes:
            if adjacency[source]:
                share = damping * rank[source] / len(adjacency[source])
                for target in sorted(adjacency[source]):
                    updated[target] += share
        delta = sum(abs(updated[n] - rank[n]) for n in nodes)
        rank = updated
        if delta < tolerance:
            break
    return rank


def quality(row):
    health = {"reachable": 1.0, "unchecked": 0.2}.get(row["link_status"], 0.0)
    complete = sum(bool(v) for v in row["quality_components"].values()) / 4
    return health, complete, 0.7 * health + 0.3 * complete


def ordering(row):
    return (-row["ranking"]["score"], row["name"].casefold(), row["canonical_url"])


def md(text):
    return re.sub(r"([\\\[\]<>|])", r"\\\1", " ".join(str(text).split()))


def sentence(text, lang="en"):
    text = str(text).strip().rstrip(".。")
    return text + ("。" if lang == "zh-CN" else ".")


def entry(row, lang="en", prefix="", score=False):
    supporting = "Supporting resource" if lang == "en" else "基础支持资源"
    label = f" **[{supporting}]**" if row["relevance"] == "supporting" else ""
    line = f"- [{md(row['name'])}]({row['website']}){label} — {md(sentence(row['summary'][lang], lang))}"
    if score:
        repo_license = row["licenses"]["repository"]["identifier"] or ("Unknown" if lang == "en" else "未知")
        line += (f" 目录分数：{row['ranking']['score']:.6f}；仓库许可：{md(repo_license)}。" if lang == "zh-CN" else f" Internal score: {row['ranking']['score']:.6f}; repository license: {md(repo_license)}.")
    detail = "Details" if lang == "en" else "详情"
    line += f" [{detail}]({row['detail_url']})"
    return line


def read_reviews():
    paths = sorted((DATA / "reviews").glob("*.json"))
    if not paths:
        raise ValueError("Editorial review inputs are missing")
    reviews = {}
    for path in paths:
        for row in load(path):
            key = row["canonical_url"]
            if key in reviews:
                raise ValueError(f"Duplicate editorial decision: {key}")
            if row["decision"] not in {"include", "exclude", "needs_review"}:
                raise ValueError(f"Invalid decision: {key}")
            reviews[key] = row
    return reviews


def build_resources(snapshot, registry, reviews):
    nodes = snapshot["resources"]
    ids = set(snapshot.get("ranking_node_ids", [r["id"] for r in nodes]))
    edges = [(e["source"], e["target"]) for e in snapshot["edges"]]
    rank = pagerank(ids, edges, {r["id"]: 0.2 + quality(r)[2] for r in nodes})
    peak = max(rank.values())
    by_url = defaultdict(list)
    for row in nodes:
        by_url[row["canonical_url"]].append(row)
        if row["id"] not in ids:
            continue
        expected = round(100 * (0.7 * rank[row["id"]] / peak + 0.3 * quality(row)[2]), 6)
        if row["stored_score"] != expected:
            raise ValueError(f"Stored score cannot be reproduced: {row['slug']}")
    categories = {r["slug"]: r for r in snapshot["categories"]}
    subcats = {(r["category__slug"], r["slug"]): r for r in snapshot["subcategories"]}
    override_path = DATA / "classification-overrides.json"
    override_rows = load(override_path) if override_path.exists() else []
    overrides = {r["canonical_url"]: r for r in override_rows}
    if len(overrides) != len(override_rows):
        raise ValueError("Duplicate classification override")
    if not set(overrides) <= {r["canonical_url"] for r in registry["urls"]}:
        raise ValueError("Classification override outside candidate registry")
    names_path = DATA / "display-names.json"
    display_names = load(names_path) if names_path.exists() else {}
    if not set(display_names) <= {r["canonical_url"] for r in registry["urls"]}:
        raise ValueError("Display name outside candidate registry")
    curated, decisions = [], []
    for registered in registry["urls"]:
        url = registered["canonical_url"]
        review = reviews[url]
        aliases = sorted(by_url[url], key=lambda r: (-r["stored_score"], r["name"].casefold(), r["slug"]))
        row = aliases[0]
        decisions.append({**review, "source_category": registered["category"], "registered_detail_path": registered["detail_path"]})
        if review["decision"] != "include":
            continue
        override = overrides.get(url)
        cat = override["category"] if override else row["category"]
        assigned_subs = override["subcategories"] if override else row["subcategories"]
        if cat not in categories or (override and any((cat, s) not in subcats for s in assigned_subs)):
            raise ValueError(f"Invalid editorial taxonomy: {url}")
        sub_names = [s for s in assigned_subs if (cat, s) in subcats]
        sub_names.sort(key=lambda s: (subcats[(cat, s)]["position"], s))
        repo = row.get("repository_metadata") or {}
        raw_license = repo.get("license")
        if isinstance(raw_license, dict):
            raw_license = raw_license.get("spdx_id")
        license_id = raw_license if isinstance(raw_license, str) and raw_license not in {"", "NOASSERTION"} else None
        license_entry = {"identifier": license_id, "status": "repository_reported" if license_id else "unknown", "source_url": row["website"] if repo else None, "observed_at": row["checked_at"] if repo else None}
        unverified = {"identifier": None, "status": "not_individually_verified", "source_url": None, "observed_at": None}
        health, completeness, q = quality(row)
        resource = {
            "id": "url-" + hashlib.sha256(url.encode()).hexdigest()[:16],
            "name": display_names.get(url, row["name"].split(" · ")[0]), "kind": KINDS[cat],
            "website": row["website"], "canonical_url": url, "detail_url": row["detail_url"],
            "summary": {"en": review["summary_en"].strip(), "zh-CN": review["summary_zh"].strip()},
            "category": {"slug": cat, "name": {"en": categories[cat]["name"], "zh-CN": CATEGORY_ZH[cat]}},
            "subcategories": [{"slug": s, "name": {"en": subcats[(cat, s)]["name"], "zh-CN": SUBCATEGORY_ZH[s]}} for s in sub_names],
            "primary_subcategory": sub_names[0] if sub_names else None,
            "classification": {
                "method": "editorial_override" if override else "imported_directory_taxonomy",
                "source_category": row["category"], "registry_category": registered["category"],
                "source_subcategories": row["subcategories"],
                "reason": {"en": override["reason_en"] if override else "Imported public directory classification; inclusion review establishes topic relevance, not every subtype or capability.", "zh-CN": override["reason_zh"] if override else "沿用公开目录分类；收录复核确认主题相关性，不表示每个子类型或能力标签均已单独验证。"},
            },
            "relevance": review["relevance"],
            "editorial": {"decision": "include", "reviewed_on": review.get("reviewed_on", DATE), "method": "Saved publisher metadata and catalogue review; no fresh installation or endpoint test", "reason": {"en": review["reason_en"], "zh-CN": review["reason_zh"]}},
            "access": {"label": row["access"], "status": "directory_reported"},
            "licenses": {"repository": license_entry, "weights": unverified.copy(), "dataset": unverified.copy(), "hardware_design": unverified.copy()},
            "repository": {"owner": repo.get("owner"), "name": repo.get("name"), "language": repo.get("language"), "archived": repo.get("archived"), "pushed_at": repo.get("pushed_at"), "created_at": repo.get("created_at")},
            "company": row.get("resource_metadata", {}) if cat == "companies" else None,
            "verification": {"link_status": row["link_status"], "checked_at": registered["checked_at"], "method": registered["check_method"], "new_check_performed": False},
            "sources": row["sources"],
            "tags": row["tags"], "topics": row["topics"], "applications": row["applications"],
            "aliases": [{"id": a["id"], "slug": a["slug"], "name": a["name"], "detail_url": a["detail_url"]} for a in aliases],
            "ranking": {"score": row["stored_score"], "pagerank": rank[row["id"]], "normalized_pagerank": rank[row["id"]] / peak, "quality": q, "link_health": health, "completeness": completeness, "rank_updated_at": row["stored_rank_updated_at"], "representative_node_id": row["id"], "graph_scope": "approved_public_directory_records", "algorithm_version": "botstest-pr-v1"},
        }
        if resource["relevance"] not in {"direct", "supporting"}:
            raise ValueError(f"Included resource lacks relevance class: {url}")
        curated.append(resource)
    curated.sort(key=lambda r: (categories[r["category"]["slug"]]["position"], *ordering(r)))
    decisions.sort(key=lambda r: r["canonical_url"])
    return curated, decisions


def build_markdown(resources, snapshot, summary, lang):
    zh = lang == "zh-CN"
    suffix = ".zh-CN" if zh else ""
    readme = ROOT / ("README.zh-CN.md" if zh else "README.md")
    companion = "[English](README.md)" if zh else "[简体中文](README.zh-CN.md)"
    lines = ["# Awesome Embodied AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)", "", companion, ""]
    lines += (["具身智能与机器人资源精选：连接元器件、硬件、软件、模型、数据集、基准、公司、产品和研究资料。", "", "由 [BotsTest](https://botstest.com/) 维护。收录机器人直接相关资源，以及明确标注的基础支持工具；所有链接直接指向资源来源。"] if zh else ["A curated guide to embodied AI and robotics: components, hardware, software, models, datasets, benchmarks, companies, products and research.", "", "Maintained by [BotsTest](https://botstest.com/). Includes directly relevant resources and explicitly labeled supporting tools. Resource links point to their source."])
    lines += ["", (f"当前版本：{DATE}。{summary['curated']} 个收录网址，{summary['categories']} 个主类，55 个可用子类。完整候选档案保留{summary['candidates']}个历史核验网址；{summary['excluded']}个已排除，{summary['needs_review']}个待复核。" if zh else f"Edition: {DATE}. **{summary['curated']} curated URLs**, {summary['categories']} categories and 55 available subcategories. The full candidate archive preserves {summary['candidates']} historically checked URLs: {summary['excluded']} excluded and {summary['needs_review']} awaiting review."), ""]
    lines += (["收录、编辑复核与链接核验是不同状态。核验日期来自历史记录；本次没有重新联网检测所有网址或安装测试。排序是目录发现线索，不能替代任务适用性、性能或采购验证。", "", "## 导航", ""] if zh else ["Editorial inclusion and endpoint checks are separate. Check dates are historical; this edition does not imply fresh network checks or installation tests. Scores help discovery; they do not measure performance or suitability for a specific deployment.", "", "## Contents", ""])
    for cat in snapshot["categories"]:
        count = sum(r["category"]["slug"] == cat["slug"] for r in resources)
        label = CATEGORY_ZH[cat["slug"]] if zh else cat["name"]
        lines.append(f"- [{label} ({count})](lists/{cat['slug']}{suffix}.md)")
    lines += ["", ("## 如何使用" if zh else "## How to use this list"), ""]
    lines += (["从任务对应的分类与子类进入。确认软件版本、接口、训练/推理条件与许可证范围后，再访问官方资料。英文与中文使用同一数据及同一排序。", "", "以下为各类分数最高的最多8个入口；分类文件提供全部收录资源。按同类分数降序展示，同分按名称、规范网址排序。精选门槛及完整公式见[排序说明](RANKING.zh-CN.md)。"] if zh else ["Start with the category and subcategory for your task. Check versions, interfaces, training/inference requirements and license scope at the source. English and Chinese editions use the same data and order.", "", "The sections below show up to eight top-scoring entries per category; category files contain every included resource. Entries sort by descending score within their category, then name and canonical URL. See [ranking and curation rules](RANKING.md)."])
    for cat in snapshot["categories"]:
        selected = sorted([r for r in resources if r["category"]["slug"] == cat["slug"]], key=ordering)
        label = CATEGORY_ZH[cat["slug"]] if zh else cat["name"]
        file = ROOT / "lists" / f"{cat['slug']}{suffix}.md"
        page = [f"# {label}", "", ("[返回首页](../README.zh-CN.md) · [English](" + cat["slug"] + ".md)" if zh else "[Back to overview](../README.md) · [简体中文](" + cat["slug"] + ".zh-CN.md)"), "", (f"收录 {len(selected)} 个去重网址。子类内按目录分数降序；分数不是性能或采购评价。许可证列仅为观测到的仓库许可，权重、数据、硬件设计许可需分别核对。" if zh else f"{len(selected)} unique curated URLs. Descending directory score within each subcategory. Scores are not performance ratings. Repository license observations do not establish weight, dataset or hardware-design license terms."), ""]
        buckets = [s for s in snapshot["subcategories"] if s["category__slug"] == cat["slug"]]
        for sub in buckets + [{"slug": None, "name": "Other resources"}]:
            group = sorted([r for r in selected if r["primary_subcategory"] == sub["slug"]], key=ordering)
            if not group:
                continue
            sub_label = (SUBCATEGORY_ZH[sub["slug"]] if sub["slug"] else "其他资源") if zh else sub["name"]
            page += [f"## {sub_label}", ""]
            page.extend(entry(r, lang, score=True) for r in group)
            page += [""]
        file.write_text("\n".join(page) + "\n", encoding="utf-8")
        lines += ["", f"## {label}", "", (f"[全部{len(selected)}个资源](lists/{file.name})" if zh else f"[All {len(selected)} resources](lists/{file.name})"), ""]
        lines.extend(entry(r, lang) for r in selected[:FEATURED_PER_CATEGORY])
    lines += ["", ("## 数据与维护" if zh else "## Data and maintenance"), ""]
    lines += (["- [结构化资源](data/resources.json) · [CSV](data/resources.csv) · [JSON Schema](schema/resource.schema.json)", "- [全体候选](data/url-registry.json) · [收录与排除记录](data/decisions.json) · [来源与时间](PROVENANCE.md)", "- [排序规则](RANKING.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md) · [Meta字段说明](METADATA.zh-CN.md)"] if zh else ["- [Structured resources](data/resources.json) · [CSV](data/resources.csv) · [JSON Schema](schema/resource.schema.json)", "- [Candidate archive](data/url-registry.json) · [Editorial decisions](data/decisions.json) · [Provenance](PROVENANCE.md)", "- [Ranking rules](RANKING.md) · [Contributing](CONTRIBUTING.md) · [Metadata contract](METADATA.md)"])
    lines += ["", "```sh", "# Python 3.10+; install validation dependencies once", "python3 -m pip install -r requirements-dev.lock", "python3 scripts/build.py", "python3 scripts/validate.py", "python3 -m unittest discover -s tests", "```", ""]
    lines += (["依赖安装完成后，生成与校验离线执行，不需要账户凭据。维护者修改原始数据后应同步更新对应来源、审核决定与两种语言，不手工修改生成文件。", "", "## 许可", "", "原创清单文字与整理成果采用 [CC0 1.0](LICENSE)，Python工具采用 [MIT](LICENSE-CODE)。外部项目、第三方来源片段、名称与商标仍遵循各自权利及许可。"] if zh else ["After dependencies are installed, generation and validation run offline and require no credentials. Update the relevant source data, review decision and both languages together; generated files should not be edited manually.", "", "## License", "", "Original list text and compilation: [CC0 1.0](LICENSE). Python tooling: [MIT](LICENSE-CODE). External projects, third-party source excerpts, names and trademarks retain their own rights and licenses."])
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    snapshot = load(DATA / "source-snapshot.json")
    registry = load(DATA / "url-registry.json")
    reviews = read_reviews()
    keys = {r["canonical_url"] for r in registry["urls"]}
    if not keys or set(reviews) != keys:
        raise ValueError("Exactly one editorial decision is required for each candidate URL")
    resources, decisions = build_resources(snapshot, registry, reviews)
    counts = Counter(r["decision"] for r in decisions)
    summary = {"edition": DATE, "curated": len(resources), "candidates": len(keys), "excluded": counts["exclude"], "needs_review": counts["needs_review"], "categories": len(snapshot["categories"]), "subcategories_available": len(snapshot["subcategories"]), "ranking_nodes": len(snapshot.get("ranking_node_ids", snapshot["resources"])), "ranking_edges": len(snapshot["edges"]), "max_score_reproduction_error": 0.0, "new_endpoint_checks": 0, "relevance_counts": dict(Counter(r["relevance"] for r in resources)), "per_category": dict(Counter(r["category"]["slug"] for r in resources))}
    write_json(DATA / "resources.json", resources)
    write_json(DATA / "decisions.json", decisions)
    write_json(DATA / "summary.json", summary)
    with (DATA / "resources.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "kind", "category", "subcategory", "relevance", "website", "detail_url", "summary_en", "summary_zh", "repository_license", "language", "archived", "checked_at", "rank_updated_at", "score"])
        writer.writeheader()
        for r in resources:
            writer.writerow({"id": r["id"], "name": r["name"], "kind": r["kind"], "category": r["category"]["slug"], "subcategory": r["primary_subcategory"], "relevance": r["relevance"], "website": r["website"], "detail_url": r["detail_url"], "summary_en": r["summary"]["en"], "summary_zh": r["summary"]["zh-CN"], "repository_license": r["licenses"]["repository"]["identifier"], "language": r["repository"]["language"], "archived": r["repository"]["archived"], "checked_at": r["verification"]["checked_at"], "rank_updated_at": r["ranking"]["rank_updated_at"], "score": r["ranking"]["score"]})
    build_markdown(resources, snapshot, summary, "en")
    build_markdown(resources, snapshot, summary, "zh-CN")
    paths = sorted([*DATA.glob("*.json"), *DATA.glob("reviews/*.json"), DATA / "resources.csv"])
    write_json(DATA / "checksums.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name != "checksums.json"})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
