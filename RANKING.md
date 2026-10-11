# Ranking rules

[简体中文](RANKING.zh-CN.md)

Edition: 11 October 2026. The curated list contains **2,000 distinct resource URLs**. Its historical candidate archive contains 2,042 endpoints: 2,000 included, 26 awaiting review and 16 excluded. The saved website ranking graph uses **2,023 approved public records and 1,790 publisher-reference edges**. Company and product aliases may share a URL; list deduplication chooses the highest-scored representative. Candidate archive membership does not make a resource eligible for ranking.

Only editorially approved public records vote. Pending, excluded and private resources do not participate. Each directed edge is an actual saved publisher README reference; a source casts at most 12 publisher-reference votes. Duplicate edges and self-links are omitted. Manual editorial links are supported but none exist in this snapshot.

Quality is `Q = 0.7 × link health + 0.3 × completeness`. Recorded reachable links receive health 1; unchecked links receive 0.2; other states receive 0. Completeness is the fraction of four fields present: tagline, description of at least 100 characters, sources and a subcategory.

PageRank uses damping 0.85, teleport prior `0.2 + Q`, proportional redistribution of dangling mass, tolerance 1e-10 and at most 100 iterations. The recommendation score is `100 × (0.7 × PageRank / max(PageRank) + 0.3 × Q)`, rounded to six decimals. It is an internal discovery score, not external Google PageRank, a product test or a performance rating. GitHub stars and payments add no ranking weight.

Within a subcategory, order is descending score, then case-insensitive resource name, then canonical URL. Categories and subcategories retain the saved taxonomy order. English and Chinese editions use exactly the same records and order. `scripts/build.py` recomputes the graph and checks every approved record against its stored score; `scripts/validate.py` verifies reproducibility, metadata and bilingual parity.

The export is dated 11 October; ranking timestamps remain those actually saved on 10 October. Endpoint checks retain their individual dates. Exporting or recomputing the snapshot does not perform a new network check. Read the [provenance](PROVENANCE.md), [metadata contract](METADATA.md) and [machine-readable snapshot](data/source-snapshot.json) for the evidence and scope.
