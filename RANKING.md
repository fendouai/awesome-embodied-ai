# Ranking and curation

This project uses a small, reproducible internal link graph to order relevant resources. The score helps readers navigate the list; it is not Google PageRank, a search ranking prediction, a product quality rating, or official certification. Payments, sponsorships, GitHub stars, and follower counts do not enter the calculation.

## Three different scopes

The input files describe three scopes that must not be confused:

- [The public source snapshot](data/source-snapshot.json) contains **1,039 resource records** and **668 directed relationships**. These records refer to **1,024 distinct canonical URL endpoints**. This is the graph used to reproduce the original scores.
- [The maintained URL registry](data/url-registry.json) contains **1,000 candidate URL endpoints**. A resource must belong to this registry before it can enter this project's candidate pool. Extra graph nodes do not automatically become list entries.
- The curated Awesome list is a **topic-relevant subset of that candidate pool**. Its current size is reported by the generated project artifacts. The 1,024 graph endpoints are not an Awesome inclusion count.

Editorial review first asks whether an endpoint fits embodied AI and robotics, has a useful official or maintainer source, and provides a meaningful resource for this list. Only candidates that pass this topic gate are included. Ranking orders the resulting entries; a high score does not cause automatic inclusion. The Awesome list uses the existing scores from the full 1,039-node source graph. Topic screening changes only what is displayed: it does not recompute PageRank on a curated subgraph. Nodes marked `exclude` or `needs_review` can therefore still influence the original graph scores through their recorded relationships. These scores are not a fully isolated measure of authority within the curated embodied-AI domain.

## What contributes to a score

For a resource record, let `H` be its stored link-health value:

| Stored link status | H |
|---|---:|
| `reachable` | 1.0 |
| `unchecked` | 0.2 |
| Any other status | 0.0 |

Let `C` be the fraction of four metadata checks that are true:

1. A tagline is present.
2. The description contains at least 100 characters.
3. At least one source is recorded.
4. At least one subcategory is assigned.

Each check contributes one quarter to `C`. The metadata/health component is:

```text
Q(i) = 0.7 × H(i) + 0.3 × C(i)
```

The source implementation names this component `quality`; it measures these particular health and completeness signals, not product performance, research validity, safety, or the correctness of every metadata field. Screenshots and logos are not additional ranking votes.

## The internal graph

A directed edge `A → B` means that resource A references resource B within this directory's graph. Allowed origins are a captured, real publisher repository README reference (`publisher_reference`) or an explicitly maintained editorial relationship (`editorial`). Shared tags, the same category, keyword similarity, payment, and popularity do not create graph edges.

Machine-derived README references are normalized to known published resource URLs, deduplicated per source, and self-references are removed. At most the first **12 distinct captured targets** are considered per source resource when rebuilding publisher references. Existing editorial relationships are retained and a publisher reference is not added again for an editorial pair. The cap limits the number of links one README can contribute; it is not a cap of 12 for all relationships across an organization. Each node's outgoing contribution is divided equally across its distinct targets.

The exported snapshot currently contains 668 publisher-reference edges and no editorial edges. It has no duplicate directed pairs, self-edges, or edges with missing endpoints, and its maximum publisher-reference outdegree is 12. Editorial relationships remain a supported provenance type for future updates.

## Reproducing the calculation

The graph is computed over the original 1,039 resource-record IDs. Define a positive prior and normalize it across all nodes:

```text
prior(i) = 0.2 + Q(i)
p(i) = prior(i) / sum(prior)
d = 0.85
```

The source routine also applies a minimum prior of 0.01; it is inactive here because `0.2 + Q(i)` is already at least 0.2. Start with `PR(i) = p(i)`. On each iteration, let `D` be the sum of rank held by nodes with no outgoing edges:

```text
PR_next(i) = (1 − d) × p(i)
           + d × D × p(i)
           + d × sum(PR(j) / outdegree(j), for each j → i)
```

Dangling rank is redistributed according to the same teleport distribution `p`, not uniformly. Stop when the sum of absolute changes across all nodes is less than `1e-10`, or after **100 iterations**, whichever happens first.

Normalize link rank against the maximum rank in the **whole source graph**, then combine it with `Q`:

```text
score(i) = round(100 × (0.7 × PR(i) / max(PR) + 0.3 × Q(i)), 6)
```

This is a relative internal recommendation score, not a probability or a percentage of product quality. It is not recalculated separately for each category.

Offline reproduction of this snapshot matches all stored six-decimal scores: **maximum absolute score difference = 0.0**. [scripts/build.py](scripts/build.py) generates the project artifacts from local data; [scripts/validate.py](scripts/validate.py) checks the result. These local operations do not perform fresh network verification.

## Deduplication and display order

Canonical endpoint deduplication is a presentation step after scores have been reproduced from the original graph. When several resource records share an endpoint, retain the **highest-scoring representative**. Do not add their scores together or pool aliases into extra votes. The representative must still be in the maintained registry and pass the topic gate.

Within the same category, and within the same subcategory, entries are sorted by:

1. Score, descending.
2. `name.casefold()`, ascending, for equal scores.
3. Canonical URL, ascending, for a remaining tie.

Categories follow the taxonomy rather than a contest for a global “best resource.” A model, dataset, actuator, and company serve different tasks. Use their categories, descriptions, licenses, and official documentation to decide whether they fit a project.

## Dates and limitations

The snapshot was **exported on 2026-10-09**. Its stored ranking timestamps remain **2026-10-07**. Link-health status and `checked_at` retain their individual historical values. Exporting or reproducing a score does not mean that a website, README reference, compatibility claim, or price was checked again on the export date.

A small internal graph can favor well-connected ecosystem resources and items with complete directory metadata. An unlinked but useful new project may score lower. Editorial topic review and source verification therefore remain necessary. Contributions should supply factual evidence for corrections or missing resources; purchasing a listing or gaining stars cannot buy a higher score.
