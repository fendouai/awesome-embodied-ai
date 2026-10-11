# Metadata contract

[简体中文](METADATA.zh-CN.md)

Every included record represents one canonical endpoint, not one domain or one organization. English and Chinese descriptions live in the same record and produce the same ordered list. The contract is defined by [JSON Schema 2020-12](schema/resource.schema.json); additional undeclared fields are rejected.

| Field | Meaning |
|---|---|
| `id` | Stable `url-` identifier derived from the canonical URL's SHA-256. |
| `name`, `kind` | Original resource name and its primary resource type. Brands and project names are not translated. |
| `website`, `canonical_url` | Direct resource link and deduplication identity. Repository file/branch aliases and tracking fragments are collapsed by the source registry. |
| `detail_url` | The public BotsTest detail page for the representative record. |
| `summary.en`, `summary.zh-CN` | Short editorial descriptions of actual purpose, in English then Chinese. Not a performance, compatibility or purchasing claim. |
| `category`, `subcategories` | Stable taxonomy slugs with bilingual display names. |
| `primary_subcategory` | One display bucket to avoid duplicate list entries; other subcategories remain discoverable in structured data. `null` means no precise bucket. |
| `classification` | Imported classification or an explicit editorial correction, original category/subcategories and bilingual reasoning. Imported subtype/capability labels are not independently certified. |
| `relevance` | `direct` for embodied AI/robotics; `supporting` for enabling tools such as physics, RL, geometry or control. |
| `editorial` | Inclusion decision, review date, method and bilingual rationale. Exclusions/pending decisions are preserved separately. |
| `access` | Historical directory label: `open`, `commercial` or `research`. It is not a price or legal permission. |
| `licenses.repository` | Observed repository license identifier and source/date, or explicit `unknown`. |
| `licenses.weights`, `.dataset`, `.hardware_design` | Separate scopes; unverified identifiers stay `null`. A repository label is never copied into these scopes. |
| `repository` | Public owner/name, primary language, archived flag and last observed push timestamp. Non-repository resources use `null`. |
| `verification` | Historical endpoint status, date and method. `new_check_performed=false` for this edition. |
| `sources` | Public discovery/provenance sources. They are not replacements for the resource's own documentation. |
| `tags`, `topics`, `applications` | Imported directory discovery facets; they do not prove support for a capability or deployment. |
| `aliases` | Public records sharing the same endpoint. Their scores are not summed. |
| `ranking` | Original internal score, raw/normalized PageRank, original quality components, representative node, algorithm and timestamp. |

Unknown is distinct from false or zero. A missing license is not permission to use a project. An archived repository is not automatically unusable, and a reachable endpoint is not an installation test. Dates retain their original timezone offsets; review date and observation date are different fields.

## Files and authority

- `data/source-snapshot.json`: public source graph and observed directory fields, used to reproduce historical scores.
- `data/url-registry.json`: the 2,042-endpoint historical candidate pool.
- `data/reviews/*.json`: one bilingual editorial decision per candidate.
- `data/classification-overrides.json`: evidence-based corrections; the source taxonomy and score remain intact.
- `data/display-names.json`: publisher name/capitalization corrections; original names remain in aliases.
- `data/resources.json`: generated included resources conforming to the schema.
- `data/decisions.json`: generated full include/exclude/needs-review ledger.
- `data/resources.csv`: portable flattened table; empty cells mean missing values, never automatic zeros.
- `data/summary.json`, `data/checksums.json`: generated counts and SHA-256 integrity manifest.

Edit source/review inputs, then run the documented build and validation commands. Do not edit generated descriptions in only one language. Adding fresh source observations requires their actual check date and method; regenerating files does not renew a check.


Company metadata preserves public portfolio association, reported year and country when available; unknown facts remain null. A portfolio year is not asserted as a verified incorporation date. Repository creation dates describe repositories, not company founding dates. New Chinese summaries retain the precise English purpose alongside Chinese taxonomy context; see PROVENANCE.md.
