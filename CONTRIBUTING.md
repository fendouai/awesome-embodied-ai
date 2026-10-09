# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Suggest resources that serve a concrete embodied AI or robotics task. Include the canonical public endpoint, a concise purpose, publisher evidence, proposed category/subcategory and known license scope. Describe enabling tools as supporting resources; do not imply that they are robot-specific.

## Review rules

1. Use a primary project, documentation, dataset, laboratory or vendor source. Reject tracking links, private endpoints, forks without distinct value and unrelated bots.
2. Explain the actual task. General AI popularity, stars and payment do not establish relevance or change ranking.
3. Verify ownership and endpoint evidence. Save the true date and method; a transport error or challenge means review is needed, not confirmed removal.
4. Keep code, model weights, datasets and hardware design license scopes separate. Unknown remains unknown.
5. Write the English summary first, then the equivalent Chinese summary. Keep original project names and avoid marketing superlatives or unsupported compatibility/performance claims.
6. Update the relevant input decision, provenance and any taxonomy correction. A new URL needs a source record and registry entry before it can enter the generated list.
7. Run generation, schema/consistency checks and ranking tests. Commit generated output together with its source changes.

The current release intentionally preserves the 1,000-candidate snapshot. Extending that pool is a maintainer operation: update the snapshot/version, count assertions and documented scope together. Never silently replace a candidate to preserve an arbitrary count.

## A resource suggestion

Provide:

- Name and canonical URL.
- Specific robotics task and primary evidence URL.
- Category, subcategory and direct/supporting relevance.
- English and Chinese short summaries.
- Repository/license information with its scope and source, if known.
- What was checked, when, and what remains unknown.

Use the issue template or propose a focused pull request. Source credentials, private data and deployment configuration do not belong in this repository. Maintainers review inclusion and evidence; the build does not publish to a website or accept payments.

## Corrections and removal

Report an incorrect description, classification, ownership, license observation or inactive endpoint with evidence. Preserve the previous observation in the decision/provenance record; do not rewrite its date to appear newly checked. Excluded and pending candidates stay in the audit ledger.

Original contributions to the list and metadata compilation are offered under CC0 1.0; tooling contributions use MIT. Do not submit third-party text or assets that you cannot appropriately attribute or include.
