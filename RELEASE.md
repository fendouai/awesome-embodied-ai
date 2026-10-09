# Publishing this repository

[简体中文](RELEASE.zh-CN.md)

This folder is a standalone local project prepared for publication. It has not created a GitHub repository or published a release.

1. Run the README's dependency setup, build, validation and ranking tests.
2. Review the generated English and Chinese overviews, inclusion ledger and license scopes.
3. Create your chosen public repository named `awesome-embodied-ai` and upload this folder's contents, including `.github`. Keep `.venv`, caches, local credentials and unrelated site files out.
4. Set the description, homepage and suggested topics from `repository-metadata.json`. Change `publication_status` only after publication; add the real repository URL if desired.
5. Commit source inputs, review decisions, schemas, scripts and generated lists together. The supplied CI validates reproducibility and bilingual parity; it performs no automatic URL crawling or publishing.
6. Link the repository from BotsTest only after its real public URL exists. Future submission to the central Awesome index should follow its maturity and editorial rules; using the Awesome badge does not imply acceptance there.

The project needs Python 3.10+ for tooling. Generation is independent of BotsTest's application, database, payments, server and media storage. File checksums and schema checks are the release verification record; they are not fresh upstream endpoint tests.
