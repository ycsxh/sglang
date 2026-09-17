# Agent guidance

## Repository boundaries

- The user's writable SGLang repository is `ycsxh/sglang` (`https://github.com/ycsxh/sglang.git`). All SGLang remote writes must target this fork.
- Upstream `sgl-project/sglang` is read-only and may be used only for source/version reference. Never operate on upstream PRs or issues, including creation, comments, reviews, labels, edits or closure. Never push, merge, create remote branches/releases, or perform any other remote write to upstream.
- All SGLang PR and issue operations must explicitly target `ycsxh/sglang`; pass `--repo ycsxh/sglang` to applicable `gh` commands. Do not rely on GitHub CLI's automatic repository selection for a fork.
- Before any remote mutation, verify the repository owner/name and the actual destination URL. Git writes must explicitly use the verified fork remote (`origin` currently points to `ycsxh/sglang`); never use `upstream` as a write destination.
