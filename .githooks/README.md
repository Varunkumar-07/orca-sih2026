# Pre-commit secret scanner

`pre-commit` in this directory blocks `git commit` if the staged changes look
like they contain credentials — this exists because `.netrc` and
`.copernicusmarine-credentials` were accidentally committed in plaintext
earlier in this project (later purged from history). It has no external
dependencies (no gitleaks/git-secrets install needed) and only scans staged
changes, so it stays fast.

## What it blocks

- **Filenames:** `.netrc`, `.copernicusmarine-credentials`, `.env` (but not
  `.env.example`/`.env.template`/`.env.sample`), `.pem`, `.p12`, `.pfx`,
  `id_rsa`, `id_ed25519`.
- **Known key formats in staged content:** Groq (`gsk_...`), OpenAI
  (`sk-...`), GitHub PATs (`ghp_...`), AWS access keys (`AKIA...`), Slack
  tokens (`xox...`), PEM private key blocks.
- **Generic `password`/`secret`/`token`/`api_key` assignments** in staged
  content — with an allowlist for obvious placeholders
  (`your-...-here`, `<...>`, `example`, `changeme`, `REDACTED`, etc.) and for
  normal "read from env" code (`os.getenv(...)`, `process.env...`, etc.),
  so it doesn't flag legitimate code that reads a key without hardcoding one.

## Activating it (once per checkout)

Git hooks aren't enabled automatically by `git clone`, for security reasons —
each checkout needs to opt in once:

```bash
git config core.hooksPath .githooks
```

This repo already has it set locally. If you clone this repo fresh elsewhere,
run that command once and every commit from then on is checked.

## If it blocks something that isn't actually a secret

Either fix the pattern in `pre-commit` (prefer this — it means the rule is
too broad and will misfire again), or bypass a single commit with
`git commit --no-verify` (not recommended — only use this once you're sure).
