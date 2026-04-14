# omnivorous

![omni](images/omni.png)

> Turn your documents into agent-ready Markdown context.

omnivorous converts PDF, DOCX, HTML, Markdown, and plain text files into clean, structured Markdown that AI coding agents can consume directly. It handles format-specific cleanup, extracts metadata, counts tokens, chunks documents intelligently, and builds navigation and relationship files that help agents read less and find the right context faster.

## Install

Requirements:
- Python 3.10+

Use `pip` if you are installing inside a virtual environment or project environment:

```bash
pip install omnivorous
```

Use `pipx` if you want a global CLI install:

Install [pipx](https://pipx.pypa.io/stable/) first, then run:

```bash
pipx install omnivorous
```

Use `uv tool` if you already use `uv` and want a global CLI install:

Install [uv](https://docs.astral.sh/uv/) first, then run:

```bash
uv tool install omnivorous
```

Verify the installation:

```bash
omni --help
```

All three install commands include scientific PDF extraction and local semantic relationships.
The first `omni <folder>` run may download the local multilingual embedding model, so it can take longer and requires network access unless the model is already cached.

## Quick Start

```bash
# Generate a full agent context pack (defaults to Claude Code)
omni docs/ -o agent-context/

# Generate for a specific agent
omni docs/ --agent codex
omni docs/ --agent cursor
```

## How It Works

omnivorous processes documents through a three-stage pipeline:

1. **Convert** — Each format has a dedicated converter that produces clean Markdown. PDFs use pymupdf4llm for accurate layout extraction with ligature repair and header/footer removal (or marker-pdf in `--mode scientific` for LaTeX formula reconstruction); HTML gets nav, script, and boilerplate stripping; DOCX preserves structure while dropping styling.
   When processing folders, omnivorous converts files in parallel by default where it is safe to do so.
2. **Extract metadata** — Page count, headings, tables, and token count are recorded as YAML frontmatter.
3. **Pack** — Agent instruction files, a project context map, a chunk-aware manifest, chunked docs, full converted docs, and deterministic cross-document relationship hints are assembled into a ready-to-use context pack. Relationships combine explicit references like file paths, IDs, and section numbers with lexical similarity, and always add local embeddings without relying on third-party LLM APIs.

## Supported Formats

- PDF (`.pdf`)
- Word (`.docx`)
- HTML (`.html`, `.htm`)
- Markdown (`.md`, `.markdown`)
- Plain text (`.txt`)

## CLI

omnivorous exposes one CLI workflow:

```bash
omni <folder> [options]
```

It generates a full agent context pack with:
- Agent instruction file (varies by target agent)
- `PROJECT_CONTEXT.md` — Documentation map, navigation hints, and cross-document bridges
- `manifest.json` — Chunk-aware file manifest with hard-link, lexical, and semantic relationship metadata
- `docs/chunks/` — Focused context for agents
- `docs/full/` — Full converted documents for fallback reading

Semantic relationships are always generated. The first pack run may download the fixed local embedding model once, then reuse it from the user cache on later runs.

Options:
- `-o, --output`: Output directory for agent context
- `-a, --agent`: Target agent(s) — can be specified multiple times (default: `claude`)
- `-m, --mode`: PDF conversion mode — `fast` (default) or `scientific` (LaTeX formula extraction)
- `--chunk-size`: Target chunk size in tokens (default: 500)
- `--chunk-by`: Strategy — `heading` or `tokens` (default: heading)

#### Supported Agents

| Agent | Key | Generated File |
|-------|-----|----------------|
| Claude Code | `claude` | `CLAUDE.md` |
| Codex  | `codex` | `AGENTS.md` |
| Cursor | `cursor` | `.cursor/rules/omnivorous.md` |
| GitHub Copilot | `copilot` | `.github/copilot-instructions.md` |
| Google Antigravity | `antigravity` | `.agent/skills/omnivorous.md` |

Use `--agent all` to generate instruction files for every supported agent at once.

## PDF Conversion Modes

omnivorous supports two PDF conversion engines, selected via the `--mode` / `-m` flag:

| Mode | Use case | ML required |
|------|----------|-------------|
| `fast` (default) | General documents — accurate layout, tables, ligature repair, header/footer removal | No |
| `scientific` | Research papers — LaTeX formula reconstruction (`$...$`, `$$...$$`), advanced layout analysis | Yes (lightweight, not a VLM) |

## Development

[uv](https://docs.astral.sh/uv/) is required for development.

### Setup

```bash
uv sync --extra dev
```

This installs all runtime dependencies plus dev tools (pytest, pytest-cov, ruff).

### Running Tests

```bash
uv run pytest                              # Run full test suite
uv run pytest tests/test_converters/test_pdf.py  # Run a specific test module
uv run pytest -v                           # Verbose output
uv run pytest --cov=src/omnivorous         # With coverage report
```

Test fixtures live in `tests/fixtures/`.

### Linting

```bash
uv run ruff check src/                     # Check for lint issues
uv run ruff check src/ --fix               # Auto-fix lint issues
```

### CI

CI runs tests, linting, and builds across Python 3.10–3.13 on Ubuntu, and also runs package-install smoke checks for built artifacts on Ubuntu, macOS, and Windows on every push to `main` and on pull requests.

## License

MIT
