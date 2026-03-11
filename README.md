# omnivorous

![omni](images/omni.png)

> Turn your documents into agent-ready Markdown context.

omnivorous converts PDF, DOCX, HTML, Markdown, and plain text files into clean, structured Markdown that AI coding agents can consume directly. It handles format-specific cleanup, extracts metadata, counts tokens, and chunks documents intelligently — so your agents get the context they need without the noise.

## Install

```bash
pip install omnivorous
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install omnivorous
```

This makes the `omni` command available globally.

## Quick Start

```bash
# Generate a full agent context pack (defaults to Claude Code)
omni pack docs/ -o agent-context/

# Generate for a specific agent
omni pack docs/ --agent codex
omni pack docs/ --agent cursor

# Generate for multiple agents at once
omni pack docs/ --agent claude --agent codex --agent copilot

# Generate for all supported agents
omni pack docs/ --agent all
```

## How It Works

omnivorous processes documents through a four-stage pipeline:

1. **Convert** — Each format has a dedicated converter that produces clean Markdown. PDFs get ligature repair and header/footer removal; HTML gets nav, script, and boilerplate stripping; DOCX preserves structure while dropping styling.
2. **Extract metadata** — Page count, headings, tables, and token count are recorded as YAML frontmatter.
3. **Chunk** — Documents are split by heading structure or by token budget, keeping sections coherent and right-sized for agent context windows.
4. **Pack** — An agent instruction file, project context summary, file manifest, and chunked docs are assembled into a ready-to-use context pack.

## Supported Formats

- PDF (`.pdf`)
- Word (`.docx`)
- HTML (`.html`, `.htm`)
- Markdown (`.md`, `.markdown`)
- Plain text (`.txt`)

## Commands

All commands accept `--encoding` to select the tiktoken encoding used for token counting (default: `o200k_base`).

### `omni pack <folder>`
Generate a full agent context pack with:
- Agent instruction file (varies by target agent)
- `PROJECT_CONTEXT.md` — Documentation summary
- `manifest.json` — File manifest
- `docs/` — Converted and chunked documents

```bash
# Generate for a specific agent
omni pack docs/ --agent codex
omni pack docs/ --agent cursor

# Generate for multiple agents at once
omni pack docs/ --agent claude --agent codex --agent copilot

# Generate for all supported agents
omni pack docs/ --agent all
```

Options:
- `-o, --output`: Output directory for agent context
- `-a, --agent`: Target agent(s) — can be specified multiple times (default: `claude`)
- `--chunk-size`: Target chunk size in tokens (default: 500)
- `--chunk-by`: Strategy — `heading` or `tokens` (default: heading)
- `--encoding`: Tiktoken encoding name (default: `o200k_base`)

#### Supported Agents

| Agent | Key | Generated File |
|-------|-----|----------------|
| Claude Code | `claude` | `CLAUDE.md` |
| Codex CLI | `codex` | `AGENTS.md` |
| Cursor | `cursor` | `.cursor/rules/omnivorous.md` |
| GitHub Copilot | `copilot` | `.github/copilot-instructions.md` |
| Google Antigravity | `antigravity` | `.agent/skills/omnivorous.md` |

Use `--agent all` to generate instruction files for every supported agent at once.

### `omni ingest <folder>`
Scan a folder and convert all supported documents.

```bash
omni ingest docs/ -o output/
```

Options:
- `-o, --output`: Output directory
- `--encoding`: Tiktoken encoding name (default: `o200k_base`)

### `omni convert <file>`
Convert a single document to Markdown with YAML frontmatter.

```bash
omni convert document.pdf -o output.md

# Use a different token encoding
omni convert document.pdf --encoding cl100k_base -o output.md
```

Options:
- `-o, --output`: Output file path
- `--encoding`: Tiktoken encoding name (default: `o200k_base`)

### `omni inspect <file>`
Display document metadata: pages, headings, tables, token count, and encoding.

```bash
omni inspect document.pdf

# Use a different token encoding
omni inspect document.pdf --encoding cl100k_base
```

Options:
- `--encoding`: Tiktoken encoding name (default: `o200k_base`)

## Token Encoding

Token counts vary across models because each uses a different tokenizer. By default, omnivorous uses `o200k_base` (GPT-4o, o1, o3). You can switch to `cl100k_base` (GPT-4 / GPT-3.5) with the `--encoding` flag.

Supported encodings:
- `o200k_base` — GPT-4o, o1, o3 (default)
- `cl100k_base` — GPT-4, GPT-3.5

The encoding name is recorded in each document's metadata so downstream tools know which tokenizer was used.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/
```

## License

MIT
