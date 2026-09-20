---
name: explain
description: Use when the user invokes /explain to analyze a codebase, directory, file, symbol, or feature and produce a layered explanation report from 5 parallel analysis lenses.
version: 1.0.0
metadata:
  mcpmarket-version: 1.0.0
---
# Explain Skill

Dispatch 5 parallel analysis subagents — each a domain lens (structure, behavior, domain & data, external dependencies, health & risk) — against the target scope. Collect their findings and synthesize a layered report that goes from 30-second overview to deep understanding.

## Invocation

Parse the user's `/explain` arguments to determine scope and options:

| Invocation | Scope |
|---|---|
| `/explain` | Whole project (default) |
| `/explain src/auth/` | Directory/module |
| `/explain src/auth/oauth.ts` | Single file |
| `/explain MyClassName` | Class/symbol (resolved via LSP or Grep) |
| `/explain "how does payment processing work?"` | Feature trace (natural language question) |
| `/explain --save` | Any of the above + write output to file |
| `/explain --save --path docs/onboarding/` | Custom save location |

Arguments are combinable. Examples:
- `/explain src/auth/ --save` — explain the auth module and save the report
- `/explain --save --path docs/` — explain the whole project and save to `docs/`

If the invocation is ambiguous or the argument is unrecognizable, ask the user to clarify before proceeding.

### Scope Detection Logic

1. If the argument matches an existing file path (exact match) — **single file** mode
2. If the argument matches an existing directory (exact match) — **directory** mode
3. If no exact match, search recursively via Glob (e.g., `**/Dockerfile`, `**/MyClassName.*`) — if a unique file or directory is found, use that mode. If multiple matches, list them and ask user to clarify.
4. If no file/directory match, treat as a **symbol** name (PascalCase, camelCase, snake_case, no spaces) — resolve via LSP or Grep. If multiple matches found, list them and ask user to clarify. If zero matches, report the error.
5. If the argument contains spaces or is a question — **feature trace** mode
6. If no argument — **whole project** mode

---

## Phase 1: Preparation

### 1.0 Load User Preferences

Read `shared/skill-context.md` for the full protocol. In brief:

1. Read `.claude/skill-context/preferences.md`.
   - If missing: invoke `/preferences` (streamlined mode — core questions only, then return here).
   - If found: load preferences.
2. Read `.claude/skill-context/explain.md` (if it exists) for explain-specific preferences.

**How preferences shape this skill:**

| Preference | Effect on Explain |
|---|---|
| Detail level: concise | Shorter TL;DR, fewer inline examples, top-level findings only |
| Detail level: detailed | Expanded sections, more code references, richer context |
| Style: analogy-heavy | Subagents use real-world parallels when describing architecture and flows |
| Style: visual-with-diagrams | More ASCII diagrams, expanded dependency graphs, flow charts for every workflow |
| Style: example-driven | Subagents include concrete usage examples alongside descriptions |
| Assumed knowledge: beginner | Define domain terms, explain framework conventions, expand the glossary |
| Assumed knowledge: expert | Skip obvious patterns, focus on non-obvious design decisions and gotchas |
| Explain-specific: top-down | Structure agent starts with highest-level modules, drills down |
| Explain-specific: bottom-up | Structure agent starts with leaf files, builds up to the big picture |

Pass the relevant subset of preferences to each subagent prompt in Phase 2 (append after the Guiding Principles block). Subagents receive only the preferences that affect their lens — don't dump the entire file into every prompt.

### 1.1 Language Detection

Infer the primary language from file extensions in scope:

| Extension | Language |
|---|---|
| `.ts`, `.tsx` | TypeScript |
| `.js`, `.jsx` | JavaScript |
| `.py` | Python |
| `.rs` | Rust |
| `.go` | Go |
| `.java` | Java |
| `.cs` | C# |
| `.rb` | Ruby |
| `.php` | PHP |
| `.swift` | Swift |
| `.kt` | Kotlin |
| `.c` | C |
| `.cpp`, `.cc`, `.h` | C++ |

If the codebase is polyglot, note all languages.

### 1.2 LSP Availability Check

Check if LSP tools are available for the detected language(s). If not, print a one-line suggestion:

> Tip: Install an LSP MCP server for [Language] for deeper analysis.

Then proceed without it. LSP is an accelerator, not a requirement.

### 1.3 File Gathering (non-project scopes only)

For **directory**, **single file**, and **symbol** scopes, resolve the file list before dispatch and pass it to all agents. This avoids 5 agents independently Globbing the same paths.

For **whole project** and **feature trace** scopes, agents explore freely — the scope is too broad to pre-load.

### 1.4 Scope Size Check

For **whole project**, **directory**, and **feature trace** scopes, estimate the number of source files in scope. If the count exceeds **500 source files**, warn the user:

> This project has ~[N] source files. A full analysis may take several minutes. Consider narrowing scope (e.g., `/explain src/core/`). Proceed anyway? (y/n)

Proceed only after confirmation.

### 1.5 Progress Indication

Before dispatching agents, print a brief status message:

> Analyzing [scope description] across 5 lenses...

---

## Phase 2: Parallel Analysis — 5 Lenses

Dispatch **5 subagents simultaneously** via the Agent tool — all 5 in a single response (5 parallel Agent tool calls).

**IMPORTANT:** Every Agent tool call **must** use `model: "opus"` to ensure high-quality analysis.

Each subagent receives a prompt containing:
- The resolved scope (what to analyze)
- The file list (for non-project scopes)
- The full Guiding Principles block (see below) — prepend it to every subagent prompt before dispatch
- **User preferences** (loaded in Phase 1.0) — append the relevant subset after the Guiding Principles. Format as a `## User Preferences` section with only the fields that affect that subagent's lens. For example, Agent 1 (Structure) gets explanation style and detail level; Agent 5 (Health) gets experience level and project phase.

### Guiding Principles (included in every subagent prompt)

1. **Evidence over guesswork.** Every claim about what the code does must reference a specific file, function, or config. No vague assertions.
2. **Flag uncertainty.** Distinguish what the code definitely does vs. what seems intended vs. what is unclear. Uncertainty is valid output — say "unclear" rather than speculate.
3. **Use LSP tools if available, fall back to Glob/Grep/Read.** Do not fail if LSP is unavailable.
4. **Respect scope boundaries.** Focus on the requested scope. Reference external context where necessary ("this module is called from `src/api/routes.ts`") but don't produce full analysis of unrelated modules.
5. **Don't reproduce source code.** Reference code with `file:line_number` pointers and short snippets for clarity. The user has the code — they need understanding, not a copy.
6. **ASCII graphs must follow the shared ASCII Graph Standards.** Read `references/ascii-standards.md` for the full specification. Use box-drawing characters, enforce alignment, and scale detail to project size per the adaptive rules.

### Subagent Roster

| # | Lens | Focus |
|---|---|---|
| 1 | Structure & Entry Points | Module boundaries, organization pattern, entry points, dependency graph, file conventions |
| 2 | Behavior — Key Workflows | Trace the 3 most important user-facing workflows end-to-end |
| 3 | Domain & Data | Data models, state transitions, domain glossary, storage mapping |
| 4 | External Dependencies | External services, infrastructure, integration patterns |
| 5 | Health & Risk | Hotspots, churn, debt, test signals, git archaeology, fastest path to competence |

### Subagent Prompt Templates

Each template below shows the domain-specific portion. When constructing the actual prompt, **prepend the full Guiding Principles block** (the 6-item list above) before the role line. The abbreviated instructions in each template are intentionally kept as reinforcement — the full principles block provides the authoritative version.

#### Agent 1: Structure & Entry Points

~~~
<GUIDING_PRINCIPLES>

You are a senior software architect analyzing a codebase for structural understanding.

## Instructions
1. Analyze the code at the specified scope for structural patterns.

## Scope
<SCOPE_DESCRIPTION>
<FILE_LIST if applicable>

## Your Analysis Must Cover
- **Organization pattern**: monorepo vs single project, feature-based vs layer-based folder structure
- **Module boundaries**: what are the major modules/packages and their responsibilities
- **Entry points**: main files, route registrations, CLI commands, public exports, event handlers
- **Dependency relationships**: how modules depend on each other (imports/requires graph)
- **File conventions**: naming patterns, co-location patterns (tests next to source? separate tree?)
- **Module dependency graph**: Include an ASCII graph showing how the major modules/packages depend on each other. Follow the ASCII Graph Standards. This graph will also be used by the synthesizer to compose the top-level Architecture Overview.

## Output Format
Return structured markdown with clear headings for each area above.
Include the module dependency graph near the top of your output under a `### Module Dependency Graph` heading.
Reference specific files with `file:line_number` format.
If something is unclear or you cannot determine it, say so explicitly.
~~~

#### Agent 2: Behavior — Key Workflows

~~~
<GUIDING_PRINCIPLES>

You are a senior software engineer tracing how this system actually behaves at runtime.

## Instructions
1. Identify the 3 most important user-facing workflows in the codebase.
2. Trace each end-to-end: entry point → validation → business logic → persistence → response.
3. Use LSP tools if available (especially go-to-definition and find-references), otherwise fall back to Glob/Grep/Read.
4. Flag uncertainty — distinguish "definitely" from "seems like" from "unclear."

## Scope
<SCOPE_DESCRIPTION>
<FILE_LIST if applicable>

## How to Pick the 3 Workflows
For **project and directory scopes**: prefer the most common or business-critical user-facing paths. If unclear, default to:
1. The most common "create" or "write" operation
2. The core "read" or "query" path — what users do most often
3. A mutation with side effects — something that triggers emails, webhooks, queues, or external API calls

For **single file and symbol scopes**: reframe as "the 3 most important code paths or behaviors" — these may not be user-facing workflows. Trace how this code is called, what it does, and what it calls.

If fewer than 3 distinct workflows/paths exist at this scope, trace what's available.

## For Each Workflow, Report
- The complete call chain: `file:function → file:function → ...`
- **Flow diagram**: An ASCII flow diagram showing the call chain for this workflow. Use horizontal flow for the primary path, vertical branches for error/alternate paths. Follow the ASCII Graph Standards.
- Key business logic decisions (branches that matter)
- External dependencies touched (DB queries, API calls, queue publishes)
- Error handling approach at each step
- Side effects (what else happens beyond the primary response)
- Implicit assumptions or potential failure points you notice

## Output Format
Return structured markdown with a section per workflow.
Reference specific files with `file:line_number` format.
If you cannot fully trace a flow, trace what you can and note where you lost the thread.
~~~

#### Agent 3: Domain & Data

~~~
<GUIDING_PRINCIPLES>

You are a senior domain analyst mapping the data model and business concepts of a codebase.

## Instructions
1. Identify the core domain objects, their relationships, and how data flows through the system.
2. Use LSP tools if available, otherwise fall back to Glob/Grep/Read.
3. Flag uncertainty — distinguish "definitely" from "seems like" from "unclear."

## Scope
<SCOPE_DESCRIPTION>
<FILE_LIST if applicable>

## Your Analysis Must Cover
- **Core domain objects**: models, schemas, entities, types — what are the important "things" in this system
- **Relationships**: how domain objects relate to each other (one-to-many, references, composition)
- **State transitions**: lifecycle states and how objects move through them (e.g., Order: draft → confirmed → shipped → delivered)
- **Domain glossary**: map code names to business concepts — especially non-obvious ones where the code name differs from the business term
- **Storage mapping**: what data lives where (relational DB, document store, cache, queue, file system, in-memory)
- **Validation and constraints**: invariants, required fields, business rules enforced at the data layer
- **State transition diagram** (optional): If domain objects have clear lifecycle states (e.g., Order: draft → confirmed → shipped → delivered), include an ASCII state transition diagram. Follow the ASCII Graph Standards. If no clear state transitions exist, skip — don't force a graph where prose is clearer.

## Output Format
Return structured markdown with clear headings for each area above.
Reference specific files with `file:line_number` format.
If the domain is unclear or naming is ambiguous, note your best interpretation and flag the uncertainty.
~~~

#### Agent 4: External Dependencies

~~~
<GUIDING_PRINCIPLES>

You are a senior infrastructure engineer mapping how this system integrates with the outside world.

## Instructions
1. Identify everything this system communicates with externally.
2. Use LSP tools if available, otherwise fall back to Glob/Grep/Read.
3. Flag uncertainty — distinguish "definitely" from "seems like" from "unclear."

## Scope
<SCOPE_DESCRIPTION>
<FILE_LIST if applicable>

## Your Analysis Must Cover
- **External services**: databases, caches, message queues, third-party APIs, auth providers, CDNs, monitoring, logging services, file storage
- **Integration points**: for each external dependency — what it's used for, where in the code it's called (file:line_number), communication pattern (REST, gRPC, SDK, raw TCP, etc.)
- **Infrastructure config**: Terraform, CloudFormation, Docker, Kubernetes, Helm charts — what infrastructure is defined in code
- **Sync vs async**: which integrations are synchronous (blocking) vs asynchronous (queues, events, webhooks)
- **Resilience patterns**: retry logic, circuit breakers, fallbacks, timeouts — or their absence
- **Integration map**: Include an ASCII integration map with the system at center and external services arranged around it. Follow the ASCII Graph Standards.

## Output Format
Return structured markdown.
Reference specific files with `file:line_number` format.
If you find config references to services but cannot determine their purpose, note them as "unclear."
~~~

#### Agent 5: Health & Risk

~~~
<GUIDING_PRINCIPLES>

You are a senior tech lead assessing the health, risk profile, and onboarding path of a codebase.

## Instructions
1. Assess code health signals, identify risks, and recommend the fastest path to competence.
2. Use LSP tools if available, otherwise fall back to Glob/Grep/Read.
3. Flag uncertainty — distinguish "definitely" from "seems like" from "unclear."

## Scope
<SCOPE_DESCRIPTION>
<FILE_LIST if applicable>

## Your Analysis Must Cover

### Hotspots & Complexity
- Largest files by line count (top 5)
- Files with the most imports/dependencies
- God objects — classes/modules with too many responsibilities

### Git Archaeology (skip if no git history)
- Recent commit narrative (last 30 commits)
- Active contributors and their areas of focus
- Most-churned files (most frequently changed in last 6 months)

### Test Signals
- Test framework(s) in use
- Test file count vs source file count (rough coverage signal)
- Types of tests present (unit, integration, e2e, snapshot)
- Important behavior that appears untested

### Technical Debt
- TODOs/FIXMEs/HACKs grouped by theme (not exhaustive listing)
- Dead code signals (unused exports, commented-out blocks)
- Dependency health signals (outdated lock files, known patterns of vulnerable packages)

### Fastest Path to Competence
- The next 5 files a new developer should read (in order, with why)
- The next 3 workflows to trace after reading the initial explanation
- Areas to avoid touching without deeper understanding first

## No Value Judgments on Architecture
Describe health signals and risks factually. Do not editorialize about whether the architecture is "good" or "bad" — that is not your lens. Focus on: what's fragile, what's unclear, what's risky to change, and what's the fastest way to get oriented.

## Output Format
Return structured markdown with clear headings for each area above.
Reference specific files with `file:line_number` format.
~~~

---

## Phase 3: Synthesis

After all agents complete (or after handling failures — see Graceful Degradation), synthesize findings into the final layered report.

### Synthesis Rules

1. **TL;DR generation**: Derive from all agents — touch structure, behavior, and health. 3 sentences maximum.
2. **Orientation Cheat Sheet**: Populated by pulling specific answers from whichever agent found them. If an answer can't be determined, write "unclear" rather than guessing. Omitted for single file and symbol scopes.
3. **Deduplication**: If multiple agents mention the same file or concept, merge into the most relevant section. Don't repeat.
4. **Uncertainty preservation**: If an agent flagged something as uncertain, preserve that flag in the synthesis. Don't silently upgrade "seems like" to "definitely."
5. **Section ordering**: Always present in the fixed order below — the layering is intentional (overview → specifics → actionable).
6. **Graph composition**: Combine Agent 1 and Agent 4 graphs into the top-level Architecture Overview. Deduplicate shared nodes (e.g., if both agents reference "Postgres," show it once). Internal modules at center, external services at periphery. Preserve agent-produced inline graphs in their respective sections as-is — only the top-level overview is a synthesizer composition.

### Output Structure

~~~
## TL;DR
3 sentences. What it is, why it exists, how it's organized.

## Orientation Cheat Sheet
(omitted for single file and symbol scopes)

| Question | Answer |
|---|---|
| Tech stack | ... |
| Primary language / framework | ... |
| Where are the entry points? | path/to/file |
| Where is the business logic? | path/to/dir |
| Where is the data model? | path/to/file |
| Where are the tests? | path/to/dir |
| Where is the config / env? | path/to/file |
| Who are the active contributors? | names from git |
| What's the most fragile part? | from Health agent |
| What should I not touch without care? | from Health agent |

## Architecture Overview
(omitted for single file and symbol scopes)
[Composed from Agent 1 module dependency graph + Agent 4 integration map.
Internal modules at center, external services at periphery. Deduplicate shared nodes.]

## Structure & Entry Points
[Agent 1 findings, edited for flow and deduplication — includes module dependency graph inline]

## Behavior — Key Workflows
[Agent 2 findings — the 3 traced flows with call chains and flow diagrams]

## Domain & Data
[Agent 3 findings — models, glossary, state transitions, state transition diagram if applicable]

## External Dependencies
[Agent 4 findings — includes integration map inline]

## Health & Risk
[Agent 5 findings — hotspots, debt, uncertainty]

## Fastest Path to Competence
[Extracted from Agent 5]
- Next 5 files to read
- Next 3 workflows to trace
- Areas to avoid touching without understanding first
~~~

When `--save` is not present, present the full report directly in the conversation.

---

## Phase 4: Save (optional)

Triggered only when `--save` is present.

### Default save location
`docs/explain/<scope-name>.md`

Scope name derivation:
- Whole project → `project.md`
- Directory `src/auth/` → `src-auth.md`
- Single file `src/auth/oauth.ts` → `src-auth-oauth-ts.md`
- Symbol `MyClassName` → `myclassname.md`
- Feature trace → extract 2-4 key nouns from the question, kebab-case them (e.g., "how does payment processing work?" → `payment-processing.md`)

### Custom save location
`--path <dir>` overrides the default directory. The filename derivation remains the same.

### Save behavior
1. Write the full synthesized report to the resolved file path
2. Print to terminal: a brief summary (TL;DR + Cheat Sheet only) followed by: `Full report saved to <path>`

---

## Graceful Degradation

| Situation | Behavior |
|---|---|
| 1-2 agents fail or time out | Synthesize from successful agents. Note which lenses are missing in the report. Offer to retry failed agents. |
| All agents fail | Report the failure. Suggest retrying or narrowing scope. |
| No git history available | Health & Risk agent skips git archaeology, notes "no git history available" |
| Empty/minimal repo (< 5 source files) | Skip parallel dispatch entirely. Do a single-pass direct analysis — 5 agents for a trivial codebase is wasteful. |
| Massive repo (1000+ files at scope) | Agents focus on the most significant files. Structure maps top 2 levels. Behavior traces 3 flows without exhaustive coverage. Health focuses on top hotspots. |
| Binary/generated files in scope | Skip and note their presence |
| LSP unavailable | Agents fall back to Glob/Grep/Read. No degradation in output structure, only in precision of navigation. |

---

## ASCII Graph Standards

Read `references/ascii-standards.md` for the full box-drawing character set,
alignment rules, adaptive detail rules, and graph types per section. These
standards are prepended to every subagent prompt that produces graphs.

---

## Guardrails

1. **Evidence over guesswork.** Every claim references a specific file, function, or config.
2. **Uncertainty is valid output.** Agents say "unclear" rather than speculate.
3. **No value judgments in non-Health lenses.** Agents 1-4 describe and explain. Only Agent 5 assesses quality and risk.
4. **Respect scope boundaries.** Focus on the requested scope. Reference but don't fully analyze unrelated modules.
5. **Don't reproduce source code.** Use `file:line_number` pointers and short snippets. The user has the code.

---

## Error Handling

| Error | Behavior |
|---|---|
| File path argument does not exist | Report: "File not found: `<path>`. Check the path and try again." |
| Directory argument does not exist | Report: "Directory not found: `<path>`. Check the path and try again." |
| Symbol argument matches zero results | Report: "No matches found for `<symbol>`. Try a different name or use `/explain <file-path>` to target a specific file." |
| Symbol argument matches too many results (>10) | List the first 10 matches and ask user to narrow the scope. |
| Unrecognized flags | Report: "Unrecognized flag: `<flag>`. Supported flags: `--save`, `--path <dir>`." |
| `--path` points to a non-existent directory | Create the directory, or report if creation fails. |
| `--path` points to a file, not a directory | Report: "`<path>` is a file, not a directory. `--path` must point to a directory." |
| Permission error reading files | Skip unreadable files, note them in the report as "skipped (permission denied)." |
