# ASCII Graph Standards

A shared visual vocabulary for all ASCII diagrams in the report. These standards are prepended to every subagent prompt that produces graphs.

## Box Characters

```
┌───────┐
│ Label │
└───────┘
```

## Arrow Conventions

| Symbol | Meaning |
|---|---|
| `────▶` | Horizontal flow (left-to-right) |
| `◀────` | Reverse horizontal flow |
| `│` / `▼` / `▲` | Vertical connection / downward / upward |
| `───` | Plain connection (no direction) |
| `┬` | Top branch point (connects down) |
| `┴` | Bottom branch point (connects up) |

## Alignment Rules

- Box internal width = longest label + 2 (1 space padding each side)
- Vertical connectors (`│`, `▼`, `▲`) must align to the exact center character of the `┬` or `┴` they connect to
- All boxes in the same row share the same height
- Minimum 3 characters horizontal gap between adjacent boxes
- Verify alignment before finalizing — count characters explicitly

## Adaptive Detail Rules

| Source files in scope | Top-level overview | Inline diagrams |
|---|---|---|
| < 20 files | Show all modules, entry points, and data stores | Full detail per section |
| 20-100 files | Major modules + key entry points (up to ~12 boxes) | Moderate detail |
| 100-500 files | High-level layers/modules only (up to ~8 boxes) | Key paths only |
| 500+ files | Top-level architectural layers (~5 boxes) | Abbreviated |

## Graph Types by Section

| Section | Graph type | What it shows |
|---|---|---|
| Architecture Overview (top-level) | Dependency/layer graph | Modules, data stores, external services, flow between them |
| Structure & Entry Points | Module dependency graph | How modules import/depend on each other |
| Behavior — Key Workflows | Flow/sequence diagram | Call chain per workflow, with branching |
| Domain & Data | State transition diagram (optional) | Lifecycle states if they exist |
| External Dependencies | Integration map | System at center, external services around it |
