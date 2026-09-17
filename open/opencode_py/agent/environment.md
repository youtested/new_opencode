# Environment block — one line of facts about this run

Template with `{placeholders}`, filled at runtime. Keep it to one line.

```
Model: {provider}/{model} | dir: {directory} | root: {worktree} | git: {git} | {platform} | {date}
```

- `{provider}` / `{model}` — the active lane, e.g. opencode/muse-spark
- `{directory}` — where the prompt was sent from
- `{worktree}` — workspace root (git top level, else the directory)
- `{git}` — yes/no
- `{platform}` — Linux/Darwin/Windows
- `{date}` — today, YYYY-MM-DD
