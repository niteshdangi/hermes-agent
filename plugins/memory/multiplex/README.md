# multiplex

Memory provider that fans out to multiple child providers (mem0 + honcho by
default), bypassing Hermes' one-active-provider rule by being itself the only
registered provider.

## Config

```yaml
# ~/.hermes/config.yaml
memory:
  provider: multiplex
  multiplex:
    children: [mem0, honcho]
```

Each child is configured exactly as it would be standalone (`mem0.json`,
`honcho.json`, `MEM0_*` / `HONCHO_*` env vars). Children that aren't
configured are skipped at init — multiplex stays alive as long as ≥1 child is
available.

## Behavior

- **system prompt / prefetch / sync_turn / on_*** hooks fan out to every
  active child in registration order.
- **tool schemas** are unioned: each child contributes its own tools
  (`mem0_search`, `honcho_search`, ...). Tool calls route back to the
  originating child.
- **search()** queries all children in parallel, merges results, dedupes by
  `id` (or text hash), and sorts by score descending.
- **add()** fans out best-effort; per-child failures log + continue.

## Notes

- Score normalization is left to the children. mem0 returns cosine [0, 1];
  honcho's score semantics differ. Cross-child ordering is heuristic.
- This plugin imports child provider classes directly to bypass the registry's
  single-active-provider gate.
