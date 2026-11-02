## What this changes

<!-- A sentence or two. Link the issue it addresses, if there is one. -->

## Why

<!-- What problem does it solve, or what does a learner gain? -->

## How I tested it

<!-- Commands you ran and what you saw. For anything visible, say that you looked at it in the browser. -->

## Checklist

- [ ] `cargo test --workspace` passes, with no new warnings
- [ ] I added or updated tests for the change (for a new model operation: a finite-difference gradient check)
- [ ] I ran the relevant browser tests (`scripts/run-e2e.mjs`) and looked at the result
- [ ] For textbook changes: it works with the keyboard and on a phone-width screen, in light and dark themes
- [ ] I did not commit generated or downloaded files (`target/`, `data/`, `runs/`, `web/pkg/`, `web/data/`)
- [ ] I updated the README or textbook if behaviour or instructions changed
