# ROSI Shareable Package

Status: `[PUBLIC BRIDGE / SHAREABLE / NOT CORE SOURCE]`

This folder contains public material that can be shared with collaborators,
reviewers, LLM assistants, or publication venues without exposing the protected
ROSI implementation.

The goal is to make the method understandable and citable while keeping the
private engine separate.

## Contents

| File | Purpose |
| --- | --- |
| `ROSI_PUBLIC_BRIEF.md` | Short public description of ROSI as a relational method. |
| `CLOSURE_NODES_BRIEF.md` | Public bridge from closure-node language to the redox paper. |
| `LLM_COLLABORATOR_PROMPT.md` | Prompt for asking another AI to critique or extend the public layer. |
| `rosi_cycle_output.schema.json` | Minimal JSON output shape for public examples. |
| `PUBLISHING_BOUNDARIES.md` | What can be published and what should remain private. |

## Safe to share

- Conceptual definitions.
- Epistemic tags and status discipline.
- Synthetic examples.
- Output schemas.
- Falsifiers and open problems.
- Zenodo links and DOI references.
- Public browser sandbox links.

## Not included

- ROSI core source code.
- Private route scoring.
- Internal prompts.
- Private datasets or benchmarks.
- API keys or provider configuration.
- Claims of physical validation beyond the cited records.
- Any automatic moral, ethical, or truth detector.

## Citation targets

- ROSI latest metadata record: https://zenodo.org/records/20363352
- Closure-Node Registration: https://zenodo.org/records/20573845
- Closure-node post-audit local companion:
  `../../03_predictions/closure_node_post_audit/`
- Navier-Stokes as relational difference redistribution:
  https://zenodo.org/records/20578888

## Suggested publication bundle

For a public submission or reviewer handoff, share these files first:

1. `ROSI_PUBLIC_BRIEF.md`
2. `CLOSURE_NODES_BRIEF.md`
3. `rosi_cycle_output.schema.json`
4. `PUBLISHING_BOUNDARIES.md`

Use `LLM_COLLABORATOR_PROMPT.md` when the receiver is another AI system.
