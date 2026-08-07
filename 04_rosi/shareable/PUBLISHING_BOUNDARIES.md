# Publishing Boundaries for ROSI Material

Status: `[PUBLIC GOVERNANCE / PRACTICAL BOUNDARY]`

This document states what can be included in public publications and what
should remain private until a separate release decision is made.

## Publishable now

- Public definitions of ROSI primitives.
- Epistemic tags and falsifiers.
- Synthetic examples.
- Public browser sandbox descriptions.
- Output schemas.
- Zenodo links and DOI references.
- Closure-node public summaries.
- Limits, failures, and open derivations.

## Do not publish yet

- Full ROSI source code.
- Private route scoring.
- Internal prompts.
- API orchestration details.
- Provider keys or configuration.
- Private benchmarks.
- Non-public datasets.
- Exact heuristics used for protected route selection.
- Any claim that ROSI detects truth, morality, harm, or intent.

## Publication rule

`[RULE]`

If a piece of material lets an outside user reproduce the protected core
behavior rather than understand the public method, keep it private.

If a piece of material explains the method, status, boundaries, or falsifiers
without exposing the protected engine, it can be shared.

## Suggested wording

Use:

> ROSI is presented here as a relational method for preserving unresolved
> difference and naming discriminating information. The protected implementation
> is not included in this publication.

Avoid:

> ROSI proves the model.

Avoid:

> ROSI detects truth.

Avoid:

> ROSI automates moral discernment.

## Patent and disclosure note

Public release establishes authorship and timestamp, but it may also count as
public disclosure. If a module is potentially patentable or commercially
sensitive, do not publish implementation details before legal review.

