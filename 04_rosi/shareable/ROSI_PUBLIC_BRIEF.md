# ROSI Public Brief

Status: `[PUBLIC BRIDGE / METHOD DESCRIPTION / NOT VALIDATION]`

ROSI, the Relational Orientation System for Information, is the computational
companion to the Relational Geometry Model document set. In its public form,
ROSI should be understood as a method for comparing candidate structures while
preserving unresolved differences instead of hiding them under consensus,
authority, or probability language.

This brief describes the shareable method layer. It is not the full ROSI engine
and does not expose private scoring, prompts, benchmarks, or source code.

## Central purpose

`[DEFINITION]`

ROSI organizes information as routes through partially resolved structures.
Each route is evaluated by what it preserves, what it closes, what residue it
leaves, and what discriminating information would resolve the remaining
indefinitions.

The system is most useful when it does not produce a final answer immediately,
but instead says:

- which route is currently strongest;
- which roots were conserved;
- which roots are missing;
- which residue remains visible;
- which information would discriminate between routes.

## Public primitives

`[DEFINITION] Latent root`

A partially resolved structure with enough identity to participate in relations
while still carrying named indefinitions.

`[DEFINITION] Indefinition`

A named ambiguity or unresolved difference. A valid indefinition must include a
closing condition: the type of information that would resolve, split, merge, or
invalidate it.

`[DEFINITION] Route`

An ordered traversal through roots and partial closures. A route is not a vote
for truth. It is a structural path whose residue can be inspected.

`[DEFINITION] Common root`

A root shared by two or more routes. Common roots couple routes: resolving one
root can affect every route passing through it.

`[DEFINITION] Projective residue`

The structural content that remains unresolved after a route is projected. It is
not a probability and not a confidence score.

`[DEFINITION] Seed`

An orientation that changes traversal priority without determining truth.

`[DEFINITION] Discriminating information`

Information that would close one or more named indefinitions. The request for
this information is often the most useful output.

## Public cycle

`[PROGRAMMATIC]`

A public ROSI cycle can be represented as:

1. Input a problem and candidate routes.
2. Extract visible roots.
3. Compare conservation of prompt roots.
4. Name missing roots and residue.
5. Identify common roots between routes.
6. Produce discriminating-information requests.
7. Return a structured output for review or a next cycle.

The browser sandbox in `04_rosi/tools/rosi-lab.html` implements a small public
toy version of this cycle. It does not run the protected ROSI core.

## Forbidden public semantics

ROSI public outputs should avoid:

- probability of truth;
- author or source authority as weight;
- majority vote as closure;
- consensus as validation;
- moral verdicts as automated classifications;
- claims of final certainty without traceable discriminants.

## Falsifiers

`[FALSIFIER]`

The public method layer fails if:

- it cannot name what information would resolve an indefinition;
- residue decreases without new information or computation;
- seeds determine results instead of only orienting traversal;
- provenance changes route score;
- consensus is treated as closure;
- a route closes without recording discriminants.

## Citation

ROSI latest metadata record:

https://zenodo.org/records/20363352

