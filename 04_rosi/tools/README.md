# ROSI Tools

Reserved for browser demos, notebooks, or small reproducible examples that
connect Relational Geometry documents to the ROSI implementation.

## Current public sandbox

- `rosi-lab.html` - client-side route comparison sandbox.

The current sandbox is intentionally small:

- it runs fully in the browser;
- it does not upload user data;
- it does not expose the protected ROSI engine;
- it can export consented JSON examples for later curation.

Next stage: connect the same UI to a private `/api/rosi/analyze` endpoint that
runs the protected ROSI core and stores only consented, anonymized examples.

## Planned interface

- LLM conciliation interface - compare model answers, preserve visible
  residues, and avoid treating consensus as truth.
