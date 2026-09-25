# PizProbe — Library for SPM Image Processing and Analysis

## What this is

A Python toolkit for SPM image processing and analysis, developed in the
LANES research group at EPFL. The Python package name is `pizprobe` (no accent).
Maintained by one person (Brandon), but written to become a complementary analysis
workflow for the group (~15 people max). Errors, defaults, and docstrings must be
maintained and clear. This is to be a shared-library, not a personal script.

## Current goal
The goal of the library is to remain small and do the bare minimum of what is required.
A lot of the image processing and analysis can be done with the software that comes with the AFM -- no need to reinvent the wheel.
At present, I am slowly building out the library with the features required.

## Design rules

The argument for each, with worked examples, is in `dev/design-principles.md`.

**Corrections are opt-in.** The researcher decides whether a further correction is
warranted; the package never decides for them.

**Parameters earn their place.** A function exposes the minimum set its callers cannot
readily supply themselves.

**Reuse before adding.** Search for the concept before
writing it — if a helper is wanted in two modules, the first one is in the wrong place;
move it and import. Prefer composing an existing entry point over adding a
near-duplicate.

## Working style

Changes are made **one at a time**. Report adjacent problems found along the way rather
than fixing them unasked. For physics or analysis judgment calls, state the mechanism
and ask — don't pick a default.

**Depth of explanation follows the domain.** The same person is expert on one side of a
file and new to the other; calibrate per topic, and ask rather than assume.

- *Physics, optics, analysis maths* — full speed. This is Brandon's field.
- *Everyday programming* — competent. Don't explain syntax, control flow, NumPy
  indexing, or what a dataclass is.
- *Library design and long-term maintenance* — the real gap. Designing signatures other people will depend on, deprecation and
  versioning, packaging and dependency resolution, release process, and CI / GitHub
  Actions especially. 
  Here: slow down, say what each piece does and why before adding
  it, go a step at a time, and aim for him being able to debug it himself afterwards
  rather than for a working config landing in one move.

## Terminal replies

The output message in the terminal to the user must use simple, plain, and clear
English. Use short sentences and everyday words. Code blocks, facts, names, numbers, and
file paths must be unchanged. Avoid technical and coding jargon. If you reference a
function or variable in your response, be explicit and remind the user what it does and
what it's for. Avoid quick and snappy sentences which are unclear ("it's a real defect"
or "genuinely good" or "the thing that matters most" or "silently/quietly important").
These phrases are too vague. Do not editorialise your output.

**Close with the action needed.** End every reply by saying what the user has to do.
Keep it to the last line or two, and never bury it in the middle of the reply. If
nothing is needed from them, say that. The body of the reply can be as long as the task
warrants, with the reasoning, the proposed fix, and any supporting detail. The closing
line only has to state plainly what is being asked of the user, or that nothing is.

This governs the reply text only. It does not change how code, docstrings, comments, or
documents are written — those follow the rules above.
