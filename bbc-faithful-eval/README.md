# faithful-cuts

## Project Overview

**Research question.** Vision-language models can now generate rich natural-language
descriptions of video at scale, but these descriptions are not always faithful to
the source video — they hallucinate objects, actions, relationships, and events
that are plausible in language but unsupported by the visual content.
[FIFA](https://aclanthology.org/2026.findings-acl.555.pdf) (Jing et al. 2026)
proposes a fact-level faithfulness metric for evaluating such descriptions, but
it was developed and validated on ~15-second single-shot web clips (MSR-VTT).

This project asks: **can FIFA reliably detect hallucinations in vision-language
model descriptions of long-form cinematic video?**

**Approach.** We apply the full FIFA pipeline (fact extraction → question
generation → dependency graph → VQA verification) to 670 scenes from the BBC
scene-segmentation dataset (mean length ~48 seconds), and evaluate its
behavior along three goals from the project proposal:

1. **Goal 1 — Establish a FIFA benchmark on cinematic content.** Characterize
   the distribution of fact-level and description-level faithfulness scores on
   BBC content to see whether the methodology transfers from short web clips.
2. **Goal 2 — Evaluate against controlled hallucinations.** Inject controlled
   factual errors into scene descriptions and measure whether FIFA assigns
   lower faithfulness scores to the corrupted versions.
3. **Goal 3 — Validate against a small human-evaluated subset.** Manually
   evaluate a small sample and compare human judgments to FIFA scores.
   *(Future work — not yet run.)*

**Headline finding.** The pipeline works on long-form cinematic content in
general (baseline mean FIFA score 0.911 across 253 stratified BBC scenes; 72%
detection rate on controlled corruptions), but shows one clear limitation:
**it cannot reliably detect when the order of events is wrong.** Four
independent measurements converge on this — see the `# Conclusion` section
at the end of the notebook.

## Notebook Table of Contents

`CSED504_Faithful_Cuts.ipynb` runs top-to-bottom. Section-level structure:

- **# Setup** — imports, `cleanup_gpu` helper
  - ### Download Data — BBC videos + GT from Google Drive
- **# Extract Scenes** — carve BBC videos into per-scene MP4s using ground-truth boundaries
- **# Generate Scene Descriptions** — Qwen3-VL generates one natural-language description per scene
  - ### Model Setup
- **# FIFA Metadata Extraction** — the three FIFA pre-verification stages
  - ### Model Setup — GLM-4.7-Flash / Qwen3-32B / OpenAI backends via `llm.py`
  - ### Extract Facts — decompose each description into atomic tuples
    - #### Fact-extraction analysis — facts per scene, DSG category distribution, subcategory breakdown
    - #### Analysis Conclusions
  - ### Generate Questions — rewrite each fact as a yes/no VQA question
    - #### Question-generation analysis — coverage, yes/no shape, question length by category
    - #### Analysis Conclusions
  - ### Dependency Graph Generation — build the Spatio-Temporal Semantic Dependency Graph (STSDG)
    - #### Dependency-graph analysis — root rate, DAG validity, category root rate, event fan-out
    - #### Analysis Conclusions
  - ### Visualize Graph — render one scene's STSDG
- **## Controlled Hallucination Injection** — Goal 2 setup
  - ### Overview
  - ### Injection-readiness table — per-tier expected feasibility
- **# Verify Facts** — Goal 1 + Goal 2 results
  - ### Overview
  - #### Goal 2 — score drops (per injection type, per DSG landing category)
  - #### Goal 1 — baseline faithfulness distribution (per-scene + per-episode)
- **# Conclusion** — summary of findings against the proposal
  - ## What we set out to investigate
  - ## Pipeline performance on cinematic content
  - ## Sensitivity to controlled hallucinations
  - ## The temporal-modeling limitation

## Supporting modules

- `parse.py` — parse LLM outputs (tuples, questions, dependencies) and validate DSGs
- `graph.py` — build and render STSDG graphs with `networkx` + `matplotlib`
- `llm.py` — pluggable LLM backends (`GLM47FlashLLM`, `Qwen3LLM`, `OpenaiLLM`) with a shared `completion(prompt, config)` interface
- `utils_video2text.py` — TIFA160 in-context examples and prompt-assembly helpers (from the FIFA reference implementation)
- `video_context_examples_0417.csv` — TIFA160 in-context examples used for FIFA's few-shot prompts

## Running in Colab

Open the notebook via the Colab badge at the top of `CSED504_Faithful_Cuts.ipynb`.
The first code cell clones this repo into the Colab session and adds the cloned
directory to `sys.path` so `import utils_video2text` / `import parse` resolve
against the checked-out files. The current working directory stays at `/content`;
the TIFA160 CSV is loaded via a `__file__`-relative path inside
`utils_video2text.py`, so nothing depends on `%cd`-ing into the clone.

