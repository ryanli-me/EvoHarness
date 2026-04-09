# Findings

- Iter 1: Added pre-completion verification protocol to system_prompt — 9 tasks flipped (adaptive-rejection-sampler, cancel-async-tasks, extract-elf, make-doom-for-mips, make-mips-interpreter, protein-assembly, raman-fitting, sanitize-git-repo, torch-pipeline-parallelism). Pattern: agents were skipping test commands before calling task_complete.
