# APS360 Project — Domain-Specific GRPO for Mathematical Reasoning

This project investigates whether **Group Relative Policy Optimization (GRPO)** applied directly to a base LLM (a "Zero RL" setup) can produce a stronger mathematical reasoner when the training signal is restricted to a single domain.

We filter the [WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified) dataset used by the General Reasoner paper down to mathematics-only questions, then run standard LLM-as-a-judge GRPO training using the General Reasoner verifier as the judge.

**Research question:** Can a math-only GRPO training signal produce a model that outperforms its cross-domain–trained counterpart on math benchmarks (MMLU-Pro, BBEH), and how much out-of-domain reasoning capability is sacrificed in exchange?

**Hypothesis:** Concentrating the reward signal on a single domain reduces noise from semantically dissimilar problems, allowing the RL policy to develop deeper, more consistent mathematical reasoning strategies.
