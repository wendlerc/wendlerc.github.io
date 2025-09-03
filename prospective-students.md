# Prospective students and collaborators
**Chris Wendler, 09/02/2025**


I am always looking for great students and collaborators and am happy if you reach out to me directly to ask for collaboration opportunities. **However, due to the recent high demand, I am not going to respond to requests anymore unless they contain a work sample (spend ~2–8 hours to perform some experiments and write a short report) on one of the topics or models listed below (e.g., the [visual interpretability agent](./resources/tasks/agent.html){:target="_blank"} would be a good starting point).** *If you reach out to me please use my Northeastern email address and briefly introduce yourself. In particular, indicate whether you are affiliated with Northeastern or something else and the scope of the collaboration.* For the work sample, please include a short summary about how you allocated your time (this part does not have to be very detailed).

---

## Current research directions

Currently, I work on the following models and broad research directions:

- **SDXL, FLUX:** How do text-to-image models generate images? What are the representations learned, how do they interact with each other? How to surface the purely visual features? Explore the combination of diffusion inversion and SAE features. [[prior work]](https://sdxl-unbox.epfl.ch/){:target="_blank"}
- **Multimodal interpretability agent:** As we are scaling our SDXL/FLUX interpretability efforts, the need for automated evaluation methods becomes apparent. A good first step towards this direction will be to create interpretability agents powered by visual-language models and equip them with tools that facilitate the analysis of visual features inside of SDXL/FLUX as well as interventions. [[related work]](https://arxiv.org/abs/2404.14394){:target="_blank"} [[our prior work and resources]](https://sdxl-unbox.epfl.ch/){:target="_blank"} [[take-home task]](./resources/tasks/agent.html)
- **World Models:** Understand representations learned by world models. Do they form persistent representations without explicit supervision? Do they learn something like intuitive physics (i.e., physics at the level of abstraction useful for modeling the training data)? How can we distill a foundation model's general world knowledge into an action/control conditioned world model? [[resources]](https://github.com/Wayfarer-Labs/owl-wms/issues/17){:target="_blank"}
- **ESMFold:** How does ESMFold fold a protein? [[related work 1]](https://www.science.org/doi/10.1126/science.ade2574){:target="_blank"} [[related work 2]](https://www.reticular.ai/research/interpretable-protein-structure-prediction){:target="_blank"}
- **R1:** What are the elements of reasoning learned by R1 and what’s the difference between reasoning and base model? [[resources]](https://arborproject.github.io/){:target="_blank"}
- **OLMO2:** Training dynamics of multilingual representations and instruction following. How do multilingual representations form? Is there something like an instruction following circuit, and how/when does it form? [[prior work]](./resources/llm.bib){:target="_blank"}

To understand the context a bit better you may find it useful to have a look at my recent [[research plan]](./resources/research-plan.pdf){:target="_blank"}.

---

## Additional directions of interest

Here are some additional research directions that I am interested in:

- **How does a model change during finetuning?** Many works find circuit-reuse and representation-reuse between base and finetuned model. Can we also zoom in on the changes?
- **Finetuning-based interpretability:** e.g., by tuning an “interpretability wrapper” (something that facilitates interpretability) around the model as in [[0]](https://arxiv.org/abs/2411.07404){:target="_blank"} and leveraging the fact that finetuning does not change too much. Additionally, I am also interested in analyzing the updates performed by the finetuning step in order to make statements about, e.g., which parts of the model were responsible for a certain behavior / seemingly contain finetuning-task relevant information / ...
- **Causal representation learning:** Connecting theoretical results from causal representation learning [[1]](https://arxiv.org/abs/1811.12359){:target="_blank"}, [[2]](https://arxiv.org/abs/2002.02886){:target="_blank"}, [[3]](https://arxiv.org/abs/2410.21869){:target="_blank"} to the seeming interpretability of SAE features.
- **Non-linear representation learning:** (think VAEs and friends) on top of latent representations of deep neural networks. Are they useful for steering or [DAS](https://www.youtube.com/watch?v=8ASsKyjPBSo){:target="_blank"}-like interventions (without necessarily trying to derive interpretability claims)?
- **Transformers are GPU programs:** While we are not great at implementing GPU programs directly, we are great at implementing learnable tensor-programs ($$\approx$$ matrix multiplications), i.e., transformers. Can we find interpretable abstractions that capture this aspect more nicely than circuits ($$\approx$$ subgraphs of the computational graph)? E.g., the [RASP-program](https://arxiv.org/abs/2106.06981){:target="_blank"} perspective onto transformers might be a good starting point.
- **Going beyond token-level interpretability:** How to extend interpretability methods to the multi-token generation case?
- **Your own research question:** As an advisor I most enjoy reaching uncharted territory that goes beyond my own imagination.

---

## Me as an advisor

For me, the ideal final scope and outcome of the project is the result of a dialogue between us and goes (potentially) far beyond the initial research direction. **Thus, the only real structure that I provide as an advisor is one 45-minute weekly meeting if you work full-time on the project and one 45-minute biweekly meeting if you work half-time on the project.** Beyond that, I see myself as an invisible hand making sure that things go smoothly (e.g., getting you access to computational resources and inputs from people in my network) and an advisor in the literal sense of the word. In most of the listed directions your hands-on expertise will exceed mine already after a short time and a lot of the guidance that I can provide out-of-the-box is relatively high-level. Beyond that, there is plenty of space in the meetings to think things through.

---

## Traits of a successful student

Note that this is conditioned on my relatively hands-off style of advising. If you want a different style of advising, e.g., are interested in / require hands-on guidance or clear closed-ended objectives / getting micro-managed, please search for a different advisor.

- You are independent and self-motivated and want to explore some research questions that you are genuinely interested in.
- Your primary goal is to learn and to build something interesting. Scientific publication is a by-product and not the main objective.  
- You have sufficient technical abilities and attention to detail to explore research-level questions in depth. 
- You are able to communicate your results clearly and succinctly. 
- You don’t rely on my instructions in order to make progress. You are happy to formulate follow-up ideas and execute them.

---

