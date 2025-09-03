# Visual interpretability agent

Akin to [MAIA](https://arxiv.org/abs/2404.14394){:target="_blank"} equip a VLM, e.g., Qwen 2.5 VLM or Gemma 3 VLM or some model behind an API with tools for (basically what we provide in our [demos](https://huggingface.co/spaces/surokpro2/Unboxing_SDXL_with_SAEs){:target="_blank"}), e.g.,
- generating images using SDXL/FLUX,
- visualizing features active during the generation,
- for a given feature, compute images for which this feature was on,
- performing feature interventions by turning on / off features

and prompt it / provide it with additional scaffolding to come up with hypotheses about features as well as functionality to validate those hypotheses. Our current **goal** for this agent is not so much only coming up with explanations of features but more to **perform targetted interventions on generated images** as is required for our recent [representation-based editing benchmark](https://github.com/wendlerc/RIEBench){:target="_blank"}. 

We provide extensive resources including desriptions and code for our SDXL/FLUX SAE feautures [here](https://sdxl-unbox.epfl.ch){:target="_blank"}.
