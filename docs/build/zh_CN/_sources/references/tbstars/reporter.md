# TBstars 2.0

# Overview
- Taobao Star is a general-purpose large language model jointly developed by Taobao Group and Aicheng Technology
- The design principles of Taobao Star are: massive data, e-commerce enhancement, business focus, and platform usability
- MoE Architecture: The latest version of Taobao Star adopts a Mixture of Experts (MoE) architecture, reducing active parameters to less than 10% of total parameters, significantly improving training and inference efficiency and reducing computational resource consumption
- Massive Training Data: Taobao Star is trained on large-scale, diverse text data, with specialized reinforcement for mathematical, coding, and reasoning capabilities, using over 10T tokens trained from scratch
- Strongest E-commerce LLM: Taobao Star enhances its e-commerce foundation capabilities with trillion-token-level e-commerce corpora and constructs billion-token-level e-commerce domain synthetic data, continuously using internal Taobao scenario flywheel data for synthetic training
- Business Applications: It has achieved superior performance compared to peer open-source models in fields such as query understanding, relevance, category prediction, recall, feature extraction, and product understanding. It has been applied in business operations at Taobao Search, Recommendation, Advertising, and Product Library teams, while also providing conversational functions for e-commerce QA application scenarios
- Ultimate Efficiency: Compared to dense models with equivalent capabilities, the TBstars-MoE model reduces inference time by 20% in short prompt, long output scenarios. In the commonly used personalized behavior long prompt, short output scenarios in Taobao's internal business, inference time is reduced by 35%

The latest dense series of Taobao Star is TBstars-008, and the MoE series is TBstars2.0-MoE, including 3B, 7B, and 13B dense models as well as three MoE versions: 15B-A1.5B-MoE, 42B-A3.5B-MoE, and 96B-A23B-MoE

Base models have only undergone pretraining and do not possess direct conversation or instruction-following capabilities. They typically require business-specific SFT before use.
Chat version models have undergone post-training and can directly complete instructions when called.