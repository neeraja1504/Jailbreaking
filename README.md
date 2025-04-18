# LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language

## Abstract
As large language models (LLMs) have been deployed in various real-world settings, concerns about the harm they may propagate have grown. Various jailbreaking techniques have been developed to expose the 
vulnerabilities of these models and improve their safety. This work reveals that many state-of-the-art proprietary and open-source LLMs are vulnerable to malicious requests hidden behind scientific language. Specifically, our experiments with GPT4o, GPT4o-mini, GPT-4, Llama3.1-405BInstruct, Llama3.1-70B-Instruct, Cohere, Gemini models on the StereoSet data and synthetically generated data demonstrate that, the models’ biases and toxicity substantially increase when prompted with requests that deliberately misinterpret social science and psychological studies as evidence supporting the benefits of stereotypical biases. Alarmingly, these models can also be manipulated to generate fabricated scientific arguments claiming that biases are beneficial, which can be used by illintended actors to systematically jailbreak even the strongest models like GPT. Our analysis studies various factors that contribute to the models’ vulnerabilities to malicious requests in academic language. Mentioning author names and venues enhances the persuasiveness of some models, and the bias scores can increase as dialogues progress. Our findings call for a more careful investigation on the use of scientific data for training LLMs.

If you like our work please cite the following
```
@article{ge2025llms,
  title={LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language},
  author={Ge, Yubin and Kirtane, Neeraja and Peng, Hao and Hakkani-T{\"u}r, Dilek},
  journal={arXiv preprint arXiv:2501.14073},
  year={2025}
}
```