# MPC (Modular Prompted Chatbot)

This is the implementation for [Prompted LLMs as Chatbot Modules for Long Open-domain Conversation](https://arxiv.org/abs/2305.04533)

## Getting started

Run the following to properly import package gpt3chat.

```bash
python3 setup.py install --user
pip install -r requirements.txt
```

To use 🤗 models, you have to install additional dependencies.
```
pip install -r requirements_hf.txt
python -c "import nltk;nltk.download('punkt')"
```

## Documentation

See the [OpenAI API docs](https://beta.openai.com/docs/api-reference?lang=python).

## Usage

The library needs to be configured with your account's secret key which is available on the [website](https://beta.openai.com/account/api-keys). Either set it as the `OPENAI_API_KEY` environment variable before using the library:

```bash
export OPENAI_API_KEY='sk-...'
```

Or set `openai.api_key` to its value:

```python
import openai
openai.api_key = "sk-..."

# list engines
engines = openai.Engine.list()

# print the first engine's id
print(engines.data[0].id)

# create a completion
completion = openai.Completion.create(engine="ada", prompt="Hello world")

# print the completion
print(completion.choices[0].text)
```
## LICENSE
The code is released under the Apache-2.0 License. See `LICENSE` for full terms.
The generated data is subject to the model owner's policy.

## Citation
[Prompted LLMs as Chatbot Modules for Long Open-domain Conversation]()  
[[PDF](MPC_ver4.pdf)] [[Arxiv](https://arxiv.org/abs/2305.04533)]
```bibtex
@misc{lee2023prompted,
      title={Prompted LLMs as Chatbot Modules for Long Open-domain Conversation}, 
      author={Gibbeum Lee and Volker Hartmann and Jongho Park and Dimitris Papailiopoulos and Kangwook Lee},
      year={2023},
      eprint={2305.04533},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```