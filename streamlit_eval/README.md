# Streamlit Chatbot Evaluation Platform

## Getting started
After installing streamlit, change the following in forms/constants.py.
```bash
# Modify the two paths below
HOME_PATH = "/path/to/openai-prompt/"
FOLDER = "debug_opt" # name of eval output directory
```

Set login credentials for your evaluation form in [authy.yaml](authy.yaml). Default is `user: user1 / password: 123`.

Finally, run the command in the streamlit_eval directory.
```bash
streamlit run --server.port=5000 forms/pairwise_model_eval.py
```


## Using custom models

🤗 models are already supported. If you want to implement inference for your own models, you can take a look at `_hf_completion` in [chat.py](../gpt3chat/chat.py).


### 🤗 Models
💡 You can also run the model directly in streamlit by removing "hf_api_ip" from the config.json file. This is okay for small models but not recommended for larger models. Also, it might not work with parallelformer, in which case you have to set `"parallelformer": false` in the config file.

❗️ Make sure FastAPI and Uvicorn are installed.
It's recommended that you start your huggingface model in a separate API first. This way you avoid restarting the heavy LM every time you restart the streamlit application. Make sure that `BOT_CONFIG_PATH` in [hf_api.py](../gpt3chat/hf_api.py) points to the correct config file.
1. `cd ../gpt3chat` 
2. `uvicorn hf_api:app`
3. Add your API address to your config file (e.g. "hf_api_ip": "http://127.0.0.1:8000")

💡 For deepspeed, you have to use the flask implementation (there is a multi-processing conflict between fastapi and deespeed-mii).
`gunicorn -t 0 -w 1 -b 127.0.0.1:8000 hf_api_flask:app`