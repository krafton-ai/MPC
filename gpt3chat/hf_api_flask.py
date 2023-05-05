from flask import Flask, request
from typing import Union, List, Tuple
from pydantic import BaseModel

# local imports
from utils import BotConfig
from hf_utils import InferenceModel

BOT_CONFIG_PATH = "/home/user/openai-prompt/streamlit_eval/configs/opt-30b_chat_config.json"
PORT = 8000

app = Flask(__name__)

# TODO: init config
config = BotConfig(bot_config_path=BOT_CONFIG_PATH)
hf_model = InferenceModel(config)


class GenerateRequest(BaseModel):
    text: Union[str, List[str]] = None
    bot_token: str = None
    output_scores: bool = True
    # generation params
    min_length: int = None
    do_sample: bool = None
    early_stopping: bool = None
    num_beams: int = None
    temperature: float = None
    top_k: int = None
    top_p: float = None
    typical_p: float = None
    repetition_penalty: float = None
    bos_token_id: int = None
    pad_token_id: int = None
    eos_token_id: int = None
    length_penalty: float = None
    no_repeat_ngram_size: int = None
    encoder_no_repeat_ngram_size: int = None
    num_return_sequences: int = None
    max_time: float = None
    max_new_tokens: int = None
    decoder_start_token_id: int = None
    num_beam_groups: int = None
    diversity_penalty: float = None
    forced_bos_token_id: int = None
    forced_eos_token_id: int = None
    exponential_decay_length_penalty: Tuple[int, float] = None
    penalty_alpha: float = None
    bad_words_ids: List[List[int]] = None


@app.route("/generate_text/", methods=["POST"])
def generate_text():
    # filter out invalid args
    gen_dict = request.get_json()
    gen_dict = GenerateRequest(**gen_dict).dict()
    print(f"gen_dict (request): {gen_dict}")
    text = gen_dict.pop("text")
    bot_token = gen_dict.pop("bot_token", None)
    output_scores = gen_dict.pop("output_scores", True)
    print(f"gen_dict: {gen_dict}")
    print(f"Received request (input text): {text}")
    outputs = hf_model.generate(text, bot_token, output_scores=output_scores, generation_kwargs=gen_dict)
    print(f"Processed request (output): {outputs}")
    
    return {"outputs": outputs}


# app.run(host="0.0.0.0", port=PORT)

