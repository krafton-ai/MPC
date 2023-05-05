# fastapi app
from typing import Union, List, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

# local imports
from gpt3chat.utils import BotConfig
from gpt3chat.hf_utils import InferenceModel

BOT_CONFIG_PATH = "/home/user/openai-prompt/streamlit_eval/configs/opt-30b_chat_config.json"

# model init
config = BotConfig(bot_config_path=BOT_CONFIG_PATH)
hf_model = InferenceModel(config)

app = FastAPI()


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

class PPLRequest(BaseModel):
    text: Union[str, List[str]] = None
    bot_token: str = None
    upr_topk: int = None
    retrieved_summ: Union[str, List[str]] = None

@app.post("/generate_text")
def generate_text(text_gen_req: GenerateRequest):
    print(f"Received request (input text): {text_gen_req.text}")
    # seperate generation kwargs from other request info
    gen_dict = text_gen_req.dict()
    text = gen_dict.pop("text")
    bot_token = gen_dict.pop("bot_token", None)
    output_scores = gen_dict.pop("output_scores", True)
    # inference model
    outputs = hf_model.generate(text, bot_token, output_scores=output_scores, generation_kwargs=gen_dict)

    # -inf value cannot be posted in json format
    for key,value in outputs[0]['logprobs']['top_logprobs'][0].items() :
        if value == float("-inf"):
            outputs[0]['logprobs']['top_logprobs'][0][key] = -100
    print(f"Processed request (outputs): {outputs}")
    
    return {"outputs": outputs}

@app.post("/ppl")
def get_ppl(ppl_get_req: PPLRequest):
    re_retrieved_summ = hf_model.call_upr_facts(ppl_get_req.text, ppl_get_req.retrieved_summ, ppl_get_req.bot_token, ppl_get_req.upr_topk)
    return {"outputs":re_retrieved_summ}


@app.on_event("shutdown")
def shutdown_event():
    hf_model.terminate()