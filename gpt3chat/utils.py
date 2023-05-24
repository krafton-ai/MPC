import os
import json
import random
import logging
import torch
import Levenshtein
from typing import List
# Chatbot defaults
DEFAULT_NAME = "chatbot"

# OpenAI defaults (you can change params in chatbot_config.json)
OPENAI_MODEL = 'text-davinci-003'
OPENAI_MAX_COMPLETION_TOKENS = 256
OPENAI_TEMPERATURE = 0.7
OPENAI_FREQUENCY_PENALTY = 0.0
OPENAI_PROMPT_SUFFIX = None
OPENAI_PROMPT_PREFIX = None
OPENAI_PROMPT_LINE_NUM = False
OPENAI_PROMPT_USER_TOKEN = "A"
OPENAI_PROMPT_BOT_TOKEN = "B"

class BotConfig:
    def __init__(self, bot_config_path: str = None, persona: List[str] = [], bot_token: str=None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = DEFAULT_NAME
        # context stuff
        self.prompt_prefix = OPENAI_PROMPT_PREFIX
        self.prompt_suffix = OPENAI_PROMPT_SUFFIX
        self.prompt_line_num = OPENAI_PROMPT_LINE_NUM
        self.use_slack_names = False
        self.prompt_user_token = OPENAI_PROMPT_USER_TOKEN
        self.prompt_bot_token = OPENAI_PROMPT_BOT_TOKEN
        # completion stuff
        self.model = OPENAI_MODEL
        self.frequency_penalty = OPENAI_FREQUENCY_PENALTY
        self.temperature = OPENAI_TEMPERATURE
        self.max_completion_tokens = OPENAI_MAX_COMPLETION_TOKENS
        # others
        self.max_context_words = 384  # equals around 512 tokens for gpt-3 (ref: https://beta.openai.com/tokenizer)
        self.line_sep_token = "\n"
        self.prompt_sep_token = "\n"
        self.repeat_turn_after_suffix = False
        #memory
        self.attach_clarifier = True
        self.attach_gpt_retriever = True
        self.attach_memory_after_dialogue = False
        self.persona_summary = []
        self.memory_module = True
        self.dpr_topk = 5
        self.summary_every_n_turns = 6 # 15tokens/turn x 6 turns x 2
        self.remove_cot = False
        self.persona_prefix = False
        self.main_language = "english"

        if bot_config_path is not None:
            self._init_from_json(bot_config_path)

        self._init_from_dict(kwargs)

        # Random selection of bot token and persona
        if bot_token is None:
            self.prompt_bot_token = select_str_options(self.prompt_bot_token)
        elif bot_token is not None:
            self.prompt_bot_token = bot_token

        if len(persona)==0:
            self.persona_summary = random.choice(self.persona_summary)
        elif len(persona)!=0:
            self.persona_summary = persona

        for i in range(len(self.persona_summary)):
            self.persona_summary[i] = select_str_options(self.persona_summary[i]).replace("[BOT_TOKEN]", self.prompt_bot_token)
        if self.prompt_prefix:
            self.prompt_prefix = self.prompt_prefix.replace("[BOT_TOKEN]", self.prompt_bot_token)
        if self.prompt_suffix:
            self.prompt_suffix = self.prompt_suffix.replace("[BOT_TOKEN]", self.prompt_bot_token)

    def _init_from_dict(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

    def _init_from_json(self, filepath):
        if os.path.exists(filepath):
            self.logger.info(f'Initializing config from: {filepath}')
            with open(filepath) as f:
                data = json.load(f)
            # init config
            self._init_from_dict(data)
        else:
            self.logger.error(f"Could not find config file at: {filepath}")

# UPR approach (memory + prompt + question; measure question perplexity)
def UPR_ppl(
    data, question, tokenizer, model, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        data,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_question = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).input_ids
    question_len = len(encoded_question[0])

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # Only consider question for perplexity
        shift_attention_mask_batch = attn_mask[..., 1:]
        cs = torch.cumsum(shift_attention_mask_batch, dim=1)
        thres = torch.max(cs, dim=1).values - question_len
        shift_attention_mask_batch = torch.where(cs >= thres.unsqueeze(1), shift_attention_mask_batch, 0)
        shift_attention_mask_batch = shift_attention_mask_batch.contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return ppls


# Compare edit distance between string and list of strings
def close_match(src_list, target, threshold=0.75, min_len=3):
    if len(target.split()) <= min_len:
        return None
    for phrase in src_list:
        if len(phrase.split()) <= min_len:
            pass
        if (1 - Levenshtein.distance(phrase, target) / max(len(phrase), len(target))) > threshold:
            return phrase
    return None

# Extract completion text from openai return
def extract_text(completion):
    if 'text' in completion['choices'][0]:
        completion_text = [choice['text'] for choice in completion['choices']][0].strip().strip("\n")
    elif 'message' in completion['choices'][0]:
        completion_text = [choice['message']['content'] for choice in completion['choices']][0].strip().strip("\n")
    return completion_text.split("\n")[0]

# Choose delimited part randomly from a string
def select_str_options(s):
    i = s.find("{")
    j = s.find("}")
    if i == -1 or j == -1:
        return s
    option = random.choice(s[i+1:j].split(","))
    return s[:i] + option + s[j+1:]