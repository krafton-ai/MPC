import re
import os
import random
import openai
import numpy as np
import math
import requests
import time
from typing import List
from copy import deepcopy

from gpt3chat import utils
from gpt3chat.memory import BiEncoderRetriever
from gpt3chat.hf_utils import InferenceModel
from transformers import GPT2TokenizerFast

try:
    import nltk
except:
    raise ImportError("Please install nltk and donwload `punkt` module. `pip install nltk` and `python -c `import nltk;nltk.download('punkt')`.")

# DEFAULTS
QUESTION_PROB = 0.34
openai.organization = None  # insert your openai api org here (if any)
openai.api_key = None  # insert your openai api key here

class Chatbot:
    def __init__(self, bot_config_path: str = 'bot_config.json', debug: bool = False, persona: List[str] = [], bot_token: str = None, **kwargs):
        self.config = utils.BotConfig(bot_config_path=bot_config_path, persona=persona, bot_token=bot_token, **kwargs)
        self.debug = debug
        
        # context stuff
        self.gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.user_token = self.config.prompt_user_token
        # self.bot_token = self.config.prompt_bot_token

        # memory retrieval module
        self.few_shot_prompt = {'clarifier': None, 'retrieval': None, 'summary': None}
        for module in self.few_shot_prompt.keys():
            if hasattr(self.config, f"few_shot_{module}"):
                if module == 'retrieval' and self.config.remove_cot:
                    with open(getattr(self.config, f"few_shot_{module}_remove_cot"), "r") as f:
                        self.few_shot_prompt['retrieval'] = f.read().replace("[USER_TOKEN]", self.user_token)
                else:
                    with open(getattr(self.config, f"few_shot_{module}"), "r") as f:
                        self.few_shot_prompt[module] = f.read().replace("[USER_TOKEN]", self.user_token)
                
        
        self.initial_summaries = getattr(self.config, "persona_summary", [])
        self.has_memory_module = getattr(self.config, "memory_module", False)
        if self.has_memory_module:
            self.memory_retriever = BiEncoderRetriever()
            if len(self.initial_summaries) > 0:
                self.initial_encodings = self.memory_retriever.encode_summaries(self.config.persona_summary).detach().cpu().numpy()
        
        # benchmark
        self.benchmark = getattr(self.config, "benchmark", False)
        self.response_times = []
        self.model_times = []
        self.toks_per_sec = []
        self.toks_per_sec_completion = []
        self.num_toks = []
        self.num_toks_completion = []

        # init local
        self.hf_model_completion = False
        checkpoint = getattr(self.config, "hf_model_name", None)
        self.hf_api_ip = getattr(self.config, "hf_api_ip", None)
        # use API if provided, otherwise init model locally
        if self.hf_api_ip is not None:
            self.hf_model_completion = True
        elif checkpoint is not None:
            self.hf_model_completion = True
            self.hf_model = InferenceModel(self.config)
        
    def _hf_completion(self, 
                       context, 
                       stop_token: str = None, 
                       max_completion_tokens: int = None, 
                       temperature: int = None, 
                       frequency_penalty: float = None,
                       remove_trailing_comments: bool = True,
                       module="response"):
        config_name = f"hf_generation_kwargs_{module}"
        generation_kwargs = {}
        if hasattr(self.config, config_name):
            generation_kwargs = deepcopy(getattr(self.config, config_name))
        if max_completion_tokens is not None:
            generation_kwargs['max_new_tokens'] = max_completion_tokens
        if temperature is not None:
            generation_kwargs['temperature'] = temperature
        if frequency_penalty is not None:
            generation_kwargs['repetition_penalty'] = frequency_penalty
        if generation_kwargs.get('temperature', None) is not None and generation_kwargs['temperature'] <= 0:
            # deactivate sampling for zero temperature
            generation_kwargs['do_sample'] = False
        if "temperature" in generation_kwargs and generation_kwargs.get('do_sample', False):
            # too low temperature with sampling will return INF/-INF prob
            generation_kwargs['temperature'] = max(generation_kwargs['temperature'], 0.1)
        if "exponential_decay_length_penalty" in generation_kwargs and "start_index" in generation_kwargs["exponential_decay_length_penalty"]:
            start_index = generation_kwargs["exponential_decay_length_penalty"]["start_index"]
            decay_factor = generation_kwargs["exponential_decay_length_penalty"]["decay_factor"]
            generation_kwargs["exponential_decay_length_penalty"] = (start_index, decay_factor)
        if self.hf_api_ip is not None:
            generation_kwargs['output_scores'] = True
            generation_kwargs['bot_token'] = self.bot_token
            generation_kwargs['text'] = context
            api_route = self.hf_api_ip + "/generate_text"
            r = requests.post(api_route, json=generation_kwargs)
            model_responses = r.json()['outputs']
        else:
            model_responses = self.hf_model.generate(context, self.bot_token, generation_kwargs=generation_kwargs)
        if "model_time" in model_responses[0]:
            self.model_times.append(model_responses[0]["model_time"])
        if "toks_per_sec" in model_responses[0]:
            self.toks_per_sec.append(model_responses[0]["toks_per_sec"])
        if "toks_per_sec_completion" in model_responses[0]:
            self.toks_per_sec_completion.append(model_responses[0]["toks_per_sec_completion"])
        if "num_toks" in model_responses[0]:
            self.num_toks.append(model_responses[0]["num_toks"])
        if "num_toks_completion" in model_responses[0]:
            self.num_toks_completion.append(model_responses[0]["num_toks_completion"])
        model_response = model_responses[0]["completion_text"]
        # API might return list (batch) of predictions
        if isinstance(model_response, list) or isinstance(model_response, tuple):
            model_response = model_response[0]
        if self.bot_token is not None and model_response.startswith(f"{self.bot_token}:"):
           model_response = model_response[len(f"{self.bot_token}:"):]
        print(f"[{module}] (full) model_response: {model_responses}")
        if stop_token is not None:
            # (1) remove leading newline, (2) split on user name (continuing dialog), (3) remove trailing new line, (4) remove trailing underscores
            bot_response = model_response.lstrip("\n").split(stop_token)[0].split(self.config.line_sep_token)[0].strip("\n").strip().rstrip("_")
            # sometimes the model mispells the user name, i.e. we have to filter for all possible utterance starts
            bot_response_filtered = []
            for w in bot_response.split():
                if w.strip().endswith(":") and nltk.edit_distance(w, f"{self.user_token}:") < len(f"{self.user_token}:") * 0.5:
                    break
                for c in [".", "?", "!"]:
                    if c in w.strip() and w.strip().find(c) < (len(w.strip()) - 1):
                        print(f"FILTERED: {bot_response}")
                        break
                bot_response_filtered.append(w)
            bot_response = " ".join(bot_response_filtered)
        else:
            bot_response = model_response.strip("\n").strip().strip("_")
        # remove trailing whitespace and newline
        bot_response = bot_response.strip().strip("\n").strip()
        if remove_trailing_comments:
            # model sometimes produces trailing parenthesis with comments
            if bot_response.endswith(")") and "(" in bot_response:
                bot_response = bot_response.rsplit("(", maxsplit=1)[0]
            elif bot_response.endswith(")*") and "*(" in bot_response:
                bot_response = bot_response.rsplit("*(", maxsplit=1)[0]
#         # cut down to max number of sentences just in case
#         if max_response_sents > 0:
#             from nltk import tokenize
#             bot_response = " ".join(tokenize.sent_tokenize(bot_response)[:max_response_sents])
        # filter out too many questions and maybe truncate number of sents
        max_response_sents = getattr(self.config, "max_response_sents", -1)
        max_response_questions = getattr(self.config, "max_response_questions", -1)
        if max_response_questions > 0 or max_response_sents > 0:
            from nltk import tokenize
            bot_response = " ".join(tokenize.sent_tokenize(bot_response)[:max_response_sents])
            filtered_sents = []
            num_questions = 0
            for sent in tokenize.sent_tokenize(bot_response):
                if "bloom" in getattr(self.config, "hf_model_name", "") and sent in filtered_sents:
                    # bloom sometimes repeats the exact same sentences in one response
                    continue
                if sent.endswith("?"):
                    if max_response_questions < 0:
                        filtered_sents.append(sent)
                    elif max_response_questions >= 0 and num_questions < max_response_questions:
                        filtered_sents.append(sent)
                    num_questions += 1
                else:
                    # always add non-questions (if shorter than max sents)
                    if max_response_sents < 0:
                        filtered_sents.append(sent)
                    elif max_response_sents >= 0 and len(filtered_sents) < max_response_sents:
                        filtered_sents.append(sent)
            if "memory" in module :
                bot_response = "\n".join(filtered_sents)
            else:
                bot_response = " ".join(filtered_sents)

        if "bloom" in getattr(self.config, "hf_model_name", ""):
            # bloom sometimes responds in blog format
            bot_response = bot_response.rstrip("Introduction").rstrip("Introduction\n")

        # TODO: why model response is empty sometimes?
        if len(bot_response) < 1:
            bot_response = "Sorry, I didn't catch that."
        print(f"[{module}] (final) bot_response: {bot_response}")
        completion = {'choices': [{'text': bot_response, 'logprobs': model_responses[0].get("logprobs", None)}]}

        return completion

    def _construct_context(self, user_name: str, user_history: List[str], bot_history: List[str], current_user_input: str):
        # we might use the real names from slack here if specified
        self.user_token = user_name if self.config.use_slack_names and user_name is not None else self.config.prompt_user_token
        # self.bot_token = bot_token if bot_token is not None else self.config.prompt_bot_token

        # self.bot_token = self.config.name if self.config.use_slack_names and self.config.name is not None else self.config.prompt_bot_token

        prefix_dialog = ""
        # add previous dialog history
        line_number = 0
        dialog_lines = []
        fake_line_sep_token = " [LINE_SEP_TOKEN] "  # to avoid newline removal during truncation

        for user_input, bot_response in zip(user_history, bot_history):
            line_number += 1
            cur_line = ""
            cur_line += f"{self.user_token}: {user_input}{fake_line_sep_token}"
            dialog_lines.append(cur_line)
            prefix_dialog += cur_line
            line_number += 1
            cur_line = ""
            cur_line += f"{self.bot_token}: {bot_response}{fake_line_sep_token}"
            dialog_lines.append(cur_line)
            prefix_dialog += cur_line
        # add current user input
        line_number += 1
        cur_line = ""
        cur_line += f"{self.user_token}: {current_user_input}"
        dialog_lines.append(cur_line)
        prefix_dialog += cur_line

        # truncate if too long (we use words here since we don't know how many tokens it will be)
        # gpt tokenizer to check number of tokens
        dialog_words_ids = self.gpt_tokenizer(prefix_dialog)['input_ids']
        if len(dialog_words_ids) > self.config.max_context_words:
            dialog_words_ids = dialog_words_ids[-self.config.max_context_words:]
            prefix_dialog = self.gpt_tokenizer.decode(dialog_words_ids)
            # Start with a speaker, e.g., "goldfish:"
            speaker_index = [prefix_dialog.find(f"{self.user_token}:"), prefix_dialog.find(f"{self.bot_token}:")]
            speaker_index = min([i for i in speaker_index if i != -1])
            prefix_dialog = prefix_dialog[speaker_index:]

        # one more line for completion start
        line_number += 1

        # add prefix/suffix before/after prompt
        suffix_added = False
        if self.config.prompt_prefix is not None and self.config.prompt_prefix.strip() != "":
            prompt_prefix = self.config.prompt_prefix.replace("[USER_TOKEN]", self.user_token).replace("[BOT_TOKEN]", self.bot_token)
            prefix_dialog = prompt_prefix + self.config.prompt_sep_token + prefix_dialog
        
        prompt_suffix = self.config.prompt_suffix if self.config.prompt_suffix is not None else ""
        if prompt_suffix is not None and prompt_suffix.strip() != "":
            suffix_added = True
            prompt_suffix = self.config.prompt_suffix.replace("[USER_TOKEN]", self.user_token).replace("[BOT_TOKEN]", self.bot_token)
            prompt_suffix = prompt_suffix.replace("[LAST_LINE_NUMBER]", str(line_number))
            if random.random() > QUESTION_PROB:
                prompt_suffix = self.config.prompt_sep_token + prompt_suffix + self.config.prompt_sep_token
            else:
                if prompt_suffix.endswith("\n"):
                    prompt_suffix_clean = prompt_suffix.rstrip("\n")
                    prompt_suffix = self.config.prompt_sep_token + prompt_suffix_clean + " End with a question." + "\n" + self.config.prompt_sep_token
                else:
                    prompt_suffix = self.config.prompt_sep_token + prompt_suffix + " End with a question." + self.config.prompt_sep_token

        # in case of suffix we might repeat the last turn to avoid repetition
        if suffix_added and self.config.repeat_turn_after_suffix:
            if len(dialog_lines) >= 2:
                prompt_suffix += "".join(dialog_lines[-2:])
            elif len(dialog_lines) == 1:
                prompt_suffix += cur_line

        # add prompt start (only add newline if there is none already)
        completion_start = f"{self.config.line_sep_token}" if not prompt_suffix.endswith(self.config.line_sep_token) else ""
        completion_start += f"{self.bot_token}:"
        prompt_suffix += completion_start
        
        # replace fake new lines by real ones
        prompt_suffix = prompt_suffix.replace(fake_line_sep_token, self.config.line_sep_token)
        prefix_dialog = prefix_dialog.replace(fake_line_sep_token, self.config.line_sep_token)

        # Assemble context prefix + dialog + suffix
        context = prefix_dialog + prompt_suffix if prompt_suffix is not None else prefix_dialog
        print(f"Input to model: {context}")

        return context, line_number, prefix_dialog, prompt_suffix

    def _get_bot_response(self, user_name, user_history, bot_history, current_user_input, history_summary, history_encodings, initial_summary=None):
        # use only first part of name
        user_name = user_name.split()[0] if user_name is not None else None

        # assemble context
        context, last_line_num, prefix_dialog, prompt_suffix = self._construct_context(user_name, user_history, bot_history, current_user_input)
        bot_response = {
            "clarifier_input": None,
            "clarifier_output": None,
            "retriever_input": None,
            "retriever_output": None,
            "generator_input": None,
            "utterance": None,
            "summary_input": None,
        }

        if self.debug:
            return "[debug mode] This is a test message."

        # retrieve relevant memory
        if last_line_num > 2 and self.has_memory_module:
            # Clarifier step
            bot_response["clarifier_output"] = current_user_input
            if self.config.attach_clarifier:
                bot_response["clarifier_input"], bot_response["clarifier_output"] = self.call_clarification(context)

            # Retrieve DPR top 
            retrieved_summ = self.memory_retriever.retrieve_top_summaries(
                bot_response["clarifier_output"], history_summary, history_encodings, topk=self.config.dpr_topk
            )

            # Memory Processor COT or UPR
            if self.config.attach_gpt_retriever == 'COT':
                bot_response["retriever_input"], retrieved_summ = self.call_relevant_facts(bot_response["clarifier_output"], retrieved_summ)
            elif self.config.attach_gpt_retriever == 'UPR':
                if self.hf_api_ip is None:
                    bot_response["retriever_input"], retrieved_summ = self.hf_model.call_upr_facts(bot_response["clarifier_output"], retrieved_summ, self.bot_token, self.config.upr_topk)
                elif self.hf_api_ip is not None:
                    ppl_kwargs={}
                    ppl_kwargs['text'] = bot_response["clarifier_output"]
                    ppl_kwargs['bot_token'] = self.bot_token
                    ppl_kwargs['retrieved_summ'] = retrieved_summ
                    ppl_kwargs['upr_topk'] = self.config.upr_topk
                    api_route = self.hf_api_ip + "/ppl"
                    r = requests.post(api_route, json=ppl_kwargs)
                    bot_response["retriever_input"], retrieved_summ = r.json()['outputs']
            bot_response["retriever_output"] = "\n".join(retrieved_summ) # it was " ".join before change
            print(f"Input summary: {bot_response['retriever_output']}")

            if not bot_response["retriever_output"].strip() == "":
                if self.config.attach_memory_after_dialogue and self.config.persona_prefix:
                    context = prefix_dialog + getattr(self.config, "memory_sep_token", "\n\n") + \
                        f"The following statements are true about {self.bot_token}.\n" + bot_response["retriever_output"] + "\n\n" + prompt_suffix.lstrip()
                else:
                    context = bot_response["retriever_output"] + getattr(self.config, "memory_sep_token", "\n\n") + context
            if getattr(self.config, "memory_module_persona", False):
                context = f"The following are persona facts about {self.bot_token}.\n" + "\n".join(initial_summary) + \
                    getattr(self.config, "memory_sep_token", "\n\n") + context
            elif getattr(self.config, "memory_module_persona_nosep", False):
                context = f"The following statements are true about {self.bot_token}.\n" + "\n".join(initial_summary) + "\n" + context
            print(f"Input to model (w/ memory): {context}")
        elif last_line_num > 2:
            # No memory module
            if self.config.attach_memory_after_dialogue and self.config.repeat_turn_after_suffix:
                context = prefix_dialog + getattr(self.config, "memory_sep_token", "\n\n") + f"The following statements are true about {self.bot_token}.\n"+\
                    "\n".join(history_summary) + getattr(self.config, "memory_sep_token", "\n\n") + prompt_suffix.lstrip()
            elif getattr(self.config, "no_persona", False):
                # in case we want to test without persona
                context = context
            else:
                context = f"The following statements are true about {self.bot_token}.\n" + "\n".join(history_summary) + \
                    getattr(self.config, "memory_sep_token", "\n\n") + context
            print(f"Input to model (w/ persona): {context}")
        
        # openai call
        stop_token = f"{self.user_token}:"

        # Davinci acts a bit different from text davinci 002
        if self.config.model == "davinci":
            logit_bias = {}
            stop_token = [stop_token, f"{self.bot_token}:"]
        else:
            logit_bias={"198": -100, "12982": -100, "15439": -100}


        if self.hf_model_completion:
            completion = self._hf_completion(context, stop_token, module="response")
        else:
            completion = openai.Completion.create(
                    model=self.config.model,
                    prompt=context,
                    max_tokens=self.config.max_completion_tokens,
                    temperature=self.config.temperature,
                    stop=stop_token,
                    n=1,
                    frequency_penalty=self.config.frequency_penalty,
                    logprobs=5,
                    logit_bias=logit_bias,
                )

        if self.config.detect_repeat and last_line_num > 2:
            completion = self.get_divergent_response(context, completion, stop_token, prefix_dialog)
        completion_text = utils.extract_text(completion)

        # in some cases stop token does not work, so we remove everything after first line
        bot_response["generator_input"] = context
        bot_response["utterance"] = completion_text.split("\n")[0]
        print(f"#3 Final utterance output: {bot_response['utterance']}")  
        return bot_response

    def get_bot_response(self, user_name: str, user_history: List[str], bot_history: List[str], current_user_input: str, history_summary: List[str], history_encodings: np.ndarray = None, bot_token: str = "", prefix: str="", suffix: str="", initial_summary=None):
        """Generate bot response"""
        print("--------------------------RESPONSE_START--------------------------")
        bot_response = {
            "clarifier_input": None,
            "clarifier_output": None,
            "retriever_input": None,
            "retriever_output": None,
            "generator_input": None,
            "utterance": None,
            "summary_input": None,
        }
        self.bot_token = bot_token if bot_token is not None else self.config.prompt_bot_token
        self.config.prompt_prefix = prefix if prefix is not None else self.config.prompt_prefix
        self.config.prompt_suffix = suffix if suffix is not None else self.config.prompt_suffix
        start_time = time.time()
        is_real_response = False
        recent_summary = []
        recent_encodings = None

        try:
            # get response from bot (API call)
            bot_response = self._get_bot_response(user_name, user_history, bot_history, current_user_input, history_summary, history_encodings, initial_summary=initial_summary)
            is_real_response = True
        except Exception as e:
            print(e)
            bot_response["utterance"] = "[ERROR] 😴 Bot is sleeping right now."
        
        if self.has_memory_module and is_real_response:
            try:
                # Summarize every n turns
                if (len(bot_history) + 1) % self.config.summary_every_n_turns == 0:
                    bot_response["summary_input"], recent_summary = self.call_summary(
                        user_history[-self.config.summary_every_n_turns:] + [current_user_input], 
                        bot_history[-self.config.summary_every_n_turns:] + [bot_response["utterance"]]
                    )
                    recent_encodings = self.memory_retriever.encode_summaries(recent_summary).detach().cpu().numpy()
                    print(f"Recent summaries: {recent_summary}")
            except Exception as e:
                print(f"Summary exception: {e}")
                
        if self.benchmark:
            # response times
            self.response_times.append(round(time.time() - start_time, 2))
            print(f"[BENCHMARK] Request processing times: {self.response_times}")
            print(f"[BENCHMARK] Request processing times (avg): {sum(self.response_times) / len(self.response_times)}")
            if len(self.response_times) >= 2:
                print(f"[BENCHMARK] Request processing times (avg, w/o first): {sum(self.response_times[1:]) / len(self.response_times[1:])}")
            # stats about model inference times
            if len(self.toks_per_sec) > 0:
                print(f"[BENCHMARK] (All Sequence) Tokens per second: {self.toks_per_sec}")
                print(f"[BENCHMARK] (All Sequence) Tokens per second (avg): {round(sum(self.toks_per_sec) / len(self.toks_per_sec), 2)}")
                if len(self.toks_per_sec) >= 2:
                    print(f"[BENCHMARK] Tokens per second (avg, w/o first): {round(sum(self.toks_per_sec[1:]) / len(self.toks_per_sec[1:]), 2)}")
            if len(self.toks_per_sec_completion) > 0:
                print(f"[BENCHMARK] (Completion) Tokens per second: {self.toks_per_sec_completion}")
                print(f"[BENCHMARK] (Completion) Tokens per second (avg): {round(sum(self.toks_per_sec_completion) / len(self.toks_per_sec_completion), 2)}")
                if len(self.toks_per_sec) >= 2:
                    print(f"[BENCHMARK] (Completion) Tokens per second (avg, w/o first): {round(sum(self.toks_per_sec_completion[1:]) / len(self.toks_per_sec_completion[1:]), 2)}")
            if len(self.num_toks) > 0:
                print(f"[BENCHMARK] (All Sequence) Number of tokens: {self.num_toks}")
                print(f"[BENCHMARK] (All Sequence) Number of tokens (avg): {round(sum(self.num_toks) / len(self.num_toks), 2)}")
            if len(self.num_toks_completion) > 0:
                print(f"[BENCHMARK] (Completion) Number of tokens: {self.num_toks_completion}")
                print(f"[BENCHMARK] (Completion) Number of tokens (avg): {round(sum(self.num_toks_completion) / len(self.num_toks_completion), 2)}")

        print("--------------------------RESPONSE_END--------------------------")
        return {
            "is_real_response": is_real_response,
            "clarifier_input": bot_response["clarifier_input"],
            "clarifier_output": bot_response["clarifier_output"],
            "retriever_input": bot_response["retriever_input"],
            "retriever_output": bot_response["retriever_output"],
            "generator_input": bot_response["generator_input"],
            "text": bot_response["utterance"],
            "summary_input": bot_response["summary_input"],
            "recent_summary": recent_summary,
            "recent_encodings": recent_encodings,
        }
    
    def get_divergent_response(self, context, completion, stop_token, prefix_dialog):
        completion_text = utils.extract_text(completion)
        lines = [re.split('[;.?!]', line) for line in prefix_dialog.replace(f"{self.user_token}:", "").replace(f"{self.bot_token}:", "").split("\n")]
        context_phrases = [phrase.strip() for line in lines for phrase in line if len(phrase) > 0] + [self.config.prompt_suffix.replace("[USER_TOKEN]", f"{self.user_token}")]
        response_phrases = re.split('[;.?!]', completion_text)
        
        matched = False
        for idx, rp in enumerate([r for r in response_phrases if len(r) > 1]):
            matched = utils.close_match(context_phrases, rp, self.config.closeness_threshold)
            if matched:
                # Below doesn't include delimiters but should be fine
                # TODO: rp might have multiple indices
                find_index = completion_text.find(rp)
                if find_index == -1:
                    return completion
                truncated_response = completion_text[:find_index].strip()
                break
        if not matched:
            return completion
        
        print(f"Non-repeated part of output: {truncated_response}")
        print(f"Repeated part of output: {completion_text[completion_text.find(rp):]}")

        if len(truncated_response) > 0:
            completion['choices'][0]['text'] = truncated_response
            return completion

        filler_words = ['So', 'Well', 'Oh', 'Ah', 'I', 'You', 'Maybe', 'You know']
        if not self.hf_model_completion:
            log_probs = completion['choices'][0]['logprobs']
            phrase_indices = [0] + [(i+1) for i, tok in enumerate(log_probs['tokens']) if tok in [';','.','?','!']]
            original_token = log_probs['tokens'][phrase_indices[idx]]
            candidates = [*log_probs['top_logprobs'][phrase_indices[idx]]]
            all_candidates = [c for c in candidates + filler_words if c != original_token]
        else:
            all_candidates = filler_words
            
        lead_word = random.choice(all_candidates)
        if lead_word == "<|endoftext|>":
            completion['choices'][0]['text'] = "End of text error!"
            return completion

        if getattr(self.config, "hf_model_name", None) and "opt" in self.config.hf_model_name :
            ctxt = context + " " + lead_word
        elif  getattr(self.config, "model", None) and self.config.model == "davinci":
            ctxt = context + " " + lead_word
        else:
            ctxt = prefix_dialog + self.config.line_sep_token + f"{self.bot_token}: " + lead_word
        print(f"Re-Input to model: {ctxt}")

        if self.hf_model_completion:
            completion = self._hf_completion(ctxt, stop_token, frequency_penalty=1.15, module="response")
        else:
            completion = openai.Completion.create(
                model=self.config.model,
                prompt=ctxt,
                max_tokens=self.config.max_completion_tokens,
                temperature=0.7,
                stop=stop_token,
                n=1,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=1.0,
                logit_bias={"198": -100},
            )
        completion['choices'][0]['text'] = truncated_response + lead_word + ' ' + completion['choices'][0]['text']

        return completion

    # GPT call
    def call_clarification(self, context):
        lines = []
        prompt_suffix = self.config.prompt_suffix.strip().replace("[USER_TOKEN]", self.user_token).replace("[BOT_TOKEN]", self.bot_token)
        for i, line in enumerate(context.split(self.config.line_sep_token)):
            if len(prompt_suffix) > 0 and prompt_suffix in line:
                break
            check_last_line = line.replace(f"{self.bot_token}:", "").strip()
            if len(check_last_line) > 1:
                lines.append(line)
        context_only = "\n".join(lines[-2:])
        # original_question = lines[-1][lines[-1].find(":")+1:].strip()

        # Clarify question
        input_text = self.few_shot_prompt['clarifier'] + context_only + '\n# Specifically,'
        print(f"Clarified input: {input_text}")
        if self.hf_model_completion:
            completion = self._hf_completion(input_text, module="clarifier")
        else:
            completion = openai.Completion.create(
                    model=self.config.model,
                    prompt=input_text,
                    max_tokens=32,
                    temperature=0.3,
                    frequency_penalty=0.3,
                    n=1,
                    logit_bias={"198": -100},
                )
        third_person_question = [choice['text'] for choice in completion['choices']][0].strip().strip('\n').split('\n')[0]
        third_person_question = third_person_question[:third_person_question.find('#')]  + ' '
        print(f"#1 Clarifier output: {third_person_question}")
        return input_text, third_person_question

    def call_relevant_facts(self, third_person_question, retrieved_summaries):
        if not self.config.remove_cot:
            numbered_summaries = [f"({i+1}) " + summ for i, summ in enumerate(retrieved_summaries)]
            summary = "\n" + "\n".join(numbered_summaries)
        elif self.config.remove_cot:
            summary = "\n" + "\n".join(retrieved_summaries)

        # Few-shot Retrieval
        input_text = self.few_shot_prompt['retrieval'] + f"# This is the list of {self.bot_token}'s knowledge." + summary + "\nQ: " + third_person_question
        if self.config.remove_cot:
            input_text += f"\nA: {self.bot_token} thinks"
        else:
            input_text += "\nA: Let's think step by step.\n(1)"
        print(f"Retrieval input: {input_text}")

        if self.hf_model_completion:
            completion = self._hf_completion(input_text, module="memory_reasoning")
        else:
            completion = openai.Completion.create(
                    model=self.config.model,
                    prompt=input_text,
                    max_tokens=128,
                    temperature=0.3,
                    frequency_penalty=0.3,
                    n=1,
                    logit_bias={"198": -100},
                )
        completion_text = [choice['text'] for choice in completion['choices']][0].strip().strip('\n').strip("#")
        # in some cases stop token does not work, so we remove everything after first line
        if self.config.remove_cot:
            retrieved = completion_text.split("\n")[0].strip()
        elif "Answer:" in completion_text:
            retrieved = completion_text[completion_text.find("Answer:")+7:].split("\n")[0].strip()
        else:
            answer = input_text + completion_text.split('\n')[0] + f"\nAnswer: {self.bot_token} thinks"
            if self.hf_model_completion:
                completion = self._hf_completion(answer, stop_token='#', module="memory_selector")
            else:
                completion = openai.Completion.create(
                        model=self.config.model,
                        prompt=answer,
                        max_tokens=32,
                        temperature=0.3,
                        frequency_penalty=0.3,
                        n=1,
                        logit_bias={"198": -100},
                    )
            completion_text = [choice['text'] for choice in completion['choices']][0].strip().strip('\n')
            retrieved = completion_text.split("\n")[0].strip()
        if getattr(self.config, "hf_model_name", None) and ("opt" in self.config.hf_model_name or "GPT-JT" in self.config.hf_model_name): 
            retrieved = retrieved
        else:
            retrieved = f"{self.bot_token} thinks " + retrieved
        print(f"#2 Retriever output: {retrieved}")
        
        return input_text, [retrieved]
    
    def call_summary(self, user_inputs, bot_responses):
        dialog_lines = []
        for user_input, bot_response in zip(user_inputs, bot_responses):
            cur_line = ""
            cur_line += f"{self.user_token}: {user_input}"
            dialog_lines.append(cur_line)
            cur_line = ""
            cur_line += f"{self.bot_token}: {bot_response}"
            dialog_lines.append(cur_line)
        # few-shot self-supervised questions
        input_text = self.few_shot_prompt['summary'] + "#Dialogue\n" + "\n".join(dialog_lines)
        # input_text += f"\n#Summarize the above conversation between {self.bot_token} and {self.user_token} in bullet points.\n" + "-"
        input_text += f"\n#Summary\n-"
        
        if self.hf_model_completion:
            print(f"Summary input: {input_text}")
            completion = self._hf_completion(input_text, 
                                            remove_trailing_comments=False, 
                                            #stop_token='#', 
                                            module="memory_summarizer"
                                            )
        else:
            completion = openai.Completion.create(
                model=self.config.model,
                prompt=input_text,
                max_tokens=128,
                temperature=0.3,
                n=1
            )
        completion_text = [choice['text'] for choice in completion['choices']][0].strip().strip("\n")
        print(f"Summary output: {completion_text}")
        summaries = []
        for s in completion_text.split('-'):
            text = s.split('\n')[0].strip().strip('#')
            if len(text) > 0:
                summaries.append(text)
            if '#' in s or len(text) == 0:
                break

        if completion_text == "Sorry, I didn't catch that.":
            # OPT sometimes predicts empty summaries
            summaries = []
            
        return input_text, summaries
