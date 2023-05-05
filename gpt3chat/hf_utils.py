import math
import nltk
import os
import time
import torch
from typing import List, Dict

from transformers.generation_stopping_criteria import StoppingCriteria, StoppingCriteriaList, STOPPING_CRITERIA_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings
from gpt3chat.utils import UPR_ppl

class StopSequenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever a certain sequence was generated.
    Args:
        min_length (`int`):
            The minimum length that the output sequence should have before checking for stop words.
    """

    def __init__(self, tokenizer, stop_sequences: List[str] = None, stop_pattern_dict: Dict[str, float] = None, min_length: int = 5):
        self.min_length = min_length
        self.stop_sequences = stop_sequences if stop_sequences is not None else []
        self.stop_pattern_dict = stop_pattern_dict if stop_pattern_dict is not None else {}
        self.tokenizer = tokenizer
        self.start_length = 0
        self.stop_sequences_ids = []
        if len(self.stop_sequences) > 0:
            for stop_seq in self.stop_sequences:
                stop_seq_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if len(stop_seq_ids) > 0:
                    self.stop_sequences_ids.append(stop_seq_ids)

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] < self.min_length + self.start_length:
            return False

        batch_gen_ids = input_ids[...,self.start_length:]
        gen_seqs_completed = [False] * batch_gen_ids.shape[0]
        for batch_idx, gen_ids in enumerate(batch_gen_ids):
            for stop_seq_ids in self.stop_sequences_ids:
                if stop_seq_ids[-1] == gen_ids[-1]:
                    # trigger potential stop sequence
                    if len(stop_seq_ids) == 1:
                        gen_seqs_completed[batch_idx] = True
                    elif len(gen_ids) >= len(stop_seq_ids):
                        stop_seq_matches = len([1 for stop_seq_id, gen_id in zip(stop_seq_ids, gen_ids[-len(stop_seq_ids):]) if stop_seq_id == gen_id])
                        if stop_seq_matches >= len(stop_seq_ids):
                            gen_seqs_completed[batch_idx] = True
            
#         # should we also ignore min_length generated tokens? input_ids[...,self.start_length + self.min_length:]
#         # Note: Make sure to set inital sequence length with `set_start_length`
#         # to ensure we check only in newly generated tokens if a sequence exists
#         gen_seqs = self.tokenizer.batch_decode(input_ids[...,self.start_length:], skip_special_tokens=True)
#         gen_seqs_completed = [False] * len(gen_seqs)
#         for batch_idx, gen_seq in enumerate(gen_seqs):
#             for stop_seq in self.stop_sequences:
#                 # check for exact match in stop sequences
#                 if stop_seq in gen_seq:
#                     gen_seqs_completed[batch_idx] = True
#                 # check for close match in patterns
#                 for stop_pattern, threshold in self.stop_pattern_dict.items():
#                     for word in gen_seq.split():
#                         if word.strip().endswith(":") and nltk.edit_distance(word, stop_pattern) < len(stop_pattern) * threshold:
#                             gen_seqs_completed[batch_idx] = True
                    
        # stop early if all sequences generated a stop sequence
        return all(gen_seqs_completed)
    
    def set_start_length(self, start_length: int = 0):
        # use this function to set initial sequence length for decoding
        self.start_length = start_length
        
    def set_stop_pattern_dict(self, stop_pattern_dict: Dict[str, float] = None):
        self.stop_pattern_dict = stop_pattern_dict
    
    
class StopTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.
    Args:
        min_length (`int`):
            The minimum length that the output sequence should have before checking for stop words.
    """

    def __init__(self, tokenizer, stop_tokens: List[int] = None, min_length: int = 5):
        self.min_length = min_length
        self.stop_tokens = stop_tokens if stop_tokens is not None else []
        self.tokenizer = tokenizer

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] < self.min_length:
            return False
        
        batch_seqs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        batch_seqs_completed = [False] * len(batch_seqs)
        for batch_idx, seq in enumerate(batch_seqs):
            for stop_word in self.stop_words:
                if stop_words in seq:
                    batch_seqs_completed[batch_idx] = True
                    
        return all(batch_seqs_completed)

    
def get_stop_criterias(tokenizer, stop_words=None, stop_pattern_dict=None):
    return StoppingCriteriaList([StopSequenceCriteria(tokenizer, stop_words, stop_pattern_dict)])


def get_filter_dict(d: Dict) -> dict:
    # d = dict(d)
    q = {}
    for i in d:
        if d[i] != None:
            q[i] = d[i]
    return q


class InferenceModel:
    def __init__(self, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
        self.config = config
        # set all hf cache dirs if custom dir is given in config
        hf_cache = getattr(config, "hf_cache", None)
        if hf_cache is not None:
            os.environ["TRANSFORMERS_CACHE"] = hf_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
            os.environ["HF_HOME"] = hf_cache
        checkpoint = getattr(self.config, "hf_model_name", None)
        print(f"🤗  Loading model: {checkpoint}")
        # device map assignment
        hf_config = AutoConfig.from_pretrained(checkpoint) 
        custom_device_map = getattr(self.config, "custom_device_map", False)
        device_map = "auto"
        num_gpus = torch.cuda.device_count()
        print(f"🔥  Available GPUs: {num_gpus}")
        if custom_device_map:
            device_map = {}
            if "facebook/opt" in checkpoint:
                layer_prefix = "model.decoder.layers"
                device_map["lm_head"] = max(0, num_gpus - 4)
                device_map["model.decoder.embed_tokens"] = max(1, num_gpus - 3)
                device_map["model.decoder.final_layer_norm"] = max(1, num_gpus - 3)
                device_map["model.decoder.embed_positions"] = max(1, num_gpus - 3)
            elif "gpt-neox" in checkpoint:
                layer_prefix = "gpt_neox.layers"
                device_map["embed_out"] = max(0, num_gpus - 2)
                device_map["gpt_neox.embed_in"] = max(1, num_gpus - 1)
                device_map["gpt_neox.final_layer_norm"] = max(1, num_gpus - 1)
            elif "bloom" in checkpoint:
                layer_prefix = "transformer.h"
                device_map["lm_head"] = max(0, num_gpus - 2)
                device_map["transformer.word_embeddings"] = max(1, num_gpus - 1)
                device_map["transformer.word_embeddings_layernorm"] = max(1, num_gpus - 1)
                device_map["transformer.ln_f"] = max(1, num_gpus - 1)
            else:
                print(f"Unsupported arch for custom_device_map: {checkpoint}.")
                layer_prefix = "decoder.layers"

            num_layers = hf_config.num_hidden_layers
            layers_per_gpu = math.ceil(num_layers / num_gpus)
            l_id_map = 0
            for gpu_id in range(num_gpus):
                for l_id in range(layers_per_gpu):
                    l_id_map = gpu_id * layers_per_gpu + l_id
                    if l_id_map < num_layers:
                        device_map[f"{layer_prefix}.{l_id_map}"] = gpu_id

        self.parallelformer = getattr(self.config, "parallelformer", False)
        self.alpa = getattr(self.config, "alpa", False)
        self.deepspeed = getattr(self.config, "deepspeed", False)
        # check model type
        self.is_seq2seq = "ConditionalGeneration" in hf_config.architectures[0]
        # load model
        model_class = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
        int8_inference = getattr(self.config, "8bit", False)
        if self.deepspeed:
            print(f"🚀  Deploying local GRPC server for Deepspeed-Inference")
            from gpt3chat.ds_utils import init_deepspeed_grpc
            self.ds_model = init_deepspeed_grpc(self.config.hf_model_name, dtype="int8" if int8_inference else "fp16")
        elif self.alpa:
            assert "facebook/opt" in self.config.hf_model_name, f"alpa only supports OPT right now. Trying to load model: {self.config.hf_model_name}"
            assert not self.parallelformer, "Cannot use Parallelformer + alpa together"

            from opt_serving.model.wrapper import get_model
            print(f"🤗  Loading alpa model (this might take a while) ...")
            alpa_model_name = self.config.hf_model_name.replace("facebook", "alpa")
            self.hf_model = get_model(model_name=alpa_model_name, path="/home/jovyan/alpa_models", do_sample=True)
        elif self.parallelformer:
            print(f"🤗  Looking for model in cache_dir: {hf_cache}")
            if int8_inference:
                print(f"🤗  Using parallelformer which does not support 8bit! Changing to fp16")
            print(f"🤗  Loading model to CPU first for parallelformer (this might take a while) ...")
            self.hf_model = model_class.from_pretrained(self.config.hf_model_name, cache_dir=hf_cache)
        elif int8_inference:
            print(f"🤗  Looking for model in cache_dir: {hf_cache}")
            print(f"🤗  Loading model using 8bit precision (this might take a while) ...")
            self.hf_model = model_class.from_pretrained(self.config.hf_model_name, device_map=device_map, cache_dir=hf_cache, load_in_8bit=True)
        else:
            print(f"🤗  Looking for model in cache_dir: {hf_cache}")
            # t0 and bloom should be inferenced with bf16 or fp32
            dtype = torch.bfloat16 if "T0" in self.config.hf_model_name or "bloom" in self.config.hf_model_name else torch.float16
            print(f"🤗  Loading model using dtype: {dtype}  (this might take a while) ...")
            self.hf_model = model_class.from_pretrained(self.config.hf_model_name, device_map=device_map, cache_dir=hf_cache, torch_dtype=dtype)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.config.hf_model_name)
        self.hf_model_completion = True
        # stop generation after producing line sep token
        # self.stop_criterias = get_stop_criterias(self.hf_tokenizer, stop_words=["\n"])
        if self.parallelformer and not self.deepspeed:
            from parallelformers import parallelize
            print("🤗  Loading HF model into parallelformer")
            parallelize(self.hf_model, num_gpus=num_gpus, fp16=True, verbose='detail')
        print("🎉  Finished model init!")
        
    def generate(self, context, bot_token=None, stop_criterias=None, output_scores=False, generation_kwargs=None):
        # if stop_criterias is None:
        #     stop_criterias = self.stop_criterias
        generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
        # filter out all None args
        generation_kwargs = get_filter_dict(generation_kwargs)
        print(f"generation_kwargs: {generation_kwargs}")
        
        start_time = time.time()
        if self.deepspeed:
            # TODO: move import to better place
            from gpt3chat.ds_utils import inference_deepspeed_grpc
            # use deepspeed-inference to get results
            output_text = inference_deepspeed_grpc(self.ds_model, context, generation_kwargs)
            # we have to manually remove the input from the output, because deepspeed returns only text
            input_ids = self.hf_tokenizer(context).input_ids
            output_ids = self.hf_tokenizer(output_text).input_ids
            # for bsz=1 tokenizer will return list instead of list of lists
            if not isinstance(output_ids[0], list):
                output_ids = [output_ids]
            if not isinstance(input_ids[0], list):
                input_ids = [input_ids]
            input_token_lengths = [len(x) for x in input_ids]
            output_token_lengths = [len(x) for x in output_ids]
            num_generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]
            completion_ids = [x[-i:] for x, i in zip(output_ids, num_generated_tokens)]
        else:
            if self.is_seq2seq and bot_token is not None:
                # remove prompt start from context and add it to decoder input
                context = context[:-len(f"{bot_token}:")]
                decoder_input = f"{self.hf_tokenizer.pad_token} {bot_token}:" # "<pad> Sarah:"
                decoder_input_ids = self.hf_tokenizer.encode(decoder_input, return_tensors="pt", add_special_tokens=False)
                inputs = self.hf_tokenizer(context, return_tensors="pt")
                input_len = decoder_input_ids.shape[-1]
                if self.config.parallelformer or self.config.alpa:
                    input_ids = inputs["input_ids"]
                    attn_mask = inputs["attention_mask"]
                else:
                    input_ids = inputs["input_ids"].to(self.hf_model.device)
                    attn_mask = inputs["attention_mask"].to(self.hf_model.device)
                    decoder_input_ids = decoder_input_ids.to(self.hf_model.device)
                # self.stop_criterias[0].set_start_length(input_len)
                # self.stop_criterias[0].set_stop_pattern_dict({f"{self.user_token}:": 0.7})
                output = self.hf_model.generate(input_ids, 
                                                attention_mask=attn_mask, 
                                                decoder_input_ids=decoder_input_ids,
                                                output_scores=True, 
                                                return_dict_in_generate=True, # stopping_criteria=stop_criterias, 
                                                **generation_kwargs)
            else:
                inputs = self.hf_tokenizer(context, return_tensors="pt")
                input_length = inputs.input_ids.size(1)
                if self.parallelformer or self.alpa:
                    input_ids = inputs["input_ids"]
                    attn_mask = inputs["attention_mask"]
                else:
                    input_ids = inputs["input_ids"].to(self.hf_model.device)
                    attn_mask = inputs["attention_mask"].to(self.hf_model.device)
                input_len = input_ids.shape[-1]
                # self.stop_criterias[0].set_start_length(input_len)
                # self.stop_criterias[0].set_stop_pattern_dict({f"{self.user_token}:": 0.7})
                if generation_kwargs.get('min_length', None) is not None:
                    generation_kwargs['min_length'] += input_length
                output = self.hf_model.generate(input_ids, 
                                                attention_mask=attn_mask, 
                                                output_scores=True, 
                                                return_dict_in_generate=True, #stopping_criteria=stop_criterias,
                                                **generation_kwargs)
            # get predictions and input
            completion_ids = output.sequences[:,input_len:]
            output_ids = output.sequences
        
        # inference stats
        model_time = time.time() - start_time
        num_toks_completion = sum([len(c_id) for c_id in completion_ids])
        num_toks = sum([len(c_id) for c_id in output_ids])
        toks_per_sec_completion = round(num_toks_completion / model_time, 2)
        toks_per_sec = round(num_toks / model_time, 2)
        # text preproc
        completion_text_batch = self.hf_tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # print("completion: ", completion_text_batch)
        if output_scores and not self.deepspeed:
            # logprob postproc
            top_logprobs_batch = []
            top1_tokens_batch = []
            for scores_batch in output.scores:
                top_logprobs = []
                top1_tokens = []
                for scores in scores_batch:
                    topk_scores = scores.topk(k=min(5, scores.shape[0]))
                    top_tokens = [self.hf_tokenizer.decode(tok_idx) for tok_idx in topk_scores.indices] 
                    top1_tokens.append(top_tokens[0])
                    top_logprobs_idx = {tt: tl for tt, tl in zip(top_tokens, topk_scores.values.tolist())}
                    top_logprobs.append(top_logprobs_idx)
                top1_tokens_batch.append(top1_tokens)
                top_logprobs_batch.append(top_logprobs)
                
            # assemble batch response
            model_responses = []
            for c_text, c_top1, c_logprobs in zip(completion_text_batch, top1_tokens_batch, top_logprobs_batch):
                model_responses.append({
                    'completion_text': c_text, 
                    'logprobs': {'top_logprobs': c_logprobs, 'tokens': c_top1},
                    'model_time': model_time,
                    'toks_per_sec_completion': toks_per_sec_completion,
                    'toks_per_sec': toks_per_sec,
                    'num_toks_completion': num_toks_completion,
                    'num_toks': num_toks,
                    })
        else:
            # assemble batch response
            model_responses = []
            for c_text in zip(completion_text_batch):
                model_responses.append({
                    'completion_text': c_text,
                    'model_time': model_time,
                    'toks_per_sec_completion': toks_per_sec_completion,
                    'toks_per_sec': toks_per_sec,
                    'num_toks_completion': num_toks_completion,
                    'num_toks': num_toks,
                    })
            
        return model_responses

    def terminate(self):
        if self.deepspeed:
            from gpt3chat.ds_utils import terminate_grpc
            terminate_grpc()

    # UPR Retrieval
    # Can NOT use for OpenAI GPT-3
    def call_upr_facts(self, third_person_question, retrieved_summaries, bot_token, topk=3):
        print(f"Retrieved summaries : {retrieved_summaries}")
        connection_prompt = f"\nPlease write a question based on this information.\n"
        concat = [summary + connection_prompt + third_person_question for summary in retrieved_summaries]

        ppl = UPR_ppl(data=concat, question=third_person_question, tokenizer=self.hf_tokenizer, model=self.hf_model)
        _, indices = torch.topk(-torch.Tensor(ppl), k=topk)
        reranked_summaries = [retrieved_summaries[i] for i in indices.tolist()]
        print(f"Reranked summaries : {reranked_summaries}")
        return concat, reranked_summaries