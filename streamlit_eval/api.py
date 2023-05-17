import copy
import os
import json
import streamlit as st
from gpt3chat.chat import Chatbot
from constants import *

SUMM_EVERY_N_TURNS = 3
model_gpt = None
model_alt_gpt = None
model_opt = None
model_alt_opt = None
model_bb3 = None

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_gpt(config_path, session_state):
    global model_gpt
    if model_gpt is None:
        model_gpt = Chatbot(config_path, persona=session_state['persona_summary'], bot_token=session_state['prompt_bot_token'])
    return model_gpt

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_alt_gpt(config_path, session_state):
    global model_alt_gpt
    if model_alt_gpt is None:
        model_alt_gpt = Chatbot(config_path, persona=session_state['persona_summary'], bot_token=session_state['prompt_bot_token'])
    return model_alt_gpt

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_opt(config_path, session_state):
    global model_opt
    if model_opt is None:
        model_opt = Chatbot(config_path, persona=session_state['persona_summary'], bot_token=session_state['prompt_bot_token'])
    return model_opt

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_alt_opt(config_path, session_state):
    global model_alt_opt
    if model_alt_opt is None:
        model_alt_opt = Chatbot(config_path, persona=session_state['persona_summary'], bot_token=session_state['prompt_bot_token'])
    return model_alt_opt

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_bb3(config_path):
    global model_bb3
    if model_bb3 is None:
        from parlai.core.agents import create_agent_from_model_file, create_agent
        with open(config_path, 'r') as f:
            opt = json.load(f)
        opt_override = copy.deepcopy(opt)
        opt['override'] = opt_override
        model_bb3 = create_agent(opt, requireModelExists=False)
        # model_bb3 = create_agent_from_model_file("zoo:bb3/bb3_3B/model", opt_overrides=opt)
    return model_bb3

# GPT3
def GPT_request(model, session_state, user_input):
    bot_response = model.get_bot_response(
        user_name = 'User', 
        user_history = session_state['past'][::2], 
        bot_history = session_state['past'][1::2],
        current_user_input = user_input,
        history_summary = session_state['summary'], 
        bot_token = session_state['prompt_bot_token'],
        prefix = session_state['prompt_prefix'],
        suffix = session_state['prompt_suffix'],
        initial_summary = session_state['persona_summary']
    )
    return bot_response

def OPT_request(model, session_state, user_input):
    bot_response = model.get_bot_response(
        user_name = 'User', 
        user_history = session_state['past'][::2], 
        bot_history = session_state['past'][1::2],
        current_user_input = user_input,
        history_summary = session_state['summary_opt'], 
        bot_token = session_state['prompt_bot_token'],
        prefix = session_state['prompt_prefix_opt'],
        suffix = session_state['prompt_suffix_opt'],
        initial_summary = session_state['persona_summary']
    )
    return bot_response

def GPT_summary(model, session_state):
    user_history = session_state['past'][::2]
    bot_history = session_state['past'][1::2]
    if len(user_history) > 0 and len(user_history) % SUMM_EVERY_N_TURNS == 0 and model.has_memory_module:
        return model.call_summary(user_history[-SUMM_EVERY_N_TURNS:], bot_history[-SUMM_EVERY_N_TURNS:])[1]
    return []

def OPT_summary(model, session_state):
    user_history = session_state['past'][::2]
    bot_history = session_state['past'][1::2]
    if len(user_history) > 0 and len(user_history) % SUMM_EVERY_N_TURNS == 0 and model.has_memory_module:
        return model.call_summary(user_history[-SUMM_EVERY_N_TURNS:], bot_history[-SUMM_EVERY_N_TURNS:])[1]
    return []

# BB3
def BB3_request(model, session_state, user_input, reset_history=True, init_persona=False, inject_memory=False):
    # only reset history in case of (1) single agent for multiple users or (2) a/b testing
    if reset_history:
        # Reset history and gather new history
        model.reset()
        personas = "\n".join(session_state['bb3_persona_summary'])
        user_input = "\n".join([personas] + session_state['past'] + [user_input])
    elif init_persona:
        # Inject persona with first message
        personas = "\n".join(session_state['bb3_persona_summary'])
        user_input = "\n".join([personas] + [user_input])

    # Model actually witnesses the human's text
    model.observe({'text': user_input, 'episode_done': False})
    print("🤖  BlenderBot's history:")
    print(model.history.get_history_str())
    print("🤖  BlenderBot's memories:")
    print(model.memories)
 
    # model produces a response
    return model.act()

def BB3_init_persona(model, session_state):
    model.memories = model.get_opening_memories(session_state['bb3_persona_summary'])

def BB3_maybe_replace_last_response(model, new_response, last_bb_response):
    # only replace if new_response is not from bb3 itself
    last_bb3_response = model.history.turns[-1].replace("Person 2: ", "")
    # print(f"🤖  new_response: {new_response}")
    # print(f"🤖  last bb3_response: {last_bb3_response}")
    if new_response == last_bb3_response:
        return

    # step 1: remove last turn from history
    last_response = model.history.turns.pop(-1)

    # step 2: remove memories from last turn
    # print(f"🤖  BB3: memories before removal: {model.memories}")
    last_memory_self = None
    last_memory_partner = None
    bb_memories = model.memories
    # we only remove bot's memory because user input stays same
    if last_bb_response['memory_generator_self'] != "no persona":
        last_memory_self = "Person 2's Persona: " + last_bb_response['memory_generator_self']
        # we only remove a memory if it was added in last turn
        # we check last two memories because it's max memories created per turn (user/bot)
        # actually dict is not strictly ordered but it should still work
        if len(bb_memories) >= 2:
            if last_memory_self == list(bb_memories.keys())[-1] or last_memory_self == list(bb_memories.keys())[-2]:
                print(f"🤖  BB3: removing (self) memory from last turn: {last_memory_self}")
                bb_memories.pop(last_memory_self)
        elif len(bb_memories) >= 1:
            if last_memory_self == list(bb_memories.keys())[-1]:
                print(f"🤖  BB3: removing (self) memory from last turn: {last_memory_self}")
                bb_memories.pop(last_memory_self)

    # print(f"🤖  BB3: updated memories after removal: {model.memories}")
    
    # step 3: insert new response and generat memories
    if hasattr(model, "batch_imitation_act"):
        from parlai.core.message import Message

        reply = [Message({
            'id': 'BlenderBot3',
            'episode_done': False,
            'text': new_response,
        })]
        response = model.batch_imitation_act([model.observation], reply)[0]
        model.self_observe(response)
        # print(f"🤖  BB3: created new memories: {response}")
    else:
        print("💥  Method `batch_imitation_act` does not exist! Unable to replace last turn.")

def get_persona_config_path(model_type):
    if model_type == "gpt":
        config_path = GPT_CONFIG_PATH
    elif model_type == "alt_gpt":
        config_path = PLAIN_GPT_CONFIG_PATH
    elif model_type == "opt":
        config_path = OPT_CONFIG_PATH
    elif model_type == "alt_opt":
        config_path = OPT66_CONFIG_PATH
    elif model_type == "bb3":
        # persona not defined in bb3 config, so we just use another config
        config_path = GPT_CONFIG_PATH
    else:
        raise ValueError(f"Model config for `model_type` {model_type} does not exist!")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config does not exist at path: {config_path}.")

    return config_path

def init_model(model_type):
    if model_type == "gpt":
        init_model = load_model_gpt(GPT_CONFIG_PATH, st.session_state)
    elif model_type == "alt_gpt":
        init_model = load_model_alt_gpt(PLAIN_GPT_CONFIG_PATH, st.session_state)
    elif model_type == "opt":
        init_model = load_model_opt(OPT_CONFIG_PATH, st.session_state)
    elif model_type == "alt_opt":
        init_model = load_model_alt_opt(OPT66_CONFIG_PATH, st.session_state)
    elif model_type == "bb3":
        init_model = load_model_bb3(BB3_CONFIG_PATH)
    else:
        raise ValueError(f"Model loading function for `model_type` {model_type} does not exist!")

    return init_model

def get_response(model, model_type, session_state, user_input, model_prefix=""):
    if model_type == 'gpt' or model_type == 'alt_opt':
        bot_response = GPT_request(model, session_state, user_input)
    elif model_type == 'opt' or model_type == 'alt_gpt':
        bot_response = OPT_request(model, session_state, user_input)
    elif model_type == 'bb3':
        bot_response = BB3_request(model, session_state, user_input, reset_history=False, init_persona=False)
    else:
        raise ValueError(f"Model response function for `model_type` {model_type} does not exist!")
    
    return bot_response

# Dummy
def dummy_request():
    return {"text": "Dummy request (debug)"}