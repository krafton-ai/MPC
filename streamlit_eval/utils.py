import random
import json
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import string
import random

import streamlit as st
from streamlit_chat import message

def show_chat(session_state):
    for i in range(len(session_state['past'])):
        is_user = (i % 2 == 0)
        if is_user:
            message(session_state['past'][i], is_user=True, key='user_' + str(i))
        else:
            message(session_state['past'][i], is_user=False, key='bot_' + str(i))

def show_chat_sessions(session_state):
    # display session1
    past = session_state['past']
    session_end_idx = session_state['session_end_idx']
    for i in range(0, session_end_idx):
        is_user = (i % 2 == 0)
        if is_user:
            message(past[i], is_user=True, key='s1_user_' + str(i))
        else:
            message(past[i], is_user=False, key='s1_bot_' + str(i))

    st.markdown("""---""")
    st.markdown("""**New session starts from here**""")

    # display session1
    for i in range(session_end_idx, len(past)):
        is_user = (i % 2 == 0)
        if is_user:
            message(past[i], is_user=True, key='user_' + str(i))
        else:
            message(past[i], is_user=False, key='bot_' + str(i))

def init_chat(session_state, force_restart=False):
    if force_restart or 'past' not in session_state:
        session_state['past'] = []
        session_state['past_order'] = []
        session_state['eval_mode'] = False
        session_state['start_page'] = True
        session_state['finish'] = False
        session_state['config'] = {'m1': '', 'm2': ''}
        session_state['bot_generated'] = {'m1': [], 'm2': []}

        # Single model evaluation (can include in A/B testing)
        session_state['single/sensibleness'] = {'m1': [], 'm2': []} #sensibleness
        session_state['single/specificity'] = {'m1': [], 'm2': []} #specificity
        session_state['single/consistency'] = {'m1': [], 'm2': []} #absolute consistency
        session_state['single/engagingness'] = {'m1': [], 'm2': []} #engagingness
        session_state['single/rating'] = {'m1': [], 'm2': []} #rating

        # Pairwise model evaluation
        session_state['pair/sensibleness'] = [] #relative sensibleness
        session_state['pair/consistency'] = [] #relative consistency
        session_state['pair/interesting'] = [] #interestingness
        session_state['pair/preference'] = [] #preference

        session_state['summary'] = []
        session_state['summary_opt'] = []
        session_state['summary_bb3'] = []
        session_state['persona_summary'] = []
        session_state['bb3_persona_summary'] = []
        session_state['prompt_bot_token'] = ""

        session_state['reward_token'] = ''
        session_state['single/rate'] = 0
        if 'session_id' in session_state:
            del session_state['session_id']

def init_persona(config_path, session_state):
    session_state['config']['m1'] = config_path
    if len(session_state['summary']) == 0:
        if os.path.exists(config_path):
            # print(f'Initializing config from: {config_path}')
            with open(config_path) as f:
                config = json.load(f)
            # init config
            config["prompt_bot_token"] = select_str_options(config["prompt_bot_token"])
            persona_idx = random.randint(0, len(config["persona_summary"]) - 1)
            # config["persona_summary"] = random.choice(config["persona_summary"])
            config["persona_summary"] = config["persona_summary"][persona_idx]
            for i in range(len(config["persona_summary"])):
                config["persona_summary"][i] = select_str_options(config["persona_summary"][i])
                config["persona_summary"][i] = config["persona_summary"][i].replace("[BOT_TOKEN]", config["prompt_bot_token"])
            if "bb3_persona_summary" in config:
                # bb3 persona format is slightly different
                config["bb3_persona_summary"] = config["bb3_persona_summary"][persona_idx]
                for i in range(len(config["bb3_persona_summary"])):
                    config["bb3_persona_summary"][i] = select_str_options(config["bb3_persona_summary"][i])
                    config["bb3_persona_summary"][i] = config["bb3_persona_summary"][i].replace("[BOT_TOKEN]", config["prompt_bot_token"])
                    config["bb3_persona_summary"][i] = "Person 2's Persona: " + config["bb3_persona_summary"][i]
                session_state['summary_bb3'].extend(config['bb3_persona_summary'])
                session_state['bb3_persona_summary'].extend(config['bb3_persona_summary'])
            session_state['summary'].extend(config['persona_summary'])
            session_state['persona_summary'].extend(config['persona_summary'])
            session_state['summary_opt'].extend(config['persona_summary'])
            session_state['prompt_bot_token'] = config["prompt_bot_token"]
            if config["prompt_prefix"] is not None:
                session_state["prompt_prefix"] = config['prompt_prefix'].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
                session_state["prompt_prefix_opt"] = config['prompt_prefix'].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
            if config["prompt_suffix"] is not None:
                session_state["prompt_suffix"] = config["prompt_suffix"].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
                session_state["prompt_suffix_opt"] = config["prompt_suffix"].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
            session_state['initialized'] = True
        else:
            print(f"Could not find config file at: {config_path}")
    else:
        session_state['initialized'] = False
    
def init_pair_persona(config_path_1, config_path_2, session_state, init_past=False, extend_persona=False):
    session_state['config']['m1'] = config_path_1
    session_state['config']['m2'] = config_path_2
    if len(session_state['summary']) == 0:
        config_path = config_path_1
        if os.path.exists(config_path):
            # print(f'Initializing config from: {config_path}')
            with open(config_path) as f:
                config = json.load(f)
            # init config
            config["prompt_bot_token"] = select_str_options(config["prompt_bot_token"])
            persona_idx = random.randint(0, len(config["persona_summary"]) - 1)
            # config["persona_summary"] = random.choice(config["persona_summary"])
            config["persona_summary"] = config["persona_summary"][persona_idx]
            for i in range(len(config["persona_summary"])):
                config["persona_summary"][i] = select_str_options(config["persona_summary"][i])
                config["persona_summary"][i] = config["persona_summary"][i].replace("[BOT_TOKEN]", config["prompt_bot_token"])
            if "bb3_persona_summary" in config:
                # bb3 persona format is slightly different
                config["bb3_persona_summary"] = config["bb3_persona_summary"][persona_idx]
                for i in range(len(config["bb3_persona_summary"])):
                    config["bb3_persona_summary"][i] = select_str_options(config["bb3_persona_summary"][i])
                    config["bb3_persona_summary"][i] = config["bb3_persona_summary"][i].replace("[BOT_TOKEN]", config["prompt_bot_token"])
                    config["bb3_persona_summary"][i] = "Person 2's Persona: " + config["bb3_persona_summary"][i]
                session_state['summary_bb3'].extend(config['bb3_persona_summary'])
                session_state['bb3_persona_summary'].extend(config['bb3_persona_summary'])
            session_state['summary'].extend(config['persona_summary'])
            session_state['persona_summary'].extend(config['persona_summary'])
            session_state['summary_opt'].extend(config['persona_summary'])
            session_state['prompt_bot_token'] = config["prompt_bot_token"]
            if config["prompt_prefix"] is not None:
                session_state["prompt_prefix"] = config['prompt_prefix'].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
            if config["prompt_suffix"] is not None:
                session_state["prompt_suffix"] = config["prompt_suffix"].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])

            # initialize from past dialog if there is one
            if init_past:
                past_dialogs = config.get("past_dialogs", None)
                past_summaries = config.get("past_summaries", None)
                if past_dialogs is not None:
                    past_idx = random.randint(0, len(past_dialogs) - 1)
                    past_dialog = past_dialogs[past_idx]
                    past_summary = past_summaries[past_idx]
                    if 'past_starters' in config:
                        past_starter = random.choice(config['past_starters'][past_idx])
                    else:
                        past_starter = "Hi there. It's great you are back. How have you been?"

                    session_state['past'] = past_dialog
                    session_state['session_end_idx'] = len(past_dialog)
                    if extend_persona:
                        session_state['summary'].extend(past_summary)
                        # session_state['summary_opt'].extend(past_summary)
                        session_state['persona_summary'].extend(past_summary)
                    else:
                        # we overwrite summary to only include past dialog summary but not the original persona list
                        session_state['summary'] = past_summary
                        session_state['summary_opt'] = past_summary
                        session_state['persona_summary'] = past_summary
                    # bot starts new session
                    session_state['past'].append(past_starter)
                        
            session_state['initialized'] = True
        else:
            print(f"Could not find config file at: {config_path}")

        if os.path.exists(config_path_2):
            with open(config_path_2) as f:
                config2 = json.load(f)
            if config2["prompt_prefix"] is not None:
                session_state["prompt_prefix_opt"] = config2['prompt_prefix'].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
            if config2["prompt_suffix"] is not None:
                session_state["prompt_suffix_opt"] = config2["prompt_suffix"].replace("[BOT_TOKEN]", session_state['prompt_bot_token'])
        else:
            print(f"Could not find config file at: {config_path_2}")
    else:
        session_state['initialized'] = False

def force_init_chat(session_state):
    init_chat(session_state, force_restart=True)

def init_session_id(session_state, direc, folder):
    if 'session_id' in session_state or 'worker_id' not in session_state:
        return
    # root = join(direc, folder)
    # Path(root).mkdir(parents=True, exist_ok=True)
    # files = [f for f in listdir(root) if (isfile(join(root, f)) and '.json' in f)]
    # filenames = [int(f.replace('.json', '').replace('session_', '')) for f in files]

    session_state['reward_token'] =token_generator(6)

    # if len(filenames) > 0:
        # session_state['session_id'] = max(filenames) + 1

    session_state['session_id'] = session_state['worker_id'] + '_' + session_state['reward_token']

def upload_session(session_state, direc, folder):
    data = {
        'session_id': session_state['session_id'],
        'worker_id': session_state['worker_id'],
        'config': session_state['config'],
        'bot_token': session_state['prompt_bot_token'],
        'past': session_state['past'],
        'past_order': session_state['past_order'],
        'bot_generated': session_state['bot_generated'],
        'single/consistency': session_state['single/consistency'],
        'single/sensibleness': session_state['single/sensibleness'],
        'single/specificity': session_state['single/specificity'],
        'single/engagingness': session_state['single/engagingness'],
        'single/rating': session_state['single/rating'],
        'pair/sensibleness': session_state['pair/sensibleness'],
        'pair/consistency': session_state['pair/consistency'],
        'pair/interesting': session_state['pair/interesting'],
        'pair/preference': session_state['pair/preference'],
        'summary': session_state['summary'],
        'summary_opt': session_state['summary_opt'],
        'summary_bb3': session_state.get('summary_bb3', []),
        'reward_token': session_state['reward_token'],
        'rate': session_state['single/rate']
    }
    root = join(direc, folder)
    save_path = join(root, str(session_state['session_id'])+'.json')
    Path(root).mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def to_binary(response):
    if response.lower() in ["yes", "y"]:
        return 1
    return 0

# Choose delimited part randomly from a string
def select_str_options(s):
    i = s.find("{")
    j = s.find("}")
    if i == -1 or j == -1:
        return s
    option = random.choice(s[i+1:j].split(","))
    return s[:i] + option + s[j+1:]

def token_generator(size=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))
