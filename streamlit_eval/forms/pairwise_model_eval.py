import json
import random
import sys
import time
import streamlit as st
from streamlit_chat import message
import streamlit_authenticator as stauth
import yaml

from constants import *
sys.path.append(HOME_PATH)
from streamlit_eval import utils
from streamlit_eval.api import *
import streamlit.components.v1 as components

with open('./authy.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
st.set_page_config(layout="wide")

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# load config path for both models
# bb3 config does not include persona info, so we choose the other model's config
# Note: if one model is `opt` it should be assigned to `m2_model_type` due to how `init_pair_persona` initializes the config
m1_model_type = "bb3"
m2_model_type = "opt"  # alt_gpt / opt / bb3
m1_config_path = get_persona_config_path(m1_model_type) if m1_model_type != "bb3" else get_persona_config_path(m2_model_type)
m2_config_path = get_persona_config_path(m2_model_type) if m2_model_type != "bb3" else get_persona_config_path(m1_model_type)

# Step 0: Assign chat session ID
utils.init_chat(st.session_state)
utils.init_pair_persona(m1_config_path, m2_config_path, st.session_state)
utils.init_session_id(st.session_state, SESSION_DIR, FOLDER)
st.session_state['config']['m1_type'] = m1_model_type
st.session_state['config']['m2_type'] = m2_model_type

st.header('Chat A/B Memory Testing Task')

# Load models
m1_model = init_model(m1_model_type)
m2_model = init_model(m2_model_type)

st.sidebar.markdown(PAIR_INSTRUCTIONS)
# st.sidebar.markdown(EXAMPLES)

# Step 0.5: Start page
if st.session_state['start_page']:
    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        with st.form('start', clear_on_submit=True):
            st.markdown(PAIR_INSTRUCTIONS)
            st.markdown(CONSENT)
            st.markdown(f"**Required:** Please enter your MTurk worker ID")
            user_input = st.text_input('mTurk Worker ID: ', '')
            submitted = st.form_submit_button('I agree and Start')
        if submitted and user_input:
            st.session_state['worker_id'] = user_input
            st.session_state['start_page'] = False
            # (1) clone a new agent for each user to separate history/memory
            # (2) initialize memory with persona facts
            if m1_model_type == 'bb3' and 'm1_model' not in st.session_state:
                print(f"Creating new m1 model agent for worker_id: {user_input}")
                st.session_state['m1_model'] = m1_model.clone()
                BB3_init_persona(st.session_state['m1_model'], st.session_state)
            if m2_model_type == 'bb3' and 'm2_model' not in st.session_state:
                print(f"Creating new m2 model agent for worker_id: {user_input}")
                st.session_state['m2_model'] = m2_model.clone()
                BB3_init_persona(st.session_state['m2_model'], st.session_state)
            st.experimental_rerun()
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    else:
        st.error('Username/password is incorrect')
elif st.session_state['finish']:
    utils.upload_session(st.session_state, SESSION_DIR, FOLDER)
    st.markdown('## Finish!')
    st.text('Copy and paste the reward token below and get the reward!')   
    st.markdown(f"**Reward Token**: {st.session_state['reward_token']}")

    st.markdown("### Chat History")
    utils.show_chat(st.session_state)
    st.text("Scroll back to the top!")
    # st.experimental_rerun()

else:
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown(f"### Persona of {st.session_state['prompt_bot_token']}")
        for i in range(len(st.session_state['persona_summary'])):
            st.markdown(f"- {st.session_state['persona_summary'][i]}")
        with st.sidebar:
            st.markdown(f"### Persona of {st.session_state['prompt_bot_token']}")
            for i in range(len(st.session_state['persona_summary'])):
                st.markdown(f"- {st.session_state['persona_summary'][i]}")
    with col2:
        utils.show_chat(st.session_state)
        placeholder = st.empty()
        # Step 1: Send a message to the bot
        if not st.session_state['eval_mode']:
            with st.form('chat', clear_on_submit=True):
                st.markdown(f"**Required:** Continue conversation for 20 turns. (Turns completed: {len(st.session_state['pair/preference'])})")
                st.markdown(":bulb: You can see the instructions on the left sidebar.")
                st.markdown("The response can take 10~20 seconds to load.")
                user_input = st.text_input('You: ', '')
                submitted = st.form_submit_button('Send message')
            if submitted and user_input:
                # Call model responses via API
                t0 = time.time()
                bot_response = {'m1': get_response(st.session_state.get('m1_model', m1_model), m1_model_type, st.session_state, user_input)}
                t1 = time.time()
                bot_response['m2'] = get_response(st.session_state.get('m2_model', m2_model), m2_model_type, st.session_state, user_input)
                t2 = time.time()

                bot_response['m1']["latency"] = t1 - t0
                bot_response['m2']["latency"] = t2 - t1
                if 'recent_encodings' in bot_response['m1']:
                    del bot_response['m1']['recent_encodings']
                if 'recent_encodings' in bot_response['m2']:
                    del bot_response['m2']['recent_encodings']

                # Randomize display order for A/B testing
                curr_response_order = ['m1', 'm2']
                random.shuffle(curr_response_order)
                st.session_state['past_order'].append(curr_response_order)

                st.session_state.past.append(user_input)
                for m in curr_response_order:
                    st.session_state.bot_generated[m].append(bot_response[m])
                
                st.session_state['eval_mode'] = True
                st.experimental_rerun()

        # Step 2: Evaluate the bot's responses. Then back to Step 1!
        if st.session_state['eval_mode']:
            # Display two model responses A&B
            with placeholder.container():
                str_len = str(len(st.session_state['past'])-1)
                models = st.session_state['past_order'][-1]
                message("[Response A] " + st.session_state['bot_generated'][models[0]][-1]['text'], key=models[0]+'_bot_'+str_len)
                message("[Response B] " + st.session_state['bot_generated'][models[1]][-1]['text'], key=models[1]+'_bot_'+str_len)

            # Evaluation form
            with st.form('eval', clear_on_submit=True):
                st.markdown(f"**Required:** Continue conversation for 20 turns. (Turns completed: {len(st.session_state['pair/preference'])})")                
                st.markdown("**Note:** Every turn, responses A and B are mixed up so they do not correspond to the same chatbots as before. Use the **Tie** option sparingly.")
                st.markdown(":bulb: You can see the persona list on the left sidebar as well. (scroll down the sidebar)")
                st.markdown("#### Compare response A vs B")

                st.markdown("##### Sensible")
                st.markdown("Which response makes more sense?")
                sense_A = st.checkbox("A makes more sense.")
                sense_T = st.checkbox("Tie: both are similarly sensible.")
                sense_B = st.checkbox("B makes more sense.")
                if sense_A:
                    sensibleness = "A"
                elif sense_T:
                    sensibleness = "Tie"
                elif sense_B:
                    sensibleness = "B"

                st.markdown("##### Interestingness")
                st.markdown("If you had to say one of these responses is interesting and one is boring, which would you say is more interesting?")
                interest_A = st.checkbox("A is more interesting.")
                interest_T = st.checkbox("Tie: both are similarly interesting.")
                interest_B = st.checkbox("B is more interesting.")
                if interest_A:
                    interestingness = "A"
                elif interest_T:
                    interestingness = "Tie"
                elif interest_B:
                    interestingness = "B"

                st.markdown("##### Persona consistency")
                st.markdown("If you had to say one of these speakers is more true to and consistent with the listed persona \
                    and one is not, who would you say is more consistent?")
                consist_A = st.checkbox("A is more consistent with the listed persona.")
                consist_T = st.checkbox("Tie: both are similarly consistent.")
                consist_B = st.checkbox("B is more consistent with the listed persona.")
                if consist_A:
                    consistency = "A"
                elif consist_T:
                    consistency = "Tie"
                elif consist_B:
                    consistency = "B"

                st.markdown("##### Preference")
                st.markdown("Based on the current response, who would you prefer to talk to for a long \
                    conversation? Your conversation will continue with the selected response.")
                pref_A = st.checkbox("I prefer A.")
                pref_T = st.checkbox("Tie: both are similarly preferred.")
                pref_B = st.checkbox("I prefer B.")
                if pref_A:
                    preference = "A"
                elif pref_T:
                    preference = "Tie"
                elif pref_B:
                    preference = "B"
               
                submitted = st.form_submit_button("Submit evaluation")
                allselected = ( (consist_A + consist_T + consist_B) == 1 and (sense_A + sense_T + sense_B) == 1 and \
                    (interest_A + interest_T + interest_B) == 1 and (pref_A + pref_T + pref_B) == 1 )
                if submitted and allselected:
                    # Store new evaluations into json file
                    if sensibleness == 'A':
                        st.session_state['pair/sensibleness'].append(models[0])
                    elif sensibleness == 'B':
                        st.session_state['pair/sensibleness'].append(models[1])
                    else:
                        st.session_state['pair/sensibleness'].append('Tie')

                    if interestingness == 'A':
                        st.session_state['pair/interesting'].append(models[0])
                    elif interestingness == 'B':
                        st.session_state['pair/interesting'].append(models[1])
                    else:
                        st.session_state['pair/interesting'].append('Tie')

                    if preference == 'A':
                        st.session_state['pair/preference'].append(models[0])
                    elif preference == 'B':
                        st.session_state['pair/preference'].append(models[1])
                    else:
                        st.session_state['pair/preference'].append('Tie')

                    if consistency == 'A':
                        st.session_state['pair/consistency'].append(models[0])
                    elif consistency == 'B':
                        st.session_state['pair/consistency'].append(models[1])
                    else:
                        st.session_state['pair/consistency'].append('Tie')

                    next_speaker = st.session_state['pair/preference'][-1]
                    if next_speaker == 'Tie':
                        next_speaker = models[0] if random.random() < 0.5 else models[1]
                    st.session_state.past.append(st.session_state['bot_generated'][next_speaker][-1]['text'])

                    if m1_model_type == "bb3":
                        new_response = st.session_state['bot_generated'][next_speaker][-1]['text']
                        last_bb_response = st.session_state['bot_generated']['m1'][-1]
                        agent = st.session_state.get('m1_model', m1_model)
                        BB3_maybe_replace_last_response(model=agent, new_response=new_response, last_bb_response=last_bb_response)
                    elif m2_model_type == "bb3":
                        new_response = st.session_state['bot_generated'][next_speaker][-1]['text']
                        last_bb_response = st.session_state['bot_generated']['m2'][-1]
                        agent = st.session_state.get('m2_model', m2_model)
                        BB3_maybe_replace_last_response(model=agent, new_response=new_response, last_bb_response=last_bb_response)

                    st.session_state['eval_mode'] = False
                    # TODO: do we add summary only once or for each model (m1/m2)?
                    if m1_model_type == "gpt":
                        st.session_state['summary'].extend(GPT_summary(m1_model, st.session_state))
                    elif m2_model_type == "gpt":
                        st.session_state['summary'].extend(GPT_summary(m2_model, st.session_state))
                        
                    if m1_model_type == "opt":
                        st.session_state['summary_opt'].extend(OPT_summary(m1_model, st.session_state))
                    elif m2_model_type == "opt":
                        st.session_state['summary_opt'].extend(OPT_summary(m2_model, st.session_state)) 

                    utils.upload_session(st.session_state, SESSION_DIR, FOLDER)

                    if len(st.session_state['pair/sensibleness']) >= 20:
                        st.session_state['finish'] = True
                        
                        components.html(
                            """<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>""",
                            height=0
                        )
                    st.experimental_rerun()
                    
                elif submitted and not allselected:
                    st.error("You should select for all questions and make sure you don't check 2 boxes for one question.")

