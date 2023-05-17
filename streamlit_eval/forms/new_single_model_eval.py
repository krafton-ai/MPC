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

st.set_page_config(layout="wide")
with open('./authy.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# load config path for both models
model_type = "gpt"  # one of: gpt / alt_gpt / opt / bb3
config_path = get_persona_config_path(model_type)

# Step 0: Assign chat session ID
utils.init_chat(st.session_state)
utils.init_persona(config_path, st.session_state)
utils.init_session_id(st.session_state, SESSION_DIR, FOLDER)
st.session_state['config']['m1_type'] = model_type

st.header('Chat Response Evaluation Task')

# Load models
model = init_model(model_type)


st.sidebar.markdown(SINGLE_INSTRUCTIONS)
st.sidebar.markdown(EXAMPLES)

# Step 0.5: Start page
if st.session_state['start_page']:
    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        with st.form('start', clear_on_submit=True):
            st.markdown(SINGLE_INSTRUCTIONS)
            st.markdown(CONSENT)
            st.markdown(f"**Required:** Please enter your MTurk worker ID")
            user_input = st.text_input('mTurk Worker ID: ', '')
            submitted = st.form_submit_button('I agree and Start')
        if submitted and user_input:
            st.session_state['worker_id'] = user_input
            st.session_state['start_page'] = False
            # (1) clone a new agent for each user to separate history/memory
            # (2) initialize memory with persona facts
            if model_type == 'bb3' and 'model' not in st.session_state:
                print(f"Creating new agent for worker_id: {user_input}")
                st.session_state['model'] = model.clone()
                BB3_init_persona(st.session_state['model'], st.session_state)
            st.experimental_rerun()
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    else:
        st.error('Username/password is incorrect')
elif st.session_state['finish']:
    components.html("""<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>""",
                            height=0
                            )
    st.markdown('### How was your chat?')
    x = st.slider('From a scale of 1 (very bad) to 5 (very good), rate the quality of the overall conversation.',1,5,5)

    col1, col2, col3=st.columns([3,2,2])
    with col1:
        st.text('Bad')
    with col2:
        if x==1:
            st.markdown(":star:")
        if x==2:
            st.markdown(":star::star:")
        if x==3:
            st.markdown(":star::star::star:")
        if x==4:
            st.markdown(":star::star::star::star:")
        if x==5:
            st.markdown(":star::star::star::star::star:")
    with col3:
        st.markdown("<p style='text-align: end;'>Good</p>",  unsafe_allow_html=True)
    st.session_state['single/rate'] = x
    
    if st.button('Send'):
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
        RUBRIC = """
        ### Evaluation Rubric

        Read the following rubric carefully. Make sure to read the persona of the chatbot below.

        1. **Not Sensible**: Does the response not make sense?
        2. **Not Specific and Generic**: Is the response generic and not specific? For example, if you say “I love tennis” then “That’s nice” would be a non-specific response.
        3. **Inconsistent Persona**: Is the response inconsistent with the information based on the persona list? Is it inconsistent with the context of the conversation?
        4. **Particularly Engaging**: Are you particularly engaged by the response?
        5. If none of the above apply, check **None of the above**.
        """
        st.markdown(RUBRIC)

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
                st.markdown(f"**Required:** Continue conversation for 20 turns. (Turns completed: {len(st.session_state['single/sensibleness']['m1'])})")
                st.markdown(":bulb: You can check the instructions on the left sidebar. (Click '>' at the top left). The response can take 5~10 seconds to load.  Trivial messages will lead to a HIT rejection.")
                if len(st.session_state['single/sensibleness']['m1']) == 0:
                    st.markdown("Start the first turn with \"Hi!\"")
                user_input = st.text_input('Type your message here: ', '')
                submitted = st.form_submit_button('Send message')
            if submitted and user_input:
                # Call model responses via API
                request_time = time.time()
                # Use ParlAI agent (clone) if available, otherwise default to model
                bot_response = get_response(st.session_state.get('model', model), model_type, st.session_state, user_input)
                bot_response["latency"] = time.time() - request_time

                if 'recent_encodings' in bot_response:
                    del bot_response['recent_encodings']
                st.session_state.past.append(user_input)
                st.session_state.bot_generated['m1'].append(bot_response)

                if len(bot_response.get('recent_summary', [])) > 0:
                    # Log memories for our chatbot
                    # OPT summaries is the same because we use only one model here
                    st.session_state['summary'].extend(bot_response['recent_summary'])
                    st.session_state['summary'] = list(set(st.session_state['summary']))
                    st.session_state['summary_opt'] = st.session_state['summary']
                if 'memories' in bot_response:
                    # Log memories for bb3
                    st.session_state['summary_bb3'] = list(bot_response['memories'].keys())
                
                st.session_state['eval_mode'] = True
                st.experimental_rerun()
                
        # Step 2: Evaluate the bot's responses. Then back to Step 1!
        if st.session_state['eval_mode']:
            # Display two model responses A&B
            with placeholder.container():
                str_len = str(len(st.session_state['past'])-1)
                message(st.session_state['bot_generated']['m1'][-1]['text'], key='m1_bot_'+str_len)

            # Evaluation form
            with st.form('eval', clear_on_submit=True):
                st.markdown(f"**Required:** Continue for 20 turns. (Turns completed: {len(st.session_state['single/sensibleness']['m1'])})")
                st.markdown(f"Rate the chatbot's response by checking (or unchecking) boxes according to the rubric.")

                nonsense_check = st.checkbox("Not Sensible")
                unspecific_check = st.checkbox("Not Specific and Generic")
                incons_check = st.checkbox("Inconsistent Persona")
                engage_check = st.checkbox("Particularly Engaging")
                none_above = st.checkbox("None of the above")

                sensibleness = "No" if nonsense_check else "Yes"
                specificity = "No" if unspecific_check else "Yes"
                consistency = "No" if incons_check else "Yes"
                engaging = "Yes" if engage_check else "No"
                
                submitted = st.form_submit_button("Submit evaluation")
                checked = nonsense_check or unspecific_check or incons_check or engage_check
                correct_eval = checked ^ none_above
                if submitted and correct_eval:
                    # Store new evaluations into json file
                    st.session_state['single/consistency']['m1'].append(consistency)
                    st.session_state['single/sensibleness']['m1'].append(sensibleness)
                    st.session_state['single/specificity']['m1'].append(specificity)
                    st.session_state['single/engagingness']['m1'].append(engaging)
                    
                    st.session_state.past.append(st.session_state['bot_generated']['m1'][-1]['text'])

                    st.session_state['eval_mode'] = False
                    utils.upload_session(st.session_state, SESSION_DIR, FOLDER)
                    if len(st.session_state['single/sensibleness']['m1']) == 20:
                        st.session_state['finish'] = True
                        
                        components.html(
                            """<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>""",
                            height=0
                        )
                    st.experimental_rerun()
                elif submitted and not correct_eval:
                    st.error("Please carefully re-read the instructions.")
