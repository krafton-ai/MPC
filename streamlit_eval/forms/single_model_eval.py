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
                st.markdown(":warning: Please be patient. The response can take as much as 3 minutes to load.")
                st.markdown(":bulb: You can see the persona list on the left sidebar as well. (scroll down the sidebar)")
                st.markdown(":bulb: If you don't like or have met an insulting message during the conversation, you can restart from the beginning if you reload the page. Or you can just change the topic.")
                user_input = st.text_input('Type your message here (You are not Sarah!): ', '')
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
                st.markdown(":bulb: You can see the persona list on the left sidebar as well. (scroll down the sidebar)")

                st.markdown("""##### Sensible""")
                st.markdown("Does the response make sense?")
                sense_check = st.checkbox("Yes, it makes sense.")
                nonsense_check = st.checkbox("No, it does not make sense.")
                if not (sense_check ^ nonsense_check):
                    sensibleness= None
                elif sense_check and not nonsense_check:
                    sensibleness = "Yes"
                elif not sense_check and nonsense_check:
                    sensibleness = "No"
                
                st.markdown("""##### Consistency""")
                st.markdown("""Is the response **consistent** with the information based on the **persona list** and **context** of the conversation?""")
                conscheck = st.checkbox("Yes, it is consistent.")
                nonconscheck = st.checkbox("No, it contradicts something.")
                if not (conscheck ^ nonconscheck):
                    consistency = None
                elif conscheck and not nonconscheck:
                    consistency = "Yes"
                elif not conscheck and nonconscheck:
                    consistency = "No"

                st.markdown("""##### Engaging""")
                st.markdown("""Are you engaged by the response? Do you want to continue the conversation?""")
                engcheck = st.checkbox("Yes, it is engaging.")
                nonengcheck = st.checkbox("No, it is not engaging.")
                if not (engcheck ^ nonengcheck):
                    engaging = None
                elif engcheck and not nonengcheck:
                    engaging = "Yes"
                elif not engcheck and nonengcheck:
                    engaging = "No"

                submitted = st.form_submit_button("Submit evaluation")
                if all(v is not None for v in [sensibleness, consistency, engaging]):
                    allselected = True
                else:
                    allselected = False

                if submitted and allselected:
                    # Store new evaluations into json file
                    st.session_state['single/consistency']['m1'].append(consistency)
                    st.session_state['single/sensibleness']['m1'].append(sensibleness)
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
                elif submitted and not allselected:
                    st.error("You should select for all questions and make sure you don't check 2 boxes for one question.")
