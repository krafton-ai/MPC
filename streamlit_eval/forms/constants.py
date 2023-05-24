import os

# Modify the two paths below
HOME_PATH = "/home/jovyan/MPC/"
FOLDER = "GPT4_mturk_batch1" # name of eval output directory

CONFIG_PATH = os.path.join(HOME_PATH, "streamlit_eval/configs/")
SESSION_DIR = os.path.join(HOME_PATH, "streamlit_eval/output")

GPT_CONFIG_PATH = os.path.join(CONFIG_PATH, "ChatGPT_chat_config.json")
OPT_CONFIG_PATH = os.path.join(CONFIG_PATH, "opt-30b_chat_config.json")
OPT66_CONFIG_PATH = os.path.join(CONFIG_PATH, "opt-66b_chat_config.json")
PLAIN_GPT_CONFIG_PATH = os.path.join(CONFIG_PATH, "plain_gpt3chat_config.json")
BB3_CONFIG_PATH = os.path.join(CONFIG_PATH, "opt_bb3.opt")

SINGLE_INSTRUCTIONS = """### Task Description
In this task, you will have a natural conversation with a chatbot and evaluate its responses for
various conversational attributes. You will be able to see a list of personal facts that the chatbot
should know about itself. You may talk about these personal facts or you may converse about other topics.
The goal is to assess the chatbot's ability to make high-quality conversation based on its persona.
We expect the task to take about 30 minutes in total. Please note that we will check responses manually
to ensure quality and accuracy.

### Instruction
- You must start the first turn with: **Hi!**
- Have a conversation: do not trivially copy or ignore responses.
- You must complete **20** turns of responses and evaluations to complete the HIT.
- You must copy and paste reward token (provided at the end) into mTurk HIT page after you finish to get reward.  

### Alert
- Do not talk about the task or MTurk, HITs, or other MTurk specific vocabulary during the conversation.
- Please do not reveal personally identifiable information. You may use a pseudonym, fake age, etc.
- Some conversation data could be made **public**. Severely aggressive or insulting chats will lead to a HIT rejection and will be reported.

Finally, make sure you enter your MTurk worker ID accurately.
"""

PAIR_INSTRUCTIONS = """### Task Description
In this task, you will have a conversation with two chatbots of identical personalities and evaluate their responses for 
**how well they maintain their personas** and **how well they remember you and the conversation**. The goal is to assess and
compare which chatbot makes better high-quality conversation based on its persona. You will be able to see a list of
personal facts that the chatbot should know about itself. To test its memory, you may talk about these personal facts
or ask facts about yourself, e.g., asking if it remembers your name.

We expect the task to take about 30 minutes in total. Please note that we will check responses manually
to ensure quality and accuracy.

### Instruction
- You must start the first turn with: **Hi!**
- Have a conversation: do not trivially copy or ignore responses.
- To test memory, you may ask and interrogate the chatbot to check if it maintains its persona.
- You must complete **20** turns of responses and evaluations to complete the HIT.
- You must copy and paste reward token (provided at the end) into mTurk HIT page after you finish to get reward.
- If you do not follow the instructions in this page, your HIT may be rejected. 

### Alert
- Do not talk about the task or MTurk, HITs, or other MTurk specific vocabulary during the conversation.
- Please do not reveal personally identifiable information. You may use a pseudonym, fake age, etc.
- Some conversation data could be made **public**. Severely aggressive or insulting chats will lead to a HIT rejection and will be reported.

Finally, make sure you enter your MTurk worker ID accurately.
"""

PAIR_INSTRUCTIONS_SESSIONS = """### Task Description
In this task, you will have a conversation with two chatbots of identical personalities and evaluate their responses for 
**how well they maintain their personas** and **how well they remember you and the conversation**. The goal is to assess and
compare which chatbot makes better high-quality conversation based on its persona and memory. 
You will be able to see a conversation history between yourself and the chatbot. 
To test its memory, you should talk about personal facts, about yourself or the chatbot, mentioned in the conversation history.

We expect the task to take about 30 minutes in total. Please note that we will check responses manually
to ensure quality and accuracy.

### Instruction
- Have a conversation: do not trivially copy or ignore responses.
- To test memory, you should ask and interrogate the chatbot about facts from the conversation history.
- You must complete **6** turns of responses and evaluations to complete the HIT.
- You must copy and paste reward token (provided at the end) into mTurk HIT page after you finish to get reward.
- If you do not follow the instructions in this page, your HIT may be rejected. 

### Alert
- Do not talk about the task or MTurk, HITs, or other MTurk specific vocabulary during the conversation.
- Please do not reveal personally identifiable information. You may use a pseudonym, fake age, etc.
- Some conversation data could be made **public**. Severely aggressive or insulting chats will lead to a HIT rejection and will be reported.

Finally, make sure you enter your MTurk worker ID accurately.
"""

EXAMPLES = """### Examples:
#### Consistent but not relevant 
A: What did you eat for dinner?  
B: My favourite dish is burritos.  
-> If there is "B likes burritos" in persona list, it's consistent, but it's not a relevant/connected response since A tried to ask what B ate for dinner but B just says B's favourite dish.

#### Relevant but not consistent
A: What did you eat for dinner?  
B: I had burritos.  
(a few turns later)  
A: What did you say you had for dinner?  
B: Fish cakes.  
-> All B's responses are well connected/relevant to what you like to say, but the last response is not consistent with B's first response (I had burritos)."""

CONSENT = """
### Consent
By participating in the chat, you consent to the use of your chat history and evaluation data, and you agree to release KRAFTON from any liability on account of such use.
"""