import openai

OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY

import os
import pickle
import json
from colorama import Fore

RESPONSE_DIR = "gpt_responses"

class ChatGPTModule:
    def __init__(self, response_dir=RESPONSE_DIR):
        self.response_dir = response_dir
        os.makedirs(response_dir, exist_ok=True)

    def save_cache(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.cache, f)

    def start_session(self, session_title):
        session_fname = os.path.join(self.response_dir, f"{session_title}.json")
        self.session_fname = session_fname
        self.message_lst = []

        if os.path.exists(session_fname):
            with open(session_fname, 'r') as f:
                self.cache = json.load(f) 
        else:
            self.cache = {}
            self.save_cache(session_fname)

    def chat(self, prompt, context=None):
        if context:
            self.message_lst.append({"role": "system", "content": context})
            print(Fore.BLUE + "System: " + context)
            print(Fore.BLACK)

        if prompt:
            self.message_lst.append({"role": "user", "content": prompt})
            print(Fore.RED + "User: " + prompt)
            print(Fore.BLACK)

        concatenated_message = '\n'.join([m["role"] + ": " + m["content"] for m in self.message_lst])
        # print(concatenated_message)
        
        if concatenated_message in self.cache:
            print(Fore.MAGENTA + "Found in cache ...")
            result = self.cache[concatenated_message]
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.message_lst
            )
            finish_reason = response["choices"][0]["finish_reason"]
            if finish_reason != "stop":
                print("warning: the response is truncated")

            result = response["choices"][0]["message"]["content"]
            self.cache[concatenated_message] = result
            self.save_cache(self.session_fname)        

        self.message_lst.append({"role": "assistant", "content": result})
        print(Fore.GREEN + "AI: " + result)
        print(Fore.BLACK)
        return result
            

if __name__ == "__main__":

    # //// Example ////////////////////////////////////////////////////////////////////////////////////////
    chat_module = ChatGPTModule()
    chat_module.start_session("test_run")

    chat_module.chat(prompt="How to place a bowl containing food into a dishwasher?", context="Your job is to tell how to accomplish some task.")
    # chat_module.chat(prompt="Nice job! Next tell me where to place the bowl after it has been washed. I see a bin and a rack.")
    # chat_module.chat(prompt="write code for drawing a circle on a paper. You can use numpy and math libraries. The function you should make use of is `move(position)`, where `position` is a 2d array and moves the pencil to `position` on the paper.")
    # ////////////////////////////////////////////////////////////////////////////////////////////////////


