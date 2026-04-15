import os
import sys
LIB_PATH = os.path.abspath(os.path.join(os.path.abspath(''), '..'))
sys.path.insert(0, LIB_PATH)
import tool_lib

from copy import deepcopy
import datetime
import time
from typing import List, Optional, Dict, Callable
import requests
import json
from glob import glob
import openai
from openai_multi_client import OpenAIMultiClient, OpenAIMultiOrderedClient
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
import random

from pathlib import Path

from tool_lib.core.generate_question.ClientManager import ClientManager, MultiClientConfig
from multiprocessing import Pool
from tool_lib.core.pdf2txt.Clean_text import read_json_file, save_json_file
from tool_lib.core.generate_question.tokenize_checker import token_length_calculator


MODELNAME = "4o"
EXPNAME = 'llama31'
# 請修改為實際的 IP 地址
ENDPONT = "http://localhost:8791/v1"
# IP 地址修改結束
# 請修改為實際的 Token
KEY = 'YOUR_ACCESS_TOKEN'
# Token 修改結束

# 請修改為實際的資料夾或檔案路徑
JSONFILE = "/home/user/workspace/data/merge.json"
SAVEPATH = "./log/dataset/"
# 路徑修改結束
TOTALCOLLECTFILE =  f"{SAVEPATH}/{EXPNAME}.json"

class AnswerGenerator:

    def __init__(self, 
                 input_json_file: str,
                 output_json_file: str,
                 refernce_chunk_file: str,
                 clients: ClientManager = None):
        
        self.input_json_file = read_json_file(input_json_file)
        self.output_json_file = output_json_file
        self.clients = clients

        self.reference_data = read_json_file(refernce_chunk_file)

    def return_answer(self, prompt:str):

        system_prompt = self.get_system_prompt()
        
        if MODELNAME == '4o':

            # 請修改為實際的 IP 地址
            url = 'http://localhost:3000'
            # IP 地址修改結束
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {KEY}'
            }

            data = {
                'model': 'gpt-4o',
                'messages': [
                    {
                        'role': 'system', 
                        'content': system_prompt
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                "temperature": 0, 
                "n": 1, 
                "top_p": 0.00001
            }

            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()
            output = response_json['choices'][0]['message']['content']



        else:
            client_qwen = OpenAI(
                api_key='EMPTY',
                base_url=ENDPONT
            )

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }            
            ]
            response = client_qwen.chat.completions.create(model=MODELNAME, messages=messages, temperature=0, n=1, top_p=0.00001)
   
            output = response.choices[0].message.content

        return output
    


    def answer_request(self, api: OpenAIMultiClient = None, 
                       reference_key: str = "",
                       ignore_same_file: bool =  False):
        
        assert reference_key != "", TypeError("please provide key to extract refernce from refernce json")
        
        output_list = []

        for idx, data in enumerate(self.input_json_file):
            
            filename = data['filename'] 
            question = data['question']
            user_prompt = self.create_prompt(data['question'], data[reference_key])
            answer = self.return_answer(user_prompt)
            
            output_dict = {
                "filename": filename, 
                "retriver_chunk_text": data[reference_key],
                "question": question, 
                "answer": data['answer'] if 'answer' in data else None, 
                "model_response": answer 
            }

            output_list.append(output_dict)

        save_json_file(self.output_json_file, output_list)

    def get_reference(self, search_pair: tuple, reference_key: str):
        index = next((index for index, item in enumerate(self.reference_data) if item.get(search_pair[0]) == search_pair[1]), None)
        return self.reference_data[index][reference_key]

    def answer_request_multiClient(self, 
                                    answer_role_prompt: Callable, 
                                    answer_user_prompt: Callable, 
                                    reference_key: str = "retriver_chunk_text",
                                    refernce_type: str = "from_chunk",
                                    ignore_same_file: bool =  False):
        
        assert reference_key != "", TypeError("please provide key to extract refernce from refernce json")
        

        def make_requests():
            for idx, data in enumerate(self.input_json_file):
                if refernce_type == "from_chunk": 
                    reference = self.get_reference(search_pair = ("filename", data['filename']) , reference_key = reference_key)
                    
                    token_len = token_length_calculator(" ".join(reference))
                    assert token_len < 4000, ValueError("Input refernce too large (> 4k token)")
                    
                else:
                    reference = data['chunk']

                user_prompt = answer_user_prompt(data['question'], reference)
                system_prompt = answer_role_prompt()

                messages =  [{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': user_prompt}]
                api.request(data={"messages": messages, "temperature": 0.7, "n": 1, "top_p": 0.1}, 
                            metadata=data)
                
                
            

        api = self.clients[0] 

        api.run_request_function(make_requests)

    
    def dump_data(self):

        final_result = []
        for result in self.clients[0]:
            try:
                response:ChatCompletion = result.response
                question_data = result.metadata

                ori_response = response.choices[0].message.content 
                if ori_response is None:
                    print(f"response error {response}")
                    continue

            
            except json.decoder.JSONDecodeError:
                print("json.decoder.JSONDecodeError")
                continue

            question_data["answer"] = ori_response
            
            save_json_file(question_data["question_file"], question_data)
            savefile = question_data["question_file"]
            print(f"save answer {savefile}")

            final_result.append(question_data)

        
        save_json_file(self.output_json_file, final_result)




if __name__ == "__main__":

    # 請修改為實際的資料夾或檔案路徑
    input_json_file = "/home/user/workspace/data/final_question.json"
    output_json_file = "/home/user/workspace/data/final_question_answer.json"
    refernce_json_file = "/home/user/workspace/data/chunkfile.json"
    # 路徑修改結束

    USER_LIST = [("user@example.com", "YOUR_PASSWORD")]
    # 請修改為實際的 IP 地址
    apiUrl = 'http://localhost:3000'
    # IP 地址修改結束
    

    _multi_client_config = MultiClientConfig()
    _multi_client_config.concurrency = 1
    _multi_client_config.max_retries = 1


    _multi_client_config.set_datatemplate({"model": "gpt-4o"})
    clientManager = ClientManager(users=USER_LIST, apiUrl=apiUrl, multi_client_config = _multi_client_config)
    api_list :List[OpenAIMultiClient] = clientManager.get_apiList()

    answerGenerator = AnswerGenerator(input_json_file = input_json_file,
                                      output_json_file = output_json_file,
                                      refernce_chunk_file = refernce_json_file,
                                      clients = api_list)

    answerGenerator.answer_request_multiClient(reference_key= "chunk_text")
    answerGenerator.dump_data()
