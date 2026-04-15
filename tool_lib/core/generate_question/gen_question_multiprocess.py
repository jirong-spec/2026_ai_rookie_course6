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
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import random
import os
from pathlib import Path

from tool_lib.core.generate_question.ClientManager import ClientManager, MultiClientConfig
from multiprocessing import Pool


DataIndex = 0
class DataFormat:
    def __init__(self, filename, chunk_text, title, idx: Optional[int] = None, **kwargs) -> None:
        
        assert filename, TypeError("[DataFormat Error] 'filename' not found, please follow format of ['filename', 'chunk_text', 'title']")
        self.filename: str = filename
        assert chunk_text, TypeError("[DataFormat Error] 'chunk_text' not found, please follow format of ['filename', 'chunk_text', 'title']")
        self.chunk_text: List[str] = chunk_text
        self.chunk_len: int = len(chunk_text)
        assert title, TypeError("[DataFormat Error] 'title' not found, please follow format of ['filename', 'chunk_text', 'title']")
        self.title: str = title

        global DataIndex
        if not idx:
            self.idx = DataIndex
            DataIndex += 1
        else:
            self.idx = idx

        self.temp_chunk_list : List[str] = None

        self.history_response: List[str] = []

    def get_chunk(self, start_index: int, window_size: int, shuffle: bool = False):
        assert len(self.chunk_text[start_index : start_index + window_size]) > 0, \
                            f"[ERROR] chunk slice is empty ! {start_index}, {start_index + window_size}, {self.chunk_len}"

        if shuffle:
            random.shuffle(self.chunk_text)

        temp_chunk_list = deepcopy(self.chunk_text[start_index : min(start_index + window_size, len(self.chunk_text))])
        
        return temp_chunk_list
    
    def create_question_prompt(self):
        return create_user_prompt(self.title, self.temp_chunk_list )
    

    def get_metadata(self, p_idx, times, startChunk) -> Dict:
        return {'index': f"{p_idx}_{times}_{startChunk}",
                 'filename':self.filename, 
                 'title':self.title, 
                 'chunks':self.temp_chunk_list, 
                 'filename_idx':self.idx}



class DataPool:
    def __init__(self, inputFile: str) -> None:

        self.inputFile = inputFile
        self.paperData : List[DataFormat] = []

        self.load_data(self.inputFile)

    def load_data(self, data_path:str):
        with open(data_path, "r", encoding="utf-8") as fp:
            paper_data = json.load(fp)

        if any(["id" not in p for p in paper_data]):
            for p_idx, p in enumerate(paper_data):
                p["id"] = p_idx
            print("add idx to chunk json file")

            with open(data_path, "w", encoding="utf-8") as fp:
                json.dump(paper_data, fp, indent=4, ensure_ascii=False)

        for pd in paper_data:
            self.paperData.append(DataFormat(**pd))

    def get_data(self, index: Optional[int] = None) -> List[DataFormat]:
        if index :
            return [self.paperData[index]]
        else:
            return self.paperData

        



    
class DatasetGenerator:
    def __init__(self,
                 input_data: List[DataFormat], 
                 output_path: str,
                 clients: List[OpenAIMultiClient], 
                 chunk_windows: int = 5,
                 times_per_paper: int = 1 , 
                 start_id:int = 0, end_id:Optional[int]=None) -> None:

        self.paperData = input_data

        self.outputPath = output_path
        if not os.path.exists(self.outputPath):
            path = Path(self.outputPath)
            path.mkdir(parents=True, exist_ok=True)

        self.clients = clients
        self.QuestionMap = {}

        self.times_per_paper = times_per_paper
        self.chunk_windows = chunk_windows
        
        self.generate_files = []


    
    def question_request(self, role_prompt_generator: Callable, user_prompt_generator: Callable,  
                         paperStartIdx: int = 0, paperEndIdx: Optional[int] = None,
                         ignore_same_file: bool =  False):
    
        def make_requests():

            for p_idx, paper in enumerate(self.paperData[paperStartIdx: paperEndIdx]):
                role_prompt = role_prompt_generator(paper)
                chunk_window_local = min(self.chunk_windows, paper.chunk_len)

                for times in range(self.times_per_paper):
                    for startChunk in range(0, paper.chunk_len-chunk_window_local+1):

                        if ignore_same_file:
                            filename_clean = f"file_{paper.idx}"
                            sub_folder = os.path.join(self.outputPath, filename_clean)
                            file_index = paper.get_metadata(p_idx, times, startChunk)["index"]
                            dump_file = f"generate_question_{filename_clean}_{file_index}.json"
                            if os.path.exists(os.path.join(sub_folder, dump_file)):
                                continue


                        
                        paper.get_chunk(start_index = startChunk, 
                                        window_size = chunk_window_local, 
                                        shuffle = times > 0)
                        
                        all_input = user_prompt_generator(paper)
                        messages =  [{'role': 'system', 'content': role_prompt},{'role': 'user', 'content': all_input}]
                        api.request(data={"messages": messages, "temperature": 0.7, "n": 1, "top_p": 0.5}, 
                                    metadata=paper.get_metadata(p_idx, times, startChunk))

        api = self.clients[0] 

        api.run_request_function(make_requests)



    def question_request_mp(self, paperStartIdx: int = 0, paperEndIdx: Optional[int] = None):
        sliceLen = len(self.paperData[paperStartIdx:paperEndIdx])//len(self.clients)
        sliceLen = sliceLen if len(self.paperData[paperStartIdx:paperEndIdx]) % len(self.clients) else sliceLen + 1

        with Pool(min(4, len(self.clients))) as pool:
            for api_idx, api in enumerate(self.clients):
                pool.apply_async(
                    self.question_request,
                    (api), {"paperStartIdx": api_idx*sliceLen,
                            "paperEndIdx" : min(len(self.paperData[paperStartIdx:paperEndIdx]), (api_idx+1)*sliceLen)}
                )
            pool.close()
            pool.join()

    
    
    def dump_data(self):
        for result in self.clients[0]:
            try:
                response:ChatCompletion = result.response
                filename = result.metadata['filename']
                title = result.metadata['title']
                chunks = result.metadata['chunks']
                index = result.metadata['index']
                filename_idx = result.metadata['filename_idx']
                ori_response = response.choices[0].message.content 
                if ori_response is None:
                    print(f"response error {response}")
                    continue

            
            except json.decoder.JSONDecodeError:
                print("json.decoder.JSONDecodeError")
                continue

            question = ""      
            explanation = ""
            chunkidx = ""

            q_json = {"filename" :filename, "title" : title, "chunk" : chunks , "file_idx" : filename_idx,
                    "question" : question, "explanation": explanation, "chunkidx" : chunkidx, 'full_response':ori_response}

            filename_clean = f"file_{filename_idx}"
            sub_folder = os.path.join(self.outputPath, filename_clean)
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder, exist_ok=True)
            dump_file = f"generate_question_{filename_clean}_{index}.json"
            
            q_json["question_file"] = os.path.join(sub_folder, dump_file)
            with open(q_json["question_file"], "w", encoding="utf-8") as fp:
                json.dump(q_json, fp, indent=4, ensure_ascii = False)
                self.generate_files.append(q_json["question_file"])
            
            print(f"[SUCCESS] dump {os.path.join(sub_folder, dump_file)}")



QUESTION_EVOLVE = ["be more difficult and in-depth than", "be more diverse and different about"]
def create_user_prompt(title:str, chunks:List[str], duplicated_question_list:List[str]):
    print(LIB_PATH)

    with open(LIB_PATH + R"/tool_lib/core/generate_question/userPrompt.json", "r", encoding="utf-8") as fp:
        user_prompt_json = json.load(fp)

    chunk_text = ""
    for c_idx , c in enumerate(chunks):
        chunk_text += f"<Chunk {c_idx}>"
        chunk_text += c
        chunk_text += f"</Chunk {c_idx}>\n"

    all_input = user_prompt_json["input"] % (title, chunk_text)
        
    questions = ""
    for q_idx, q in enumerate(duplicated_question_list):
        questions +=f"<Q:{q_idx}> "
        questions +=f"{q}"
    
    if len(duplicated_question_list) > 0:
        choice = len(duplicated_question_list) %2
        all_input += user_prompt_json["optinalInput"][0] % (QUESTION_EVOLVE[choice], questions)
    if len(duplicated_question_list)%3 == 0:
        all_input += user_prompt_json["optinalInput"][1]

    return all_input


def QuestionGenerator_RolePrompt():

    with open(LIB_PATH + R"/tool_lib/core/generate_question/RolePrompt.json", "r", encoding="utf-8") as fp:
        role_prompt_json = json.load(fp)
 
    
    Roles = role_prompt_json["Roles"]
    role_type = Roles[random.randint(0,2)]
    role_prompt = role_prompt_json["role_prompt"] % (role_type[0], role_type[2], role_type[1])
    role_prompt += role_prompt_json["example"]

    return role_prompt

def ParseLLMQuestion(output_path_folders: List[str], filetered: Callable = None):
    
    final_question = []
    
    question_list = []

    for output_path_folder in output_path_folders:
        assert len(glob(output_path_folder + "/*.json")), FileNotFoundError("question file not found")

        for q_file in glob(output_path_folder + "/*.json"):
            with open(q_file, "r", encoding="utf-8") as fp:
                q_json = json.load(fp)

            if q_json and q_json["question"] not in question_list:
                final_question.append(q_json)
                question_list.append(q_json["question"])


    with open("/".join(output_path_folders[0].split("/")[:-1]) + "/final_question.json", "w", encoding="utf-8") as fp:
        json.dump(final_question, fp, indent=4, ensure_ascii=False)






if __name__ == "__main__":

    startTime = time.time()

    USER_LIST = [("user@example.com", "YOUR_PASSWORD")]
    # 請修改為實際的 IP 地址
    apiUrl = 'http://localhost:3000'
    # IP 地址修改結束


    _multi_client_config = MultiClientConfig()
    _multi_client_config.concurrency = 10
    _multi_client_config.set_datatemplate({"model": "gpt-4o"})
    clientManager = ClientManager(users=USER_LIST, apiUrl=apiUrl, multi_client_config = _multi_client_config)

    api_list :List[OpenAIMultiClient] = clientManager.get_apiList()




    current_datetime = datetime.datetime.now()
    # 請修改為實際的資料夾或檔案路徑
    output_path = "generateData/chunkData" + f"/{current_datetime.strftime('%Y%m%d_%H%M%S')}"

    paperData = DataPool(inputFile = "generateData/chunkData.json").get_data()
    # 路徑修改結束
    datasetGenerator = DatasetGenerator(input_data=paperData,
                                        output_path=output_path,
                                        times_per_paper = 3,
                                        clients = api_list)
    datasetGenerator.question_request(paperStartIdx = 0, paperEndIdx = 20)
    datasetGenerator.dump_data()

    endTime = time.time()

    print("total time" , startTime - endTime)
