from copy import deepcopy
import datetime
import time
from typing import List, Optional
import requests
import json
from openai_multi_client import OpenAIMultiClient, OpenAIMultiOrderedClient, Payload
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import random
import os
from dataclasses import dataclass, asdict


@dataclass
class MultiClientConfig:
    concurrency: int = 10
    max_retries: int = 10
    wait_interval: float = 0
    retry_multiplier: float = 1
    retry_max: float = 60
    endpoint: Optional[str] = "chat.completions"
    data_template: Optional[dict] = None
    metadata_template: Optional[dict] = None
    custom_api=None

    def set_datatemplate(self, data_template: dict):
        self.data_template = data_template

    def set_metadata_template(self, metadata_template: dict):
        self.metadata_template = metadata_template

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}
    

class ClientManager:
    # 請修改為實際的 IP 地址
    def __init__(self, users: List[tuple], 
                 apiUrl: str = 'http://localhost:3000', 
                 multi_client_config: Optional[MultiClientConfig] = None) -> None:
        # IP 地址修改結束
        
        self._url = apiUrl
        self._users = users

        self._tokens = []
        self.get_tokens(LoginURL = apiUrl + "/login")

        self.apiList = []
        self.apiCount = 0

        self._multi_client_config = multi_client_config.to_dict()
        self.create_multi_client()


    
    def get_tokens(self, LoginURL):

        headers = {'Content-Type': 'application/json'}
        for user in self._users:
            data = {    
                "username": user[0],    
                "password": user[1]
            }
            try:
                response = requests.post(LoginURL, headers=headers, data=json.dumps(data))
                respone_token = response.json()
                self._tokens.append(respone_token["token"])
            except requests.exceptions.JSONDecodeError:
                print(f"The {user[0]} is invalid!")
                continue

        print(f"Total valid users:{len(self._tokens)}")
    

    def custom_api(self, payload: Payload):
        if self._multi_client_config["endpoint"] == "chat.completions":   
            return self.aclient.chat.completions.create(**payload.data)
        else:
            TypeError("endpoint not support yet")


    def create_multi_client(self, clean_api = True):
        assert len(self._tokens) > 0, "[create_multi_client error] no valid user !!!!"
        
        if clean_api:
            self.clean_apiList()

        for token in self._tokens:
            self.aclient = AsyncOpenAI(api_key=token, base_url=self._url, timeout=120)

            self._multi_client_config["custom_api"] = self.custom_api

            try:
                api = OpenAIMultiClient(**self._multi_client_config)
            except TypeError:
                api = OpenAIMultiClient(self.aclient, 
                                        **self._multi_client_config)
            
            self.apiList.append(api)
        self.apiCount = len(self.apiList)
        print(f"request {self.apiCount} client complete")

        return self.apiList

    def clean_apiList(self):
        self.apiList = []

    def get_apiList(self) -> List[OpenAIMultiClient]:
        return self.apiList
    
if __name__ == "__main__":
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

    print(USER_LIST)
