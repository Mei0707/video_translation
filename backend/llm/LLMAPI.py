# -*- coding: utf-8 -*-
import openai
from typing import List, TypedDict, Dict, Any
import json
import os
import time
import re

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##############################openai####################################

# 配置openai密钥和使用梯子以后的http端口

os.environ["OPENAI_API_KEY"] = "自己的key"
OPENAI_API_KEY="自己的key"
#openai.organization = "org-FpXulnQI2QmOCefMxkwPqlvL"
openai.api_key = "自己的key"
openai.proxy="http://127.0.0.1:21882"


################################openai######################################

################################星火#########################################
import SparkAPI as SparkApi
#星火api不太贵啦
#以下密钥信息从控制台获取
appid = "a875925a"     #填写控制台中获取的 APPID 信息
api_secret = "MDIyNDZjYWQ5ZmJhODJjMTM3ZTNhNDUx"   #填写控制台中获取的 APISecret 信息
api_key ="443c3a0287b60892a4780d3f60f71354"    #填写控制台中获取的 APIKey 信息

#用于配置大模型版本,默认“general/generalv2”
#domain = "general"   # v1.5版本
domain = "generalv2"    # v2.0版本
#云端环境的服务地址
#Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址

text =[]

# length = 0

def getText(role,content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text
    
#################################星火#######################################



class ModelManager:
    def __init__(self):

        self.total_tokens = 0

    def Get_Response_Spark(self,messages):
        # 星火如果要返回json格式需要做一点prompt优化
        SparkApi.answer =""
    
        SparkApi.main(appid,api_key,api_secret,Spark_url,domain,messages)
        response=SparkApi.answer
        print("LLM `s response:")
        print(response)
       
        response=response.replace("\n", " ")
        #answer=json.loads(response)
        # print(str(text))

        return response
        #return answer   
    
    def Get_Response_OpenAI_GPT4(self,messages):
        time.sleep(60)
        response = openai.ChatCompletion.create(
            temperature=0,
            model="gpt-4-1106-preview",
            messages = messages
        )
        print("\nLLM `s response:")
        print(response)
        #print(response.choices[0].message.content)
        response_str=response.choices[0].message.content        
        response_str=response_str.replace("\n", " ")
        print("\nresponse_str:")
        print(response_str)

        try:
            answer = json.loads(response_str)
        except (ValueError, json.JSONDecodeError):          
            # 使用正则表达式精确删除 "json"
            response_str = re.sub(r'\bjson\b', '', response_str)
            
            print("\nsilce response_str:")
            print(response_str)
            try:
                answer = json.loads(response_str)
            except (ValueError, json.JSONDecodeError):
                answer = {"reflex":response_str,
                 "value":5,"error":1}  #出错的设置
           

        return answer
    
    def Get_Response_OpenAI_GPT3_16k(self,messages):
        response = openai.ChatCompletion.create(
            temperature=0,
            model="gpt-3.5-turbo-16k-0613",
            messages = messages
        )
        print("LLM `s response:")
        print(response)
        #print(response.choices[0]["message"]["content"])
        response_str=response.choices[0]["message"]["content"]
        response_str=response_str.replace("\n", " ","'")
        try:
            answer = json.loads(response_str)
        except (ValueError, json.JSONDecodeError):
            # 使用正则表达式精确删除 "json"
            response_str = re.sub(r'\bjson\b', '', response_str)
            
            print("\nsilce response_str:")
            print(response_str)
            try:
                answer = json.loads(response_str)
            except (ValueError, json.JSONDecodeError):
                answer = {"reflex":response_str,
                 "value":5,"error":1}  #出错的设置
        return answer
    
    def get_total_tokens(self) -> int:
        return self.total_tokens




def gpt_3_16k_test():
    mm=ModelManager()
    message=[{
        "role":"user",
        "content":"""不要有任何多余输出，返回括号中的内容（ {"content":"hi"} ) """
    }]
    mm.Get_Response_OpenAI_GPT3_16k(messages=message)

def gpt_4_test():
    mm=ModelManager()
    message=[{
        "role":"user",
        "content":"""不要有任何多余输出，返回括号中的内容（ {"content":"hi"} ) """
    }]
    mm.Get_Response_OpenAI_GPT4(messages=message)

def spark_test():
    mm=ModelManager()
    message=[{
        "role":"user",
        "content":"""不要有任何多余输出，返回括号中的内容（ {"content":"hi"} ) """
    }]
    mm.Get_Response_Spark(messages=message)


if __name__ == "__main__":
    spark_test()