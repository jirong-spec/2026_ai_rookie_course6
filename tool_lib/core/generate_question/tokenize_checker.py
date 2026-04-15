from transformers import AutoTokenizer
from glob import glob
import numpy as np
import tqdm
import json
from tool_lib.core.pdf2txt.Clean_text import  read_json_file, save_json_file


def token_length_calculator(text, tokenizer_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenized_text = tokenizer.tokenize(text)
    token_length = len(tokenized_text)
    
    return token_length

def tokensizeFilter(data, tokenizer_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct" , token_limit = 4000):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    token_limit -= 50
    
    filtered_data = []
    
    for d in data:
        q = d["question"]
        a = d["answer"]
        c = d["hybrid_chunks"]
        
        all_text = " ".join(q+a+c)

        tokenized_text = tokenizer.tokenize(all_text)
        token_length = len(tokenized_text)
        if token_length < token_limit:
            filtered_data.append(d)
    
    
    print(f"oridata len {len(data)}, filter len {len(filtered_data)}")
    
    return filtered_data
        
if __name__ == "__main__":
    
    # 請修改為實際的資料夾或檔案路徑
    question_file = R"/path/to/inference_data.json"
    # 路徑修改結束
    q_json = read_json_file(question_file)
    tksize = []
    for q in tqdm.tqdm(q_json):
        lens = token_length_calculator(q["question"] + " ".join(q["RAG_chunks"][:45]))
        tksize.append(lens)
        
    
    print(np.mean(tksize))
    print(np.max(tksize))
    print(np.min(tksize))
        
    exit()
    
    # 請修改為實際的資料夾或檔案路徑
    trainjsonfile = R"/path/to/training_data.json"
    testjsonfile = R"/path/to/testing_data.json"
    # 路徑修改結束

    with open(trainjsonfile, "r") as fp:
        trainjsonData = json.load(fp)
    with open(testjsonfile, "r") as fp:
        testjsonData = json.load(fp)
        
    # Load the tokenizer for the Alibaba-NLP/gte-Qwen2-7B-instruct model
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct")
    token_limit = 4000

    trainjsonData_filt = tokensizeFilter(trainjsonData )   
    testjsonData_filt = tokensizeFilter(testjsonData)  

    save_json_file(trainjsonfile.replace(".json", "_filter.json"), trainjsonData_filt)
    save_json_file(testjsonfile.replace(".json", "_filter.json"), testjsonData_filt)
        
    exit()
        

    # token_counter = []
    # for f in tqdm.tqdm(glob(R"/path/to/TempFile/*")):

    # 請修改為實際的資料夾或檔案路徑
    jsonfile = R"/path/to/final_question_answer_conversation.json"
    # 路徑修改結束

    with open(jsonfile, "r") as fp:
        jsonData = json.load(fp)
        
    # Load the tokenizer for the Alibaba-NLP/gte-Qwen2-7B-instruct model
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        
    test = jsonData[0]
    chunk = test["chunk"][0]
    # Tokenize the input text
    tokenized_text = tokenizer.tokenize(chunk)
    token_length = len(tokenized_text)
    print(f"single chunk length {token_length}")


    tk_size_q = []
    tk_size_a_max = []
    tk_size_a_min = []
    tk_size_a_all = []

    tk_chunk = []
    for j in jsonData:
        q = j["question"]
        q = " ".join(q)
        tokenized_text = tokenizer.tokenize(q)
        tk_size_q.append(len(tokenized_text))
        
        a_size = []
        for a in j["answer"]:
            tokenized_text = tokenizer.tokenize(a)
            a_size.append(len(tokenized_text))
        tk_size_a_min.append(min(a_size))
        tk_size_a_max.append(max(a_size))
        tk_size_a_all.append(sum(a_size))
        
        tk_chunk += [len( tokenizer.tokenize(c)) for c in j["chunk"]]
        

    print("question size")
    print(np.median(tk_size_q), np.mean(tk_size_q), np.max(tk_size_q), np.min(tk_size_q))
    print("answer size")
    print(np.median(tk_size_a_all), np.mean(tk_size_a_all), np.mean(tk_size_a_max), np.mean(tk_size_a_min))
    print("tk_chunk size")
    print(np.median(tk_chunk), np.mean(tk_chunk), np.max(tk_chunk), np.min(tk_chunk))