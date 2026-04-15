
'''
this file is for cleaning redundence word or misleading phrase in training data 
'''

import os
import json
from typing import List

def title_first(text: str):
    textList = text.split(" ")
    if len(textList) > 1:
        textList[0] = textList[0].title()
        text = " ".join(textList)
    else:
        text = text.title()

    return text


def create_word_conbination(word: str, suffix : List[str] = [":", "s", "s:"]):
    """ create all possibility of a single word that might exist in the article"""
    word = word.lower()
    all_combination = []
    for v in suffix:  #[":", "s", "s:", "\n", ":\n", "s\n", "s:\n"]:
        if (word[-1] == "." or word[-1] == "s") and "s" in v:
            continue
        all_combination.append(word + v)
    
    all_combination += ["\n" + c for c in all_combination]
    all_combination += [title_first(c) for c in all_combination]

    return all_combination


def append_appendix_discription(data:str, appendix_keyword:List[str], discription: str):
    '''
    add more discription at the end of data if meet appendix 
    ex. if 'figure' in data, output would become '<data> Where the {figure} is from {discription}'
    '''
    
    exist_appendix = []
    for a in appendix_keyword:
        if a in data:
            exist_appendix.append(a)

    if len(exist_appendix) == 1:
        data += f" Where the {exist_appendix[0]} is from {discription}"
    elif len(exist_appendix) > 1:
        verbose_text = ", ".join(exist_appendix)
        data += f" Where the {verbose_text} are from {discription}"

    return data


def replace_vague_text(data:str, vague_text:List[List[str]], replace_text: List[str], extend_combination: bool = False):
    '''
    replace misleading text using target text
    vague_text list should be paired with replace text
    input :
        vague_text : keyword list of vague text, need to be paired with replace text 
        replace_text : text to replace to the paired vague text
        extend_combination : add 's', ':' , capitialize the vaguetext
    ex. if 'presented content' in data, output would become 'presented content of {replace text}'
    '''
    assert len(vague_text) == len(replace_text), TypeError("vague_text list should be paired with replace text (need to be same length)")

    for v_idx, vague in enumerate(vague_text):
        if extend_combination:
            vague = create_word_conbination(vague)
        for v in vague:
            if v in data:
                data = data.replace(v, f"{v} " + replace_text[v_idx])
                return data
            
    return data


def remove_redundence_section(data:str, suffix_list:List[str]):
    '''remove all the text after when meet keyword for redundence section'''

    new_data = ""
    for suffix in suffix_list:
        if suffix in data:
            if len(data.split(suffix))==2:
                new_data = data.split(suffix)[0]
                break
            else:
                print(f"[WARN] '{suffix}', '{data}'")
                print("------------------")
    if not new_data:
        print(f"[WARN] The data doesn't have suffix keyword. '{data}'")
        return data
    
    return new_data


def drop_redundance_suffix(data:str, 
                        backcheck_reject_list:List[str], 
                        backcheck_accept_list:List[str]):
    
    '''drop redundance symbol or charactor at the end of texts'''
    for i in range(1, len(data)):
        if data[-i] in backcheck_accept_list or data[-i] not in backcheck_reject_list:
            if i == 1:
                return data
            else:
                i = i-1
                return data[:-i]
            
    return data


def read_json_file(data_path:str):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json_file(file_path:str, json_list:list):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(json_list, file, indent = 4,ensure_ascii=False)

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def clean_bopomofo(text):
    # Mapping of Bopomofo characters to their corresponding alphabetic representations
    bopomofo_mapping = {
        'ㄅ': 'b', 'ㄆ': 'p', 'ㄇ': 'm', 'ㄈ': 'f', 'ㄉ': 'd', 'ㄊ': 't',
        'ㄋ': 'n', 'ㄌ': 'l', 'ㄍ': 'g', 'ㄎ': 'k', 'ㄏ': 'h', 'ㄐ': 'j',
        'ㄑ': 'q', 'ㄒ': 'x', 'ㄓ': 'zh', 'ㄔ': 'ch', 'ㄕ': 'sh', 'ㄖ': 'r',
        'ㄗ': 'z', 'ㄘ': 'c', 'ㄙ': 's', 'ㄚ': 'a', 'ㄛ': 'o', 'ㄜ': 'e',
        'ㄝ': 'eh', 'ㄞ': 'ai', 'ㄟ': 'ei', 'ㄠ': 'ao', 'ㄡ': 'ou', 'ㄢ': 'an',
        'ㄣ': 'en', 'ㄤ': 'ang', 'ㄥ': 'eng', 'ㄦ': 'er', 'ㄧ': 'yi', 'ㄨ': 'wu',
        'ㄩ': 'yu', 'ˇ': '3', 'ˋ': '4', '˙': '.'
    }

    cleaned_text = ''
    has_bpm = False
    for char in text:
        
        if char in bopomofo_mapping:
            has_bpm = True
            continue
        else:
            cleaned_text += char
    if has_bpm:
        return cleaned_text.replace("\n", "")
    else:
        return cleaned_text

if __name__ == "__main__":
    json_file = R"script\chunktextFilter\file_0.json"
    jsondata = read_json_file(json_file)[0]

    json_title = jsondata["title"]
    json_chunk = jsondata["chunk"]
    json_question = jsondata["question"]
    json_answer = jsondata["answer"]

    


    ####### replace vague phrase ########
    appendix_keyword = ["figure", "fig.", "table", "tab.", "theorem", "algorithm" , "lemma", "equation"]
    appendix_keyword_allcombine = [k_combine for keyword in appendix_keyword for k_combine in create_word_conbination(keyword)]
    
    vague_phrase = [["section" , "in the content" ,"presented content", "given content", "provided text", "the text", "proposed method"],
                    ["experiments" , "experiment" , "content provided", "text provided", "provided text", "information provided"],
                    ["the paper" , "this paper", "the study"]]
    

    for c_idx,  chunk in enumerate(json_chunk):
        ## add discription for the appendix to the input data
        chunk = append_appendix_discription(data=chunk, appendix_keyword=appendix_keyword, discription=json_title)
        ## replace vague text in chunk
        chunk = replace_vague_text(data=chunk,
                                   vague_text=vague_phrase,
                                   replace_text=[f"of '{json_title}'", f"in '{json_title}'", f"'{json_title}'"])

        json_chunk[c_idx] = chunk



    ####### remove redundance text in answer ########
    redundance_keyword = ["reference"]
    redundance_keyword_allcombine = [r_combine for keyword in redundance_keyword for r_combine in create_word_conbination(keyword)]
    json_answer = remove_redundence_section(data=json_answer, suffix_list=redundance_keyword_allcombine)

    REJECT_LIST = [
        "#",
        " ",
        "<",
        ">",
    ]
    ACCEPT_LIST = [
        ".",
        "\n",
        "?",
        "!",
    ]
    json_answer = drop_redundance_suffix(data=json_answer, 
                                            backcheck_reject_list=REJECT_LIST, 
                                            backcheck_accept_list=ACCEPT_LIST)
