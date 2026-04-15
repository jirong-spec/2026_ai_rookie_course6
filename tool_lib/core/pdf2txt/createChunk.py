import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json 
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def ChunkText(docs, chunk_size = 1024, chunk_overlap = 256):
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function = len)
    texts = text_splitter.split_text(docs)

    return texts

def create_chunk_from_txt(txt_folder, chunk_size = 1024, chunk_overlap = 256, contentLenLimit = 100, save_file_name = None, plot = False, add_metadata = True):
    if not save_file_name:
        save_file_name = txt_folder + "/chunkData.json"

    finalJson = []

    chunkCountList = []

    totalPaper = 0
    failedPaper = 0
    titleJson = {}

    if os.path.exists(txt_folder + "/titleJson.json"):
        with open(txt_folder + "/titleJson.json", "r", encoding="utf-8") as fp:
            titleJson = json.load(fp)
    else:
        assert FileNotFoundError("please place titleJson.json(create in PDFParser) under txt files folder")


    for txtf in tqdm(glob(txt_folder + "/docs_txt/*txt")):
        filename = txtf.split("/")[-1].replace(".txt", "")
        totalPaper += 1
        if filename in titleJson:
            title = titleJson[filename]

            with open(txtf, "r", encoding="utf-8") as fp:
                content = fp.read()

            if len(content) < contentLenLimit:
                failedPaper += 1
                continue

            chunks = ChunkText(content, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            if add_metadata == True:
                chunks = ["file name:" + filename + "\ncontent: " + s for s in chunks]

            chunkCountList.append(len(chunks))
            
            ProcessData = {"filename" : title,
                           "chunk_text" : chunks,
                           "chunk_length" : len(chunks), 
                           "title" : title}
            finalJson.append(ProcessData)


        else:
            failedPaper += 1
            continue

    print("totalPaper :", totalPaper, "failedPaper :", failedPaper)
    print("mean chunkCount", np.mean(chunkCountList),
          "max chunkCount", np.max(chunkCountList), 
          "min chunkCount",np.min(chunkCountList))
        
    with open(save_file_name, "w", encoding="utf-8") as fp:
        json.dump(finalJson, fp, indent=4, ensure_ascii=False)
        print(f"chunk file save at {save_file_name}")

    
    if plot:
        plt.hist(chunkCountList, bins=50)
        plt.xlim([0, 200])
        plt.xlabel("chunk count")
        plt.ylabel("paper with this chunk count")
        plt.show()


if __name__ == "__main__":
    pass
    # txt_folder = R"D:\work\Dev\chunkGenerator\nand_paper_10_txt\nand_2_paper_TEMPTXT"
    # create_chunk_from_txt(txt_folder, chunk_size = 1024, chunk_overlap = 256, plot = False)