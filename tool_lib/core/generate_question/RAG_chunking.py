import json
import os
from glob import glob
from datetime import datetime
import logging

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from time import time

from copy import deepcopy
from tool_lib.core.pdf2txt.Clean_text import  read_json_file, save_json_file

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4

class RAGChunking:
    def __init__(self,  
                 chromaDB_file: Optional[str] = None,
                 db_collection_name = "db-for-fine-tuning",
                 model_name = "intfloat/multilingual-e5-large",
                 device = "cuda",
                 chunk_size= 256,
                 chunk_overlap = 32) -> None:


        self.embedding_model = self.create_embedding_model(model_name, device)
        self.text_splitter = self.create_textsplitter(chunk_size, chunk_overlap)
        
        self.db_collection_name = db_collection_name
        # 請修改為實際的資料夾或檔案路徑
        self.chromaDB_file = "/home/user/workspace/data/chroma_db/"
        # 路徑修改結束
        
    def init(self, txtfileFolder: str,
            src_data_type_pattern = ["*.txt"],):
        self.txtfileFolder = txtfileFolder
        loader = DirectoryLoader(path=txtfileFolder + "/docs_txt", glob=src_data_type_pattern, loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
        docs = loader.load()
        logging.info("RAG DB load %d files", len(docs))
        self.vectorstore = self.create_vectorstore_by_docs(docs = docs, embedding_model = self.embedding_model, 
                                        text_splitter = self.text_splitter, collection_name = self.db_collection_name)

    def create_vectorstore_by_docs(self, docs:List[Document], embedding_model:Embeddings,   text_splitter:TextSplitter, collection_name:str):
        chunked_docs = text_splitter.split_documents(docs)
        
        index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        uuids = [str(uuid4()) for _ in range(len(chunked_docs))]

        vector_store.add_documents(documents=chunked_docs, ids=uuids)
        return vector_store


    def create_vectorstore_by_docs_ori(self, docs:List[Document], embedding_model:Embeddings, text_splitter:TextSplitter, collection_name:str): 
        print("[Start] create db")
        self.chromaDB_file = self.txtfileFolder + "/chroma_db_" + datetime.now().strftime('%Y%m%d_%H%M%S')
        
        chunked_docs = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embedding_model,
                                            collection_name=collection_name, persist_directory=self.chromaDB_file)
        
        
        print(f"vectore store at {self.chromaDB_file}")
        print("[End] create db")
        return vectorstore

    def create_vectorstore(self, embedding_model:Embeddings, persist_directory:str, collection_name:str):
        print("[Start] create db")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)
        print("[End] create db")
        return vectorstore

    def create_embedding_model(self, model_name:str, device:str):
        embed_model_kwargs = {'device': device, 
                            "trust_remote_code": "True"}
        encode_kwargs = {'normalize_embeddings': True}
        hf_embedding = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs=embed_model_kwargs,
                                                encode_kwargs=encode_kwargs)
        return hf_embedding

    def create_textsplitter(self, chunk_size:int, overlap:int):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return text_splitter
    

    def create_topk_retriver(self, search_type:str,top_k:int, fetch_k:int):
        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': top_k, 'fetch_k': fetch_k})
    

    def retrive_RAG_data(self, retriever: VectorStoreRetriever, tagetText: str) -> List[str]:
        high_score_chunks = retriever.get_relevant_documents(query=tagetText)

        RAG_chunks = []
        for chunk in high_score_chunks:
            page_content = chunk.page_content
            metadata = chunk.metadata  

            chunks = "file name:" + metadata["source"].split("/")[-1] + " content: " + page_content
            RAG_chunks.append(chunks)

        return RAG_chunks
    

    def merge_chunk(self, golden_chunk: List[str], RAG_chunks: List[str], total_chunk_count : int):
        merge_chunk = []

        if len(golden_chunk) >= total_chunk_count:

            return golden_chunk[:total_chunk_count]
        
        else:
            merge_chunk = deepcopy(golden_chunk)
            total_chunk_count -= len(golden_chunk)
            for r_chunk in RAG_chunks:
                
                if r_chunk in merge_chunk:
                    continue
                
                else:
                    merge_chunk.append(r_chunk)
                    total_chunk_count -= 1

                if total_chunk_count == 0:
                    break
            
            if total_chunk_count > 0:
                print("final chunk less than set total chunk")

        return merge_chunk[len(golden_chunk):] + merge_chunk[:len(golden_chunk)] 
     

def split_train_test_dataset(data, testset_count, saveFolder = "",  save = False):

    if saveFolder != "":
        save = True

    fileCount = {}
    trainSet = []
    testSet = []

    for d in data:
        if d["filename"] in fileCount:
            fileCount[d["filename"]].append(d)
        else:
            fileCount[d["filename"]] = [d]

    
    for d in data: 
        if len(fileCount[d["filename"]]) > testset_count:
            testSet += fileCount[d["filename"]][:testset_count]
            trainSet += fileCount[d["filename"]][testset_count:]
            
        else:
            trainSet += fileCount[d["filename"]]
    
    
    if save :
        with open(saveFolder + "/train_dataset.json", "w", encoding= "utf-8") as fp:
            json.dump(trainSet, fp, indent=4)
        with open(saveFolder + "/test_dataset.json", "w", encoding= "utf-8") as fp:
            json.dump(testSet, fp, indent=4)

    return trainSet, testSet




if __name__ == "__main__":

    # 請修改為實際的資料夾或檔案路徑
    all_folder = ["/home/user/workspace/dataset/source"]
    # 路徑修改結束

    for txt_folder in all_folder:
        txt_folder = txt_folder.replace("prescription", "prescription/temptxt")
        chunk_output_path = txt_folder + "/question_temp"

        output_json_file = chunk_output_path + "/final_question_answer_conversation.json"
        chunk_json_file = txt_folder + "/chunkfile.json"

        questionJson = read_json_file(output_json_file)

        rAGChunking = RAGChunking()
        retriever = rAGChunking.create_topk_retriver(search_type= "mmr", top_k=8, fetch_k=30)

        for data in questionJson:
            question = data["question"][0]
            goldenchunk = data["chunk"]

            RAG_chunks = rAGChunking.retrive_RAG_data(retriever = retriever, tagetText = question)

            merge_chunk = rAGChunking.merge_chunk(golden_chunk=goldenchunk, RAG_chunks=RAG_chunks, total_chunk_count=5)
            data["hybrid_chunks"] = merge_chunk


        trainSet, testSet = split_train_test_dataset(data = questionJson, testset_count = 2)
        
        save_json_file(txt_folder + "/trainingData.json", trainSet)
        save_json_file(txt_folder + "/testingData.json", testSet)
