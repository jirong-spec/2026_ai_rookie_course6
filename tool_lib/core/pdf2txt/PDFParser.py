import json 
from glob import glob
import os 
from pathlib import Path
from typing import Callable
from PyPDF2 import PdfReader, PdfFileReader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import pymupdf
import re
from tool_lib.core.pdf2txt.Clean_text import clean_bopomofo
from tqdm import tqdm
import logging
from tool_lib.core.pdf2txt.ocr_client import llm_response
from PIL import Image
from pdf2image import convert_from_path
import tool_lib.core.pdf2txt.fill_markdown_table as fill_markdown_table

def pdfMining(file_path):

    Extract_Data=[]

    fontSizeCollection = {}

    for page_layout in extract_pages(file_path, maxpages=3):
        for element in page_layout:
            if isinstance(element, LTTextContainer):

                try:
                    for text_line in element:
                        for character in text_line:
                            if isinstance(character, LTChar):
                                Font_size=character.size
                except:
                    Font_size = 0

                Extract_Data.append([Font_size,(element.get_text())])

                if round(Font_size, 1) in fontSizeCollection:
                    fontSizeCollection[round(Font_size, 1)] += len(Extract_Data[-1][1])
                else:
                    fontSizeCollection[round(Font_size, 1)] = len(Extract_Data[-1][1])

    averagesize = max(fontSizeCollection, key=fontSizeCollection.get)
    maxsize = max(list(fontSizeCollection.keys()))

    print("fontSizeCollection", fontSizeCollection)

    title = getTitleByFont(Extract_Data, averagesize, maxsize)
    return title


def getTitleByFont(Extract_Data, avgFontSize, maxsize):

    title = ""
    for ed in Extract_Data[:10]:
        if ed[0] < avgFontSize + 1 :
            continue
        if any(element in ed[1].lower() for element in ["vol.", "no.", "session", "journal"]):
            continue

        title += ed[1]

        if "," in ed[1] and len(title) > 0:
            break
    
    return title


def get_pdftxt(file_path):
    doc = pymupdf.open(file_path) 
    text = ""
    for page in doc: 

        tabs = page.find_tables()
        for tab in tabs:
            df = tab.to_pandas()
            text += df.to_csv(index=False)

            page.add_redact_annot(tab.bbox)  
        page.apply_redactions()  

        text += page.get_text() 


    return text


def CleanTxt(txtData, startKeyword:list = None, endKeyword:list = None):
    if not startKeyword and not endKeyword:
        return txtData

    cleandata = []
    startRecord = False
    txtData = txtData.split("\n")

    meet_conclusion = False

    for td_idx,  td in enumerate(txtData):
        td = td.lower()
        if not startRecord and any([sk in td for sk in startKeyword]):  
            
            for sk in startKeyword:
                if sk in td:
                    containKeyWord = sk
            
            startRecord = True
            if td == containKeyWord:
                continue
            elif containKeyWord in td:
                cleandata += td.split(containKeyWord)[1:]

        elif startRecord:
            if td_idx > len(txtData)*5 // 10:
                if any([ek in td for ek in endKeyword]):
                    meet_conclusion = True
                    for ek in endKeyword:
                        if ek in td:
                            containKeyWord = ek
                    cleandata.append(td.split(containKeyWord)[0])
                    break

                

            cleandata.append(txtData[td_idx])
            
    cleandata = "\n".join(cleandata)

    return cleandata


def gettitle(articleContent):
    articleContent = articleContent.lower()
    articleContent = articleContent.split("\n")

    if any(element in articleContent[0] for element in ["vol.", "no.", "session", "journal", "paper", "IEEE"]):
        articleContent.pop(0)

    articleContent[0] = articleContent[0].strip()
    for x in range(0, 5):

        articleContent[x] = articleContent[x].strip()

        if len(articleContent[x]) <= 1:
            continue

        if "," in articleContent[x]:
            title = " ".join(articleContent[0:x])

    return title


def fixPdf(file):
    doc = pymupdf.open(file)
    num_pages = len(doc)
    doc.close()
    return num_pages


def get_pdftxt_by_ocr(file_path, PDFFolder, chunk_size = 4):
    print("file_path", file_path)
    logging.info("file_path: %s", file_path)
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    print("pdf_name", pdf_name)
    image_dir =  PDFFolder + f"/docs_txt/{pdf_name}"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    images = convert_from_path(file_path)

    image_path_list = []
    for i, image in enumerate(images):
        image_path = f"{image_dir}/{i+1}.png"
        image.save(image_path, "PNG")
        image_path_list.append(image_path)
    
    all_responses = []
    for i in tqdm(range(0, len(image_path_list), chunk_size), desc="Processing"):
        chunk = image_path_list[i:i + chunk_size]
        response = llm_response(chunk)
        if response is not None:
            all_responses.append(response)

    final_response = ''.join(all_responses)  

    pdf_text = fill_markdown_table.process_text(final_response)
    pdf_text = fill_markdown_table.remove_consecutive_spaces(pdf_text)

    return pdf_text

def PDFParser(PDFFolder, Temp_txt_folder,
              cleantxt_startkey: list = None, 
              cleantxt_endkey: list = None, 
              extract_title_rule: Callable = None,
              method: str = "fast",
              ocr_chunk_size: int = 5):

    types = ('*.pdf', '*.PDF')
    articleFile = []
    for files in types:
        articleFile.extend(glob(PDFFolder + "/" + files))
    
    titleJson = {}
    if not os.path.exists(Temp_txt_folder):
        path = Path(Temp_txt_folder)
        path.mkdir(parents=True, exist_ok=True)


    if not os.path.exists(Temp_txt_folder + "/titleJson.json"):
        titleJson = {}
    else:
        with open(Temp_txt_folder + "/titleJson.json", "r", encoding="utf-8") as fp:
            titleJson = json.load(fp)

    directory = os.path.dirname(Temp_txt_folder + "/docs_txt")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for af in articleFile:
        print(f"[INFO] 現在開始處理第 {articleFile.index(af) + 1} / {len(articleFile)} 個檔案: {af}")
        try:
            num_pages = fixPdf(af) 
            fileName = af.split("/")[-1].lower().replace(".pdf", "")
            txtFile = Temp_txt_folder + f"/docs_txt/{fileName}.txt"
            
            tag = 0
            if os.path.exists(txtFile):
                with open(txtFile, "r", encoding="utf-8") as fp:
                    ori_txtData = fp.read()
            else:
                if method == "ocr":
                    ori_txtData = get_pdftxt_by_ocr(af, Temp_txt_folder, ocr_chunk_size)
                if method == "fast":
                    ori_txtData = get_pdftxt(af)
                
                ori_txtData = clean_bopomofo(ori_txtData)
                
                txtData = CleanTxt(ori_txtData, startKeyword=cleantxt_startkey, endKeyword=cleantxt_endkey)
                with open(txtFile, "w", encoding="utf-8") as fp:
                    fp.write(txtData)
            if extract_title_rule:
                title = extract_title_rule(af, ori_txtData)
            else:
                try : 
                    title = pdfMining(af) 
                except Exception as e:
                    print(f"Error parsing title from {af}: {e}")
                    title = ''
                    if len(title) < 5:  
                        try:
                            pdf = PdfReader(af)
                            info = pdf.getDocumentInfo()
                            title = info.title_raw  
                            author = info.author_raw
                        except Exception as e2:
                            print(f"Error from meta data in {af}: {e2}")
                            title = ""
            if not title or title == "":
                title = fileName.replace("_", " ")
            titleJson[fileName] = title
            
        except Exception as e:
            print(f"[WARNING] Failed to process {af}: {e}")
            with open(Temp_txt_folder + "/failed_pdf.log", "a", encoding="utf-8") as f:
                f.write(f"{af}\t{e}\n")
            continue
        
    
    with open(Temp_txt_folder + "/titleJson.json", "w", encoding="utf-8") as fp:
        json.dump(titleJson, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    # 請修改為實際的資料夾或檔案路徑
    folder = "/home/user/workspace/dataset/source"
    Temp_txt_folder = "/home/user/workspace/dataset/data"
    # 路徑修改結束

    def extract_title(txt):
        for t in txt.split("\n"):
            if t.strip() != "":
                return t.strip()

        return ""

    PDFParser(folder, Temp_txt_folder, extract_title_rule=extract_title)
