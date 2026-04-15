import os
import pandas as pd
import json

def xlsx_parser(xlsx_folder_list, temp_txt_folder, method="fast"):
    """
    會自動把所有 XLSX/text 輸出到 temp_txt_folder/docs_txt
    並自動同 PDF 寫入 temp_txt_folder/titleJson.json
    """
    output_folder = os.path.join(temp_txt_folder, "docs_txt")
    os.makedirs(output_folder, exist_ok=True)
    title_json_path = os.path.join(temp_txt_folder, "titleJson.json")

    if os.path.exists(title_json_path):
        with open(title_json_path, "r", encoding="utf-8") as fp:
            titleJson = json.load(fp)
    else:
        titleJson = {}

    for xlsx_folder in xlsx_folder_list:
        for root, _, files in os.walk(xlsx_folder):
            for file in files:
                if file.lower().endswith(".xlsx"):
                    file_path = os.path.join(root, file)
                    try:
                        excel = pd.ExcelFile(file_path)
                        texts = []
                        for sheet in excel.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet, dtype=str)
                            txt = f"[Sheet: {sheet}]\n"
                            txt += df.to_csv(sep='\t', index=False)
                            texts.append(txt)
                        all_txt = "\n\n".join(texts)
                        base_name = os.path.splitext(file)[0]
                        txt_file = os.path.join(output_folder, f"{base_name}.txt")
                        with open(txt_file, "w", encoding="utf-8") as f:
                            f.write(all_txt)
                        # 建議key帶副檔名防覆蓋
                        titleJson[f"{base_name}.xlsx"] = {
                            "source": "xlsx",
                            "original_file": file
                        }
                        print(f"[xlsx_parser] Done: {file_path} -> {txt_file}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process {file_path}: {e}")
    with open(title_json_path, "w", encoding="utf-8") as fp:
        json.dump(titleJson, fp, indent=4, ensure_ascii=False)