import os
import base64
from openai import OpenAI, OpenAIError
import logging
import re

openai_api_key = "EMPTY"
# 請修改為實際的 IP 地址
openai_api_base = "http://localhost:7788/v1"
# IP 地址修改結束

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def llm_response(image_path_list):
    messages=[{"role": "system", "content": "You are a careful reader."}]
    content = [{"type": "text", "text": prompt}]
    for image_path in image_path_list:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        content.append({"type": "image_url","image_url": {"url": base64_qwen}})

    messages.append({"role": "user", "content": content})

    client = OpenAI(
        api_key=openai_api_key,
        # 請修改為實際的 IP 地址
        base_url= "http://localhost:7788/v1",
        # IP 地址修改結束
    )

    try:
        chat_response = client.chat.completions.create(
            model="Qwen2-VL-72B-Instruct-AWQ/",
            messages=messages,
            temperature=0.0,
            timeout=10
        )
        return chat_response.choices[0].message.content
    except OpenAIError as e:
        logging.error("An error occurred while calling the OpenAI API: %s", e)
        logging.info("image_path_list: %s", image_path_list)
        return None
    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        return None


def main():
    # 請修改為實際的資料夾或檔案路徑
    data_path = './dataset/source'
    # 路徑修改結束

    png_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))

    png_files.sort(key=lambda x: int(re.search(r'(?<=\/|\\)(\d+)(?=\.png)', x).group(0)) if re.search(r'(?<=\/|\\)(\d+)(?=\.png)', x) else float('inf'))
    
    pngs = png_files[1:3]
    print(pngs)
    response = llm_response(pngs)
    print(response)
    
if __name__ == '__main__':
    main()
