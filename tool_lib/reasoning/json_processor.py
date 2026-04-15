import json
import os
import sys
from tqdm import tqdm

def check_file_path(file_path):
    if not file_path.endswith('.json'):
        raise ValueError("The input file is not a JSON file. Please provide a valid .json file path.")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError("The specified file does not exist. Please check the file path.")

def process_question(question):
    separator = "\n</think>\n\n"
    if separator in question:
        return question.split(separator, 1)[1]
    return question

def process_base_answer(base_answer):
    prefix = "<think>\n"
    if not base_answer.startswith(prefix):
        return prefix + base_answer
    return base_answer

def process_json(file_path, output_path):
    check_file_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Processing progress", unit="item"):
                question = item.get('question')
                base_answer = item.get('base_answer')
                if question is not None and base_answer is not None:
                    if 'question' in item:
                        item['question'] = process_question(item['question'])
                    if 'base_answer' in item:
                        item['base_answer'] = process_base_answer(item['base_answer'])
                    new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            question = data.get('question')
            base_answer = data.get('base_answer')
            if question is not None and base_answer is not None:
                if 'question' in data:
                    data['question'] = process_question(data['question'])
                if 'base_answer' in data:
                    data['base_answer'] = process_base_answer(data['base_answer'])
            else:
                data = {}

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Processing completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def process_json_for_sim(file_path, output_path):
    check_file_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Processing progress", unit="item"):
                simple_question = item.get('simple_question')
                if simple_question is not None:
                    if 'simple_question' in item:
                        item['simple_question'] = process_question(item['simple_question'])
                    new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            simple_question = data.get('simple_question')
            if simple_question is not None:
                if 'simple_question' in data:
                    data['simple_question'] = process_question(data['simple_question'])
            else:
                data

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Processing completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def process_json_for_neg(file_path, output_path):
    check_file_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="處理進度", unit="項"):
                negative_question = item.get('negative_question')
                if negative_question is not None:
                    if "問題: " in negative_question:
                        negative_question = negative_question.split("問題: ", 1)[1]
                    if 'negative_question' in item:
                        item['negative_question'] = process_question(negative_question)
                    new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            negative_question = data.get('negative_question')
            if negative_question is not None:
                if "問題: " in negative_question:
                    negative_question = negative_question.split("問題: ", 1)[1]
                if 'negative_question' in data:
                    data['negative_question'] = process_question(negative_question)
            else:
                data

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Processing completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def extract_clean_answer(base_answer):
    separator1 = "<ANSWER>: "
    separator2 = "\nANSWER: "
    if separator1 in base_answer:
        return base_answer.split(separator1, 1)[1]
    elif separator2 in base_answer:
        return base_answer.split(separator2, 1)[1]
    return base_answer

def clear_json(file_path, output_path):
    check_file_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Clearing progress", unit="item"):
                question = item.get('question')
                base_answer = item.get('base_answer')
                if question is not None and base_answer is not None:
                    if 'question' in item:
                        item['question'] = process_question(item['question'])
                    if 'base_answer' in item:
                        item['clean_answer'] = extract_clean_answer(item['base_answer'])
                    new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            question = data.get('question')
            base_answer = data.get('base_answer')
            if question is not None and base_answer is not None:
                if 'question' in data:
                    data['question'] = process_question(data['question'])
                if 'base_answer' in data:
                    data['clean_answer'] = extract_clean_answer(data['base_answer'])
            else:
                data = {}

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Clearing completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while clearing the file: {e}")

def clear_json_dev(file_path, output_path):
    check_file_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Clearing progress", unit="item"):
                question = item.get('question')
                base_answer = item.get('base_answer')
                if question is not None and base_answer is not None:
                    if 'question' in item:
                        item['question'] = process_question(item['question'])
                    if 'base_answer' in item:
                        item['base_answer'] = extract_clean_answer(item['base_answer'])
                    new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            question = data.get('question')
            base_answer = data.get('base_answer')
            if question is not None and base_answer is not None:
                if 'question' in data:
                    data['question'] = process_question(data['question'])
                if 'base_answer' in data:
                    data['base_answer'] = extract_clean_answer(data['base_answer'])
            else:
                data = {}

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Clearing completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while clearing the file: {e}")

def clean_question(file_path, output_path):
    check_file_path(file_path)  
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Cleaning progress", unit="item"):
                question = item.get('question')
                if question is not None:
                    item['question'] = process_question(question)
                new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            question = data.get('question')
            if question is not None:
                data['question'] = process_question(question)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)  

        print(f"Cleaning completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while cleaning the file: {e}")

def clean_negative_question(file_path, output_path):
    check_file_path(file_path)  
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  

        if isinstance(data, list):
            new_data = []
            for item in tqdm(data, desc="Cleaning progress", unit="item"):
                question = item.get('negative_question')
                if question is not None:
                    item['negative_question'] = process_question(question)
                new_data.append(item)
            data = new_data

        elif isinstance(data, dict):
            question = data.get('negative_question')
            if question is not None:
                data['negative_question'] = process_question(question)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)  

        print(f"Cleaning completed. The results have been saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while cleaning the file: {e}")

def adjust_hybrid_chunks_length(input_path, output_path, target_length=20):
    import json
    import logging

    logging.info(f"Adjusting hybrid_chunks length to {target_length}...")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        adjusted_data = []
        for item in data:
            hybrid_chunks = item.get("hybrid_chunks", [])
            
            if len(hybrid_chunks) > target_length:
                adjusted_chunks = hybrid_chunks[:target_length]
            else:
                adjusted_chunks = hybrid_chunks + [{}] * (target_length - len(hybrid_chunks))
            
            item["hybrid_chunks"] = adjusted_chunks
            adjusted_data.append(item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(adjusted_data, f, ensure_ascii=False, indent=4)

        logging.info(f"Successfully adjusted hybrid_chunks length and saved to {output_path}")

    except Exception as e:
        logging.error(f"Error adjusting hybrid_chunks length: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file_path> <output_file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    clear_json_dev(file_path, output_path)
