import json
from datasets import load_dataset
from tqdm import tqdm
import time
import os
import re
from kg_construction import REExtractor
from llm_completion import ENRICH_PROMPT
from llm_completion import LLMResponse
from ultis import parse_json_safe
from dotenv import load_dotenv
import argparse

load_dotenv()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Xoá các tag dạng <...>
    text = re.sub(r'<[^>]+>', '', text)

    # Xoá các ký tự đặc biệt không cần thiết (giữ lại chữ, số, dấu chấm, dấu phẩy, dấu hỏi, dấu chấm than)
    text = re.sub(r'[^\w\s.,!?]', '', text)

    # Chuyển nhiều khoảng trắng thành 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    # Xoá khoảng trắng đầu/cuối
    text = text.strip()

    return text
    
def save_dataset(split, path, num_samples=27000):
    hf_dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', trust_remote_code=True)
    
    no_samples = min(num_samples, len(hf_dataset[split]))
    dataset = hf_dataset[split].select(range(no_samples))

    base, ext = os.path.splitext(path)
    out_path = f"{base}_{split}{ext}"

    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc=f"Processing {split} set"):
            question = sample['question']
            answer = sample['answer']
            
            contexts = []
            full_context = ""
            
            for title in sample['supporting_facts']['title']:
                title_idx = sample['context']['title'].index(title)
                context = ""
                for sentence in sample['context']['sentences'][title_idx]:
                    sentence = clean_text(sentence)
                    context += sentence 
                    full_context += sentence + " "
                contexts.append(context)

            contexts = list(set(contexts))
            
            dct = {
                "context": full_context,
                "contexts": contexts,
                "question": question,
                "answer": answer
            }

            f.write(json.dumps(dct, ensure_ascii=False) + '\n')

    print(f"Saved {no_samples} samples to {out_path}")

def create_dataset_triplets(src_path, out_path):
    print("============ PROCESSING MUSIQUE ============")
    print(f"Creating triplets from {src_path} to {out_path}")
    print("=============================================")

    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    re_extractor = REExtractor(device='cuda')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc=f"[TRIPLETS]"):
            triplets = []
            for cont in sample['contexts']:
                triplet = re_extractor.paragraph_extractor(cont)
                triplets.extend(triplet)

            triplets = [list(t) for t in set(tuple(triplet) for triplet in triplets)]
            dct = {
                "context": sample['context'],
                "contexts": sample['contexts'],
                "triplets": triplets,
                "question": sample['question'],
                "answer": sample['answer']
                
            }
            f.write(json.dumps(dct, ensure_ascii=False) + '\n')

    print(f"Saved triplets of {src_path} samples to {out_path}")

def enrich_dataset_with_llm(llm, src_path, out_path, start_index=0, max_samples=None):
    print("============ PROCESSING MUSIQUE ============")
    print(f"Enrichment and path creation from {src_path} to {out_path}")
    print("=============================================")

    # Đọc dataset
    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    total_samples = len(dataset)
    end_index = total_samples if max_samples is None else min(start_index + max_samples, total_samples)

    # Tìm file out_path mới với số thứ tự tăng dần
    dir_path = os.path.dirname(out_path)
    base_name = os.path.basename(out_path)
    name_no_ext, ext = os.path.splitext(base_name)

    existing_nums = []
    for fname in os.listdir(dir_path):
        if fname.startswith(name_no_ext + "_") and fname.endswith(ext):
            try:
                num_part = fname[len(name_no_ext) + 1 : -len(ext)]
                existing_nums.append(int(num_part))
            except ValueError:
                pass
    next_index = max(existing_nums) + 1 if existing_nums else 1
    out_path = os.path.join(dir_path, f"{name_no_ext}_{next_index}{ext}")

    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset[start_index:end_index], desc=f"[LLM_ENRICH]"):
            context = sample['context']
            question = sample['question']
            answer = sample['answer']
            triplets = sample['triplets']
            llm_json = None

            prompt = ENRICH_PROMPT.format(
                context=context,
                triplets=triplets,
                question=question,
                answer=answer
            )

            max_tries = 5
            while max_tries > 0:
                try:
                    llm_response = llm.response(prompt)
                except Exception as e:
                    max_tries -= 1
                    print(f"LLM call failed: {e}. Retrying in 60s... ({max_tries} tries left)")
                    time.sleep(60)
                    continue

                llm_response_json = parse_json_safe(llm_response)

                if llm_response_json is not None:
                    llm_json = llm_response_json
                    time.sleep(1.0)
                    break
                else:
                    max_tries -= 1
                    print(f"Invalid JSON, retrying in 60s... ({max_tries} tries left)")

            if max_tries == 0:
                llm_json = llm_response
                print("Failed to get valid JSON from LLM after 5 retries")

            dct = {
                "context": context,
                "contexts": sample.get('contexts', []),
                "triplets": triplets,
                "question": question,
                "answer": answer,
                "llm": llm_json
            }
            f.write(json.dumps(dct, ensure_ascii=False) + '\n')

    print(f"Saved {end_index - start_index} samples to {out_path}")

def clean_dataset(src_path, out_path):
    print("============ PROCESSING MUSIQUE ============")
    print(f"Cleaning from {src_path} to {out_path}")
    print("=============================================")
    # Open file
    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    clean_dataset = []
    invalid_dataset = []
    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc='[CLEAN]'):
            invalid = False
            
            try:
                llm_response = sample['llm']
                triplets = sample['triplets']
                triplets.extend(llm_response['inferred_triplets'])

                sample['triplets'] = triplets

                # Get all nodes
                heads = [s[0] for s in triplets]
                tails = [s[2] for s in triplets]

                nodes = heads + tails

                for combo in llm_response['reasoning_paths']:
                    # Check for length
                    if len(combo['path']) % 2 == 0:
                        invalid = True
                    # Check for consistency
                    else:
                        path = combo['path']
                        for i in range(0, len(combo['path']), 2):
                            if path[i] not in nodes:
                                invalid = True

                if invalid:
                    invalid_dataset.append(sample)
                else:
                    clean_dataset.append(sample)
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            except:
                invalid_dataset.append(sample)

    print(f"Number of valid samples from {src_path}: {(len(clean_dataset))}")
    print(f"Number of invalid samples from {src_path}: {(len(invalid_dataset))}")
    print(f"Saved processed samples to {out_path}")

def format_dataset(src_path, out_path):
    print("============ PROCESSING MUSIQUE ============")
    print(f"Format from {src_path} to {out_path}")
    print("=============================================")
    # Open file
    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc='[FORMAT]'):
            # Process questions
            original_question = sample['question']
            question = [s['sub-question'] for s in sample['llm']['reasoning_paths']]

            # Process reasoning paths
            reasoning_paths = []
            for s in sample['llm']['reasoning_paths']:
                lst = s['path']
                path = [lst[i:i+3] for i in range(0, len(lst)-3+1, 2)]
                reasoning_paths.append(path)

            # Create dictionary
            dct = {
                "context": sample['context'],
                "original_question": original_question,
                "question": question,
                "answer": sample['answer'],
                "triplets": sample['triplets'],
                "retrieval_path": reasoning_paths,
            }   
            f.write(json.dumps(dct, ensure_ascii=False) + '\n')

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Musique Dataset Creation")
    api_key = os.getenv('GEMINI_API_KEY')

    parser.add_argument('--type', choices=['triplets', 'enrichment', 'clean', 'format'], required=True, help='Creation type')
    parser.add_argument('--dataset', choices=['train', 'test'], required=True, help='Dataset type')

    args = parser.parse_args()

    if args.type == 'triplets':
        if args.dataset == 'train':
             create_dataset_triplets("dataset/musique/raw/muiqsue_ans_v1.0_train.jsonl", 
                            "dataset/musique/triplets/train_musique_0.jsonl")
        elif args.dataset == 'test':
            create_dataset_triplets("dataset/musique/raw/musique_ans_v1.0_dev.jsonl",
                            "dataset/musique/triplets/eval_musique_0.jsonl")
    
    elif args.type == 'enrichment':
        # Initialize model
        llm = LLMResponse(api_key=api_key)

        if args.dataset == 'train':
            enrich_dataset_with_llm(llm, "dataset/musique/triplets/train_musique_0.jsonl", 
                            "dataset/musique/triplets/train_musique_enrich.jsonl",
                            max_samples=18000)
        elif args.dataset == 'test':
            enrich_dataset_with_llm(llm, "dataset/musique/triplets/eval_musique_0.jsonl", 
                            "dataset/musique/triplets/eval_musique_enrich.jsonl")
            
    elif args.type == 'clean':
        if args.dataset == 'train':
            clean_dataset("dataset/musique/triplets/train_musique_enrich.jsonl", 
                            "dataset/musique/triplets/train_musique_enrich_processed.jsonl")
        elif args.dataset == 'test':
            clean_dataset("dataset/musique/triplets/eval_musique_enrich.jsonl",
                            "dataset/musique/triplets/eval_musique_enrich_processed.jsonl")
            
    elif args.type == 'format':
        if args.dataset == 'train':
            format_dataset("dataset/musique/triplets/train_musique_enrich_processed.jsonl", 
                            "dataset/musique/triplets/train_musique_enrich_format.jsonl")
        elif args.dataset == 'test':
            format_dataset("dataset/musique/triplets/eval_musique_enrich_processed.jsonl",
                            "dataset/musique/triplets/eval_musique_enrich_format.jsonl")