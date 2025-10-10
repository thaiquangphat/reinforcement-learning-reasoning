from kg_construction import RelationTriplet, REExtractor
from llm_completion import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import json
import os
import time
from tqdm import tqdm
from typing import TypedDict, List

class QASample(TypedDict):
    context: str
    question: str
    answer: str

class QADataset(object):
    def __init__(self):
        self.hf_dataset = None
        self.train_dataset = None
        self.eval_dataset = None

    def get_train_dataset(self) -> List[QASample]:
        return self.train_dataset

    def get_eval_dataset(self) -> List[QASample]:
        return self.eval_dataset

    def get_num_train_samples(self) -> int:
        return len(self.train_dataset)

    def get_num_eval_samples(self) -> int:
        return len(self.eval_dataset) 
    
    def get_sample(self, split: str='train', idx: int=0) -> QASample:
        if split == 'train':
            return self.train_dataset[idx]
        elif split == 'eval':
            return self.eval_dataset[idx]
        else:
            raise NotImplementedError(f"split `{split}` not implemented, try `train`, `eval`")
        
class Wiki2QA(QADataset):
    def __init__(self):
        super().__init__()

        self.train_path = 'dataset/2wikiqa/train_0.json'
        self.eval_path = 'dataset/2wikiqa/dev_0.json'

        self.train_dataset = self.construct_dataset(self.train_path)
        self.eval_dataset = self.construct_dataset(self.eval_path)

    def construct_dataset(self, path='dataset/2wikiqa/train_0.json'):
        dataset_json = []
        with open(path, 'r', encoding='utf-8') as file:
            dataset_json = json.load(file)
            
        dataset = []
        for sample in dataset_json:
            context = ""
            for sup in sample['supporting_facts']:
                title = sup[0]
                cont = next((c for c in sample['context'] if c[0] == title), None)
                for con in cont[1]:
                    context = context + ' ' + con

            dct: QASample = {
                "context": context,
                "question": sample['question'],
                "answer": sample['answer']
            }

            dataset.append(dct)

        return dataset
    
def create_dataset(split='train', start_idx=0, end_idx=-1, dir='2wikiqa'):
    print("============ PROCESSING 2WIKIQA ============")
    print(f" From index {start_idx} to index {end_idx}")
    print("============================================")
    
    # Xác định đường dẫn tới file nguồn (raw JSON)
    file_split = 'train' if split == 'train' else 'dev'
    data_path = f'data_loader/2wikiqa/{file_split}.json'
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset_json = json.load(file)

    dataset = Wiki2QA()
    re_extractor = REExtractor(device='cuda')
    os.makedirs(dir, exist_ok=True)

    # Tính tổng số sample và giới hạn chỉ số
    total_samples = dataset.get_num_train_samples() if split == 'train' else dataset.get_num_eval_samples()
    end_idx = total_samples if end_idx == -1 else min(end_idx, total_samples)
    num_samples = end_idx - start_idx

    # Tìm số lượng file đã tồn tại bắt đầu bằng "train" hoặc "dev" trong thư mục
    existing_files = [f for f in os.listdir(dir) if f.startswith(split) and f.endswith('.jsonl')]
    file_index = len(existing_files)

    # Tạo tên file mới với suffix
    output_path = os.path.join(dir, f"{split}_{file_index}.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(start_idx, end_idx), desc=f"Processing {split} samples [{start_idx}:{end_idx}]"):
            sample = dataset.get_sample(split=split, idx=i)
            paragraph = sample['context']
            raw_triplets = re_extractor.re_extractor(paragraph)
            triplets = [[t['head'], t['relation'], t['tail']] for t in raw_triplets]
            path = dataset_json[i]['evidences']

            json_line = json.dumps({
                "context": paragraph,
                "question": sample['question'],
                "answer": sample['answer'],
                "triplets": triplets,
                "path": path
            }, ensure_ascii=False)

            f.write(json_line + '\n')

    print(f"Done creating dataset '{dir}' split '{split}' [{start_idx}:{end_idx}] with {num_samples} samples.")
    print(f"Saved to: {output_path}")

def count_valid_samples(src_path):
    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_parsed = json.loads(line)
            dataset.append(json_parsed)

    valid = 0
    for sample in dataset:
        head_list = [t[0] for t in sample['triplets']]
        tail_list = [t[2] for t in sample['triplets']]

        nodes = list(set(head_list + tail_list))

        verified = True
        for p in sample['path']:
            if p[0] not in nodes or p[2] not in nodes:
                verified = False

        valid += 1 if verified else 0

    print(f"Valid samples {valid} over {len(dataset)} samples")

def process_and_expand_triplets(src_path, out_path):
    print("============= PROCESSING PATH NODES =============")
    print(f"Source path: {src_path}")
    print(f"Output path: {out_path}")
    print("=================================================")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    dataset = []
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    with open(out_path, 'w', encoding='utf-8') as fout:
        for sample in tqdm(dataset, desc="Processing"):
            triplets = sample.get('triplets', [])
            head_list = [t[0] for t in triplets]
            tail_list = [t[2] for t in triplets]
            nodes = list(set(head_list + tail_list))

            # Embed all unique nodes once
            node_embeddings = model.encode(nodes)

            updated_path = []
            for h, r, t in sample.get('path', []):
                h_emb = model.encode([h])
                t_emb = model.encode([t])

                best_h = nodes[cosine_similarity(h_emb, node_embeddings).argmax()]
                best_t = nodes[cosine_similarity(t_emb, node_embeddings).argmax()]

                # Add to triplets
                new_triplet = [best_h, r, best_t]
                triplets.append(new_triplet)

                # Also update path
                updated_path.append([best_h, r, best_t])

            sample['triplets'] = triplets
            sample['path'] = updated_path  # update path with normalized nodes
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved updated dataset to {out_path} with {len(dataset)} samples.")

def split_logical_paths(src_path, out_path):
    print("============= SPLITTING LOGICAL PATHS =============")
    print(f"Source path: {src_path}")
    print(f"Output path: {out_path}")
    print("===================================================")

    dataset = []
    with open(src_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dataset.append(json.loads(line))

    def split_path_chain(path):
        chains = []
        current_chain = []

        for triple in path:
            h, r, t = triple

            if not current_chain:
                current_chain.append([h, r, t])
            else:
                prev_tail = current_chain[-1][2]
                if h == prev_tail:
                    current_chain.append([h, r, t])
                else:
                    # break, store current and start new
                    chains.append(current_chain)
                    current_chain = [[h, r, t]]

        if current_chain:
            chains.append(current_chain)

        return chains

    with open(out_path, 'w', encoding='utf-8') as fout:
        for sample in tqdm(dataset, desc="Processing"):
            path = sample.get('path', [])
            logical_paths = split_path_chain(path)
            sample['retrieval_path'] = logical_paths
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved processed dataset with logical retrieval paths to {out_path}")

def count_retrieval_paths(src_path):
    print("============= COUNTING RETRIEVAL PATHS =============")
    print(f"Source path: {src_path}")
    print("====================================================")

    total_samples = 0
    total_paths = 0
    path_length_counter = Counter()

    with open(src_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line)
            retrieval_paths = sample.get("retrieval_path", [])
            total_samples += 1
            total_paths += len(retrieval_paths)

            for path in retrieval_paths:
                path_length = len(path)
                path_length_counter[path_length] += 1

    print(f"Total samples         : {total_samples}")
    print(f"Total retrieval paths : {total_paths}")
    print(f"Average paths/sample  : {total_paths / total_samples:.2f}")
    print("\nDistribution of retrieval path lengths:")
    for length, count in sorted(path_length_counter.items()):
        print(f"  Length {length}: {count} path(s)")

def merge_datasets(list_path, out_path):
    dataset = []

    for path in list_path:
        child_dataset = []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                child_dataset.append(json.loads(line))

        for sample in child_dataset:
            dct = {
                "context": sample['context'],
                "question": sample['question'],
                "answer": sample['answer'],
                "triplets": sample['triplets'],
                "retrieval_path": sample['retrieval_path']
            }

            dataset.append(dct)

    with open(out_path, 'w', encoding='utf-8') as file:
        for item in dataset:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

    print(f"Merged dataset with {len(dataset)} samples total")
    
if __name__ == "__main__":
    # phase 1. train 0 - 6000 - done
    # phase 2. train 6000 - 18000 - done
    # phase 3. eval 0 - -1
    # create_dataset(split='eval', start_idx=0, end_idx=-1, dir='2wikiqa')

    # phase 1. 2wikiqa/train_0.jsonl, 2wikiqa/train_expand_0.jsonl - done
    # phase 2. 2wikiqa/train_1.jsonl, 2wikiqa/train_expand_1.jsonl - done
    # phase 3. 2wikiqa/eval_0.jsonl, 2wikiqa/eval_expand_0.jsonl - done
    # process_and_expand_triplets(src_path = "2wikiqa/eval_0.jsonl", out_path="2wikiqa/eval_expand_0.jsonl")
    # count_valid_samples(src_path = "2wikiqa/eval_expand_0.jsonl")

    # phase 1. 2wikiqa/train_expand_0.jsonl, 2wikiqa/train_processed_0.jsonl - done
    # phase 2. 2wikiqa/train_expand_1.jsonl, 2wikiqa/train_processed_1.jsonl - done
    # phase 3. 2wikiqa/eval_expand_0.jsonl, 2wikiqa/eval_processed_0.jsonl - done
    # split_logical_paths(src_path="2wikiqa/eval_expand_0.jsonl", out_path="2wikiqa/eval_processed_0.jsonl")

    # count_retrieval_paths(src_path="logg_dataset/hotpotqa_train_1_processed.jsonl")

    train_path_list = ["logg_dataset/hotpotqa_train_1_processed.jsonl", "2wikiqa/train_processed_0.jsonl", "2wikiqa/train_processed_1.jsonl"]
    merge_datasets(list_path=train_path_list, out_path="logg_dataset/train_0.jsonl")