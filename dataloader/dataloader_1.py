import torch
from torch.utils.data import Dataset
from collections import defaultdict, deque, Counter
import random
import copy
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

class RelGraphDataset(Dataset):
    def __init__(self, raw_data, encoder='sbert', num_samples=-1, max_nodes=10, device='cuda'):
        self.device='cuda'
        self.max_nodes = max_nodes

        self.data_raw = raw_data

        if num_samples != -1:
            self.data_raw = self.data_raw[:num_samples]

        self.encoder_name = encoder
        self.encoder, self.tokenizer = self.get_encoder(encoder)
        self.ent_dim = self.get_encoder_dim(encoder)

        self.data = []
        for idx, sample in enumerate(tqdm(self.data_raw, desc="[DATA] Processing")):
            try:
                raw_data_sample = self.process_sample(sample)
                if raw_data_sample['valid']:
                    self.data.append(raw_data_sample)
            except Exception as e:
                pass
        
        # random.shuffle(self.data)

    def get_encoder(self, encoder):
        if encoder=='sbert':
            return SentenceTransformer("all-MiniLM-L6-v2"), None
        elif encoder=='bert':
            return BertModel.from_pretrained("bert-base-uncased").to(self.device), BertTokenizer.from_pretrained('bert-base-uncased')
        raise NotImplementedError(f"Encoder {encoder} not implemented")

    def get_encoder_dim(self, encoder):
        if encoder=='sbert':
            return 384
        elif encoder=='bert':
            return 768
        raise NotImplementedError(f"Encoder {encoder} not implemented")

    def get_embedding(self, item):
        if self.encoder_name=='sbert':
            return self.encoder.encode(item, convert_to_tensor=True)
        elif self.encoder_name=='bert':
            inputs = self.tokenizer(item, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :]
        raise NotImplementedError(f"Encoder {self.encoder_name} not implemented")

    def build_graph(self, triplets):
        graph = defaultdict(list)
        for h, r, t in triplets:
            graph[h].append((t, r))
        return graph

    def build_adj_matrices(self, nodes, edges):
        N = len(nodes)
        node2idx = {n: i for i, n in enumerate(nodes)}
        adj = torch.zeros(N, N)
        rel_adj = [["" for _ in range(N)] for _ in range(N)]

        for h, r, t in edges:
            if h in node2idx and t in node2idx:
                i, j = node2idx[h], node2idx[t]
                adj[i][j] = 1
                rel_adj[i][j] = r

        return adj, rel_adj, node2idx

    def process_sample(self, sample):
        triplets = [(h, r, t) for h, r, t in sample['triplets']]
        all_nodes = set()
        for h, r, t in sample['triplets']:
            all_nodes.add(h)
            all_nodes.add(t)
        all_nodes = list(all_nodes)
        
        nodes = copy.deepcopy(all_nodes)

        valid = True
        retrieval_nodes = set()
        for path in sample['retrieval_path']:
            for triplet in path:
                if triplet[0] not in nodes:
                    valid=False
                if triplet[2] not in nodes:
                    valid=False

        edges = triplets

        # check for recursive nodes
        for path in sample['retrieval_path']:
            if not path:
                valid = True
                continue

            # Build ordered node sequence: start head of first triplet, then all tails
            node_seq = [path[0][0]] + [t for (_, _, t) in path]
            counts = Counter(node_seq)
            repeated_nodes = [n for n, c in counts.items() if c > 1]

            valid = len(repeated_nodes) == 0
            if not valid:
                break

        dct = {
            "valid": valid,
            "nodes": nodes,
            "edges": edges,
            "query": sample['retrieval_path'],
            "question": sample['question'],
            "answer": sample['answer']
        }
        
        return dct

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        nodes, edges, query, question, answer = sample["nodes"], sample["edges"], sample["query"], sample["question"], sample["answer"]

        adj, rel_adj, node2idx = self.build_adj_matrices(nodes, edges)
        node_feat = self.get_embedding(nodes)

        N = adj.shape[0]
        rel_feat = torch.zeros((N, N, self.ent_dim))  # (N, N, D)
        for i in range(N):
            for j in range(N):
                rel = rel_adj[i][j]
                if rel != "":
                    rel_vec = self.get_embedding(rel)
                    rel_feat[i][j] = rel_vec

        return {
            "adj": adj,                            # (N, N)
            "rel_adj": rel_adj,                    # list of str (N, N)
            "rel_feat": rel_feat,                  # (N, N, D)
            "node_feat": node_feat,                # (N, D)
            "nodes": nodes,
            "query": query,
            "question": question,
            "answer": answer
        }