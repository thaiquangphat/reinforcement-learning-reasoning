# ------------------------ AMR parser + wrapper ------------------------
import random
import re
from collections import defaultdict, deque

class AMRNode:
    def __init__(self, var, concept):
        self.var = var
        self.concept = concept
        self.edges = defaultdict(list)  # rel -> [target_var | literal]

def tokenize(s: str):
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace(): i += 1; continue
        if s[i] in '()': tokens.append(s[i]); i += 1; continue
        if s[i] == ':':
            j = i + 1
            while j < len(s) and s[j] not in " \t\n()":
                j += 1
            tokens.append(s[i:j]); i = j; continue
        if s[i] == '"':
            j = s.find('"', i + 1) + 1
            tokens.append(s[i:j]); i = j; continue
        j = i
        while j < len(s) and s[j] not in " \t\n()":
            j += 1
        tokens.append(s[i:j]); i = j
    return tokens

def parse_amr(amr_text: str):
    """
    Returns (nodes_dict, root_var) where nodes_dict maps var -> AMRNode.
    """
    tokens = tokenize(amr_text)
    idx = [0]
    nodes = {}

    def parse_one():
        assert tokens[idx[0]] == '('
        idx[0] += 1
        var = tokens[idx[0]]; idx[0] += 1
        assert tokens[idx[0]] == '/'; idx[0] += 1
        concept = tokens[idx[0]]; idx[0] += 1
        node = AMRNode(var, concept)
        nodes[var] = node

        while tokens[idx[0]] != ')':
            rel = tokens[idx[0]]; idx[0] += 1
            if tokens[idx[0]] == '(':
                child = parse_one()
                node.edges[rel].append(child.var)
            else:
                lit = tokens[idx[0]]; idx[0] += 1
                node.edges[rel].append(lit)
        idx[0] += 1  # consume ')'
        return node

    root = parse_one()
    return nodes, root.var

class AMRGraph:
    """
    Lightweight wrapper for parsed AMR nodes.
    Provides:
      - var_list: ordered variables
      - var2idx mapping
      - adjacency (symmetric) as torch.FloatTensor (N,N)
      - concepts: list[str]
    Accepts either pre-parsed nodes dict or raw AMR string.
    """
    def __init__(self, amr):
        # amr can be a string (PENMAN-style) or (nodes_dict, root_var)
        if isinstance(amr, str):
            nodes, root = parse_amr(amr)
            self.root = root
        elif isinstance(amr, tuple) and len(amr) == 2:
            nodes, root = amr
            self.root = root
        elif isinstance(amr, dict):
            nodes = amr
            self.root = next(iter(nodes.keys()))
        else:
            raise ValueError("Unsupported amr type")

        self.nodes = nodes
        self.var_list = list(nodes.keys())
        self.var2idx = {v: i for i, v in enumerate(self.var_list)}
        self.concepts = [nodes[v].concept for v in self.var_list]

        # build adjacency (directed then make undirected)
        import torch
        N = len(self.var_list)
        A = torch.zeros((N, N), dtype=torch.float32)
        for src_var, node in nodes.items():
            i = self.var2idx[src_var]
            for rel, tgts in node.edges.items():
                for t in tgts:
                    if t in self.var2idx:
                        j = self.var2idx[t]
                        A[i, j] = 1.0
        # make symmetric (undirected for GCN)
        A = ((A + A.t()) > 0).float()
        self.adj = A
