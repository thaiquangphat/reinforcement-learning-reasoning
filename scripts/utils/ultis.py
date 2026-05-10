import json

# --- Helper function to remove "triplets" anywhere in the object ---
def remove_key_recursive(data, key="triplets"):
    if isinstance(data, dict):
        return {k: remove_key_recursive(v, key)
                for k, v in data.items() if k != key}
    elif isinstance(data, list):
        return [remove_key_recursive(item, key) for item in data]
    else:
        return data


# ============================
# Load dataset 1
# ============================
dataset = []
with open('dataset/amr_decompose_llm_enhance.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        dataset.append(json.loads(line))

# Print sample 1 (without triplets)
clean_sample1 = remove_key_recursive(dataset[0], key="triplets")
print(json.dumps(clean_sample1, indent=2, ensure_ascii=False))

print("=" * 20)

# ============================
# Load dataset 2
# ============================
dataset2 = []
with open('dataset/train.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        dataset2.append(json.loads(line))

# Print sample 2 (without triplets)
clean_sample2 = remove_key_recursive(dataset2[0], key="triplets")
print(json.dumps(clean_sample2, indent=2, ensure_ascii=False))


# ============================
# Build retrieval_path
# ============================
for sample in dataset:
    retrieval_path = []
    for sub in sample['llm_response']:
        for subsub in sub['reasoning_path']:
            retrieval_path.append(subsub)

    sample['retrieval_path'] = retrieval_path

dataset3 = []
for sample in dataset:
    try:
        if "triplets" not in sample:
            sample["triplets"] = []

        existing = set(tuple(t) for t in sample["triplets"])

        # Flatten retrieval_path: it is [[[t1],[t2]], [[t1],[t2,t3]], ...]
        for chain in sample["retrieval_path"]:
            for triple in chain:     # triple is ["h", "r", "t"]
                triple_t = tuple(triple)
                if triple_t not in existing:
                    sample["triplets"].append(triple)
                    existing.add(triple_t)
        dataset3.append(sample)
    except:
        pass

print(f"NUMBER OF SUCCESS SAMPLES: {len(dataset3)} OUT OF {len(dataset)}.")

# ============================
# Save updated dataset
# ============================
with open('dataset/traintestamr.jsonl', 'w', encoding='utf-8') as file:
    for sample in dataset:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")


print("=" * 20)

# Print modified dataset[0] without triplets
clean_out = remove_key_recursive(dataset[0], key="triplets")
print(json.dumps(clean_out, indent=2, ensure_ascii=False))
