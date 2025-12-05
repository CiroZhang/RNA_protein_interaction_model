# dataset.py

import json
from torch.utils.data import Dataset


def read_data(
    train_seq_path="train/seq.json",
    train_summary_path="train/qa_summary_rule.json",
    test_seq_path="test/seq.json",
    test_summary_path="test/qa_summary_rule.json",
):
    with open(train_seq_path) as f:
        train_seq = json.load(f)
    with open(test_seq_path) as f:
        test_seq = json.load(f)

    with open(train_summary_path) as f:
        train_summary_json = json.load(f)
    with open(test_summary_path) as f:
        test_summary_json = json.load(f)

    train_summary_map = {str(item["Gene Id"]): item["Summary"] for item in train_summary_json}
    test_summary_map  = {str(item["Gene Id"]): item["Summary"] for item in test_summary_json}

    train_dataset = []
    for gid, seq_list in train_seq.items():
        if gid not in train_summary_map:
            continue
        summary = train_summary_map[gid]
        for dna in seq_list:
            train_dataset.append({
                "gene_id": gid,
                "dna": dna,
                "target": summary,
            })

    test_dataset = []
    for gid, seq_list in test_seq.items():
        if gid not in test_summary_map:
            continue
        summary = test_summary_map[gid]
        for dna in seq_list:
            test_dataset.append({
                "gene_id": gid,
                "dna": dna,
                "target": summary,
            })

    return train_dataset, test_dataset, train_summary_map, test_summary_map


class GeneSummaryDataset(Dataset):
    """
    Each item:
      dna: str
      target: str (summary)
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["dna"], item["target"]


def collate_fn(batch):
    """
    batch: list of (dna, target) pairs.
    """
    dna_list = [b[0] for b in batch]
    tgt_list = [b[1] for b in batch]
    return dna_list, tgt_list


