"""Leakage audit evidence for the pinned foundation-model checkpoints (needs Hugging Face access).

For each pinned revision in configs/models.yaml it records:
  - the date of the pinned commit and the date the weights file last changed at that revision
    (weights dated before the test start cannot contain test-period data);
  - the model card's declared datasets and every card line mentioning energy/grid keywords,
    as leads for the manual check of each model's training-data list.

    python -m tgc.leakage_check            # writes out/env/leakage_check.json
"""
from __future__ import annotations

import json
import re

from tgc import config

WEIGHTS = {"chronos2": "model.safetensors", "timesfm25": "model.safetensors",
           "moirai2": "model.safetensors", "tirex": "model.ckpt"}
KEYWORDS = r"(EIA|ERCOT|CAISO|ISO|electric|electricity|energy|grid|load|solar|wind|power|LOTSA|GIFT|pretrain|training data|corpus)"


def check(model: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    mc = config.load("models")["models"][model]
    repo, rev = mc["checkpoint"], mc["revision"]
    api = HfApi()
    out = {"checkpoint": repo, "revision": rev}
    info = api.model_info(repo, revision=rev)
    out["repo_created"] = str(getattr(info, "created_at", None))
    out["revision_last_modified"] = str(getattr(info, "last_modified", None))
    card = getattr(info, "card_data", None)
    out["card_datasets"] = list(getattr(card, "datasets", None) or []) if card is not None else []
    try:
        paths = api.get_paths_info(repo, [WEIGHTS[model]], revision=rev, expand=True)
        lc = getattr(paths[0], "last_commit", None) if paths else None
        out["weights_file"] = WEIGHTS[model]
        out["weights_last_commit_date"] = str(getattr(lc, "date", None))
        out["weights_last_commit_title"] = getattr(lc, "title", None)
    except Exception as e:  # file name differs, or API change
        out["weights_error"] = repr(e)
    try:
        commits = api.list_repo_commits(repo)
        out["all_commits"] = [{"id": c.commit_id[:12], "date": str(c.created_at), "title": c.title} for c in commits]
    except Exception as e:
        out["commits_error"] = repr(e)
    try:
        readme = open(hf_hub_download(repo, "README.md", revision=rev), encoding="utf-8").read()
        out["card_keyword_lines"] = [l.strip()[:300] for l in readme.splitlines()
                                     if re.search(KEYWORDS, l, re.IGNORECASE)][:60]
        out["card_links"] = sorted(set(re.findall(r"https?://(?:arxiv\.org|github\.com)[^\s)\]>\"']+", readme)))
    except Exception as e:
        out["card_error"] = repr(e)
    return out


def main():
    test_start = config.load("data")["splits"]["test"][0]
    res = {"test_start": test_start}
    for m in WEIGHTS:
        try:
            res[m] = check(m)
        except Exception as e:
            res[m] = {"error": repr(e)}
        w = res[m].get("weights_last_commit_date") or res[m].get("revision_last_modified")
        res[m]["weights_before_test_start"] = (str(w)[:10] < test_start) if w and w != "None" else None
        print(f"{m:10s} weights dated {str(w)[:10]}  before test start: {res[m]['weights_before_test_start']}")
    p = config.path(config.load("models")["env_log_dir"]) / "leakage_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
