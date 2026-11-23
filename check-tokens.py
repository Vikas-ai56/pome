import json, sys
f = sys.argv[1] if len(sys.argv) > 1 else "results/test_qwen.jsonl"
rows = [json.loads(l) for l in open(f)]
u = [r.get("usage") or {} for r in rows]
def avg(k):
    v = [x.get(k) for x in u if x.get(k) is not None]
    return round(sum(v) / len(v)) if v else "n/a"
print("answers saved:", len(rows), "| usable:", sum(r["ok"] for r in rows))
print("mean tokens  prompt:", avg("prompt"), " completion:", avg("completion"), " reasoning:", avg("reasoning"))