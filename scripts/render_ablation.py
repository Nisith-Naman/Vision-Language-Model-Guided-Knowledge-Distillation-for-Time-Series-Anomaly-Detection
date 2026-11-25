"""Teacher rendering ablation on Daphnet (exploratory; results in experiments/benchmark_v2/render_ablation_daphnet.json)."""
import os, sys, time, yaml, json
sys.path.insert(0, os.getcwd())
os.environ.setdefault("HF_HOME", os.path.abspath("cache/huggingface")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
torch.set_num_threads(10)
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from src.experiments.data import prepare_dataset
from src.teacher.render import render_batch
from src.teacher.clip_teacher import CLIPEmbedder
from src.teacher.probe import LinearProbeTeacher
cfg = yaml.safe_load(open("config.yaml"))
d = prepare_dataset("daphnet", cfg)
tr, va = d.splits["train"], d.splits["val"]
ti = np.linspace(0, len(tr.X)-1, 2000).astype(int); vi = np.linspace(0, len(va.X)-1, 800).astype(int)
emb = CLIPEmbedder().load()
for name, style, clip in [("old_global_heatmap", "heatmap", 100.0), ("heatmap_clip4", "heatmap", 4.0), ("lines_clip4", "lines", 4.0)]:
    t=time.time()
    et = emb.encode_images(render_batch(tr.X[ti], style, clip)); ev = emb.encode_images(render_batch(va.X[vi], style, clip))
    zs = emb.zero_shot(ev)[:,1]
    pr = LinearProbeTeacher(); pr.fit(et, tr.y_window[ti]); pv = pr.predict_proba(ev)[:,1]
    yv = va.y_window[vi]
    print(json.dumps({"render": name, "val_prev": round(float(yv.mean()),3), "zeroshot_auprc": round(ap(yv, zs),3), "zeroshot_roc": round(auc(yv, zs),3), "probe_auprc": round(ap(yv,pv),3), "probe_roc": round(auc(yv,pv),3), "C": pr.C_, "secs": round(time.time()-t)}), flush=True)
