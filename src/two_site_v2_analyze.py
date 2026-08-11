"""Revised two-site analysis (Stage 2 of the peer-review revision).

Consumes the cached query-excluded, deduplicated, patient-tagged cells from two_site_v2.py
(--build) and addresses the remaining reviewer points:

  FIX E — MODEL-RECOMMENDED CONFIGURATION. v1 used mean pooling for everything, which
    misconfigures the models whose OUTCOME (their ranking) is the whole point:
      * MedCPT uses SEPARATE query and article encoders with CLS pooling. Implemented as
        such here (Query-Encoder for queries, Article-Encoder for targets).
      * BGE uses CLS pooling with a retrieval query instruction.
      * E5 / Nomic keep their documented query/document prefixes (already correct in v1).
      * GTE / mpnet / MiniLM / BioLORD use mean pooling (correct in v1).
      * The five MLM backbones (BERT/BioBERT/ClinicalBERT/PubMedBERT/SciBERT) have no
        canonical sentence-pooling; we standardise them to CLS and report them explicitly
        as "standardised-backbone" comparators, not as deployment-configured models.
    Truncation fraction (docs exceeding 512 tokens) is recorded per model/cell.

  FIX F — FULL STATISTICS, PATIENT-CLUSTERED. Every uncertainty interval resamples PATIENTS
    (not queries), because ~12.6% of patients contribute multiple documents. Reports:
      * MRR@10 per model/cell with patient-clustered bootstrap CI
      * Kendall tau cross-site per genre: full panel, CONTRASTIVE-ONLY, MLM-only
        (the review showed contrastive-only discharge tau ~0.57, far below the 0.82
        full-dense number, because MLM models sit reliably at the bottom)
      * top-3 / top-5 rank overlap across sites
      * CROSS-SITE SELECTION REGRET: pick the best model at site A, measure its MRR loss
        at site B vs B's own best (the reviewer's preferred, more direct argument)
      * full variance decomposition incl. site×genre and model×site×genre (v1 omitted
        these; terms now sum to 1.0), with patient-clustered CIs on each eta2 and on
        (eta2_model×site - eta2_model×genre)

Usage:
    python two_site_v2_analyze.py --run
    python two_site_v2_analyze.py --run --models bge gte e5 nomic   # subset for a quick check
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

SEED, TOP_K, N_BOOT = 42, 10, 2000
RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper19_results")))
CELLCACHE = RESULTS / "two_site_v2_cells.json"
CACHE = RESULTS / "two_site_v2_emb"

# (key, hf_or_pair, objective, pooling, q_prefix, d_prefix)
# pooling: "mean" | "cls" ; MedCPT uses a (query_hf, article_hf) pair with cls
PANEL = [
    ("bge", "BAAI/bge-base-en-v1.5", 1, "cls",
     "Represent this sentence for searching relevant passages: ", ""),
    ("gte", "thenlper/gte-base", 1, "mean", "", ""),
    ("e5", "intfloat/e5-base-v2", 1, "mean", "query: ", "passage: "),
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", 1, "mean", "search_query: ", "search_document: "),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2", 1, "mean", "", ""),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", 1, "mean", "", ""),
    ("medcpt", ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder"), 1, "cls", "", ""),
    ("biolord", "FremyCompany/BioLORD-2023", 1, "mean", "", ""),
    ("bert-base", "bert-base-uncased", 2, "cls", "", ""),
    ("biobert", "dmis-lab/biobert-v1.1", 2, "cls", "", ""),
    ("clinicalbert", "medicalai/ClinicalBERT", 2, "cls", "", ""),
    ("pubmedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 2, "cls", "", ""),
    ("scibert", "allenai/scibert_scivocab_uncased", 2, "cls", "", ""),
]
CONTRASTIVE = {"bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"}
MLM = {"bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"}


def _encode(texts, hf, pooling, prefix, tag, kind):
    c = CACHE / f"{tag}_{kind}.npy"
    if c.exists():
        return np.load(c)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kw = {"trust_remote_code": True} if "nomic" in hf else {}
    tok = AutoTokenizer.from_pretrained(hf, **kw)
    net = AutoModel.from_pretrained(hf, **kw).to(dev).eval()
    trunc = 0
    out = []
    txt = [prefix + t for t in texts]
    with torch.no_grad():
        for i in range(0, len(txt), 8):
            enc = tok(txt[i:i + 8], padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            trunc += int((enc["attention_mask"].sum(1) >= 512).sum())
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = net(**enc).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(F.normalize(v, p=2, dim=1).cpu().numpy())
    del net, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    X = np.vstack(out).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(c, X)
    np.save(CACHE / f"{tag}_{kind}_trunc.npy", np.array([trunc, len(texts)]))
    return X


def _encode_chunks(texts, hf, pooling, prefix, tag, kind, chunk_tokens=510, stride=384):
    """Encode each document as overlapping token windows; return (vectors, doc_ids).
    Discharge notes truncate at 99.9% under single-window encoding, so single-vector
    retrieval on them is really first-512-token retrieval. Chunking removes that confound."""
    c = CACHE / f"{tag}_{kind}_chunk.npz"
    if c.exists():
        z = np.load(c); return z["v"], z["ids"]
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kw = {"trust_remote_code": True} if "nomic" in hf else {}
    tok = AutoTokenizer.from_pretrained(hf, **kw)
    net = AutoModel.from_pretrained(hf, **kw).to(dev).eval()
    windows, ids = [], []
    for di, txt in enumerate(texts):
        toks = tok(prefix + txt, add_special_tokens=False)["input_ids"]
        if not toks:
            toks = tok(prefix + " ", add_special_tokens=False)["input_ids"]
        for s in range(0, len(toks), stride):
            windows.append(toks[s:s + chunk_tokens]); ids.append(di)
            if s + chunk_tokens >= len(toks):
                break
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad = tok.pad_token_id or 0
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), 8):
            batch = windows[i:i + 8]
            seqs = [([cls_id] if cls_id is not None else []) + w +
                    ([sep_id] if sep_id is not None else []) for w in batch]
            mx = max(len(x) for x in seqs)
            att = [[1] * len(x) + [0] * (mx - len(x)) for x in seqs]
            ids_p = [x + [pad] * (mx - len(x)) for x in seqs]
            input_ids = torch.tensor(ids_p).to(dev); mask = torch.tensor(att).to(dev)
            h = net(input_ids=input_ids, attention_mask=mask).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                mm = mask.unsqueeze(-1).float()
                v = (h * mm).sum(1) / mm.sum(1).clamp(min=1e-9)
            out.append(F.normalize(v, p=2, dim=1).cpu().numpy())
    del net, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    V = np.vstack(out).astype(np.float32); ID = np.array(ids)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(c, v=V, ids=ID)
    return V, ID


def embed_cell(key, spec, pooling, qpre, dpre, queries, targets, tag, chunk=False):
    dual = isinstance(spec, tuple)
    qhf, dhf = (spec[0], spec[1]) if dual else (spec, spec)
    qp = "cls" if dual else pooling
    dp = "cls" if dual else pooling
    qpref = "" if dual else qpre
    dpref = "" if dual else dpre
    Q = _encode(queries, qhf, qp, qpref, f"{tag}_{key}", "q")
    if chunk:
        Dv, Did = _encode_chunks(targets, dhf, dp, dpref, f"{tag}_{key}", "d")
        return Q, (Dv, Did)
    D = _encode(targets, dhf, dp, dpref, f"{tag}_{key}", "d")
    return Q, D


def rr_vector(Q, D):
    S = Q @ D.T
    rr = np.zeros(len(Q), np.float32)
    k = min(TOP_K, len(D) - 1)
    for i in range(len(Q)):
        t = np.argpartition(-S[i], k)[:k]
        for r, j in enumerate(t[np.argsort(-S[i, t])]):
            if j == i:
                rr[i] = 1.0 / (r + 1)
                break
    return rr


def rr_vector_chunked(Q, Dv, Did, n_docs):
    """max-sim: document score = its best chunk's similarity to the query. Target i = doc i."""
    S = Q @ Dv.T
    doc_score = np.full((len(Q), n_docs), -np.inf, np.float32)
    for c in range(Dv.shape[0]):
        d = Did[c]
        np.maximum(doc_score[:, d], S[:, c], out=doc_score[:, d])
    rr = np.zeros(len(Q), np.float32)
    k = min(TOP_K, n_docs - 1)
    for i in range(len(Q)):
        tt = np.argpartition(-doc_score[i], k)[:k]
        for r, jj in enumerate(tt[np.argsort(-doc_score[i, tt])]):
            if jj == i:
                rr[i] = 1.0 / (r + 1); break
    return rr


def patient_clusters(patients):
    """Return list of index-arrays, one per unique patient, for clustered bootstrap."""
    groups = defaultdict(list)
    for i, p in enumerate(patients):
        groups[p].append(i)
    return [np.array(v) for v in groups.values()]


def clustered_boot_mean(rr, clusters, n=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    C = len(clusters)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, C, C)
        idx = np.concatenate([clusters[i] for i in pick])
        means[b] = rr[idx].mean()
    return float(rr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--chunk", action="store_true", help="chunk long targets, removes 512 truncation")
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    if not a.run:
        raise SystemExit("--run")
    if not CELLCACHE.exists():
        raise SystemExit(f"missing {CELLCACHE} — run two_site_v2.py --build first")

    blob = json.loads(CELLCACHE.read_text())
    N, cells = blob["N"], blob["cells"]
    panel = [p for p in PANEL if (a.models is None or p[0] in a.models)]
    mode = "CHUNKED" if a.chunk else "truncated@512"
    print(f"N={N} per cell | {len(panel)} models | query-excluded | {mode} | patient-clustered CIs\n")

    # rr vectors and MRR per model/cell
    results = {}      # (cellkey, model) -> {mrr, lo, hi, rr, patients}
    trunc_report = {}
    for key, spec, obj, pooling, qpre, dpre in panel:
        for cellkey, recs in cells.items():
            queries = [r["query"] for r in recs]
            targets = [r["target"] for r in recs]
            patients = [r["patient"] for r in recs]
            try:
                ctag = cellkey.replace("|", "_") + ("_chunk" if a.chunk else "")
                Q, D = embed_cell(key, spec, pooling, qpre, dpre, queries, targets, ctag, chunk=a.chunk)
                if a.chunk:
                    Dv, Did = D
                    rr = rr_vector_chunked(Q, Dv, Did, len(queries))
                else:
                    rr = rr_vector(Q, D)
                m, lo, hi = clustered_boot_mean(rr, patient_clusters(patients))
                results[(cellkey, key)] = {"mrr": m, "lo": lo, "hi": hi, "rr": rr,
                                           "patients": patients, "obj": obj}
                rrdir = RESULTS / ("two_site_v2_rr" + ("_chunk" if a.chunk else ""))
                rrdir.mkdir(parents=True, exist_ok=True)
                np.save(rrdir / f"{cellkey.replace('|','_')}__{key}.npy", rr)
                tf = CACHE / f"{cellkey.replace('|','_')}_{key}_d_trunc.npy"
                if tf.exists():
                    tr = np.load(tf)
                    trunc_report[(cellkey, key)] = tr[0] / tr[1]
                print(f"  {cellkey:<18} {key:<13} MRR={m:.4f} [{lo:.4f},{hi:.4f}]")
            except Exception as e:
                print(f"  {cellkey:<18} {key:<13} SKIP {type(e).__name__}: {str(e)[:50]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dump = {f"{c}|{m}": {k: v for k, v in d.items() if k != "rr" and k != "patients"}
            for (c, m), d in results.items()}
    suffix = "_chunk" if a.chunk else ""
    (RESULTS / f"two_site_v2_mrr{suffix}.json").write_text(json.dumps(dump, indent=2, default=float))

    # ---------------- RQ1: rank transfer, by model family ----------------
    from scipy.stats import kendalltau
    models_present = [p[0] for p in panel]
    def mrr_map(cellkey, subset):
        return {m: results[(cellkey, m)]["mrr"] for m in subset if (cellkey, m) in results}

    print("\n" + "=" * 92)
    print("RQ1 — CROSS-SITE RANK TRANSFER, BY MODEL FAMILY (query-excluded task)")
    print("=" * 92)
    print(f"  {'genre':<10} {'panel':<18} {'tau':>7} {'top3':>6} {'top5':>6}  models")
    for genre in ["discharge", "imaging"]:
        A, B = f"BIDMC|{genre}", f"UCSF|{genre}"
        for lbl, subset in [("full panel", models_present),
                            ("contrastive only", [m for m in models_present if m in CONTRASTIVE]),
                            ("MLM only", [m for m in models_present if m in MLM])]:
            a_ = mrr_map(A, subset); b_ = mrr_map(B, subset)
            common = sorted(set(a_) & set(b_))
            if len(common) < 3:
                continue
            tau, _ = kendalltau([a_[m] for m in common], [b_[m] for m in common])
            ra = sorted(common, key=lambda m: -a_[m]); rb = sorted(common, key=lambda m: -b_[m])
            top3 = len(set(ra[:3]) & set(rb[:3])) / 3
            top5 = len(set(ra[:5]) & set(rb[:5])) / min(5, len(common))
            print(f"  {genre:<10} {lbl:<18} {tau:>+7.3f} {top3:>6.2f} {top5:>6.2f}  {len(common)}")

    # ---------------- cross-site selection regret ----------------
    print("\n" + "=" * 92)
    print("CROSS-SITE SELECTION REGRET (contrastive models; MRR loss from using other site's pick)")
    print("=" * 92)
    contra = [m for m in models_present if m in CONTRASTIVE]
    for genre in ["discharge", "imaging"]:
        A, B = f"BIDMC|{genre}", f"UCSF|{genre}"
        a_ = mrr_map(A, contra); b_ = mrr_map(B, contra)
        if len(a_) < 3 or len(b_) < 3:
            continue
        bestA = max(a_, key=a_.get); bestB = max(b_, key=b_.get)
        regret_at_B = b_[bestB] - b_.get(bestA, min(b_.values()))
        regret_at_A = a_[bestA] - a_.get(bestB, min(a_.values()))
        print(f"  {genre}:")
        print(f"    BIDMC-best={bestA} deployed at UCSF -> regret {regret_at_B:+.4f} "
              f"(UCSF-best={bestB})")
        print(f"    UCSF-best={bestB} deployed at BIDMC -> regret {regret_at_A:+.4f} "
              f"(BIDMC-best={bestA})")

    # ---------------- RQ2: full variance decomposition, patient-clustered CI ----------------
    print("\n" + "=" * 92)
    print("RQ2 — FULL VARIANCE DECOMPOSITION (dense models; patient-clustered bootstrap CIs)")
    print("=" * 92)
    sites, genres = ["BIDMC", "UCSF"], ["discharge", "imaging"]
    dense = [m for m in models_present if m != "bm25"]

    def decompose(mrr_lookup):
        vals = [mrr_lookup[(s, g, m)] for s in sites for g in genres for m in dense]
        grand = np.mean(vals); sstot = sum((v - grand) ** 2 for v in vals)
        def fss(f):
            grp = defaultdict(list)
            for s in sites:
                for g in genres:
                    for m in dense:
                        grp[f(m, s, g)].append(mrr_lookup[(s, g, m)])
            return sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())
        ssm = fss(lambda m, s, g: m); sss = fss(lambda m, s, g: s); ssg = fss(lambda m, s, g: g)
        ms = fss(lambda m, s, g: (m, s)) - ssm - sss
        mg = fss(lambda m, s, g: (m, g)) - ssm - ssg
        sg = fss(lambda m, s, g: (s, g)) - sss - ssg
        msg = sstot - ssm - sss - ssg - ms - mg - sg
        return {k: v / sstot for k, v in
                [("model", ssm), ("site", sss), ("genre", ssg), ("model×site", ms),
                 ("model×genre", mg), ("site×genre", sg), ("model×site×genre", msg)]}

    point = {(s, g, m): results[(f"{s}|{g}", m)]["mrr"]
             for s in sites for g in genres for m in dense if (f"{s}|{g}", m) in results}
    base = decompose(point)

    # patient-clustered bootstrap of the whole decomposition + the interaction difference
    rng = np.random.default_rng(SEED)
    cl = {c: patient_clusters([r["patient"] for r in cells[c]]) for c in cells}
    boot_terms = defaultdict(list); boot_diff = []
    for _ in range(N_BOOT):
        mr = {}
        resample = {}
        for c in cells:
            C = len(cl[c]); pick = rng.integers(0, C, C)
            resample[c] = np.concatenate([cl[c][i] for i in pick])
        for s in sites:
            for g in genres:
                idx = resample[f"{s}|{g}"]
                for m in dense:
                    if (f"{s}|{g}", m) in results:
                        mr[(s, g, m)] = results[(f"{s}|{g}", m)]["rr"][idx].mean()
        d = decompose(mr)
        for k, v in d.items():
            boot_terms[k].append(v)
        boot_diff.append(d["model×site"] - d["model×genre"])

    print(f"  {'term':<20} {'eta2':>8}   {'95% CI':>18}")
    for k in ["model", "genre", "model×site", "model×genre", "site", "site×genre", "model×site×genre"]:
        lo, hi = np.percentile(boot_terms[k], [2.5, 97.5])
        print(f"  {k:<20} {base[k]:>8.3f}   [{lo:>7.3f}, {hi:>7.3f}]")
    print(f"  {'SUM':<20} {sum(base.values()):>8.3f}")
    dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
    print(f"\n  eta2(model×site) - eta2(model×genre) = {base['model×site']-base['model×genre']:+.3f}"
          f"  95% CI [{dlo:+.3f}, {dhi:+.3f}]")
    if dlo > 0:
        print("  -> CI excludes zero (positive): model×site reliably EXCEEDS model×genre.")
    elif dhi < 0:
        print("  -> CI excludes zero (negative): model×genre reliably EXCEEDS model×site.")
        print("     Institution perturbs rankings LESS than documentation genre.")
    else:
        print("  -> CI includes zero: the two interactions are statistically indistinguishable.")

    # truncation report
    if trunc_report:
        print("\n" + "=" * 92)
        print("TRUNCATION (share of target docs hitting the 512-token limit)")
        print("=" * 92)
        for genre in ["discharge", "imaging"]:
            worst = max((v for (c, m), v in trunc_report.items() if genre in c), default=0)
            print(f"  {genre:<12} max across models: {worst:.1%}")


if __name__ == "__main__":
    main()
