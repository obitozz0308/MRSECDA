"""
Case Study: Top-10 circRNA predictions for Breast/Lung/Colorectal Cancer
with PubMed PMID validation.

Trains MRSECDA (v3.3 + no lncRNA + 2-feature PSE) on ALL circR2Cancer
associations (no CV split), then predicts scores for all unlabeled
circRNA-disease pairs per target disease.

Usage: python scripts/run_case_study_mrsecda.py
"""
import os, sys, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from data.dataset import CDADataset
from data.hetero_data import build_hetero_graph
from data.structural_features import MultiViewStructuralFeatureComputer
from data.metapath_encoder import compute_metapath_degrees
from scripts.run_mrsecda import build_metapath_hgt_edges, focal_bce_loss, filter_node_type_from_graph

DEVICE = "cpu"
SEED = 42

# Target diseases (3 cancers, matching v3.3 case study)
TARGETS = {
    "Breast Cancer": "breast cancer",
    "Lung Cancer": "lung cancer",
    "Colorectal Cancer": "colorectal cancer",
}


# Disease-name synonyms (common PubMed spellings) used as fallback query terms.
DISEASE_SYNONYMS = {
    "breast cancer": ["breast cancer", "breast carcinoma", "breast tumor", "breast tumours",
                      "breast adenocarcinoma", "triple-negative breast cancer", "mammary carcinoma"],
    "lung cancer": ["lung cancer", "lung carcinoma", "non-small cell lung cancer",
                    "non-small-cell lung cancer", "small cell lung cancer", "lung adenocarcinoma",
                    "lung squamous cell carcinoma", "pulmonary carcinoma"],
    "colorectal cancer": ["colorectal cancer", "colorectal carcinoma", "colon cancer",
                          "colon carcinoma", "colon adenocarcinoma", "rectal cancer"],
}

# Well-established circRNA aliases (canonical gene symbol).
CIRCRNA_ALIASES = {
    "hsa_circ_0000284": "circHIPK3",
    "circRNA_100290": "hsa_circ_100290",
    "circ-Foxo3": "circFOXO3",
    "CDR1as": "ciRS-7",
    # circR2Disease "hsa_circRNA_000479" = circBase hsa_circ_0000479 = circEPSTI1
    "hsa_circRNA_000479": "circEPSTI1",
}


def _circrna_variants(name: str) -> list:
    """Return plausible PubMed spellings for a circRNA name (deduped, ordered)."""
    name = name.strip()
    out = [name]
    low = name.lower()
    # circ-XXX -> circXXX / circXXX(UPPER) / circRNA-XXX / XXX
    if name.startswith("circ-"):
        core = name[5:]
        out += ["circ" + core, "circ" + core.upper(), "circRNA-" + core, core]
    # circXXX (no dash) -> circ-XXX / circRNA-XXX / XXX
    elif low.startswith("circ") and not low.startswith(("circrna", "circ_", "circ-")):
        core = name[4:]
        if core:
            out += ["circ-" + core, "circRNA-" + core, core]
    # Numeric circRNAs: try common prefix spellings.
    num = None
    for prefix in ("circRNA_", "hsa_circRNA_", "hsa_circ_", "circ_"):
        if name.startswith(prefix):
            num = name[len(prefix):]
            break
    if num is not None:
        for prefix in ("circRNA_", "hsa_circRNA_", "hsa_circ_", "circ_",
                       "circRNA-", "circ-", "hsa_circRNA-"):
            out.append(prefix + num)
        out.append("circ" + num)
        # leading-zero normalization (hsa_circ_00001666 -> hsa_circ_0001666)
        stripped = num.lstrip("0") or "0"
        for w in (7, 6):
            pn = stripped.zfill(w)
            if pn != num:
                for prefix in ("hsa_circ_", "circRNA_", "circ_", "circ-", "circ"):
                    out.append(prefix + pn)
    if name.startswith("circRNA-"):
        out.append(name[len("circRNA-"):])
    if name in CIRCRNA_ALIASES:
        out.append(CIRCRNA_ALIASES[name])
    seen, uniq = set(), []
    for v in out:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq


def _efetch_titles(pmids) -> dict:
    """Fetch article titles for a list of PMIDs (single batched efetch call)."""
    if not pmids:
        return {}
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        root = ET.fromstring(resp.read().decode("utf-8"))
    titles = {}
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        if (pmid_el is not None and pmid_el.text
                and title_el is not None and title_el.text):
            titles[pmid_el.text] = title_el.text
    return titles


def _title_confirms(title: str, variants: list, synonyms: list) -> bool:
    """A paper counts as confirming evidence only if its title names BOTH the
    circRNA (under some spelling) AND the disease (under some synonym).

    Two acceptance paths:
      1. title contains a circ-bearing variant (e.g. "circMTO1", "circ-MTO1");
      2. title contains the bare gene symbol together with the "circular RNA"
         phrase (e.g. "circular RNA MTO1 ..."), or a canonical circRNA proper
         name (CDR1as / ciRS-7).

    Requiring the "circular RNA" phrase (not just "circ") rejects titles where
    "circ" comes from a DIFFERENT circRNA and the gene symbol is a downstream
    effector (e.g. "hsa_circ_0000729 ... FOXO3 axis" is not circFOXO3). This is
    the field-standard (see MTGCDA, MNMDCDA, SAGCN).
    """
    tl = title.lower()
    if any(w in tl for w in ("retraction", "retracted", "withdrawn")):
        return False
    if not any(d.lower() in tl for d in synonyms):
        return False
    circ_variants = [v for v in variants if "circ" in v.lower()]
    if any(v.lower() in tl for v in circ_variants):
        return True
    core_variants = [v for v in variants if "circ" not in v.lower()]
    if not any(v.lower() in tl for v in core_variants):
        return False
    if any(v.lower() in tl for v in ("cdr1as", "cirs-7")):
        return True
    return "circular rna" in tl


def _core_symbol(name: str) -> str:
    name = name.strip()
    low = name.lower()
    for p in ("hsa_circrna_", "hsa_circ_", "circrna_", "circ_",
              "hsa_circrna-", "circrna-", "circ-", "circ"):
        if low.startswith(p):
            return name[len(p):]
    return name


def search_pubmed(circrna: str, disease: str, max_results: int = 5) -> list:
    """Return PMIDs that CONFIRM the circRNA-disease association.

    Retrieves candidate PMIDs via esearch, then keeps only those whose article
    title names both the circRNA and the disease (see _title_confirms).
    """
    synonyms = DISEASE_SYNONYMS.get(disease.lower().strip(), [disease])
    variants = _circrna_variants(circrna)
    core = _core_symbol(circrna)
    dis_q = " OR ".join(f'"{d}"' for d in synonyms)

    # Numeric circRNAs: use precise phrases (hsa_circ_NNNN etc.) — a combined
    # OR of many numeric variants dilutes relevance and drops true hits.
    # Gene-symbol circRNAs: use the bare core + "circ" to catch varied
    # spellings (circMTO1 / circ-MTO1 / circular RNA MTO1).
    if core.isdigit():
        # circBase IDs are 7-digit; the dataset sometimes has 6/8-digit variants.
        nums = [core]
        stripped = core.lstrip("0") or "0"
        for w in (7, 6):
            p = stripped.zfill(w)
            if p not in nums:
                nums.append(p)
        queries = []
        for num in nums:
            queries += [f'"{p}{num}" AND ({dis_q})'
                        for p in ("hsa_circ_", "circRNA_", "circ_", "circ-", "circ")]
        # gene-symbol alias for a numeric circRNA (hsa_circRNA_000479 -> circEPSTI1)
        for alias in [a for k, a in CIRCRNA_ALIASES.items()
                      if circrna.lower() == k.lower() and not a.isdigit()]:
            queries.append(f'"{alias}" AND ({dis_q})')
            queries.append(f'"{alias}" AND circ AND ({dis_q})')
    else:
        core_terms = [core]
        for k, v in CIRCRNA_ALIASES.items():
            if circrna.lower() in (k.lower(), v.lower()):
                for t in (k, v):
                    if t not in core_terms:
                        core_terms.append(t)
        queries = []
        for ct in core_terms:
            queries.append(f'"{ct}" AND circ AND ({dis_q})')
            queries.append(f'"{ct}" AND ({dis_q})')
        queries.append(f'"circ{core}" AND ({dis_q})')
        queries.append(f'"circ-{core}" AND ({dis_q})')
        queries.append(f'"circRNA-{core}" AND ({dis_q})')

    candidate_ids, seen = [], set()
    for query in queries:
        try:
            params = urllib.parse.urlencode({
                "db": "pubmed", "term": query,
                "retmax": 20, "retmode": "xml", "sort": "relevance",
            })
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
            with urllib.request.urlopen(url, timeout=20) as resp:
                root = ET.fromstring(resp.read().decode("utf-8"))
            for e in root.findall(".//Id"):
                if e.text and e.text not in seen:
                    seen.add(e.text)
                    candidate_ids.append(e.text)
        except Exception:
            continue
        time.sleep(0.35)  # respect NCBI rate limit (3 req/s w/o API key)

    # Batch-fetch titles and keep only those that confirm the association.
    confirmed = []
    for i in range(0, len(candidate_ids), 30):
        batch = candidate_ids[i:i + 30]
        try:
            titles = _efetch_titles(batch)
        except Exception:
            titles = {}
        for pmid in batch:
            if _title_confirms(titles.get(pmid, ""), variants, synonyms):
                confirmed.append(pmid)
                if len(confirmed) >= max_results:
                    return confirmed
        time.sleep(0.35)
    return confirmed


def main():
    print("=" * 70)
    print("  MRSECDA Case Study: Top-10 circRNA Predictions")
    print("=" * 70)

    # Fix seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Config ──
    config = Config()
    config.train.device = DEVICE
    config.model.use_metapath_edges = True
    config.train.finetune_lr = 5e-4
    config.model.hidden_dim = 96
    config.model.dropout = 0.3
    config.model.gcn_dropout = 0.3
    config.model.decoder_dropout = 0.3
    config.model.focal_gamma = 2.0
    config.train.finetune_epochs = 400
    config.train.batch_size = 4096
    config.data.neg_ratio = 1
    config.model.lap_pe_dim = 16
    config.model.rwse_dim = 16
    config.model.gcn_layers = 3    # new params: L=3 (HGT layers)
    config.model.use_hgt = True
    config.model.hgt_heads = 8     # new params: H=8 (attention heads)
    config.model.decoder_hidden = [96, 48]

    # ── 1. Load data ──
    print("\n[1/4] Loading circR2Cancer...")
    ds = CDADataset(
        name="circR2Cancer",
        data_dir=os.path.join(ROOT, "data", "CompareExperiment", "Data"),
        seed=SEED, use_external_knowledge=True, use_kmer_features=True,
    )
    N_c, N_d = ds.adj.shape
    adj = ds.adj
    print(f"  circRNAs: {N_c}, Diseases: {N_d}, Associations: {int(adj.sum())}")

    # Find target disease indices
    disease_names_lower = [str(d).lower().strip() for d in ds.disease_names]
    target_indices = {}
    for group, target_name in TARGETS.items():
        for i, dn in enumerate(disease_names_lower):
            if target_name in dn:
                target_indices[group] = i
                print(f"  [{group}] {ds.disease_names[i]} (idx={i})")
                break

    # ── 2. Build hetero graph ──
    print("\n[2/4] Building heterogeneous graph...")
    train_mask = np.ones_like(adj, dtype=bool)

    circ_mirna_h5_dir = os.path.join(ROOT, "data", "CompareExperiment",
                                      "Data", "circRNA_miRNA")
    khgat_dir = os.path.join(ROOT, "data", "CompareExperiment",
                              "AE-RF main", "KHGAT", "data")

    graph_data, x_all, node_counts = build_hetero_graph(
        ds, train_mask,
        khgat_data_dir=khgat_dir,
        lap_pe_dim=config.model.lap_pe_dim,
        rwse_dim=config.model.rwse_dim,
        use_hmdd_v4=True,
        circ_mirna_h5_dir=circ_mirna_h5_dir,
    )

    N_c_full = node_counts['N_circ']
    N_d_full = node_counts['N_disease']
    N_m = node_counts['N_mirna']
    N_l = node_counts['N_lncrna']

    # Extract per-type features
    x_circ = x_all[:N_c_full]
    x_disease = x_all[N_c_full:N_c_full + N_d_full]
    x_mirna = x_all[N_c_full + N_d_full:N_c_full + N_d_full + N_m]
    x_lncrna = x_all[N_c_full + N_d_full + N_m:]

    edge_index = graph_data.edge_index
    edge_index_by_type = getattr(graph_data, 'edge_index_by_type', None)

    # PE
    pe_lap = getattr(graph_data, 'lap_pe', None)
    pe_rw = getattr(graph_data, 'rwse', None)
    if pe_lap is not None:
        node_counts['pe_lap'] = pe_lap
    if pe_rw is not None:
        node_counts['pe_rw'] = pe_rw

    # ── MRSECDA: remove lncRNA nodes (3 types: circ/disease/miRNA) ──
    x_all, edge_index, edge_index_by_type, node_counts = filter_node_type_from_graph(
        x_all, edge_index, edge_index_by_type, node_counts, "lncrna")
    N_c_full = node_counts['N_circ']
    N_d_full = node_counts['N_disease']
    N_m = node_counts['N_mirna']
    N_l = node_counts['N_lncrna']
    x_circ = x_all[:N_c_full]
    x_disease = x_all[N_c_full:N_c_full + N_d_full]
    x_mirna = x_all[N_c_full + N_d_full:N_c_full + N_d_full + N_m]
    x_lncrna = x_all[N_c_full + N_d_full + N_m:]

    print(f"  Nodes: circ={N_c_full}, dis={N_d_full}, mir={N_m}, lnc={N_l}")
    print(f"  Features: {x_all.shape[1]}D total")

    # MetaPath edges
    if config.model.use_metapath_edges and edge_index_by_type is not None:
        mp_new, _, _ = build_metapath_hgt_edges(
            edge_index_by_type, N_c_full, N_d_full, N_m)
        edge_index_by_type = dict(edge_index_by_type)
        edge_index_by_type.update(mp_new)
        n_cmc = mp_new.get(3, torch.zeros(2, 0)).shape[1]
        n_dmd = mp_new.get(4, torch.zeros(2, 0)).shape[1]
        print(f"  MetaPath edges: CMC={n_cmc}, DMD={n_dmd}")
    num_edge_types = 5 if config.model.use_metapath_edges else 3

    # Structural features (MRSECDA: 2 features = jaccard + Adamic-Adar)
    struct_feat_computer = MultiViewStructuralFeatureComputer(
        edge_index_by_type=edge_index_by_type,
        N_total=N_c_full + N_d_full + N_m + N_l,
        N_circ=N_c_full, N_disease=N_d_full,
        node_counts=node_counts,
        drop_feature=["resource_alloc", "inv_distance"],
    )
    struct_feat_dim = struct_feat_computer.feat_dim
    print(f"  StructFeat: {struct_feat_dim}-dim")

    # ── 3. Train model on ALL associations ──
    print(f"\n[3/4] Training MRSECDA on full circR2Cancer...")

    from models.mrsecda_hetero import MRSECDAHetero

    model = MRSECDAHetero(
        input_dim_circrna=x_circ.shape[1],
        input_dim_disease=x_disease.shape[1],
        input_dim_mirna=x_mirna.shape[1],
        input_dim_lncrna=x_lncrna.shape[1],
        hidden_dim=config.model.hidden_dim,
        lap_pe_dim=config.model.lap_pe_dim,
        rwse_dim=config.model.rwse_dim,
        gcn_layers=config.model.gcn_layers,
        gcn_dropout=config.model.gcn_dropout,
        decoder_hidden=tuple(config.model.decoder_hidden),
        decoder_dropout=config.model.decoder_dropout,
        num_resolutions=3,
        struct_feat_dim=struct_feat_dim,
        use_hgt=config.model.use_hgt,
        hgt_heads=config.model.hgt_heads,
        num_edge_types=num_edge_types,
        use_metapath_edges=config.model.use_metapath_edges,
        edge_dropout=0.05,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Create full training set: all positives + equal random negatives
    pos_pairs = list(zip(*np.where(adj > 0)))
    n_pos = len(pos_pairs)

    np.random.seed(SEED)
    neg_pairs = set()
    while len(neg_pairs) < n_pos:
        ci, di = np.random.randint(0, N_c), np.random.randint(0, N_d)
        if adj[ci, di] == 0:
            neg_pairs.add((ci, di))
    neg_pairs = list(neg_pairs)

    train_c = torch.tensor([p[0] for p in pos_pairs] + [n[0] for n in neg_pairs], dtype=torch.long)
    train_d = torch.tensor([p[1] for p in pos_pairs] + [n[1] for n in neg_pairs], dtype=torch.long)
    train_l = torch.tensor([1.0] * n_pos + [0.0] * n_pos, dtype=torch.float32)
    perm = torch.randperm(len(train_c))
    train_c, train_d, train_l = train_c[perm], train_d[perm], train_l[perm]

    print(f"  Training pairs: {n_pos * 2} ({n_pos} pos + {n_pos} neg)")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.finetune_lr,
                                  weight_decay=1e-5)
    bs = config.train.batch_size
    epochs = config.train.finetune_epochs

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        perm_e = torch.randperm(len(train_c))
        total_loss = 0.0
        n_batch = 0
        for i in range(0, len(train_c), bs):
            idx = perm_e[i:i + bs]
            c_b, d_b, l_b = train_c[idx], train_d[idx], train_l[idx]

            # Compute structural features for this batch
            sf_np = struct_feat_computer.compute_features(
                c_b.numpy(), d_b.numpy())
            sf = torch.tensor(sf_np, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            logits, _ = model(
                x_circ.to(DEVICE), x_disease.to(DEVICE),
                x_mirna.to(DEVICE), x_lncrna.to(DEVICE),
                edge_index.to(DEVICE), node_counts,
                c_b.to(DEVICE), d_b.to(DEVICE),
                edge_index_by_type={
                    k: v.to(DEVICE) for k, v in edge_index_by_type.items()
                } if edge_index_by_type else None,
                struct_feat=sf,
            )
            loss = focal_bce_loss(logits.squeeze(), l_b.to(DEVICE),
                                 gamma=config.model.focal_gamma)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss / max(n_batch, 1):.6f}")

    print(f"  Training done in {time.time() - t0:.0f}s")

    # Save model for future reuse
    model_path = os.path.join(ROOT, "results", "mrsecda", "case_study_model_mrsecda.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    # ── 4. Predict Top-10 + PubMed ──
    print(f"\n[4/4] Predicting Top-10 circRNAs per cancer + PubMed validation...\n")

    model.eval()
    known_pos = set(zip(*np.where(adj > 0)))
    all_results = {}

    for group, di in target_indices.items():
        dname = str(ds.disease_names[di])
        known = [ci for ci, d in known_pos if d == di]

        # Predict for ALL circRNAs
        all_scores = []
        with torch.no_grad():
            for i in range(0, N_c, bs * 4):
                c_b = torch.arange(i, min(i + bs * 4, N_c), dtype=torch.long)
                d_b = torch.full((len(c_b),), di, dtype=torch.long)
                # Compute structural features for these pairs
                sf_np = struct_feat_computer.compute_features(
                    c_b.numpy(), d_b.numpy())
                sf = torch.tensor(sf_np, dtype=torch.float32, device=DEVICE)
                logits, _ = model(
                    x_circ.to(DEVICE), x_disease.to(DEVICE),
                    x_mirna.to(DEVICE), x_lncrna.to(DEVICE),
                    edge_index.to(DEVICE), node_counts,
                    c_b.to(DEVICE), d_b.to(DEVICE),
                    edge_index_by_type={
                        k: v.to(DEVICE) for k, v in edge_index_by_type.items()
                    } if edge_index_by_type else None,
                    struct_feat=sf,
                )
                all_scores.append(torch.sigmoid(logits).squeeze().cpu().numpy())

        scores = np.concatenate(all_scores)

        # Exclude known positives
        mask = np.ones(N_c, dtype=bool)
        mask[known] = False

        # Top-10
        candidate = [(ci, float(scores[ci])) for ci in range(N_c) if mask[ci]]
        candidate.sort(key=lambda x: x[1], reverse=True)
        top10 = candidate[:10]

        print(f"  {'='*60}")
        print(f"  {group} — {dname}")
        print(f"  Known: {len(known)}, Candidates: {len(candidate)}")
        print(f"  {'='*60}")

        preds = []
        for rank, (ci, score) in enumerate(top10, 1):
            cname = str(ds.circrna_names[ci]).strip()
            pmids = search_pubmed(cname, dname)
            pmid_str = ", ".join(pmids) if pmids else "—"
            print(f"  #{rank:2d}: {cname:30s} | {score:.4f} | PubMed: {pmid_str}")
            preds.append({
                "rank": rank, "circrna": cname, "circrna_idx": int(ci),
                "score": score, "pmids": pmids,
            })
            time.sleep(0.35)  # Rate limit PubMed API

        all_results[group] = {
            "disease": dname, "disease_idx": int(di),
            "n_known": len(known), "predictions": preds,
        }

    # ── Save ──
    out_dir = os.path.join(ROOT, "results", "mrsecda")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "case_study_3cancers_mrsecda.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown report
    lines = [
        "# Case Study: Top-10 circRNA Predictions for 3 Cancers (MRSECDA)",
        "",
        f"**Model**: MRSECDA (no lncRNA + 2-feature PSE), trained on full circR2Cancer ({n_pos} associations, {epochs} epochs)",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Params**: {n_params:,}",
        "",
    ]
    for group, gr in all_results.items():
        lines.append(f"## {group} — {gr['disease']}")
        lines.append(f"Known associations: {gr['n_known']}")
        lines.append("")
        lines.append("| # | circRNA | Score | PubMed PMIDs |")
        lines.append("|---|---------|-------|-------------|")
        with_ev = 0
        for p in gr["predictions"]:
            ps = ", ".join(p["pmids"]) if p["pmids"] else "—"
            if p["pmids"]:
                with_ev += 1
            lines.append(f"| {p['rank']} | {p['circrna']} | {p['score']:.4f} | {ps} |")
        lines.append("")
        lines.append(f"**{with_ev}/10 predictions have PubMed evidence.**")
        lines.append("")

    md_path = os.path.join(out_dir, "case_study_3cancers_mrsecda.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  JSON → {json_path}")
    print(f"  MD   → {md_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
