"""
MRSECDA 训练器 (精简版)
- 单阶段监督训练 (无对比预训练)
- 评估函数
"""

import copy
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

from config import Config


# =============================================================================
# 工具函数
# =============================================================================

def build_optimizer(model: nn.Module, config) -> optim.Optimizer:
    if config.train.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.train.finetune_lr,
            weight_decay=config.train.finetune_weight_decay,
        )
    return optim.Adam(
        model.parameters(),
        lr=config.train.finetune_lr,
        weight_decay=config.train.finetune_weight_decay,
    )


def build_scheduler(optimizer, config, n_epochs: int) -> Optional[object]:
    if config.train.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=config.train.lr_min)
    elif config.train.lr_scheduler == "step":
        return StepLR(optimizer, step_size=config.train.lr_step_size, gamma=config.train.lr_gamma)
    elif config.train.lr_scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=config.train.lr_gamma,
                                  patience=config.train.finetune_patience // 2,
                                  min_lr=config.train.lr_min)
    return None


def sample_negative_pairs(
    train_mask: np.ndarray,
    n_samples: int,
    exclude_pairs: set = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """从未知关联中采样负样本。"""
    N_c, N_d = train_mask.shape
    all_unknown = np.where(train_mask == 0)
    n_available = len(all_unknown[0])

    if exclude_pairs is not None and len(exclude_pairs) > 0:
        valid_mask = np.ones(n_available, dtype=bool)
        for idx in range(n_available):
            pair = (all_unknown[0][idx], all_unknown[1][idx])
            if pair in exclude_pairs:
                valid_mask[idx] = False
        available_idx = np.where(valid_mask)[0]
    else:
        available_idx = np.arange(n_available)

    n_samples = min(n_samples, len(available_idx))
    sampled = np.random.choice(available_idx, n_samples, replace=False)

    return all_unknown[0][sampled], all_unknown[1][sampled]


def prepare_batches(
    pos_circ: np.ndarray,
    pos_dis: np.ndarray,
    neg_circ: np.ndarray,
    neg_dis: np.ndarray,
    batch_size: int = 4096,
    shuffle: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n_pos = len(pos_circ)
    n_neg = len(neg_circ)

    all_circ = np.concatenate([pos_circ, neg_circ])
    all_dis = np.concatenate([pos_dis, neg_dis])
    all_label = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    if shuffle:
        perm = np.random.permutation(len(all_circ))
        all_circ = all_circ[perm]
        all_dis = all_dis[perm]
        all_label = all_label[perm]

    batches = []
    for start in range(0, len(all_circ), batch_size):
        end = min(start + batch_size, len(all_circ))
        batches.append((
            torch.tensor(all_circ[start:end], dtype=torch.long),
            torch.tensor(all_dis[start:end], dtype=torch.long),
            torch.tensor(all_label[start:end], dtype=torch.float32),
        ))
    return batches


# =============================================================================
# 评估函数
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_circ: torch.Tensor,
    x_disease: torch.Tensor,
    lap_pe_circ: torch.Tensor,
    lap_pe_disease: torch.Tensor,
    rwse_circ: torch.Tensor,
    rwse_disease: torch.Tensor,
    edge_index: torch.Tensor,
    N_circ: int,
    test_circ_idx: np.ndarray,
    test_dis_idx: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
    # Side branch inputs
    kmer_feat: torch.Tensor = None,
    mesh_feat: torch.Tensor = None,
    adj: torch.Tensor = None,
) -> Dict[str, float]:
    model.eval()

    h_circ, h_disease = model.encode(
        x_circ.to(device), x_disease.to(device),
        lap_pe_circ.to(device), lap_pe_disease.to(device),
        rwse_circ.to(device), rwse_disease.to(device),
        edge_index.to(device),
        N_circ=N_circ,
        adj=adj.to(device) if adj is not None else None,
    )

    kmer_dev = kmer_feat.to(device) if kmer_feat is not None else None
    mesh_dev = mesh_feat.to(device) if mesh_feat is not None else None

    all_scores = []
    n_test = len(test_labels)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        ci = torch.tensor(test_circ_idx[start:end], dtype=torch.long, device=device)
        di = torch.tensor(test_dis_idx[start:end], dtype=torch.long, device=device)

        if model.use_side_branches and hasattr(model, 'side_gate'):
            # Compute side-branch fused scores (as probabilities)
            score_orig = model.decoder(h_circ[ci], h_disease[di]).squeeze(-1)
            side_scores = model.compute_side_scores(
                h_circ, h_disease, ci, di,
                kmer_feat=kmer_dev, mesh_feat=mesh_dev,
            )
            fused_logit = model.side_gate(score_orig, side_scores)
            scores = torch.sigmoid(fused_logit)
        else:
            scores = model.predict(h_circ, h_disease, circ_indices=ci, disease_indices=di)

        all_scores.append(scores.cpu())

    scores = torch.cat(all_scores).numpy()

    auc = roc_auc_score(test_labels, scores)
    aupr = average_precision_score(test_labels, scores)

    preds = (scores > 0.5).astype(int)
    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds, zero_division=0)
    rec = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)

    return {
        "auc": auc, "aupr": aupr,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
    }


# =============================================================================
# 单阶段监督训练
# =============================================================================

def train_supervised(
    model: nn.Module,
    x_circ: torch.Tensor,
    x_disease: torch.Tensor,
    lap_pe_circ: torch.Tensor,
    lap_pe_disease: torch.Tensor,
    rwse_circ: torch.Tensor,
    rwse_disease: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    N_circ: int,
    train_pos: Tuple[np.ndarray, np.ndarray],
    test_pos: Tuple[np.ndarray, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config,
    val_pos: Tuple[np.ndarray, np.ndarray] = None,
    val_mask: np.ndarray = None,
    # Side branch inputs (optional)
    kmer_feat: torch.Tensor = None,
    mesh_feat: torch.Tensor = None,
    adj: torch.Tensor = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    单阶段监督训练: 编码 → 预测 → BCE loss → 反向传播。

    Uses VALIDATION set (val_pos/val_mask) for early stopping and checkpoint
    selection. Test set is ONLY evaluated once after training completes.
    Falls back to old behaviour (test for selection) if val_pos not provided.

    Side branch support (innovation-2): when model.use_side_branches=True,
    kmer_feat and mesh_feat are used by SeqEncoder and DO_Encoder.
    """
    device = config.train.device
    model = model.to(device)
    model.train()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, config.train.finetune_epochs)

    train_pos_circ, train_pos_dis = train_pos
    test_pos_circ, test_pos_dis = test_pos
    n_train = len(train_pos_circ)

    # ── Build exclude set for negative sampling ──
    all_pos_mask = np.clip(train_mask + test_mask, 0, 1)
    if val_mask is not None:
        all_pos_mask = np.clip(all_pos_mask + val_mask, 0, 1)
    all_pos_pairs_set = set(zip(test_pos_circ, test_pos_dis))
    all_pos_pairs_set |= set(zip(train_pos_circ, train_pos_dis))

    # ── Prepare VALIDATION data (for epoch selection) ──
    use_val = val_pos is not None and val_mask is not None
    if use_val:
        val_pos_circ, val_pos_dis = val_pos
        all_pos_pairs_set |= set(zip(val_pos_circ, val_pos_dis))

        neg_circ_val, neg_dis_val = sample_negative_pairs(
            all_pos_mask, n_samples=len(val_pos_circ),
            exclude_pairs=all_pos_pairs_set,
        )
        val_all_circ = np.concatenate([val_pos_circ, neg_circ_val])
        val_all_dis = np.concatenate([val_pos_dis, neg_dis_val])
        val_all_labels = np.concatenate([np.ones(len(val_pos_circ)), np.zeros(len(neg_circ_val))])

    # ── Prepare TEST data (ONLY for final evaluation) ──
    neg_circ_test, neg_dis_test = sample_negative_pairs(
        all_pos_mask, n_samples=len(test_pos_circ),
        exclude_pairs=all_pos_pairs_set,
    )
    test_all_circ = np.concatenate([test_pos_circ, neg_circ_test])
    test_all_dis = np.concatenate([test_pos_dis, neg_dis_test])
    test_all_labels = np.concatenate([np.ones(len(test_pos_circ)), np.zeros(len(neg_circ_test))])

    # ── Which set to use for epoch selection ──
    if use_val:
        sel_circ, sel_dis, sel_labels = val_all_circ, val_all_dis, val_all_labels
        sel_name = "Val"
    else:
        sel_circ, sel_dis, sel_labels = test_all_circ, test_all_dis, test_all_labels
        sel_name = "Test"

    # BCE loss (带正样本权重)
    pos_weight = torch.tensor([max(1.0, config.data.neg_ratio)]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = defaultdict(list)
    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.train.finetune_epochs):
        model.train()

        # 负采样
        n_neg = n_train * config.data.neg_ratio
        neg_circ, neg_dis = sample_negative_pairs(
            train_mask, n_samples=n_neg,
            exclude_pairs=all_pos_pairs_set,
        )

        batches = prepare_batches(
            train_pos_circ, train_pos_dis,
            neg_circ, neg_dis,
            batch_size=config.train.batch_size,
            shuffle=True,
        )

        n_batches = len(batches)
        epoch_loss = 0.0

        # 一次编码, 所有batch共用
        h_circ, h_disease = model.encode(
            x_circ.to(device), x_disease.to(device),
            lap_pe_circ.to(device), lap_pe_disease.to(device),
            rwse_circ.to(device), rwse_disease.to(device),
            edge_index.to(device),
            edge_weight.to(device) if edge_weight is not None else None,
            N_circ=N_circ,
            adj=adj.to(device) if adj is not None else None,
        )

        # Move side-branch inputs to device
        kmer_dev = kmer_feat.to(device) if kmer_feat is not None else None
        mesh_dev = mesh_feat.to(device) if mesh_feat is not None else None

        optimizer.zero_grad()
        for circ_idx, dis_idx, labels in batches:
            circ_idx = circ_idx.to(device)
            dis_idx = dis_idx.to(device)
            labels = labels.to(device)

            if model.use_side_branches and hasattr(model, 'side_gate'):
                # Side-branch path: decoder logits + side scores → gate → fused logit
                score_orig = model.decoder(
                    h_circ[circ_idx], h_disease[dis_idx]
                ).squeeze(-1)
                side_scores = model.compute_side_scores(
                    h_circ, h_disease,
                    circ_idx, dis_idx,
                    kmer_feat=kmer_dev, mesh_feat=mesh_dev,
                )
                logits = model.side_gate(score_orig, side_scores)
            else:
                logits = model.decoder(
                    h_circ[circ_idx], h_disease[dis_idx]
                )
            batch_loss = bce_loss(logits.squeeze(-1), labels)

            # Include side gate reg in batch loss (must be same backward call)
            if model.use_side_branches and hasattr(model, 'side_gate'):
                gate_reg = model.get_side_gate_loss()
                if gate_reg.item() != 0.0:
                    batch_loss = batch_loss + gate_reg

            # Gradient accumulation: sum losses, one optimizer step per epoch
            (batch_loss / n_batches).backward(retain_graph=True)
            epoch_loss += bce_loss(logits.squeeze(-1), labels).item()

        # Side branch gate stats (reg already included in batch loss above)
        if model.use_side_branches and hasattr(model, 'side_gate'):
            gate_reg = model.get_side_gate_loss()
            if gate_reg.item() != 0.0:
                history.setdefault("side_gate_reg", []).append(gate_reg.item())

        # Single optimizer step after accumulating all batch gradients
        if config.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(best_auc)
            else:
                scheduler.step()

        epoch_loss /= n_batches
        history["loss"].append(epoch_loss)

        # ── Evaluate on SELECTION set (val or test) for early stopping ──
        if epoch % config.train.log_interval == 0 or epoch == config.train.finetune_epochs - 1:
            metrics = evaluate(
                model, x_circ, x_disease,
                lap_pe_circ, lap_pe_disease,
                rwse_circ, rwse_disease,
                edge_index, N_circ,
                sel_circ, sel_dis, sel_labels,
                batch_size=config.train.batch_size,
                device=device,
                kmer_feat=kmer_feat,
                mesh_feat=mesh_feat,
                adj=adj,
            )
            for k, v in metrics.items():
                history[k].append(v)

            best_str = ""
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_str = " [NEW BEST]"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch:4d}/{config.train.finetune_epochs} | "
                  f"Loss: {epoch_loss:.6f} | {sel_name} AUC: {metrics['auc']:.4f} | "
                  f"{sel_name} AUPR: {metrics['aupr']:.4f} | Acc: {metrics['accuracy']:.4f}{best_str}")

        if patience_counter >= config.train.finetune_patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Load best checkpoint (selected by val AUC) ──
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── FINAL evaluation on TEST set (untouched during training) ──
    final_metrics = evaluate(
        model, x_circ, x_disease,
        lap_pe_circ, lap_pe_disease,
        rwse_circ, rwse_disease,
        edge_index, N_circ,
        test_all_circ, test_all_dis, test_all_labels,
        batch_size=config.train.batch_size * 4,
        device=device,
        kmer_feat=kmer_feat,
        mesh_feat=mesh_feat,
        adj=adj,
    )

    print(f"\n  === Final Results (on Test) ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    return dict(history), final_metrics


# =============================================================================
# 完整训练入口
# =============================================================================

def train_mrsecda(
    model: nn.Module,
    x_circ: torch.Tensor,
    x_disease: torch.Tensor,
    lap_pe_circ: torch.Tensor,
    lap_pe_disease: torch.Tensor,
    rwse_circ: torch.Tensor,
    rwse_disease: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    N_circ: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config,
    do_pretrain: bool = False,       # 保留兼容性, 始终忽略
    use_contrastive: bool = False,   # 保留兼容性, 始终忽略
    val_ratio: float = 0.2,          # NEW: fraction of train for validation
    # Side branch inputs
    kmer_feat: torch.Tensor = None,
    mesh_feat: torch.Tensor = None,
    adj: torch.Tensor = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    单阶段监督训练。

    在 train_mask 内部切出 val_ratio 的正样本作为验证集，
    用于 early stopping 和 checkpoint 选择。
    Test 集仅在训练结束后评估一次，不参与模型选择。
    """
    from data.dataset import train_val_split

    # ── Split train → train_inner + val ──
    train_inner, val_mask = train_val_split(train_mask, val_ratio=val_ratio)

    train_pos_circ, train_pos_dis = np.where(train_inner > 0)
    val_pos_circ, val_pos_dis = np.where(val_mask > 0)
    test_pos_circ, test_pos_dis = np.where(test_mask > 0)

    n_train = len(train_pos_circ)
    n_val = len(val_pos_circ)
    n_test = len(test_pos_circ)
    print(f"\n{'='*60}")
    print(f"Training: {n_train} pos | Validation: {n_val} pos | Test: {n_test} pos")
    print(f"{'='*60}")

    print(f"\n--- Supervised Training ---")
    history, final_metrics = train_supervised(
        model, x_circ, x_disease,
        lap_pe_circ, lap_pe_disease,
        rwse_circ, rwse_disease,
        edge_index, edge_weight, N_circ,
        (train_pos_circ, train_pos_dis),
        (test_pos_circ, test_pos_dis),
        train_inner, test_mask, config,
        val_pos=(val_pos_circ, val_pos_dis),
        val_mask=val_mask,
        kmer_feat=kmer_feat,
        mesh_feat=mesh_feat,
        adj=adj,
    )

    return model, final_metrics
