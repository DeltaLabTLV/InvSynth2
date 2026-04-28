"""Normalized Temperature-scaled Cross-Entropy (NT-Xent / InfoNCE) loss.

Used by the Transformer encoder during SSL pre-training (Eq. 1 in the paper).

For each masked position i, we compute cosine similarity between the encoder's
output z_i and a candidate set C_i = {q_i} ∪ N_i, where:
    q_i — positive target: same time step, computed on an UNMASKED view of the
           spectrogram, with stop-gradient.
    N_i — negative distractors: drawn from unmasked time steps in the same
           spectrogram (within-clip negatives, as motivated in §3.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z_masked: torch.Tensor,
    q_pos: torch.Tensor,
    q_negs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent loss for one batch of masked positions.

    Parameters
    ----------
    z_masked: (M, D) encoder output at masked positions
    q_pos:    (M, D) positive targets (already stop-gradient'd by caller)
    q_negs:   (M, K, D) negative distractors per masked position
    temperature: τ in Eq. (1).

    Returns
    -------
    Scalar loss tensor.
    """
    if z_masked.numel() == 0:
        # No masked positions in the batch — return 0 with grad.
        return z_masked.sum()

    z_masked = F.normalize(z_masked, dim=-1)
    q_pos = F.normalize(q_pos, dim=-1)
    q_negs = F.normalize(q_negs, dim=-1)

    # Cosine similarities
    pos_sim = (z_masked * q_pos).sum(dim=-1, keepdim=True)        # (M, 1)
    neg_sim = torch.einsum("md,mkd->mk", z_masked, q_negs)        # (M, K)

    # Stack [pos | negs] into one logit matrix; the positive is index 0.
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature  # (M, 1+K)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def sample_negatives_within_spectrogram(
    embeddings: torch.Tensor,            # (B, T, D) full encoder outputs on unmasked view
    masked_positions: torch.Tensor,      # (B, T) bool — True at masked time steps
    n_negatives: int = 50,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather (z_masked, q_pos, q_negs) tuples from a batch.

    z_masked and q_pos differ only in that q_pos comes from an UNMASKED forward pass.
    We pass the unmasked-view embeddings as `embeddings` and select positives/
    negatives from there — both for q_pos at masked positions and for negatives
    sampled from unmasked positions in the same row.
    """
    B, T, D = embeddings.shape
    device = embeddings.device

    # Flat list of (b, t) for masked positions.
    flat_mask = masked_positions.flatten()        # (B*T,)
    masked_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)

    if masked_idx.numel() == 0:
        empty = embeddings.new_empty(0, D)
        empty_neg = embeddings.new_empty(0, n_negatives, D)
        return empty, empty, empty_neg

    bs = masked_idx // T
    ts = masked_idx % T

    # Positive targets: same (b, t) on the unmasked view.
    q_pos = embeddings[bs, ts]  # (M, D)
    # Stop-gradient on the target branch.
    q_pos = q_pos.detach()

    # For each masked (b, t), draw `n_negatives` time steps from unmasked positions in row b.
    unmasked_per_batch = (~masked_positions)  # (B, T)
    q_negs_list = []
    for i in range(masked_idx.numel()):
        b = bs[i].item()
        unmasked_t = unmasked_per_batch[b].nonzero(as_tuple=False).squeeze(-1)
        if unmasked_t.numel() == 0:
            # Degenerate edge case: every frame masked. Use full row.
            unmasked_t = torch.arange(T, device=device)
        if unmasked_t.numel() >= n_negatives:
            choose = unmasked_t[
                torch.randperm(unmasked_t.numel(), device=device, generator=rng)[:n_negatives]
            ]
        else:
            # Sample with replacement to fill K slots.
            choose = unmasked_t[
                torch.randint(0, unmasked_t.numel(), (n_negatives,), device=device, generator=rng)
            ]
        q_negs_list.append(embeddings[b, choose])  # (K, D)
    q_negs = torch.stack(q_negs_list, dim=0).detach()  # (M, K, D)

    z_at_masked = embeddings[bs, ts]  # (M, D) — but we want masked-VIEW outputs here.
    # NOTE: caller passes the masked-view embeddings via z_masked separately;
    # this helper just returns positives + negatives sampled from the unmasked view.
    return z_at_masked, q_pos, q_negs
