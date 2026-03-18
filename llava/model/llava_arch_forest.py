from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print

import os
import random
import numpy as np
from typing import Dict, Optional, Any
import json
   
def build_region_map(H, W, num_regions=10, mode="interleave", device="cpu"):
    if mode == "interleave":
        # 更推荐：每个区域分布在整张图上，而不是一整块
        rr = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        cc = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
        region_map = (3 * rr + 7 * cc) % num_regions
        return region_map.long()

    elif mode == "tile":
        # 按空间块划分，10个区域时大概是 2x5
        best_r, best_c = 1, num_regions
        best_gap = abs(best_c - best_r)
        for r in range(1, int(math.sqrt(num_regions)) + 1):
            if num_regions % r == 0:
                c = num_regions // r
                gap = abs(c - r)
                if gap < best_gap:
                    best_r, best_c, best_gap = r, c, gap

        row_sizes = [H // best_r + (i < H % best_r) for i in range(best_r)]
        col_sizes = [W // best_c + (j < W % best_c) for j in range(best_c)]

        region = torch.empty(H, W, dtype=torch.long, device=device)
        hs = 0
        rid = 0
        for rh in row_sizes:
            ws = 0
            for cw in col_sizes:
                region[hs:hs+rh, ws:ws+cw] = rid
                rid += 1
                ws += cw
            hs += rh
        return region

    else:
        raise ValueError(f"Unknown region mode: {mode}")

def collect_prune_stats(per_frame_keep_orig, H, W, num_regions=10, region_mode="interleave"):
    T = len(per_frame_keep_orig)
    device = per_frame_keep_orig[0].device if T > 0 else "cpu"
    N = H * W

    keep_masks = []
    per_frame_keep_indices = []
    per_frame_prune_indices = []
    keep_centroid_rc = []
    prune_centroid_rc = []

    for t in range(T):
        keep = torch.zeros(N, dtype=torch.bool, device=device)
        idx = per_frame_keep_orig[t].long()
        idx = idx[(idx >= 0) & (idx < N)].unique(sorted=True)
        keep[idx] = True

        prune_idx = (~keep).nonzero(as_tuple=False).squeeze(1)

        keep_masks.append(keep.view(H, W))
        per_frame_keep_indices.append(idx.detach().cpu().tolist())
        per_frame_prune_indices.append(prune_idx.detach().cpu().tolist())

        if idx.numel() > 0:
            kr = (idx // W).float()
            kc = (idx % W).float()
            keep_centroid_rc.append([kr.mean().item(), kc.mean().item()])
        else:
            keep_centroid_rc.append([None, None])

        if prune_idx.numel() > 0:
            pr = (prune_idx // W).float()
            pc = (prune_idx % W).float()
            prune_centroid_rc.append([pr.mean().item(), pc.mean().item()])
        else:
            prune_centroid_rc.append([None, None])

    keep_masks = torch.stack(keep_masks, dim=0)      # [T, H, W]
    prune_masks = ~keep_masks

    region_map = build_region_map(H, W, num_regions=num_regions, mode=region_mode, device=device)
    region_flat = region_map.reshape(-1)
    region_sizes = torch.bincount(region_flat, minlength=num_regions).float().cpu()

    keep_region_hist = []
    prune_region_hist = []

    for t in range(T):
        keep_idx = keep_masks[t].reshape(-1).nonzero(as_tuple=False).squeeze(1)
        prune_idx = prune_masks[t].reshape(-1).nonzero(as_tuple=False).squeeze(1)

        keep_region_hist.append(
            torch.bincount(region_flat[keep_idx], minlength=num_regions).cpu().tolist()
        )
        prune_region_hist.append(
            torch.bincount(region_flat[prune_idx], minlength=num_regions).cpu().tolist()
        )

    stats = {
        "T": T,
        "H": H,
        "W": W,
        "num_regions": num_regions,
        "region_mode": region_mode,
        "keep_rate_per_frame": keep_masks.float().mean(dim=(1, 2)).cpu().tolist(),
        "prune_rate_per_frame": prune_masks.float().mean(dim=(1, 2)).cpu().tolist(),
        "keep_heatmap": keep_masks.float().mean(dim=0).cpu().tolist(),
        "prune_heatmap": prune_masks.float().mean(dim=0).cpu().tolist(),
        "per_frame_keep_indices": per_frame_keep_indices,
        "per_frame_prune_indices": per_frame_prune_indices,
        "keep_centroid_rc": keep_centroid_rc,
        "prune_centroid_rc": prune_centroid_rc,
        "keep_region_hist": keep_region_hist,
        "prune_region_hist": prune_region_hist,
        "keep_region_ratio": (torch.tensor(keep_region_hist).float() / region_sizes.unsqueeze(0)).tolist(),
        "prune_region_ratio": (torch.tensor(prune_region_hist).float() / region_sizes.unsqueeze(0)).tolist(),
        "region_map": region_map.cpu().tolist(),
    }

    if T > 1:
        inter = (keep_masks[1:] & keep_masks[:-1]).float().sum(dim=(1, 2))
        union = (keep_masks[1:] | keep_masks[:-1]).float().sum(dim=(1, 2)).clamp(min=1)
        flip = (keep_masks[1:] ^ keep_masks[:-1]).float().mean(dim=(1, 2))

        stats["adjacent_keep_jaccard"] = (inter / union).cpu().tolist()
        stats["adjacent_flip_ratio"] = flip.cpu().tolist()

    return stats

@dataclass
class GraphPrunerConfig:
    temperature: float = 0.03
    similarity: float = 0.5
    iterate_num: int = 5
    top_m: Optional[int] = None
    diversify: bool = False
    lambda_div: float = 0.5
    use_feature_norm_weight: bool = True
    ensure_connectivity: bool = False
    k_intra: float = 0.5


class GraphPruner:

    def __init__(self, config: Optional[GraphPrunerConfig] = None, **kwargs: Any):
        self.config = config or GraphPrunerConfig()
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
                else:
                    raise ValueError(f"Unknown config field: {k}")
    
    def _idx_to_rc01(self, idx: torch.Tensor, H, W) -> torch.Tensor:
            r = (idx // W).float() / max(H - 1, 1)
            c = (idx %  W).float() / max(W - 1, 1)
            return torch.stack([r, c], dim=-1)

    def gprune(self, feats: torch.Tensor,remain_tokens_num=None) -> Dict[str, torch.Tensor]:
        N = feats.shape[0]
        k = int(round(N * self.config.k_intra))

        if feats.numel() == 0 or k == 0:
            return torch.empty(0, dtype=torch.long, device=feats.device)
        if k > N:
            k = N
        if remain_tokens_num:
            if remain_tokens_num > N:
                k = N
            else:
                k = remain_tokens_num
        normed_feats = F.normalize(feats, p=2, dim=1)
        A = normed_feats @ normed_feats.t()
        A[A < self.config.similarity] = 0
        A = A / self.config.temperature
        A[torch.abs(A) < 1e-6] = -5000
        A = F.softmax(A, dim=1)
        degree = (A > 0).sum(dim=1).float()
        A_n = torch.matrix_power(A, self.config.iterate_num)
        W = feats.norm(dim=1, keepdim=True).t()
        scores = (W @ A_n).squeeze(0) / degree
        scores[torch.isnan(scores)] = 0
        topk_indices = torch.topk(scores, k, sorted=False).indices
        return {"indices": topk_indices, "scores": scores}

    def prune_temporal_forest_spatial(
        self,
        video_feats,
        *,
        per_frame_orig_idx,
        grid_hw=(14, 14),
        keep_ratio: float = 0.1,
        tau: float = 0.9,
        tau_spatial: float = 0.8,
        normalize: bool = True,
    ):
        if isinstance(video_feats, torch.Tensor) and video_feats.ndim == 3:
            T = video_feats.shape[0]
            frames = [video_feats[t] for t in range(T)]
        elif isinstance(video_feats, (list, tuple)):
            frames = list(video_feats)
            T = len(frames)
        else:
            raise ValueError("video_feats should be List[Tensor] or Tensor[T, N, D]")

        device = frames[0].device
        dtype = frames[0].dtype
        N_list = [f.shape[0] for f in frames]
        total  = int(sum(N_list))
        if total == 0:
            return dict(
                keep_idx_global=torch.empty(0, dtype=torch.long, device=device),
                per_frame_keep_local=[torch.empty(0, dtype=torch.long, device=device) for _ in range(T)],
                per_frame_keep_orig=[torch.empty(0, dtype=torch.long, device=device) for _ in range(T)],
                depth=torch.zeros(0, dtype=torch.long, device=device),
                parent=torch.zeros(0, dtype=torch.long, device=device),
                parent_sim=torch.zeros(0, dtype=torch.float32, device=device),
                frame_id=torch.zeros(0, dtype=torch.long, device=device),
            )

        frames_n = [F.normalize(f, dim=1) if (normalize and f.numel() > 0) else f for f in frames]
        offset = torch.zeros(T + 1, dtype=torch.long, device=device)
        offset[1:] = torch.cumsum(torch.tensor(N_list, dtype=torch.long, device=device), dim=0)


        H, W = grid_hw
        coords_per_frame = [self._idx_to_rc01(per_frame_orig_idx[t].long(), H, W) for t in range(T)]
        psim_all      = torch.zeros(total, dtype=torch.float32, device=device)
        depth_all     = torch.full((total,), -1, dtype=torch.long, device=device)
        parent_all    = torch.full((total,), -1, dtype=torch.long, device=device)
        local_idx_all = torch.full((total,), -1, dtype=torch.long, device=device)
        orig_idx_all  = torch.full((total,), -1, dtype=torch.long, device=device)
        frame_id_all  = torch.full((total,), -1, dtype=torch.long, device=device)
        hist_feats    = torch.empty(0, frames_n[0].shape[1], device=device, dtype=dtype)  # (M, D)
        hist_coords   = torch.empty(0, 2, device=device)                     # (M, 2)
        hist_tree_id  = torch.empty(0, dtype=torch.long, device=device)      # (M,)
        tree_tail_gid = []
        tree_count    = 0
        tree_id_all   = torch.full((total,), -1, dtype=torch.long, device=device)

        g = 0
        d_thresh = 1.0 - float(tau_spatial)
        Nt0 = N_list[0]
        if Nt0 > 0:
            s_gid = g
            e_gid = g + Nt0
            depth_all[s_gid:e_gid]     = 0
            parent_all[s_gid:e_gid]    = -1
            psim_all[s_gid:e_gid]      = 1.0
            frame_id_all[s_gid:e_gid]  = 0
            local_idx_all[s_gid:e_gid] = torch.arange(Nt0, device=device, dtype=torch.long)
            orig_idx_all[s_gid:e_gid]  = per_frame_orig_idx[0].long()

            new_tids = torch.arange(tree_count, tree_count + Nt0, device=device, dtype=torch.long)
            tree_count += Nt0
            tree_id_all[s_gid:e_gid] = new_tids
            tree_tail_gid.extend(torch.arange(s_gid, e_gid, device=device, dtype=torch.long).tolist())

            hist_feats   = torch.cat([hist_feats, frames_n[0]], dim=0)
            hist_coords  = torch.cat([hist_coords, coords_per_frame[0]], dim=0)
            hist_tree_id = torch.cat([hist_tree_id, new_tids], dim=0)
            g = e_gid

        for t in range(1, T):
            Nt = N_list[t]
            if Nt == 0:
                continue

            Ft = frames_n[t]                    # [Nt, D]
            Ct = coords_per_frame[t]            # [Nt, 2]

            cutoff = hist_feats.shape[0]

            S = torch.matmul(hist_feats[:cutoff], Ft.T)                           # [M, Nt]
            D = torch.amax((hist_coords[:cutoff, None, :] - Ct[None, :, :]).abs(), dim=2)  # [M, Nt]
            Feas = (S >= tau) & (D <= d_thresh)                                   # [M, Nt]

            idx_tree_1d = hist_tree_id[:cutoff].long()                            # [M]
            idx_tree_2d = idx_tree_1d[:, None].expand(-1, Nt)                     # [M, Nt]
            num_trees   = tree_count
            NEG_INF = torch.tensor(float('-inf'), dtype=S.dtype, device=device)
            POS_INF = torch.tensor(float('inf'),  dtype=D.dtype, device=device)

            s_src   = torch.where(Feas, S, NEG_INF)                               # [M, Nt]
            s_tree  = torch.full((num_trees, Nt), NEG_INF, dtype=S.dtype, device=device)
            s_tree.scatter_reduce_(0, idx_tree_2d, s_src, reduce='amax', include_self=False)  # [Ttrees, Nt]

            s_tree_gather = s_tree.index_select(0, idx_tree_1d)                   # [M, Nt]
            near_max = Feas & (S == s_tree_gather)                                # [M, Nt]
            d_src    = torch.where(near_max, D, POS_INF)
            d_tree   = torch.full((num_trees, Nt), POS_INF, dtype=D.dtype, device=device)
            d_tree.scatter_reduce_(0, idx_tree_2d, d_src, reduce='amin', include_self=False)  # [Ttrees, Nt]

            s_max_tok, _ = torch.max(s_tree, dim=0)                               # [Nt]
            has_match = torch.isfinite(s_max_tok)                                  # [Nt]
            cand_mask = (s_tree == s_max_tok[None, :])                             # [Ttrees, Nt]
            d_cand    = torch.where(cand_mask, d_tree, POS_INF)                    # [Ttrees, Nt]
            tid_best  = torch.argmin(d_cand, dim=0)                                # [Nt]

            old_tails = torch.as_tensor(tree_tail_gid, device=device, dtype=torch.long)  # [Ttrees]
            parent_gid = torch.full((Nt,), -1, dtype=torch.long, device=device)
            parent_gid[has_match] = old_tails.index_select(0, tid_best[has_match])

            s_gid = g
            e_gid = g + Nt
            g_ids = torch.arange(s_gid, e_gid, device=device, dtype=torch.long)    # [Nt]

            depth_frame = torch.zeros(Nt, dtype=torch.long, device=device)
            if has_match.any():
                depth_frame[has_match] = depth_all.index_select(0, parent_gid[has_match]) + 1

            depth_all[s_gid:e_gid]    = depth_frame
            parent_all[s_gid:e_gid]   = parent_gid
            frame_id_all[s_gid:e_gid] = t
            local_idx_all[s_gid:e_gid]= torch.arange(Nt, device=device, dtype=torch.long)
            orig_idx_all[s_gid:e_gid] = per_frame_orig_idx[t].long()

            ps = torch.zeros(Nt, dtype=psim_all.dtype, device=device)
            if has_match.any():
                P = hist_feats.index_select(0, parent_gid[has_match])              # [K, D]
                ps_val = (P * Ft.index_select(0, has_match.nonzero(as_tuple=True)[0])).sum(dim=1)
                ps[has_match] = ps_val.to(ps.dtype)
            else:
                ps[:] = 1.0
            psim_all[s_gid:e_gid] = ps

            curr_tree_ids = torch.empty(Nt, dtype=torch.long, device=device)
            curr_tree_ids[has_match] = tid_best[has_match]
            root_mask = ~has_match
            n_new = int(root_mask.sum().item())
            if n_new > 0:
                new_ids = torch.arange(tree_count, tree_count + n_new, device=device, dtype=torch.long)
                curr_tree_ids[root_mask] = new_ids
                tree_count += n_new

            tree_id_all[s_gid:e_gid] = curr_tree_ids

            hist_feats   = torch.cat([hist_feats, Ft], dim=0)
            hist_coords  = torch.cat([hist_coords, Ct], dim=0)
            hist_tree_id = torch.cat([hist_tree_id, curr_tree_ids], dim=0)

            last_gid_exist = torch.full((num_trees,), -1, dtype=torch.long, device=device)
            if has_match.any():
                last_gid_exist.scatter_reduce_(
                    0,
                    curr_tree_ids[has_match],
                    g_ids[has_match],
                    reduce='amax',
                    include_self=False
                )
            tail_tensor = torch.as_tensor(tree_tail_gid, device=device, dtype=torch.long)
            upd_mask = last_gid_exist >= 0
            if upd_mask.any():
                tail_tensor[upd_mask] = last_gid_exist[upd_mask]
            tree_tail_gid = tail_tensor.tolist()
            if n_new > 0:
                tree_tail_gid.extend(g_ids[root_mask].tolist())

            g = e_gid

        K = int(round(total * float(keep_ratio) / self.config.k_intra))
        print(total, keep_ratio, self.config.k_intra)
        root_idx = torch.nonzero(parent_all < 0, as_tuple=False).squeeze(1)   # [R]
        n_root = root_idx.numel()

        if n_root > K:
            Xn = torch.cat(frames_n, dim=0)          # [total, D]
            Xr = Xn[root_idx]                        # [R, D]
            R  = n_root
            sims_rr = Xr @ Xr.t()                    # [R, R]
            sims_rr.fill_diagonal_(float('-inf'))
            ii = torch.arange(R, device=device).unsqueeze(1)
            jj = torch.arange(R, device=device).unsqueeze(0)
            forward_mask = (root_idx[ii] < root_idx[jj])
            sims_rr = sims_rr.masked_fill(~forward_mask, float('-inf'))
            remain = torch.ones(R, dtype=torch.bool, device=device)
            merges_needed = n_root - K

            for _ in range(merges_needed):
                mask_rows = remain.view(R,1).expand(R,R)
                mask_cols = remain.view(1,R).expand(R,R)
                M = sims_rr.masked_fill(~(mask_rows & mask_cols), float('-inf'))
                flat = M.view(-1)
                flat_idx = torch.argmax(flat)
                if flat[flat_idx] == float('-inf'):
                    break 

                parent_pos = int(flat_idx // R)
                child_pos  = int(flat_idx %  R)
                parent_g = int(root_idx[parent_pos].item())
                child_g  = int(root_idx[child_pos].item())
                parent_all[child_g] = parent_g
                remain[child_pos] = False

            depth_all.zero_()
            for g in range(total):
                p = int(parent_all[g].item())
                if p >= 0:
                    depth_all[g] = depth_all[p] + 1

        valid_mask = (frame_id_all >= 0) & (local_idx_all >= 0)
        num_valid  = int(valid_mask.sum().item())
        if num_valid == 0:
            print("[Forest][fatal] no valid slots; returning empty selection.", flush=True)
            K = 0
            keep_global = torch.empty(0, dtype=torch.long, device=device)
        else:
            K = min(K, num_valid)

            key = depth_all.float() + frame_id_all.float() / 1e3
            key = torch.where(valid_mask, key, torch.tensor(float('inf'), device=key.device))
            key_safe = torch.nan_to_num(key, nan=1e9, posinf=1e9, neginf=-1e9)

            keep_global = torch.topk(-key_safe, k=K, largest=True).indices

        per_frame_keep_local, per_frame_keep_orig = [], []
        for t in range(T):
            s, e = offset[t].item(), offset[t+1].item()
            g_idx_t = keep_global[(keep_global >= s) & (keep_global < e)]
            if g_idx_t.numel() == 0:
                per_frame_keep_local.append(torch.empty(0, dtype=torch.long, device=device))
                per_frame_keep_orig.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            loc = local_idx_all.index_select(0, g_idx_t)       # 0..Nt-1
            Nt = e - s
            good = (loc >= 0) & (loc < Nt)
            if (~good).any():
                drop = int((~good).sum().item())
                print(f"[Forest][fix] frame {t}: drop {drop} invalid local_idx before mapping.", flush=True)
                loc = loc[good]
                g_idx_t = g_idx_t[good]
            per_frame_keep_local.append(loc)

            idx_orig = per_frame_orig_idx[t].to(device=device, dtype=torch.long).index_select(0, loc)
            per_frame_keep_orig.append(idx_orig)
        
        kept_per_frame = [len(x) for x in per_frame_keep_local]
        print("per-frame kept:", kept_per_frame, "kept token nums:", sum(kept_per_frame))

        depth_cpu = depth_all.detach().to('cpu')
        if depth_cpu.dtype != torch.long:
            depth_cpu = depth_cpu.long()

        neg = int((depth_cpu < 0).sum().item())
        if neg > 0:
            print(f"[Forest][warn] depth_all has {neg} negatives (invalid slots) — excluding from histogram.", flush=True)

        depth_valid = depth_cpu[depth_cpu >= 0]
        depth_hist = torch.bincount(depth_valid).tolist()
        print(f"[Forest] depth histogram (ALL, >=0): {depth_hist}", flush=True)

        return dict(
            keep_idx_global=keep_global,
            per_frame_keep_local=per_frame_keep_local,
            per_frame_keep_orig=per_frame_keep_orig,
            depth=depth_all,
            parent=parent_all,
            parent_sim=psim_all,
            frame_id=frame_id_all,
            offset=offset,
        )
    
    def prune_region_cycle(
        self,
        video_feats,
        *,
        grid_hw=(14, 14),
        num_regions=10,
        keep_regions_per_frame=1,
        region_mode="interleave",
        shuffle_schedule=False,
        seed=0,
    ):
        if isinstance(video_feats, torch.Tensor) and video_feats.ndim == 3:
            frames = [video_feats[t] for t in range(video_feats.shape[0])]
        elif isinstance(video_feats, (list, tuple)):
            frames = list(video_feats)
        else:
            raise ValueError("video_feats should be List[Tensor] or Tensor[T, N, D]")

        T = len(frames)
        H, W = grid_hw
        N = H * W
        device = frames[0].device

        for t in range(T):
            assert frames[t].shape[0] == N, f"frame {t} token num = {frames[t].shape[0]}, expected {N}"

        region_map = build_region_map(H, W, num_regions=num_regions, mode=region_mode, device=device)
        region_flat = region_map.reshape(-1)

        schedule = torch.arange(num_regions, device=device)
        if shuffle_schedule:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            perm = torch.randperm(num_regions, generator=g, device=device)
            schedule = schedule[perm]

        per_frame_keep_orig = []
        kept_feats = []

        for t in range(T):
            start = (t * keep_regions_per_frame) % num_regions
            active_regions = [schedule[(start + j) % num_regions] for j in range(keep_regions_per_frame)]
            active_regions = torch.stack(active_regions, dim=0)

            keep_mask = (region_flat[:, None] == active_regions[None, :]).any(dim=1)
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

            per_frame_keep_orig.append(keep_idx)
            kept_feats.append(frames[t].index_select(0, keep_idx))

        return {
            "per_frame_keep_orig": per_frame_keep_orig,
            "kept_feats": kept_feats,
            "region_map": region_map,
            "schedule": schedule,
        }

    def prune_pixel_diff(
        self,
        video_pixels,
        *,
        grid_hw=(14, 14),
        keep_ratio: float = 0.1,
        first_keep_ratio: Optional[float] = None,
        diff_thresh_q: float = 0.7,
        seed: int = 0,
    ):
        if not isinstance(video_pixels, torch.Tensor) or video_pixels.ndim != 4:
            raise ValueError("video_pixels should be a Tensor of shape [T, C, H, W]")

        x = video_pixels.float()
        if x.max() > 1.5:
            x = x / 255.0
        
        T, C, H_img, W_img = x.shape
        H, W = grid_hw
        N = H * W
        device = x.device

        if first_keep_ratio is None:
            first_keep_ratio = min(1.0, keep_ratio * 2.0)
        
        keep_num = max(1, int(round(N * keep_ratio)))
        first_keep_num = max(1, int(round(N * first_keep_ratio)))

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        per_frame_keep_orig = []
        diff_maps = []

        # 第一帧多保留一点
        prob0 = torch.ones(N, device=device)
        idx0 = torch.multinomial(prob0, first_keep_num, replacement=False, generator=g)
        idx0 = idx0.sort().values

        prev_keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
        prev_keep_mask[idx0] = True

        per_frame_keep_orig.append(idx0)
        diff_maps.append(torch.zeros(H, W, dtype=torch.float32, device=device))

        # 对之后的帧做帧差
        for t in range(1, T):
            diff = (x[t] - x[t - 1]).abs().mean(dim=0, keepdim=True)
            diff_grid = F.adaptive_avg_pool2d(diff, (H, W)).view(-1)
            diff_maps.append(diff_grid.view(H, W))

            # 如果差值全是 0
            if diff_grid.sum() <= 1e-8:
                diff_grid = torch.ones_like(diff_grid)
            
            # 用分位数设置阈值，取高差值区域作为候选
            thr = torch.quantile(diff_grid, diff_thresh_q)
            cand_mask = diff_grid >= thr

            # 首先强制当前帧和上一帧不同
            cand_mask = cand_mask & (~prev_keep_mask)
            cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(1)

            # 从候选里按照差值大小采样
            if cand_idx.numel() >= keep_num:
                prob = diff_grid[cand_idx]
                if prob.sum() <= 1e-8:
                    prob = torch.ones_like(prob)
                prob = prob / prob.sum()

                choose = torch.multinomial(prob, keep_num, replacement=False, generator=g)
                cur_idx = cand_idx[choose]
            else:
                # 候选不足时，从非上一帧位置里补足， 仍按照差值大小采样
                selected = []
                if cand_idx.numel() > 0:
                    selected.append(cand_idx)
                
                remain = keep_num - cand_idx.numel()
                rest_mask = (~prev_keep_mask).clone()
                if cand_idx.numel() > 0:
                    rest_mask[cand_idx] = False
                
                rest_idx = rest_mask.nonzero(as_tuple=False).squeeze(1)

                if remain > 0 and rest_idx.numel() > 0:
                    if rest_idx.numel() <= remain:
                        selected.append(rest_idx)
                    else:
                        prob = diff_grid[rest_idx]
                        if prob.sum() <= 1e-8:
                            prob = torch.ones_like(prob)
                        prob = prob / prob.sum()

                        choose = torch.multinomial(prob, remain, replacement=False, generator=g)
                        selected.append(rest_idx[choose])
                
                cur_idx = torch.cat(selected, dim=0) if len(selected) > 0 else torch.empty(0, dtype=torch.long, device=device)
            
            cur_idx = cur_idx.unique(sorted=True)
            if cur_idx.numel() > keep_num:
                cur_idx = cur_idx[:keep_num]
            
            # 更新上一帧 mask
            prev_keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
            prev_keep_mask[cur_idx] = True

            per_frame_keep_orig.append(cur_idx)
        
        return {
            "per_frame_keep_orig": per_frame_keep_orig,
            "diff_maps": torch.stack(diff_maps, dim=0)
        }

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images) # image_features.shape = torch.Size([16, 729, 1152])
        image_features = self.get_model().mm_projector(image_features) # [16, 729, 3584]
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature, per_frame_indices=None, H=14, W=14):
        from torch.nn.utils.rnn import pad_sequence
        device = image_feature.device
        add_faster_video = getattr(self.config, "add_faster_video", False)

        newline = self.model.image_newline.to(device)
        if newline.dim() == 1:
            newline = newline.unsqueeze(0)   # (1, C)
        elif newline.dim() == 2 and newline.size(0) == 1:
            pass
        else:
            newline = newline.reshape(1, -1)
        feature_dim = image_feature.size(-1)

        if per_frame_indices is not None:
            if isinstance(per_frame_indices, torch.Tensor):
                flat_idx = per_frame_indices.to(device).long()
                grid_size = H * W
                if flat_idx.numel() > 0:
                    frame_ids = torch.div(flat_idx, grid_size, rounding_mode='floor')
                    spatial_idx = flat_idx % grid_size
                    F = int(frame_ids.max().item()) + 1
                    lengths_tensor = torch.bincount(frame_ids, minlength=F)
                    lengths = lengths_tensor.tolist()
                else:
                    F, lengths, spatial_idx = 0, [], flat_idx

                per_frame_indices_list, frame_chunks = [], []
                start = 0
                for k in lengths:
                    end = start + k
                    frame_chunks.append(image_feature[start:end])          # (K_i, C)
                    per_frame_indices_list.append(spatial_idx[start:end])  # (K_i,)
                    start = end
                assert start == image_feature.size(0), \
                    f"{start} is not same as image_feature.size(0)={image_feature.size(0)}"

            elif isinstance(per_frame_indices, (list, tuple)):
                F = len(per_frame_indices)
                lengths = [int(len(idx)) for idx in per_frame_indices]
                assert sum(lengths) == image_feature.size(0), \
                    f"sum(per_frame_lengths)={sum(lengths)}, but image_feature.size(0)={image_feature.size(0)}"
                frame_chunks, per_frame_indices_list = [], []
                start = 0
                for idx in per_frame_indices:
                    k = int(len(idx))
                    end = start + k
                    frame_chunks.append(image_feature[start:end])
                    per_frame_indices_list.append(idx.to(device).long())
                    start = end
            else:
                raise TypeError(f"per_frame_indices has a wrong shapr: {per_frame_indices.shape}")

            Ni = H * W
            per_frame_out = []
            for t, (chunk, idx) in enumerate(zip(frame_chunks, per_frame_indices_list)):
                if idx.numel() > 0:
                    valid = (idx >= 0) & (idx < Ni)
                    if not valid.all():
                        drop = int((~valid).sum().item())
                        print(f"[Grid][warn] frame {t}: drop {drop} invalid orig_idx (Ni={Ni})", flush=True)
                        idx = idx[valid]
                        if idx.numel() != chunk.shape[0]:
                            m = min(idx.numel(), chunk.shape[0])
                            print(f"[Grid][warn] frame {t}: trim after drop -> {m}", flush=True)
                            idx = idx[:m]
                            chunk = chunk[:m]

                    order = torch.argsort(idx)
                    idx = idx.index_select(0, order)
                    chunk = chunk.index_select(0, order)

                    row_ids = torch.div(idx, W, rounding_mode='floor')
                    legal_row = (row_ids >= 0) & (row_ids < H)
                    if not legal_row.all():
                        drop = int((~legal_row).sum().item())
                        print(f"[Grid][warn] frame {t}: drop {drop} invalid row ids (H={H})", flush=True)
                        row_ids = row_ids[legal_row]
                        chunk   = chunk[legal_row]

                    rows = [[] for _ in range(H)]
                    for tok, r in zip(chunk, row_ids.tolist()):
                        rows[r].append(tok)

                    parts = []
                    for r in range(H):
                        if len(rows[r]) > 0:
                            parts.append(torch.stack(rows[r], dim=0))        # (N_r, C)
                        parts.append(newline.expand(1, feature_dim))         # (1, C)
                    frame_seq = torch.cat(parts, dim=0)                       # (K_i + H, C)
                else:
                    frame_seq = newline.expand(H, feature_dim)

                per_frame_out.append(frame_seq)

            if add_faster_video:
                padded = pad_sequence(per_frame_out, batch_first=True)       # (F, L_max, C)
                lens = torch.tensor([x.size(0) for x in per_frame_out], device=device)
                L_max = padded.size(1)
                arange = torch.arange(L_max, device=device).unsqueeze(0)
                self.attention_mask = (arange < lens.unsqueeze(1))           # (F, L_max) bool
                return padded
            else:
                return torch.cat(per_frame_out, dim=0)

        num_frames = image_feature.shape[0]
        tokens_per_frame = image_feature.shape[1]

        image_feature = image_feature.view(num_frames, 1, H, W, feature_dim)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)

        image_feature = torch.cat(
            (
                image_feature,
                newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(device)
            ),
            dim=-1
        )

        if add_faster_video:
            image_feature = image_feature.view(feature_dim, num_frames, H, -1)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            image_feature = image_feature.flatten(1, 2)
            return image_feature

        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)
            encoded_image_features = torch.split(encoded_image_features, split_sizes) # type(encoded_image_features) == tuple
            image_features = []
            kept_token_index = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            try:
                method = os.environ.get("METHOD", "baseline")
                keep_ratio = float(os.environ.get("KEEP_RATIO", 0.1))

                tau = float(os.environ.get("TAU", 0.9))
                tau_spatial = float(os.environ.get("TAU_SPATIAL", 0.8))
                gprune_ratio = float(os.environ.get("GPRUNE_RATIO", 0.5))

                first_keep_ratio = float(os.environ.get("FIRST_KEEP_RATIO", keep_ratio * 2.0))
                diff_thresh_q = float(os.environ.get("DIFF_THRESH_Q", 0.7))
            except:
                pass

            if method != "baseline":
                video_pos = video_idx_in_batch[0]
                video_feat = image_features[video_pos]

                T = video_feat.shape[0]
                pruned_image_features = []
                pruner = GraphPruner()
                H = W = int(math.sqrt(video_feat.shape[1]))

                pruner = GraphPruner(k_intra=gprune_ratio)
                
                if method == "region_cycle":
                    num_regions = int(os.environ.get("NUM_REGIONS", 10))
                    keep_regions_per_frame = int(os.environ.get("KEEP_REGIONS_PER_FRAME", 1))
                    region_mode = os.environ.get("REGION_MODE", "interleave")

                    out = pruner.prune_region_cycle(
                        video_feats=image_features[0],   # 直接对原始每帧 token 做
                        grid_hw=(H, W),
                        num_regions=num_regions,
                        keep_regions_per_frame=keep_regions_per_frame,
                        region_mode=region_mode,
                        shuffle_schedule=True,
                        seed=0,
                    )

                    kept = out["kept_feats"]
                    pruned_image_features = torch.cat(kept, dim=0)
                    keep_indices = out["per_frame_keep_orig"]

                elif method == "pixel_diff":
                    video_pixels = images_list[video_pos].to(device=video_feat.device)

                    out = pruner.prune_pixel_diff(
                        video_pixels=video_pixels,
                        grid_hw=(H, W),
                        keep_ratio=keep_ratio,
                        first_keep_ratio=first_keep_ratio,
                        diff_thresh_q=diff_thresh_q,
                        seed=0,
                    )

                    keep_indices = out["per_frame_keep_orig"]
                    kept = [video_feat[i].index_select(0, keep_indices[i]) for i in range(T)]
                    pruned_image_features = torch.cat(kept, dim=0)

                    print(f"[PixelDiff] first_keep_ratio={first_keep_ratio}, keep_ratio={keep_ratio}, diff_thresh_q={diff_thresh_q}", flush=True)
                    print(f"[PixelDiff] per-frame kept: {[len(x) for x in keep_indices]}", flush=True)

                else:
                    frames = []
                    orig_idx_list = []
                    for i in range(T):
                        feats = image_features[0][i]
                        out = pruner.gprune(feats)
                        frames.append(feats[out["indices"]])
                        orig_idx_list.append(out["indices"].long())

                    out = pruner.prune_temporal_forest_spatial(
                        video_feats=frames,
                        per_frame_orig_idx=orig_idx_list,
                        grid_hw=(H, W),
                        keep_ratio=keep_ratio,
                        tau=tau,
                        tau_spatial=tau_spatial,
                        normalize=True,
                    )

                    kept = [image_features[0][i].index_select(0, out["per_frame_keep_orig"][i]) for i in range(T)]
                    pruned_image_features = torch.cat(kept, dim=0)
                    keep_indices = out["per_frame_keep_orig"]
                    stats_path = os.environ.get("PRUNE_STATS_PATH", "")
                    if stats_path:
                        stats = collect_prune_stats(
                            keep_indices,
                            H,
                            W,
                            num_regions=int(os.environ.get("NUM_REGIONS", 10)),
                            region_mode=os.environ.get("REGION_MODE", "tile"),
                        )
                        with open(stats_path, "w") as f:
                            json.dump(stats, f, indent=2)
                        print(f"[PruneStats] saved to {stats_path}", flush=True)
                        print(f"[PruneStats] keep_rate_per_frame: {stats['keep_rate_per_frame']}", flush=True)
                        if "adjacent_keep_jaccard" in stats:
                            print(f"[PruneStats] adjacent_keep_jaccard: {stats['adjacent_keep_jaccard']}", flush=True)

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_idx in video_idx_in_batch:
                        if mm_newline_position == "grid":
                            # Grid-wise
                            if method == "baseline":
                                pruned_image_features = image_features[0].reshape(-1, image_features[0].shape[-1])
                                keep_indices = torch.arange(len(pruned_image_features))
                                H = W = int(math.sqrt(image_features[0].shape[1]))
                                image_feature = self.add_token_per_grid(pruned_image_features, keep_indices, H=H, W=W)
                            else:
                                image_feature = self.add_token_per_grid(pruned_image_features, keep_indices, H=H, W=W)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                concat_slow_fater_token = []
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                image_feature = torch.cat(concat_slow_fater_token)
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)
                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features

            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            if isinstance(image_features,list):
                print(image_features[0].shape)
            else:
                print(image_features.shape)
        else:
            image_features = self.encode_images(images)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False