import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from typing import List


from typing import List, Optional, Tuple
from transformers.cache_utils import Cache
import run_longbench as run
import global_vars



def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M 

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected] 
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')
        
    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent

class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        sample_id = global_vars.current_sample_id
        segment_boundaries = global_vars.global_segment_boundaries.get(sample_id, None)

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        max_capacity_prompt = max_capacity_prompt - self.window_size

        print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices1 = indices.unsqueeze(-1)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            return key_states, value_states

class SemBlockCluster(): 
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64,
                 kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80,
                 layer_idx=None, merge=None, threshold: float = 0.85):
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.beta = beta
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.merge = merge
        self.threshold = float(threshold)

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def compute_segment_scores(self, key_states, query_states, segment_boundaries, eta=1.0, theta_omega=0.5, theta_L=40):
        bsz, num_heads, q_len, head_dim = query_states.shape
        K = key_states.size(2)

        q_last = query_states[:, :, -1:, :].float()
        k_all  = key_states.float()
        attn_logits = torch.matmul(q_last, k_all.transpose(2, 3)) / math.sqrt(head_dim)

        attn = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
        attn = attn.squeeze(2).mean(dim=0)                         

        result = []
        for start, end in segment_boundaries:
            if end > K or end <= start:
                continue

            L_k = end - start
            attn_seg = attn[:, start:end]  # [H, L_k], fp32

            I_k = attn_seg.sum(dim=0).mean().item()

            p = attn_seg 
            logp = torch.zeros_like(p)
            pos = p > 0
            logp[pos] = torch.log(p[pos])
            entropy = (-p[pos] * logp[pos]).sum(dim=0).mean()
            D_k = entropy.item()

            omega_k = I_k * (1 + eta * D_k)
            retain = (omega_k > theta_omega) and (L_k < theta_L)

            result.append({
                "start": start, "end": end, "L_k": L_k, "I_k": I_k, "D_k": D_k, "omega_k": omega_k, "retain": retain})
        return result

    def _clip_segments_to_past(self, segments, past_len):
        clipped = []
        for s, e in segments or []:
            s = max(0, min(int(s), past_len))
            e = max(0, min(int(e), past_len))
            if e > s:
                clipped.append((s, e))
        return clipped

    @torch.no_grad()
    def _adaptive_segment_indices_per_head_vec(self, token_scores, capacity_total, segment_boundaries, eps=1e-12):
        B, H, T = token_scores.shape
        device = token_scores.device
        dtl = torch.long
        K_total = min(int(capacity_total), T)

        if not segment_boundaries:
            return token_scores.topk(K_total, dim=-1).indices

        segs = self._clip_segments_to_past(segment_boundaries, T)
        if not segs:
            return token_scores.topk(K_total, dim=-1).indices

        S = len(segs)
        seg_starts = torch.tensor([s for s, _ in segs], device=device, dtype=dtl)
        seg_ends   = torch.tensor([e for _, e in segs], device=device, dtype=dtl)
        seg_lens   = (seg_ends - seg_starts).clamp_min(0)
        if int(seg_lens.max().item()) == 0:
            return token_scores.topk(K_total, dim=-1).indices

        out_idx = torch.empty((B, H, K_total), dtype=dtl, device=device)

        unique_L = torch.unique(seg_lens)
        groups_L = {}
        for L in unique_L.tolist():
            if L <= 0:
                continue
            m = (seg_lens == L)
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            groups_L[L] = {
                "idx": idx,
                "starts": seg_starts[idx],
                "lens": seg_lens[idx]
            }

        for b in range(B):
            s_all = token_scores[b]

            gh_head = s_all.topk(K_total, dim=-1).indices
            base_mask = torch.zeros((H, T), dtype=torch.bool, device=device)
            base_mask.scatter_(1, gh_head, True)

            n_selects_h = torch.zeros((H, S), dtype=dtl, device=device)
            base_sum_h  = torch.zeros((H, S), dtype=token_scores.dtype, device=device)

            for L, G in groups_L.items():
                idxL   = G["idx"]
                starts = G["starts"]
                r = torch.arange(L, device=device, dtype=dtl)
                idx_grid = starts.unsqueeze(1) + r.unsqueeze(0)

                hit = base_mask[:, idx_grid]
                vals = s_all[:, idx_grid]
                n_sel = hit.sum(dim=-1)
                base_sum = (vals * hit).sum(dim=-1)

                n_selects_h[:, idxL] = n_sel
                base_sum_h[:, idxL] = base_sum

            active = (n_selects_h > 0) & (seg_lens.unsqueeze(0) > 0)
            bs0 = torch.minimum(torch.full_like(n_selects_h, 13), n_selects_h.clamp(min=1))
            bs_cur = torch.where(active, bs0, torch.zeros_like(bs0))

            cand_tokens_hs = [[None for _ in range(S)] for _ in range(H)]
            accepted = torch.zeros((H, S), dtype=torch.bool, device=device)

            for bs_val in [13, 11, 9, 7, 5, 3, 2, 1]:
                bucket_hs = (bs_cur == bs_val) & (~accepted) & active
                if not bucket_hs.any():
                    continue

                for L, G in groups_L.items():
                    if L <= 0:
                        continue
                    idxL = G["idx"]
                    mask_L = bucket_hs[:, idxL]
                    if not mask_L.any():
                        continue

                    hs_idx = torch.nonzero(mask_L, as_tuple=False)
                    if hs_idx.numel() == 0:
                        continue

                    h_e = hs_idx[:, 0]
                    s_local = hs_idx[:, 1]
                    s_abs = idxL[s_local]

                    starts_e = seg_starts[s_abs]
                    nsel_e = n_selects_h[h_e, s_abs]
                    base_e = base_sum_h[h_e, s_abs]

                    n_blocks_L = (L + bs_val - 1) // bs_val
                    k_blocks_all = torch.clamp((nsel_e + bs_val - 1) // bs_val,
                                                min=0, max=n_blocks_L)
                    if not (k_blocks_all > 0).any():
                        bs_cur[h_e, s_abs] = torch.clamp(bs_cur[h_e, s_abs] - 1, min=1)
                        continue

                    block_starts = torch.arange(n_blocks_L, device=device, dtype=dtl) * bs_val
                    tok_off = torch.arange(bs_val, device=device, dtype=dtl)

                    unique_k = torch.unique(k_blocks_all)
                    for k_need in unique_k.tolist():
                        if k_need <= 0:
                            continue
                        sel_mask = (k_blocks_all == k_need)
                        if not sel_mask.any():
                            continue

                        idx_sub   = torch.nonzero(sel_mask, as_tuple=False).squeeze(-1)
                        h_sub     = h_e[idx_sub]
                        s_sub     = s_abs[idx_sub]
                        nsel_sub  = nsel_e[idx_sub]
                        base_sub  = base_e[idx_sub]
                        starts_sub= starts_e[idx_sub]
                        G2 = idx_sub.numel()

                        r = torch.arange(L, device=device, dtype=dtl)
                        idx_g = starts_sub.unsqueeze(1) + r.unsqueeze(0)
                        seg_buf = s_all[h_sub].gather(1, idx_g)

                        blk_pos = block_starts.view(1, n_blocks_L, 1) + tok_off.view(1, 1, bs_val)
                        blk_pos = blk_pos.expand(G2, -1, -1)
                        valid_blk = blk_pos < L

                        pos_rel = blk_pos.clamp_max(L - 1)
                        seg_buf_exp = seg_buf.unsqueeze(1).expand(G2, n_blocks_L, L)
                        seg_blk = torch.gather(seg_buf_exp, 2, pos_rel)
                        seg_blk = seg_blk * valid_blk
                        block_sums = seg_blk.sum(dim=-1)

                        _, top_blk = torch.topk(block_sums, k=k_need, dim=-1)
                        blk_start_off = starts_sub.unsqueeze(1) + (top_blk * bs_val)

                        pos_abs = blk_start_off.unsqueeze(-1) + tok_off.view(1, 1, bs_val)
                        rel_abs = pos_abs - starts_sub.unsqueeze(1).unsqueeze(2)
                        valid_tok = (rel_abs >= 0) & (rel_abs < L)
                        pos_abs = pos_abs.clamp_(min=0, max=T - 1)
                        cand_all = pos_abs.view(G2, -1)

                        nsel_max = int(nsel_sub.max().item())
                        if nsel_max > 0:
                            cand_scores = s_all[h_sub].gather(1, cand_all)
                            invalid_mask = ~valid_tok.view(G2, -1)
                            cand_scores = cand_scores.masked_fill(invalid_mask, torch.finfo(cand_scores.dtype).min)
                            _, topi2 = torch.topk(cand_scores, k=nsel_max, dim=-1)
                            sel_max = cand_all.gather(1, topi2)
                            mask_len = torch.arange(nsel_max, device=device).view(1, -1) < nsel_sub.view(-1, 1)
                            sel_scores = s_all[h_sub].gather(1, sel_max)
                            block_sum = (sel_scores * mask_len).sum(dim=-1)
                        else:
                            block_sum = torch.zeros(G2, device=device, dtype=s_all.dtype)

                        ratio = (block_sum + eps) / (base_sub + eps)
                        accept = (ratio >= self.threshold) | (bs_val == 1)

                        if accept.any():
                            acc_rows = torch.nonzero(accept, as_tuple=False).squeeze(-1)
                            for ri in acc_rows.tolist():
                                hh = int(h_sub[ri].item()); ss = int(s_sub[ri].item())
                                cand_tokens_hs[hh][ss] = cand_all[ri]  # (M_raw,) LongTensor（合法绝对索引）
                            accepted[h_sub[accept], s_sub[accept]] = True

                        still = (~accept)
                        if still.any():
                            bs_cur[h_sub[still], s_sub[still]] = torch.clamp(
                                bs_cur[h_sub[still], s_sub[still]] - 1, min=1
                            )

                if (accepted | (~active)).all():
                    break

            for h in range(H):
                pairs = []
                for s in range(S):
                    ki = int(n_selects_h[h, s].item())
                    cand = cand_tokens_hs[h][s]
                    if ki > 0 and (cand is not None) and cand.numel() > 0:
                        pairs.append((ki, s, cand))

                if not pairs:
                    gh = gh_head[h]
                    out_idx[b, h] = gh[:K_total]
                    continue

                buckets = {}
                for ki, s, cand in pairs:
                    buckets.setdefault(ki, []).append((s, cand))

                chosen_all = []
                for ki, items in buckets.items():
                    Gk = len(items)
                    maxM = max(c.numel() for _, c in items)

                    cand_mat = torch.zeros((Gk, maxM), dtype=dtl, device=device)
                    cand_mat_mask = torch.zeros((Gk, maxM), dtype=torch.bool, device=device)
                    for i, (_, c) in enumerate(items):
                        m = c.numel()
                        cand_mat[i, :m] = c
                        cand_mat_mask[i, :m] = True

                    safe_cand = cand_mat.clone()
                    safe_cand[~cand_mat_mask] = 0
                    safe_cand.clamp_(0, T - 1)

                    cand_scores = s_all[h].gather(0, safe_cand.view(-1)).view(Gk, maxM)
                    cand_scores = cand_scores.masked_fill(~cand_mat_mask, torch.finfo(cand_scores.dtype).min)

                    _, topi = torch.topk(cand_scores, k=min(ki, maxM), dim=-1)
                    top_tok = cand_mat.gather(1, topi)  # (Gk, ki)
                    chosen_all.append(top_tok.reshape(-1))

                selected = torch.cat(chosen_all, dim=0) if chosen_all else torch.empty(0, dtype=dtl, device=device)

                if selected.numel() > 1:
                    try:
                        _, first_idx = torch.unique(selected, sorted=False, return_index=True)
                        selected = selected[first_idx.sort().indices]
                    except TypeError:
                        sel_list = selected.detach().cpu().tolist()
                        seen = set()
                        kept = []
                        for x in sel_list:
                            if x not in seen:
                                seen.add(x)
                                kept.append(x)
                        selected = torch.tensor(kept, dtype=dtl, device=device)

                if selected.numel() < K_total:
                    gh = gh_head[h]
                    remain = gh[~torch.isin(gh, selected)] if selected.numel() > 0 else gh
                    need = K_total - selected.numel()
                    selected = torch.cat([selected, remain[:need]], dim=0)
                elif selected.numel() > K_total:
                    selected = selected[:K_total]

                out_idx[b, h] = selected

        return out_idx  # (B,H,K_total)


    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        sample_id = global_vars.current_sample_id
        segment_boundaries = global_vars.global_segment_boundaries.get(sample_id, None)
        if segment_boundaries:
            segment_stats = self.compute_segment_scores(key_states, query_states, segment_boundaries)

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps

        print(f"SemBlock max_capacity_prompt {max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # (B,H,window,past) -> sum over window => (B,H,past_len)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)

        if self.pooling == 'avgpool':
            token_scores = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            token_scores = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        past_len = token_scores.size(-1)

        alpha = getattr(self, "alpha_prior", 0.9)
        prior = torch.zeros(past_len, device=token_scores.device, dtype=torch.float32)

        if segment_boundaries:
            for seg_idx, (start, end) in enumerate(segment_boundaries):
                s = max(0, min(start, past_len))
                e = max(0, min(end,   past_len))
                if e > s:
                    prior[s:e] = segment_stats[seg_idx]['omega_k']

        prior = (1.0 + alpha * prior).view(1, 1, past_len)

        scores = token_scores * prior

        past_len = key_states.size(2) - self.window_size

        if q_len < (self.max_capacity_prompt - self.window_size) * 2:
            K_total = self.max_capacity_prompt - self.window_size
        else:
            K_total = int(max_capacity_prompt)

        segs = None
        if segment_boundaries:
            segs = []
            for s, e in segment_boundaries:
                segs.append((int(s), int(e)))
            segs = self._clip_segments_to_past(segs, past_len)

        indices = self._adaptive_segment_indices_per_head_vec(
            scores, K_total, segs
        )

        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = key_states[:, :, :-self.window_size, :]
        v_past = value_states[:, :, :-self.window_size, :]
        k_past_compress = k_past.gather(dim=2, index=indices_exp)
        v_past_compress = v_past.gather(dim=2, index=indices_exp)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states


class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states

class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SentenceKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        sample_id = global_vars.current_sample_id
        segment_boundaries = global_vars.global_segment_boundaries.get(sample_id, None)

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        w = self.window_size

        max_capacity_prompt = self.max_capacity_prompt
        print(f"SentenceKV max_capacity_prompt {max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        def build_attn_cache(key_states, query_states):
            attn_weights = torch.matmul(
                query_states[..., -w:, :],
                key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)

            mask = torch.full((w, w), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            attn_weights[:, :, -w:, -w:] += mask[None, None, :, :]

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_cache = attn_weights[:, :, -w:, : -w].sum(dim=-2)
            return attn_cache

        attn_cache = build_attn_cache(key_states, query_states)
        past_len = attn_cache.size(-1)
        if past_len <= 0:
            return key_states, value_states

        def sentencekv_select_and_concat(L_max_tokens: int):
            if L_max_tokens <= 0:
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                out_k = torch.cat([key_states[:, :, :0, :], k_cur], dim=2)
                out_v = torch.cat([value_states[:, :, :0, :], v_cur], dim=2)
                return out_k, out_v

            tok_scores = attn_cache.mean(dim=1)  # [B, past_len]

            if not segment_boundaries:
                topk_tok = tok_scores.topk(k=min(L_max_tokens, past_len), dim=-1).indices  # [B, K]
                idx = topk_tok.unsqueeze(1).expand(-1, num_heads, -1).unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past = key_states[:, :, : -w, :].gather(dim=2, index=idx)
                v_past = value_states[:, :, : -w, :].gather(dim=2, index=idx)
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                out_k = torch.cat([k_past, k_cur], dim=2)
                out_v = torch.cat([v_past, v_cur], dim=2)
                return out_k, out_v

            clipped = []
            for (s, e) in segment_boundaries:
                s2 = max(0, min(s, past_len))
                e2 = max(0, min(e, past_len))
                if e2 > s2:
                    clipped.append((s2, e2))
            if len(clipped) == 0:
                return key_states, value_states

            B = tok_scores.size(0)
            per_b_indices = [] 
            for b in range(B):
                scores = []
                lengths = []
                for (s, e) in clipped:
                    lengths.append(e - s)
                    scores.append(tok_scores[b, s:e].sum())
                scores = torch.stack(scores, dim=0)
                lengths = torch.tensor(lengths, device=tok_scores.device, dtype=torch.long)

                order = torch.argsort(scores, descending=True)

                budget = int(L_max_tokens)
                chosen_tokens = []
                for idx in order.tolist():
                    s, e = clipped[idx]
                    Ls = e - s
                    if Ls <= 0:
                        continue
                    if Ls <= budget:
                        chosen_tokens.append(torch.arange(s, e, device=tok_scores.device))
                        budget -= Ls
                        if budget == 0:
                            break
                    else:
                        chosen_tokens.append(torch.arange(s, s + budget, device=tok_scores.device))
                        budget = 0
                        break

                if len(chosen_tokens) == 0:
                    per_b_indices.append(torch.empty(0, dtype=torch.long, device=tok_scores.device))
                else:
                    per_b_indices.append(torch.cat(chosen_tokens, dim=0))

            max_keep = max((x.numel() for x in per_b_indices), default=0)
            if max_keep == 0:
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                out_k = torch.cat([key_states[:, :, :0, :], k_cur], dim=2)
                out_v = torch.cat([value_states[:, :, :0, :], v_cur], dim=2)
                return out_k, out_v

            idx_nohead = torch.empty((B, max_keep), dtype=torch.long, device=key_states.device)
            for b in range(B):
                t = per_b_indices[b]
                Lb = t.numel()
                if Lb == 0:
                    idx_nohead[b].fill_(0)
                else:
                    idx_nohead[b, :Lb] = t
                    if Lb < max_keep:
                        idx_nohead[b, Lb:] = t[Lb - 1]

            idx = idx_nohead.unsqueeze(1).expand(-1, num_heads, -1).unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past = key_states[:, :, : -w, :].gather(dim=2, index=idx)
            v_past = value_states[:, :, : -w, :].gather(dim=2, index=idx)

            k_cur = key_states[:, :, -w:, :]
            v_cur = value_states[:, :, -w:, :]
            out_k = torch.cat([k_past, k_cur], dim=2)
            out_v = torch.cat([v_past, v_cur], dim=2)
            return out_k, out_v

        if q_len < (self.max_capacity_prompt - self.window_size) * 2:
            L_max_tokens = self.max_capacity_prompt - self.window_size
            key_states, value_states = sentencekv_select_and_concat(L_max_tokens)
            return key_states, value_states
        else:
            L_max_tokens = self.max_capacity_prompt - self.window_size
            key_states, value_states = sentencekv_select_and_concat(L_max_tokens)
            return key_states, value_states


class ChunkKVCluster(): 
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        max_capacity_prompt = self.max_capacity_prompt
        print(f"ChunkKV max_capacity_prompt {max_capacity_prompt}")

        def build_chunk_ranges(past_len: int, chunk_size: int, device):
            C = (past_len + chunk_size - 1) // chunk_size
            starts = torch.arange(0, C * chunk_size, chunk_size, device=device)[:C]
            ends = torch.clamp(starts + chunk_size, max=past_len)
            return starts, ends

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        w = self.window_size
        attn_weights = torch.matmul(
            query_states[..., -w:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        mask = torch.full((w, w), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -w:, -w:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_cache = attn_weights[:, :, -w:, : -w].sum(dim=-2)

        def chunkkv_select_and_concat(L_max_tokens: int):
            past_len = attn_cache.size(-1)
            if past_len <= 0 or L_max_tokens <= 0:
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                return torch.cat([key_states[:, :, :0, :], k_cur], dim=2), torch.cat([value_states[:, :, :0, :], v_cur], dim=2)

            chunk_size = 10
            starts, ends = build_chunk_ranges(past_len, chunk_size, attn_cache.device)
            C = starts.numel()

            chunk_scores = []
            for i in range(C):
                s, e = starts[i].item(), ends[i].item()
                chunk_scores.append(attn_cache[:, :, s:e].sum(dim=-1))
            chunk_scores = torch.stack(chunk_scores, dim=-1)

            K = max(0, min(C, L_max_tokens // chunk_size))
            if K == 0:
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                return torch.cat([key_states[:, :, :0, :], k_cur], dim=2), torch.cat([value_states[:, :, :0, :], v_cur], dim=2)

            topk_chunks = chunk_scores.topk(K, dim=-1).indices

            B, H = attn_cache.shape[:2]
            gather_lists = []
            max_keep = 0
            for b in range(B):
                row = []
                for h in range(H):
                    ids = topk_chunks[b, h]  # [K]
                    token_ids_list = []
                    for ci in ids.tolist():
                        s = starts[ci].item()
                        e = ends[ci].item()
                        if e > s:
                            token_ids_list.append(torch.arange(s, e, device=key_states.device))
                    if len(token_ids_list) == 0:
                        tokens_bh = torch.empty(0, dtype=torch.long, device=key_states.device)
                    else:
                        tokens_bh = torch.cat(token_ids_list, dim=0)
                    row.append(tokens_bh)
                    if tokens_bh.numel() > max_keep:
                        max_keep = tokens_bh.numel()
                gather_lists.append(row)

            if max_keep == 0:
                k_cur = key_states[:, :, -w:, :]
                v_cur = value_states[:, :, -w:, :]
                return torch.cat([key_states[:, :, :0, :], k_cur], dim=2), torch.cat([value_states[:, :, :0, :], v_cur], dim=2)

            idx_tensor = torch.empty((B, H, max_keep), dtype=torch.long, device=key_states.device)
            for b in range(B):
                for h in range(H):
                    t = gather_lists[b][h]
                    L = t.size(0)
                    if L == 0:
                        idx_tensor[b, h].fill_(0)
                    else:
                        idx_tensor[b, h, :L] = t
                        if L < max_keep:
                            idx_tensor[b, h, L:] = t[L - 1]

            idx_exp = idx_tensor.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B,H,max_keep,D]
            k_past = key_states[:, :, :-w, :].gather(dim=2, index=idx_exp)
            v_past = value_states[:, :, :-w, :].gather(dim=2, index=idx_exp)

            k_cur = key_states[:, :, -w:, :]
            v_cur = value_states[:, :, -w:, :]

            return torch.cat([k_past, k_cur], dim=2), torch.cat([v_past, v_cur], dim=2)

        if q_len < (self.max_capacity_prompt - self.window_size) * 2:
            L_max_tokens = self.max_capacity_prompt - self.window_size
            key_states, value_states = chunkkv_select_and_concat(L_max_tokens)
        else:
            L_max_tokens = max_capacity_prompt - self.window_size
            key_states, value_states = chunkkv_select_and_concat(L_max_tokens)

        return key_states, value_states

def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = PyramidKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_semblock(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = SemBlockCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_sentencekv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = SentenceKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_chunkkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = ChunkKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )
 
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )


def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = StreamingLLMKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )