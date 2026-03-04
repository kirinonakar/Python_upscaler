import torch
import torch.nn as nn
from torch.nn import functional as F

def get_attn_mask(bias, mask, B_windows, num_heads, N):
    """
    bias: (nH, N, N) - Relative Position Bias
    mask: (nW, N, N) or None - Window/Shift Mask
    B_windows: Total windows in batch (B * nW)
    """
    # 1. Start with bias: (1, nH, N, N)
    attn_mask = bias.unsqueeze(0)
    
    if mask is not None:
        nw = mask.shape[0]
        # (nW, 1, N, N)
        m_ = mask.unsqueeze(1)
        # Combine: (nW, nH, N, N)
        attn_mask = attn_mask + m_
        
        # If batch size B > 1
        if B_windows > nw:
            B = B_windows // nw
            # (B, nW, nH, N, N) -> (B*nW, nH, N, N)
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B_windows, num_heads, N, N)
    
    return attn_mask

def patch_swinir():
    try:
        from spandrel.architectures.SwinIR.__arch.SwinIR import WindowAttention
        
        def new_forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Relative Position Bias
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ).permute(2, 0, 1).contiguous() # (nH, N, N)

            dropout_p = self.attn_drop.p if self.training else 0.0

            try:
                attn_mask = get_attn_mask(bias, mask, B_, self.num_heads, N)
                attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)
                
                x = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    dropout_p=dropout_p,
                    scale=self.scale
                )
            except Exception:
                # Fallback to original
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn + get_attn_mask(bias, mask, B_, self.num_heads, N).to(q.device, q.dtype)
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        WindowAttention.forward = new_forward
        print("SwinIR SDPA patch applied (with fallback).")
    except Exception as e:
        print(f"Failed to patch SwinIR: {e}")

def patch_hat():
    try:
        from spandrel.architectures.HAT.__arch.HAT import WindowAttention
        
        def new_forward(self, x, rpi, mask=None):
            b_, n, c = x.shape
            qkv = (
                self.qkv(x)
                .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ).permute(2, 0, 1).contiguous()

            dropout_p = self.attn_drop.p if self.training else 0.0
            
            try:
                attn_mask = get_attn_mask(bias, mask, b_, self.num_heads, n)
                attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)
                
                x = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    dropout_p=dropout_p,
                    scale=self.scale
                )
            except Exception:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn + get_attn_mask(bias, mask, b_, self.num_heads, n).to(q.device, q.dtype)
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(b_, n, c)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        WindowAttention.forward = new_forward
        print("HAT SDPA patch applied (with fallback).")
    except Exception as e:
        print(f"Failed to patch HAT: {e}")

def patch_dat():
    try:
        from spandrel.architectures.DAT.__arch.DAT import Spatial_Attention, Adaptive_Channel_Attention, windows2img
        
        def spatial_forward(self, qkv, H, W, mask=None):
            q_raw, k_raw, v_raw = qkv[0], qkv[1], qkv[2]
            B_full, L, C = q_raw.shape
            
            q = self.im2win(q_raw, H, W)
            k = self.im2win(k_raw, H, W)
            v = self.im2win(v_raw, H, W)
            # Shapes: (B*nw, nH, N, C_h)
            B_windows = q.shape[0]
            N = q.shape[2]

            attn_mask = None
            if self.position_bias or mask is not None:
                bias = torch.zeros((self.num_heads, N, N), device=q.device, dtype=q.dtype)
                if self.position_bias:
                    pos = self.pos(self.rpe_biases)
                    bias = pos[self.relative_position_index.view(-1)].view(
                        self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
                    ).permute(2, 0, 1).contiguous()
                
                # Use robust mask getter
                attn_mask = get_attn_mask(bias, mask, B_windows, self.num_heads, N)
                attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)

            dropout_p = self.attn_drop.p if self.training else 0.0
            
            try:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    scale=self.scale
                )
            except Exception:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if attn_mask is not None:
                    attn = attn + attn_mask
                attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
                attn = self.attn_drop(attn)
                x = attn @ v
            
            x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
            x = windows2img(x, self.H_sp, self.W_sp, H, W)
            return x

        Spatial_Attention.forward = spatial_forward

        def channel_forward(self, x, H, W):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            
            v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

            dropout_p = self.attn_drop.p if self.training else 0.0
            
            try:
                attened_x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout_p,
                    scale=self.temperature
                )
            except Exception:
                attn = (q @ k.transpose(-2, -1)) * self.temperature
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                attened_x = attn @ v
            
            attened_x = attened_x.permute(0, 3, 1, 2).reshape(B, N, C)

            conv_x = self.dwconv(v_)
            attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
            channel_map = self.channel_interaction(attention_reshape)
            spatial_map = (
                self.spatial_interaction(conv_x)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(B, N, 1)
            )

            attened_x = attened_x * torch.sigmoid(spatial_map)
            conv_x = conv_x * torch.sigmoid(channel_map)
            conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

            x = attened_x + conv_x
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        Adaptive_Channel_Attention.forward = channel_forward
        print("DAT SDPA patch applied (with fallback).")
    except Exception as e:
        print(f"Failed to patch DAT: {e}")

def apply_sdpa_patches():
    if hasattr(F, "scaled_dot_product_attention"):
        patch_swinir()
        patch_hat()
        patch_dat()
    else:
        print("SDPA not available in this PyTorch version.")
