# Full forward pass — từ `(B, 2, N)` đến `logit`

Tài liệu trace từng layer của `SE2RoadNet`, hiển thị shape input/output ở mỗi bước.

## Ký hiệu

- `B` = batch size (ví dụ 384)
- `N` = số point thô (raw) của road — biến thiên theo road (10-50)
- `L` = 197 — số point sau resample (cố định trong SensoDat)
- `d` = 192 — d_model
- `h` = 8 — số head
- `d/h` = 24 — chiều mỗi head
- `ff` = 512

---

## Stage A — Feature extraction (CPU, ngoài model)

[exp02_SE2Equivariant.py:53-97](exp02_SE2Equivariant.py#L53-L97)

```
INPUT:  (B, 2, N)              ← N point thô (x, y) cho mỗi road
        ↓
        (riêng từng road — N có thể khác nhau)
        ↓
A.1 Tính độ dài cung tích luỹ s
        (B, N) — số thực ∈ [0, L_total]
        ↓
A.2 Resample uniformly 197 point dọc theo s
        (B, 2, 197)
        ↓
A.3 Tính θ(s) = atan2(dy, dx) → unwrap
        (B, 197) — góc tiếp tuyến
        ↓
A.4 Tính κ = dθ/ds (curvature)
        (B, 197)
        ↓
A.5 Tính dκ/ds, d²κ/ds²
        (B, 197), (B, 197)
        ↓
A.6 Tính |dα| (góc đổi hướng)
        (B, 197)
        ↓
A.7 Tính seg (segment length)
        (B, 197)
        ↓
A.8 Tính s/L (vị trí chuẩn hoá ∈ [0,1])
        (B, 197)
        ↓
A.9 Tính lstd (local std)
        (B, 197)
        ↓
A.10 Stack 7 kênh
        (B, 7, 197)
        ↓
A.11 Z-score normalize: (x - μ) / σ
        (B, 7, 197)

OUTPUT: (B, 7, 197) = (B, 7, L)
```

---

## Stage B — Vào model

[exp02_SE2Equivariant.py:160-167](exp02_SE2Equivariant.py#L160-L167)

### B.1 Permute

```
INPUT:  x  shape (B, 7, 197)

OP:     x = x.permute(0, 2, 1)

OUTPUT: x  shape (B, 197, 7)
```

### B.2 Tách `s_norm` ra để dùng cho bias sau này

```
INPUT:  x  shape (B, 197, 7)

OP:     s_norm = x[..., 5]            ← lấy kênh thứ 6 (index 5) = s/L

OUTPUT: s_norm  shape (B, 197)        ← arc-length chuẩn hoá ∈ [0,1]
        x       shape (B, 197, 7)     ← giữ nguyên (kênh 5 vẫn còn)
```

### B.3 `proj` — Linear + LayerNorm + GELU

```
INPUT:  x  shape (B, 197, 7)

B.3.1 Linear(7 → 192): y = x · Wᵀ + b   với W (192, 7), b (192)
      Áp lên trục cuối, độc lập với (B, 197)
      → shape (B, 197, 192)

B.3.2 LayerNorm(192)
      Chuẩn hoá mỗi vector 192-d: μ = mean, σ = std trên 192 chiều
      → shape (B, 197, 192)

B.3.3 GELU element-wise
      → shape (B, 197, 192)

OUTPUT: h  shape (B, 197, 192)
```

### B.4 Prepend CLS token

```
INPUT:  h  shape (B, 197, 192)

OP:     cls = self.cls.expand(B, 1, 192)   ← broadcast learnable token
        h = torch.cat([cls, h], dim=1)

OUTPUT: h  shape (B, 198, 192)             ← row 0 = CLS, row 1..197 = points
```

---

## Stage C — Pre-compute `s_full` cho bias

```
INPUT:  s_norm  shape (B, 197)

OP:     s_full = torch.cat([zeros(B, 1), s_norm], dim=1)
        ↑ CLS được gán s=0

OUTPUT: s_full  shape (B, 198)
```

`s_full` này sẽ được **truyền vào MỖI block** để tính bias. Không thay đổi qua các block.

---

## Stage D — InvariantBlock 1 (lặp lại tương tự cho block 2..6)

Bên trong một block có **2 sub-layer**: Attention + FFN. Mỗi sub-layer có residual.

```
INPUT:  h       shape (B, 198, 192)
        s_norm  shape (B, 197)             ← để tính bias
```

### D.1 Attention sub-layer

#### D.1.1 Pre-LayerNorm

```
INPUT:  h  shape (B, 198, 192)

OP:     z = self.n1(h)
        LayerNorm(192) — chuẩn hoá mỗi vector 192-d độc lập

OUTPUT: z  shape (B, 198, 192)
```

#### D.1.2 Tính relative-arclength bias

[exp02_SE2Equivariant.py:127-138](exp02_SE2Equivariant.py#L127-L138)

```
INPUT:  s_full  shape (B, 198)

D.1.2.a Outer difference
        ds = s_full.unsqueeze(2) − s_full.unsqueeze(1)
        → shape (B, 198, 198)
        unsqueeze(-1) → shape (B, 198, 198, 1)

D.1.2.b RFF: nhân với 32 tần số fix + sin
        ds * self.rff           ds (B,198,198,1) * rff (1,32)
        → broadcast → shape (B, 198, 198, 32)
        torch.sin(...)
        → shape (B, 198, 198, 32)

D.1.2.c MLP: Linear(32→64) → GELU → Linear(64→8)
        → shape (B, 198, 198, 8)

D.1.2.d Permute để khớp PyTorch MHA
        .permute(0, 3, 1, 2)
        → shape (B, 8, 198, 198)

D.1.2.e Reshape về (B*nhead, L+1, L+1)
        attn_mask = bias.reshape(B*8, 198, 198)
        → shape (B*8, 198, 198)

OUTPUT: attn_mask  shape (B*8, 198, 198)
```

#### D.1.3 MultiheadAttention internals

```
INPUT:  z         shape (B, 198, 192)
        attn_mask shape (B*8, 198, 198)

D.1.3.a Project Q, K, V (mỗi cái là Linear 192→192)
        Q = Linear_q(z)  → shape (B, 198, 192)
        K = Linear_k(z)  → shape (B, 198, 192)
        V = Linear_v(z)  → shape (B, 198, 192)

D.1.3.b Split thành 8 head (chia chiều cuối 192 → 8 × 24)
        Q.view(B, 198, 8, 24).transpose(1, 2)
        → Q_h shape (B, 8, 198, 24)
        → K_h shape (B, 8, 198, 24)
        → V_h shape (B, 8, 198, 24)

D.1.3.c Raw attention score
        score = Q_h @ K_h.transpose(-2, -1) / sqrt(24)
        → shape (B, 8, 198, 198)

D.1.3.d ╔═════════════════════════════════════════╗
        ║ CỘNG BIAS — đây là chỗ bias hoạt động   ║
        ║                                          ║
        ║ score = score + attn_mask.view(B,8,198,198)
        ║ → shape (B, 8, 198, 198)                 ║
        ╚═════════════════════════════════════════╝

D.1.3.e Softmax theo trục cuối (token j)
        attn = softmax(score, dim=-1)
        → shape (B, 8, 198, 198)

D.1.3.f Aggregate V theo trọng số attn
        out_h = attn @ V_h
        → shape (B, 8, 198, 24)

D.1.3.g Merge 8 head lại
        out_h.transpose(1, 2).contiguous().view(B, 198, 192)
        → shape (B, 198, 192)

D.1.3.h Linear output projection (192→192)
        a = Linear_out(...)
        → shape (B, 198, 192)

OUTPUT: a  shape (B, 198, 192)
```

#### D.1.4 Dropout + Residual

```
INPUT:  h shape (B, 198, 192), a shape (B, 198, 192)

OP:     h = h + Dropout(a)         ← residual connection

OUTPUT: h shape (B, 198, 192)
```

### D.2 FFN sub-layer

#### D.2.1 Pre-LayerNorm

```
INPUT:  h  shape (B, 198, 192)

OP:     z = self.n2(h)
        LayerNorm(192)

OUTPUT: z  shape (B, 198, 192)
```

#### D.2.2 Linear(192 → 512)

```
INPUT:  z  shape (B, 198, 192)

OP:     y = z · Wᵀ + b   với W (512, 192), b (512)
        Áp lên trục cuối

OUTPUT: y  shape (B, 198, 512)
```

#### D.2.3 GELU

```
INPUT:  y  shape (B, 198, 512)

OP:     element-wise GELU

OUTPUT: y  shape (B, 198, 512)
```

#### D.2.4 Dropout (trong FF module)

```
INPUT:  y  shape (B, 198, 512)
OP:     Dropout(0.1)
OUTPUT: y  shape (B, 198, 512)
```

#### D.2.5 Linear(512 → 192)

```
INPUT:  y  shape (B, 198, 512)

OP:     out = y · Wᵀ + b   với W (192, 512), b (192)

OUTPUT: f  shape (B, 198, 192)
```

#### D.2.6 Dropout + Residual

```
INPUT:  h shape (B, 198, 192), f shape (B, 198, 192)

OP:     h = h + Dropout(f)

OUTPUT: h shape (B, 198, 192)
```

### D.3 Kết thúc block 1

```
OUTPUT của Block 1:  h  shape (B, 198, 192)
                     s_norm  shape (B, 197)  (không đổi, dùng cho block 2)
```

---

## Stage E — Lặp lại Block 2, 3, 4, 5, 6

**Mỗi block làm y hệt Stage D** với cùng input shape, cùng output shape. Khác nhau:
- Các `Linear_q, Linear_k, Linear_v, Linear_out` (trong attention)
- Các `Linear(192→512), Linear(512→192)` (trong FFN)
- Bộ `ω` (tần số RFF) và MLP bias `(Linear(32→64), Linear(64→8))`
- Các `LayerNorm`

→ Tất cả đều có **trọng số riêng** cho từng block. **6 block = 6 bộ trọng số độc lập**.

```
Sau block 2: h (B, 198, 192)
Sau block 3: h (B, 198, 192)
Sau block 4: h (B, 198, 192)
Sau block 5: h (B, 198, 192)
Sau block 6: h (B, 198, 192)
```

---

## Stage F — Head MLP

[exp02_SE2Equivariant.py:157-159, 167](exp02_SE2Equivariant.py#L157-L167)

### F.1 Lấy CLS row

```
INPUT:  h  shape (B, 198, 192)

OP:     cls_out = h[:, 0, :]            ← chỉ lấy row 0 (CLS), bỏ 197 point

OUTPUT: cls_out  shape (B, 192)
```

### F.2 Final LayerNorm

```
INPUT:  cls_out  shape (B, 192)
OP:     LayerNorm(192)
OUTPUT: cls_out  shape (B, 192)
```

### F.3 Linear(192 → 64)

```
INPUT:  cls_out  shape (B, 192)
OP:     y = cls_out · Wᵀ + b   với W (64, 192), b (64)
OUTPUT: y  shape (B, 64)
```

### F.4 GELU

```
INPUT:  y  shape (B, 64)
OP:     element-wise GELU
OUTPUT: y  shape (B, 64)
```

### F.5 Dropout(0.2)

```
INPUT:  y  shape (B, 64)
OP:     Dropout(0.2)
OUTPUT: y  shape (B, 64)
```

### F.6 Linear(64 → 1)

```
INPUT:  y  shape (B, 64)
OP:     y = y · Wᵀ + b   với W (1, 64), b (1)
OUTPUT: y  shape (B, 1)
```

### F.7 Squeeze

```
INPUT:  y  shape (B, 1)
OP:     y.squeeze(-1)
OUTPUT: logit  shape (B,)        ← một số thực cho mỗi road
```

---

## Stage G — Inference (ngoài model)

```
INPUT:  logit  shape (B,)

G.1 Sigmoid:  p = 1 / (1 + exp(-logit))
              → shape (B,) — xác suất FAIL ∈ [0, 1]

G.2 Sắp xếp test theo p giảm dần
              → thứ tự ưu tiên

G.3 Tính APFD theo thứ tự đó
              → scalar APFD ∈ [0, 1]
```

---

## Bảng tổng kết shape qua toàn pipeline

```
Stage    Layer                          Shape Input              Shape Output
─────────────────────────────────────────────────────────────────────────────────
A.0      Raw input                      —                        (B, 2, N)
A.1–A.10 Extract 7 channels             (B, 2, N)                (B, 7, 197)
A.11     Z-score normalize              (B, 7, 197)              (B, 7, 197)
B.1      Permute                        (B, 7, 197)              (B, 197, 7)
B.2      Tách s_norm = x[..., 5]        (B, 197, 7)              (B, 197)
B.3.1    Linear 7→192                   (B, 197, 7)              (B, 197, 192)
B.3.2    LayerNorm                      (B, 197, 192)            (B, 197, 192)
B.3.3    GELU                           (B, 197, 192)            (B, 197, 192)
B.4      Prepend CLS                    (B, 197, 192)            (B, 198, 192)
C        s_full = cat([0], s_norm)      (B, 197)                 (B, 198)
─── Block 1 (lặp lại tương tự cho block 2..6) ───
D.1.1    LayerNorm n1                   (B, 198, 192)            (B, 198, 192)
D.1.2.a  outer diff Δs                  (B, 198)                 (B, 198, 198, 1)
D.1.2.b  RFF: sin(Δs·ω)                 (B, 198, 198, 1)         (B, 198, 198, 32)
D.1.2.c  MLP bias 32→64→8               (B, 198, 198, 32)        (B, 198, 198, 8)
D.1.2.d  permute                        (B, 198, 198, 8)         (B, 8, 198, 198)
D.1.2.e  reshape attn_mask              (B, 8, 198, 198)         (B*8, 198, 198)
D.1.3.a  Linear Q/K/V (192→192) ×3      (B, 198, 192)            (B, 198, 192) ×3
D.1.3.b  Split heads                    (B, 198, 192)            (B, 8, 198, 24)
D.1.3.c  Q @ Kᵀ / √d                    (B, 8, 198, 24)          (B, 8, 198, 198)
D.1.3.d  + bias                         (B, 8, 198, 198)         (B, 8, 198, 198)
D.1.3.e  softmax (trục j)               (B, 8, 198, 198)         (B, 8, 198, 198)
D.1.3.f  attn @ V                       (B, 8, 198, 198)         (B, 8, 198, 24)
D.1.3.g  Merge heads                    (B, 8, 198, 24)          (B, 198, 192)
D.1.3.h  Linear_out 192→192             (B, 198, 192)            (B, 198, 192)
D.1.4    Dropout + residual             (B, 198, 192)            (B, 198, 192)
D.2.1    LayerNorm n2                   (B, 198, 192)            (B, 198, 192)
D.2.2    Linear 192→512                 (B, 198, 192)            (B, 198, 512)
D.2.3    GELU                           (B, 198, 512)            (B, 198, 512)
D.2.4    Dropout                        (B, 198, 512)            (B, 198, 512)
D.2.5    Linear 512→192                 (B, 198, 512)            (B, 198, 192)
D.2.6    Dropout + residual             (B, 198, 192)            (B, 198, 192)
─── Block 2..6: lặp lại D.1.1 → D.2.6 ───
─── Sau 6 block: (B, 198, 192) ───
F.1      Lấy CLS row h[:, 0, :]         (B, 198, 192)            (B, 192)
F.2      LayerNorm                      (B, 192)                 (B, 192)
F.3      Linear 192→64                  (B, 192)                 (B, 64)
F.4      GELU                           (B, 64)                  (B, 64)
F.5      Dropout 0.2                    (B, 64)                  (B, 64)
F.6      Linear 64→1                    (B, 64)                  (B, 1)
F.7      Squeeze                        (B, 1)                   (B,)
─── Sigmoid (ngoài model) ───
G.1      p = sigmoid(logit)             (B,)                     (B,)  ∈ [0,1]
```

---

## Tổng số phép Linear trong toàn mạng

| Vị trí | Số lần | Shape |
|---|---|---|
| Stage B.3.1 (`proj`) | 1 | 7→192 |
| Mỗi block: bias MLP | 2 | 32→64, 64→8 |
| Mỗi block: Q,K,V,out | 4 | 192→192 |
| Mỗi block: FFN | 2 | 192→512, 512→192 |
| 6 blocks × (2+4+2) | 48 | — |
| Stage F (head) | 2 | 192→64, 64→1 |
| **Tổng** | **51 Linear** | — |

Cộng thêm 13 LayerNorm (1 ở proj + 2 mỗi block × 6 + 1 ở head), 6 RFF parameter (frozen), 1 CLS parameter, vài Dropout.

---

## Câu thần chú để nhớ

> Input `(B, 2, N)` → extract 7 kênh → permute → nâng 7 lên 192 → thêm CLS → đi qua 6 block (mỗi block: attention với bias `Δs` + FFN) → lấy CLS → MLP 192→64→1 → sigmoid → xác suất FAIL → sort → APFD.
