# EXP 02 — SE(2)-Equivariant RoadNet: Walkthrough chi tiết

> Mục tiêu: mô phỏng end-to-end quá trình `exps/exp02_SE2Equivariant.py` chạy
> trên một ví dụ nhỏ 5 điểm road, để hiểu rõ từng phép tính từ JSON thô đến
> APFD và rotation probe.

---

## 0. Bối cảnh lý thuyết (1 phút đọc)

Một test SDC = một con đường. Mô hình prioritizer `f` nhận road `r` (chuỗi
điểm 2D) và trả về một điểm số (logit) cho biết khả năng xe đi chệch lane.

Yêu cầu vật lý: nếu xoay/dịch chuyển con đường trên mặt phẳng, **bản chất**
con đường không đổi (curvature, chiều dài, độ uốn không phụ thuộc hệ tọa
độ). Do đó ta cần:

```
f(R · r + t) = f(r)     với mọi R ∈ SO(2), t ∈ R^2
```

→ Đây gọi là **SE(2)-invariance**. Baseline 10-kênh dùng `sin(ang), cos(ang)`
và (x, y) raw → KHÔNG invariant (chỉ "có thể học được", nhưng không bị ràng
buộc).

Exp 02 ép invariance bằng kiến trúc — chỉ cho model nhìn vào các đại lượng
**intrinsic** (curvature, arc-length, |Δheading|), không nhìn vào hệ tọa độ
tuyệt đối. Kết quả: rotation-probe `Δ APFD = 0.0000` (exact, không phải xấp
xỉ).

---

## 1. Dataset có hình dạng như thế nào

### 1.1 Cấu trúc JSON

Mỗi file (`sensodat_train.json`, `sensodat_test.json`, `sdc-test-data.json`)
là một list các test case. Một test case mẫu (đã rút gọn):

```json
{
  "_id": {"$oid": "65a3f7c1abc..."},
  "road_points": [
    {"x":   0.0, "y":  0.0},
    {"x":  10.0, "y":  0.0},
    {"x":  20.0, "y":  5.0},
    ...
    {"x": 145.2, "y": 88.7}
  ],
  "meta_data": {
    "test_info": {
      "test_outcome": "FAIL"
    }
  }
}
```

Code đọc dữ liệu (lines [exp02_SE2Equivariant.py:95-101](exp02_SE2Equivariant.py#L95-L101)):

```python
def get_pts(tc):  return [[p['x'], p['y']] for p in tc['road_points']]
def is_fail(tc): return tc['meta_data']['test_info']['test_outcome'] == 'FAIL'
def get_id(tc):  return tc['_id']['$oid']
```

### 1.2 Kích thước SensoDat

| Split | #tests | %FAIL  |
|-------|-------:|-------:|
| train | ~4480  | ~30%   |
| test  | ~1120  | ~30%   |
| comp  | 957    | ~30%   |

Số điểm mỗi road dao động: ~50 đến ~250 → script dùng `SEQ_LEN = 197` và
trong `extract_invariant_7ch` sẽ không pad/truncate explicit (đó là điểm
khác baseline — Transformer attention tự xử lý độ dài thật khi xếp batch;
trong thực tế các batch được padding zero ở mức DataLoader).

---

## 2. Worked example: 1 road 5 điểm

Để tính tay được, ta dùng một road siêu nhỏ:

```
pts = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2)]
```

Vẽ ra:

```
y
2 |          . (4,2)
1 |       . (3,1)
0 | . . . . . . . . . . . . .
    (0,0) (1,0) (2,0)        x
```

Con đường này có 2 đoạn thẳng đầu (đi ngang), sau đó **rẽ trái 45°** ở
điểm thứ 3 và đi thẳng tiếp.

---

## 3. Bước 1 — Trích xuất 7 feature INVARIANT

Hàm `extract_invariant_7ch(pts_raw)` ở
[exp02_SE2Equivariant.py:79-93](exp02_SE2Equivariant.py#L79-L93).

### 3.0 Hiểu trực quan: 7 channel = 7 câu hỏi về con đường

> Đừng vội đọc toán. Trước hết hãy hình dung bạn đang LÁI XE dọc con đường
> đó. Tại mỗi điểm trên đường, ta hỏi 7 câu hỏi — mỗi câu trả lời là một
> "channel" số.

#### Bối cảnh: con đường 5 điểm

```
        khúc cua trái 45° ở đây
                ↓
   ●───●───●───●───●
   0   1   2   3   4
   đi thẳng    đi xiên
```

Đường có 5 điểm: 2 đoạn đầu đi thẳng (ngang), đến điểm 2 thì **bẻ trái
45°**, rồi 2 đoạn cuối đi xiên 45°.

#### Bảng tóm tắt 7 channel — bằng ngôn ngữ tài xế

| # | Tên     | Câu hỏi tại mỗi điểm                                | Giá trị tại điểm 2 (khúc cua) |
|--:|---------|-----------------------------------------------------|-------------------------------|
| 1 | `seg`   | "Đoạn sắp đi dài bao nhiêu mét?"                    | 1.41 m (đi xiên)              |
| 2 | `\|dang\|`| "Tay lái phải xoay bao nhiêu để đi tiếp?"         | 0.79 rad ≈ **45°**            |
| 3 | `k`     | "Cua gắt cỡ nào? Dương = trái, âm = phải, 0 = thẳng"| **+0.65** (cua trái)          |
| 4 | `dk`    | "Tay lái đang xoay **nhanh dần** hay chậm lại?"     | -0.65 (đang nhả lái)          |
| 5 | `ddk`   | "Có **giật** tay lái không? Dao động bất thường?"   | +0.65 (vẫn có dao động)       |
| 6 | `s/L`   | "Tôi đang ở **đoạn nào** của con đường? (0=đầu, 1=cuối)" | 0.55 (gần giữa road)     |
| 7 | `lstd`  | "**Khu vực này** có nhiều cua không?"               | 0.26 (có 1 cua trong vùng)   |

#### Vì sao chọn 7 cái này (không phải x, y, sin, cos)?

7 con số trên có một tính chất chung: **chúng không đổi khi bạn xoay tấm
bản đồ**. Nếu bạn quay con đường 30° quanh gốc tọa độ:

- Tay lái vẫn phải xoay 45° (`|dang|` không đổi).
- Cua trái vẫn là cua trái (`k` không đổi).
- Đoạn đường vẫn dài 1.41 m (`seg` không đổi).
- Vị trí "giữa road" vẫn là 0.55 (`s/L` không đổi).

Ngược lại, nếu cho mô hình thấy **`x, y` thô** hoặc **`sin(heading),
cos(heading)`** (như baseline 10-channel), khi xoay bản đồ thì các số
này thay đổi → mô hình bị "lừa". Đó là toàn bộ ý tưởng của Exp 02.

#### Phân chia "vai trò" của 7 channel

```
┌──────────────────────────────────────────────────┐
│  ĐỘ DÀI                                          │
│  ├─ seg     : đoạn sắp đi dài bao nhiêu          │
│                                                  │
│  CUA / HƯỚNG                                     │
│  ├─ |dang|  : độ lớn của cua (luôn ≥ 0)          │
│  ├─ k       : độ gắt + chiều cua (có dấu)        │
│  ├─ dk      : tốc độ thay đổi cua  (đạo hàm k)   │
│  ├─ ddk     : "giật" cua           (đạo hàm dk)  │
│                                                  │
│  VỊ TRÍ DỌC ROAD                                 │
│  ├─ s/L     : đang ở phần nào của road (0 → 1)   │
│                                                  │
│  ĐẶC ĐIỂM VÙNG                                   │
│  └─ lstd    : vùng quanh đây có nhiều cua không   │
└──────────────────────────────────────────────────┘
```

→ **3 nhóm thông tin**: (a) đoạn đường, (b) độ cua + đạo hàm các cấp,
(c) ngữ cảnh không gian/thời gian dọc road.

#### Pipeline trích xuất — sơ đồ một-trang

```
        pts (n, 2)
            │
            │ tính vector từ điểm này sang điểm kế
            ▼
        d (n-1, 2)
        ┌───────┴────────┐
        │                │
        │ độ dài         │ góc heading
        ▼                ▼
       seg              ang        ← ang KHÔNG đi vào feature
        │                │
        │                │ chênh lệch 2 góc kế tiếp
        │                ▼
        │              dang  ──┬─►  |dang|     (channel 2)
        │                │     │
        │                │     └─► k = dang / mean(seg)   (channel 3)
        │                │              │
        │                │              ├─► diff → dk    (channel 4)
        │                │              │              │
        │                │              │              └─► diff → ddk  (channel 5)
        │                │              │
        │                │              └─► sliding-std → lstd  (channel 7)
        │                │
        ▼                │
   pad → seg_full  ──────┘                                  (channel 1)
        │
        │ cumsum / total length
        ▼
    s_norm                                                  (channel 6)
```

Đọc sơ đồ: từ ma trận điểm `(n, 2)` → ra một ma trận feature `(n, 7)`.
Mọi mũi tên trong sơ đồ này là các phép toán **không đụng đến hệ tọa độ
tuyệt đối** → đầu ra cũng vậy → invariant.

#### Đọc kết quả bằng tay cho road 5 điểm

Sau khi chạy `extract_invariant_7ch(pts)`, ma trận `(5, 7)` đọc thành 5
"thẻ thông tin" — mỗi thẻ là 1 điểm trên đường:

```
Điểm 0 (đầu road, ngay sau gốc):
  seg=1.00 (đoạn sắp đi 1m)
  |dang|=0 (đi thẳng, không cua)
  k=0      (không cua)
  dk=0, ddk=+0.65 (sắp có spike curvature → cảnh báo trước)
  s/L=0.16 (đang ở 16% road)
  lstd=0.26 (vùng này có 1 cua)

Điểm 2 (đỉnh cua trái 45°):
  seg=1.41 (đoạn sắp đi 1.41m — đi xiên)
  |dang|=0.79 (BẺ LÁI 45°!)
  k=+0.65 (cua TRÁI, khá gắt)
  dk=-0.65, ddk=+0.65 (đang nhả lái sau khi vừa bẻ)
  s/L=0.55 (giữa road)
  lstd=0.26

Điểm 4 (cuối road):
  seg=1.41 (đệm bằng giá trị trước)
  |dang|=0
  k=0
  dk=0, ddk=0
  s/L=1.00 (cuối road)
  lstd=0.26
```

Đó chính là **input của Transformer** sau khi normalize. Mỗi điểm = một
"token". Transformer xem 5 token này và quyết định: con đường này có dễ
làm xe đi chệch lane không?

---

> **TL;DR**: 7 channel = (1 độ dài) + (4 mô tả cua các cấp) + (1 vị trí) +
> (1 thống kê vùng). Tất cả đều "view nội tại" của con đường — xoay/dịch
> bản đồ không thay đổi giá trị.

Phần dưới đây (3.1 → 3.8) là cách **tính tay từng channel** với các phép
toán NumPy cụ thể. Đọc nếu bạn cần code lại hoặc verify số. Còn để hiểu
"mô hình thấy gì" thì chỉ cần phần 3.0 ở trên.

---

### 3.1 → 3.8 — Tính từng channel (phiên bản ngắn)

> Mỗi mục dưới đây gồm 3 phần: **công thức** → **tính tay 5 điểm** → **1
> câu ý nghĩa**. Đó là tất cả những gì cần để code lại.

Nhắc lại: `pts = [(0,0), (1,0), (2,0), (3,1), (4,2)]`, `n = 5`.

---

#### 3.1 — `d` = vector từ điểm này sang điểm kế

```python
d = np.diff(pts, axis=0)         # shape (n-1, 2)
```

`d[i] = pts[i+1] - pts[i]`. Đây là "mũi tên" chỉ đoạn đường thứ `i`.

```
d[0] = (1,0)-(0,0) = (1, 0)
d[1] = (2,0)-(1,0) = (1, 0)
d[2] = (3,1)-(2,0) = (1, 1)
d[3] = (4,2)-(3,1) = (1, 1)
```

→ **`d` = [(1,0), (1,0), (1,1), (1,1)]**, shape `(4, 2)`.

Ý nghĩa: 2 đoạn đầu chỉ về Đông, 2 đoạn sau chỉ về Đông-Bắc (45°).

---

#### 3.2 — Channel 1 `seg` = độ dài mỗi đoạn

```python
seg      = np.linalg.norm(d, axis=1)            # shape (n-1,)
seg_full = np.pad(seg, (0, 1), mode='edge')     # shape (n,) — pad 1 ở cuối
```

`seg[i] = sqrt(dx² + dy²)`. Pad bằng "edge" (copy giá trị cuối) để về
đúng `n` phần tử.

```
seg      = [√1, √1, √2,    √2]    = [1.00, 1.00, 1.41, 1.41]
seg_full = [1.00, 1.00, 1.41, 1.41, 1.41]
                                    ↑ copy
```

→ **Channel 1**: `[1.00, 1.00, 1.41, 1.41, 1.41]`.

Invariance: xoay không đổi độ dài → `seg` giữ nguyên.

---

#### 3.2bis — `ang` (heading) — KHÔNG phải channel, chỉ là bước trung gian

```python
ang = np.arctan2(d[:,1], d[:,0])    # shape (n-1,)
```

Góc của mũi tên `d[i]` trong dải `(-π, π]`. **KHÔNG đưa vào feature** vì
nếu xoay road, mọi `ang` shift cùng một hằng → leak hệ tọa độ tuyệt đối.

```
ang[0] = arctan2(0, 1) = 0      (Đông)
ang[1] = arctan2(0, 1) = 0      (Đông)
ang[2] = arctan2(1, 1) = π/4    (Đông-Bắc)
ang[3] = arctan2(1, 1) = π/4
```

→ **`ang` = [0, 0, 0.785, 0.785]**.

---

#### 3.3 — Channel 2 `|Δheading|` = độ lớn của cua tại mỗi điểm

```python
dang = (np.diff(ang) + np.pi) % (2*np.pi) - np.pi    # wrap về (-π, π]
abs_dang_full = np.pad(np.abs(dang), (1, 1), mode='constant')   # pad 0 hai đầu
```

3 bước: (1) trừ 2 ang kế tiếp; (2) wrap để xử lý jump ±180° (vd `170° →
-170°` thật ra chỉ cua 20°); (3) lấy `abs` (chỉ giữ độ lớn, dấu giữ ở
`k` channel 3); (4) pad 0 hai đầu (điểm đầu/cuối không có "góc cua").

```
diff(ang) = [0, 0.785, 0]       (đã trong dải, wrap không đổi)
|dang|    = [0, 0.785, 0]
pad(1,1)  = [0, 0, 0.785, 0, 0]
```

→ **Channel 2**: `[0, 0, 0.785, 0, 0]` (chỉ điểm 2 có cua 45° = 0.785 rad).

Invariance: xoay road shift mọi ang cùng hằng → hiệu triệt tiêu → `dang`
không đổi.

---

#### 3.4 — Channel 3 `k` = curvature có dấu (gắt + chiều cua)

```python
denom = 0.5*(seg[:-1] + seg[1:]) + 1e-8       # trung bình 2 đoạn quanh đỉnh
k_raw = dang / denom                          # rad/m
k     = np.pad(k_raw, (1, 1), mode='constant')   # pad 0 hai đầu
```

`k = dθ/ds`: tốc độ thay đổi góc trên một đơn vị quãng đường. Dấu cho
biết trái (+) / phải (−). `+ 1e-8` để tránh div 0.

```
seg[:-1] = [1, 1, 1.41]
seg[1:]  = [1, 1.41, 1.41]
denom    = 0.5*sum    = [1.00, 1.205, 1.414]

k_raw[0] = 0     / 1.00  = 0
k_raw[1] = 0.785 / 1.205 = 0.651
k_raw[2] = 0     / 1.414 = 0

k = [0, 0, 0.651, 0, 0]
```

→ **Channel 3**: `[0, 0, 0.651, 0, 0]` (cua TRÁI tại điểm 2).

> Lưu ý: chỉ invariant với **rotation** (SO(2)). Nếu phản chiếu (mirror)
> thì `k → -k`. Đây là một subtlety đáng đề cập trong paper.

---

#### 3.5 — Channel 4 `dk` = tốc độ thay đổi curvature

```python
dk = np.pad(np.diff(k), (0, 1), mode='constant')   # diff + pad 0 cuối
```

`dk[i] ≈ k[i+1] - k[i]`. Báo hiệu curvature đang tăng/giảm.

```
diff(k) = [0-0, 0.651-0, 0-0.651, 0-0]
        = [0,  +0.651,  -0.651,    0]

dk = [0, +0.651, -0.651, 0, 0]
```

→ **Channel 4**: `[0, +0.651, −0.651, 0, 0]`. Đọc: "đang vào cua" rồi
ngay sau "đang ra khỏi cua".

---

#### 3.6 — Channel 5 `ddk` = đạo hàm bậc 2 (jerk-like)

```python
ddk = np.pad(np.diff(dk), (0, 1), mode='constant')
```

`ddk[i] ≈ dk[i+1] - dk[i]`. Bắt **dao động curvature** rõ hơn `dk` —
spike trong `k` xuất hiện thành cụm `+/-/+` trong `ddk`.

```
diff(dk) = [0.651-0,    -0.651-0.651, 0-(-0.651), 0-0]
         = [+0.651,     -1.302,       +0.651,      0]

ddk = [+0.651, −1.302, +0.651, 0, 0]
```

→ **Channel 5**: `[+0.651, −1.302, +0.651, 0, 0]`. Cụm 3 dấu xen kẽ =
đặc trưng "spike cua đơn".

---

#### 3.7 — Channel 6 `s/L` = vị trí dọc road (0 đầu → 1 cuối)

```python
s_cum  = np.cumsum(seg_full)                  # tổng dồn quãng đường
s_norm = s_cum / (s_cum[-1] + 1e-8)           # chia tổng → về [0, 1]
```

`s_cum[i]` = quãng đường tích lũy đến điểm `i`. Chia cho tổng `L` để
chuẩn hóa.

```
seg_full = [1.00, 1.00, 1.41, 1.41, 1.41]
s_cum    = [1.00, 2.00, 3.41, 4.83, 6.24]    (cộng dồn)
L = s_cum[-1] = 6.24

s_norm   = s_cum / 6.24
         = [0.160, 0.320, 0.547, 0.774, 1.000]
```

→ **Channel 6**: `[0.160, 0.320, 0.547, 0.774, 1.000]`.

Đây cũng là "đồng hồ vị trí" mà `InvariantBlock` dùng để tính bias
attention `MLP(sin((s_i - s_j)·ω))` (mục 5.4).

---

#### 3.8 — Channel 7 `lstd` = độ dao động curvature quanh điểm (cửa sổ ±5)

```python
w = 11; hw = 5
lstd = np.zeros(n)
for i in range(n):
    a, b = max(0, i - hw), min(n, i + hw + 1)    # cửa sổ ±5, clamp ở biên
    lstd[i] = np.std(k[a:b])
```

Mỗi điểm: lấy 11 giá trị `k` quanh nó (truncate ở biên), tính
standard-deviation. Đoạn nhiều cua → `lstd` lớn; đoạn thẳng đều → `lstd
≈ 0`.

```
n=5 < w=11 → mọi i đều lấy toàn bộ k = [0, 0, 0.651, 0, 0]

mean = 0.651/5 = 0.130
var  = ( 4·(0-0.130)² + (0.651-0.130)² ) / 5
     = ( 4·0.0169 + 0.2715 ) / 5
     = 0.0677
std  = √0.0677 = 0.260
```

→ **Channel 7**: `[0.260, 0.260, 0.260, 0.260, 0.260]` (đều, vì road 5
điểm quá ngắn so với cửa sổ).

> Trên road thực `n ≈ 100`, cửa sổ khác nhau cho mỗi điểm → `lstd` thay
> đổi dọc road, lớn ở vùng nhiều cua, nhỏ ở đoạn thẳng.

---

#### Tóm tắt 7 channel: ai mất bao nhiêu phần tử và pad ở đâu?

| Channel | Lý do mất phần tử                | Cách pad                | Đầu / Cuối |
|---------|----------------------------------|-------------------------|------------|
| `seg`   | `diff(pts)` mất 1 ở cuối         | `(0, 1)` `mode=edge`    | copy cuối  |
| `\|dang\|`| `diff(ang)` mất 2 (đầu+cuối)   | `(1, 1)` `mode=constant`| pad 0      |
| `k`     | `dang/denom` cùng shape, mất 2  | `(1, 1)` `mode=constant`| pad 0      |
| `dk`    | `diff(k)` mất 1 ở cuối           | `(0, 1)` `mode=constant`| pad 0      |
| `ddk`   | `diff(dk)` mất 1 ở cuối          | `(0, 1)` `mode=constant`| pad 0      |
| `s_norm`| không mất (cumsum giữ shape)     | (không pad)             | —          |
| `lstd`  | không mất (vòng for chạy đủ n)   | (không pad)             | —          |

→ Tất cả về shape `(n,)` rồi `np.column_stack` thành `(n, 7)`.

---

_(Phần chi tiết hơn về invariance, units, edge cases đã được lược; nếu
cần, xem lịch sử git của file này.)_

### 3.1.deprecated Phần chi tiết cũ (xóa được nếu thấy không cần)

<details>
<summary>Bấm để xem phiên bản cũ với 6 sub-section / channel</summary>

#### 3.1.1 `pts` là gì?

Là một **ma trận NumPy 2D** shape `(n, 2)`, mỗi hàng là một điểm `(x, y)`.
Với ví dụ của ta:

```python
pts = np.array([[0, 0],
                [1, 0],
                [2, 0],
                [3, 1],
                [4, 2]])    # shape (5, 2)
```

| index | x | y |
|------:|--:|--:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 2 |

#### 3.1.2 `np.diff(..., axis=0)` làm gì?

`np.diff` tính **hiệu giữa hai phần tử liên tiếp** dọc theo trục được chỉ
định.

- `axis=0`: trục hàng (đi từ trên xuống) → trừ hàng sau với hàng trước.
- `axis=1`: trục cột (đi từ trái sang) → trừ cột sau với cột trước (không
  phải cái ta cần).

Công thức tổng quát:

```
d[i] = pts[i+1] - pts[i]      với i = 0, 1, ..., n-2
```

→ Số hàng của `d` là `n - 1` (mất 1 hàng vì hàng đầu không có "hàng trước
nó để trừ").

#### 3.1.3 Tính từng dòng

```
d[0] = pts[1] - pts[0] = (1, 0) - (0, 0) = (1, 0)
d[1] = pts[2] - pts[1] = (2, 0) - (1, 0) = (1, 0)
d[2] = pts[3] - pts[2] = (3, 1) - (2, 0) = (1, 1)
d[3] = pts[4] - pts[3] = (4, 2) - (3, 1) = (1, 1)
```

Kết quả:

```python
d = np.array([[1, 0],
              [1, 0],
              [1, 1],
              [1, 1]])     # shape (4, 2)
```

| i | d[i] = (dx, dy) |
|---|-----------------|
| 0 | (1, 0)          |
| 1 | (1, 0)          |
| 2 | (1, 1)          |
| 3 | (1, 1)          |

#### 3.1.4 Ý nghĩa hình học

Mỗi hàng `d[i]` là **vector chỉ phương** của đoạn thẳng nối `pts[i] →
pts[i+1]`. Hai thứ rút ra được từ nó:

- **Độ dài đoạn** (channel `seg`):
  ```
  seg[i] = ||d[i]|| = sqrt(dx² + dy²)
  ```
- **Hướng đoạn** (heading angle):
  ```
  ang[i] = arctan2(d[i].y, d[i].x)
  ```

Vẽ trực quan:

```
y
2 |               (4,2)
  |              ↗ d[3]=(1,1)
1 |          (3,1)
  |         ↗ d[2]=(1,1)
0 | (0,0)→(1,0)→(2,0)
  |   d[0]   d[1]
  | =(1,0)  =(1,0)
  +-----------------> x
```

Hai vector đầu `(1,0)` → đoạn nằm ngang. Hai vector sau `(1,1)` → đoạn
nghiêng 45°. Chỗ chuyển từ `d[1]` sang `d[2]` chính là khúc cua (sẽ tính
ra `|Δheading| = 45°` ở channel 2).

#### 3.1.5 Tại sao bước này là nền cho SE(2)-invariance

Khi xoay con đường bằng `R`, mỗi điểm đi `p → R·p`. Lấy hiệu:

```
d_rot[i] = R·pts[i+1] - R·pts[i] = R·(pts[i+1] - pts[i]) = R·d[i]
```

→ `d` cũng xoay theo `R`, NHƯNG `||d||` (độ dài) không đổi và **chênh
lệch góc giữa hai d kế tiếp** cũng không đổi. Đây là lý do mọi feature ở
các bước sau (chỉ dùng `||d||` và `Δang`) đều invariant.

#### 3.1.6 Lỗi thường gặp

- **Quên `axis=0`**: `np.diff(pts)` mặc định lấy `axis=-1` → trừ giữa cột
  `x` và cột `y` trong cùng một hàng → sai (ra shape `(5, 1)` thay vì
  `(4, 2)`).
- **Pad sai cuối**: `seg` có shape `(n-1,)`; code sau đó pad `(0, 1)` để
  về lại `n`. Hay nhầm thành `(1, 0)`.

### 3.2 Channel 1 — `seg` (segment length)

```python
seg = np.linalg.norm(d, axis=1)
seg_full = np.pad(seg, (0, 1), mode='edge')
```

#### 3.2.1 `np.linalg.norm(d, axis=1)` làm gì?

Tính **Euclidean norm** (chiều dài vector) cho từng hàng của `d`. Với
`axis=1` → reduce dọc theo trục cột:

```
seg[i] = sqrt(d[i,0]^2 + d[i,1]^2) = sqrt(dx^2 + dy^2)
```

Shape: `(n-1, 2) → (n-1,)`.

#### 3.2.2 Tính tay

Với `d` từ bước 3.1:

```
seg[0] = sqrt(1^2 + 0^2) = sqrt(1) = 1.0000
seg[1] = sqrt(1^2 + 0^2) = sqrt(1) = 1.0000
seg[2] = sqrt(1^2 + 1^2) = sqrt(2) = 1.4142
seg[3] = sqrt(1^2 + 1^2) = sqrt(2) = 1.4142
```

→ `seg = [1.0000, 1.0000, 1.4142, 1.4142]`, shape `(4,)`.

#### 3.2.3 `np.pad(seg, (0, 1), mode='edge')` làm gì?

- `(0, 1)`: thêm 0 phần tử trước, 1 phần tử sau.
- `mode='edge'`: phần tử mới = giá trị cuối của mảng gốc (replicate).

```
seg_full = [1.0000, 1.0000, 1.4142, 1.4142, 1.4142]
                                          ↑
                                  copy of seg[-1]
```

Shape thành `(n,) = (5,)` — khớp với số điểm để xếp thành ma trận
`(n, 7)`.

#### 3.2.4 Ý nghĩa & tính invariance

`seg[i]` là **chiều dài đoạn thẳng** giữa hai điểm liên tiếp — một đại
lượng intrinsic của road. Khi xoay `R`:

```
||R · d[i]|| = sqrt((R·d[i])^T (R·d[i]))
             = sqrt(d[i]^T R^T R d[i])
             = sqrt(d[i]^T d[i])         (vì R^T R = I)
             = ||d[i]||
```

→ `seg` bit-exact bằng nhau sau xoay. Đây là **đại lượng đơn giản nhất**
trong 7 channel có thể chứng minh invariance bằng đại số tuyến tính.

#### 3.2.5 Tại sao pad bằng `edge` chứ không phải `0`?

`seg` là độ dài đoạn → giá trị 0 sẽ vô nghĩa (một điểm "trùng điểm cuối").
Replicate giá trị cuối nghĩa là "giả định điểm cuối kéo dài thêm một đoạn
cùng độ dài" — tránh tạo discontinuity trong phân phối feature.

### 3.2bis Bước trung gian — `ang` (heading angle)

Trước khi sang Channel 2, ta tính `ang` (góc heading của từng đoạn).
`ang` không phải là một channel của 7-feature (vì nó **KHÔNG invariant**),
nhưng là intermediate value cần để tính `Δheading` và curvature.

```python
ang = np.arctan2(d[:,1], d[:,0])
```

#### 3.2bis.1 `np.arctan2(y, x)` khác gì `arctan(y/x)`?

`arctan` thường chỉ trả về góc trong `(-π/2, π/2)` → mất thông tin về 4
góc phần tư. `arctan2(y, x)` nhìn cả dấu của `y` và `x` → trả về góc đầy
đủ trong `(-π, π]`:

| Góc phần tư | x   | y   | arctan2(y, x)       |
|------------:|----:|----:|---------------------|
| I (đông-bắc)| > 0 | > 0 | (0, π/2)            |
| II (tây-bắc)| < 0 | > 0 | (π/2, π)            |
| III (tây-nam)| < 0 | < 0 | (-π, -π/2)         |
| IV (đông-nam)| > 0 | < 0 | (-π/2, 0)          |

#### 3.2bis.2 Tính tay

```
ang[0] = arctan2(0, 1) = 0          (đông)
ang[1] = arctan2(0, 1) = 0          (đông)
ang[2] = arctan2(1, 1) = π/4 ≈ 0.7854   (đông-bắc, 45°)
ang[3] = arctan2(1, 1) = π/4 ≈ 0.7854
```

→ `ang = [0, 0, 0.7854, 0.7854]`, shape `(4,)`.

#### 3.2bis.3 Tại sao `ang` KHÔNG được đưa thẳng vào feature?

Vì nó phụ thuộc **hệ tọa độ tuyệt đối**: nếu xoay road 30°, mọi `ang[i]`
đều shift 30° → leak orientation. Đây chính là điểm baseline 10-ch sai —
nó dùng `sin(ang), cos(ang)` raw. Exp 02 chỉ giữ lại **đạo hàm** của
`ang` (= `dang`), một đại lượng intrinsic.

### 3.3 Channel 2 — `|Δheading|` (magnitude của thay đổi góc)

```python
ang  = np.arctan2(d[:,1], d[:,0])
dang = (np.diff(ang) + np.pi) % (2*np.pi) - np.pi   # wrap về (-π, π]
abs_dang_full = np.pad(np.abs(dang), (1, 1), mode='constant')
```

#### 3.3.1 `np.diff(ang)` làm gì?

Tính `ang[i+1] - ang[i]` — đo "góc cua" tại đỉnh `pts[i+1]`:

```
dang_raw[i] = ang[i+1] - ang[i]    với i = 0, 1, ..., n-3
```

Shape: `(n-1,) → (n-2,)`. Với ví dụ:

```
dang_raw[0] = ang[1] - ang[0] = 0      - 0      = 0
dang_raw[1] = ang[2] - ang[1] = 0.7854 - 0      = 0.7854
dang_raw[2] = ang[3] - ang[2] = 0.7854 - 0.7854 = 0
```

→ `dang_raw = [0, 0.7854, 0]`, shape `(3,)`.

#### 3.3.2 Phép `wrap` `(... + π) % (2π) - π` để làm gì?

`arctan2` trả về góc trong `(-π, π]`. Khi trừ hai góc có thể ra giá trị
ngoài range này, ví dụ: một đoạn có `ang = 170°` rồi đoạn sau `ang =
-170°` → `dang_raw = -340°`, nhưng "thực ra" xe chỉ cua **20°** (đi qua
±180°).

Công thức `(x + π) % (2π) - π` ép `x` về dải `(-π, π]`:

| `dang_raw` | sau wrap         |
|-----------:|------------------|
| `-340°`    | `+20°`           |
| `+200°`    | `-160°`          |
| `+30°`     | `+30°` (giữ)     |
| `-30°`     | `-30°` (giữ)     |

Ở ví dụ của ta `dang_raw ∈ {0, 0.7854}` → đều trong dải, wrap không đổi
giá trị:

```
dang = [0, 0.7854, 0]
```

#### 3.3.3 Tại sao lấy `|dang|` thay vì `dang` raw?

Lý do là chuẩn hóa **chiral**: cua trái và cua phải cùng cường độ phải
được mô hình "đối xử như nhau" về độ khó — chỉ độ lớn mới quan trọng cho
feature này. Còn dấu (trái/phải) đã được giữ trong `k` (signed
curvature) ở channel 3.

```
|dang| = [0, 0.7854, 0]
```

#### 3.3.4 `np.pad(..., (1, 1), mode='constant')` — pad ở đâu?

- `(1, 1)`: thêm **1 phần tử trước** và **1 phần tử sau**.
- `mode='constant'`, default `constant_values=0`: pad bằng `0`.

```
abs_dang_full = [0, 0, 0.7854, 0, 0]
                ↑                  ↑
            pad trước           pad sau
```

Shape: `(n-2,) → (n,) = (5,)`.

#### 3.3.5 Tại sao pad `(1, 1)` cho `dang` nhưng `(0, 1)` cho `seg`?

- `seg` có `n-1` phần tử (mỗi đoạn nối 2 điểm) → pad 1 ở cuối là đủ.
- `dang` có `n-2` phần tử (mỗi "góc cua" cần 3 điểm để xác định) → pad 1
  ở đầu (điểm 0 chưa có đoạn trước) **và** 1 ở cuối (điểm n-1 chưa có
  đoạn sau).

#### 3.3.6 Ý nghĩa

`abs_dang_full[i]` ≈ "tại điểm i, xe phải quay vô-lăng bao nhiêu radian".
Đoạn thẳng → 0; cua gắt → gần π/2. Là một trong những feature **mạnh
nhất** cho dự đoán FAIL — cua gấp dễ làm xe trượt.

#### 3.3.7 Invariance

Khi xoay road bằng `R(θ)`:

```
ang_rot[i] = arctan2((R · d[i])_y, (R · d[i])_x) = ang[i] + θ
```

→ Mọi `ang` shift cùng một hằng `θ` → hiệu `dang_rot[i] = ang_rot[i+1] -
ang_rot[i] = (ang[i+1] + θ) - (ang[i] + θ) = dang[i]`.

→ `|dang|` invariant bit-exact (chỉ lệch ở float roundoff của `arctan2`).

### 3.4 Channel 3 — `k` (signed curvature)

```python
def signed_curvature(pts):
    d = np.diff(pts, axis=0); ang = np.arctan2(d[:,1], d[:,0])
    dang = (np.diff(ang) + np.pi) % (2*np.pi) - np.pi
    seg = np.linalg.norm(d, axis=1)
    denom = 0.5*(seg[:-1] + seg[1:]) + 1e-8
    k = dang / denom
    return np.pad(k, (1, 1), mode='constant')
```

#### 3.4.1 Curvature là gì?

Trong hình học vi phân, curvature `κ(s)` của một đường cong tham số hóa
theo arc-length `s` là **tốc độ thay đổi heading**:

```
κ(s) = dθ(s) / ds
```

trong đó `θ(s)` là heading angle tại vị trí arc-length `s`. Đơn vị: rad
/ meter.

Trên một đường rời rạc (chuỗi điểm), ta xấp xỉ:

```
κ_i ≈ Δθ_i / Δs_i
    = dang[i] / (chiều dài trung bình của 2 đoạn nối quanh đỉnh i+1)
```

#### 3.4.2 `denom = 0.5*(seg[:-1] + seg[1:])` — tại sao trung bình hai đoạn?

Tại đỉnh `pts[i+1]`, có 2 đoạn liên quan: đoạn vào (`seg[i]`) và đoạn ra
(`seg[i+1]`). `Δs` tương ứng với cua tại đỉnh này = trung bình:

```
Δs_i = 0.5 · (seg[i] + seg[i+1])
```

`+ 1e-8` để tránh chia cho 0 nếu hai điểm trùng nhau.

#### 3.4.3 Tính tay

Với `seg = [1, 1, 1.4142, 1.4142]` (chưa pad) và `dang = [0, 0.7854, 0]`:

```
seg[:-1] = [1,      1,      1.4142]
seg[1:]  = [1,      1.4142, 1.4142]
sum      = [2,      2.4142, 2.8284]
denom    = sum/2 = [1.0000, 1.2071, 1.4142]   (+ 1e-8)

k_raw[0] = dang[0] / denom[0] = 0      / 1.0000 = 0.0000
k_raw[1] = dang[1] / denom[1] = 0.7854 / 1.2071 = 0.6506
k_raw[2] = dang[2] / denom[2] = 0      / 1.4142 = 0.0000
```

→ `k_raw = [0, 0.6506, 0]`, shape `(n-2,) = (3,)`.

#### 3.4.4 Pad `(1, 1)` constant 0

Giống `dang`: curvature cần 3 điểm liên tiếp để xác định → mất 1 ở đầu, 1
ở cuối. Pad 0:

```
k = [0, 0, 0.6506, 0, 0]
```

Shape `(n,) = (5,)`.

#### 3.4.5 Tại sao "signed"?

`k` giữ DẤU của `dang` (đã wrap, không lấy `abs`):

- `k > 0` ↔ cua trái (counter-clockwise theo convention `arctan2`).
- `k < 0` ↔ cua phải (clockwise).
- `k = 0` ↔ đoạn thẳng.

Khác với `|dang|` ở channel 2 chỉ giữ magnitude. Cặp `(|dang|, k)` chứa
trọn vẹn thông tin về hướng cua + cường độ — model có thể tự học cua trái
và cua phải có khác nhau về độ khó không.

#### 3.4.6 Invariance

`dang` invariant (đã chứng minh ở 3.3.7), `seg` invariant (3.2.4) → tỷ
số `dang / mean(seg)` invariant. Phép xoay toàn cục không ảnh hưởng
**dấu** của `dang` vì `R(θ)` (rotation thuần — không phải reflection)
bảo toàn orientation của mặt phẳng.

> Cảnh báo: nếu thêm **phép phản chiếu** (mirror) thì `k → -k`. Exp 02
> chỉ chứng minh invariance dưới SO(2) (rotation), không phải O(2). Đây
> là một subtlety đáng đề cập trong paper.

#### 3.4.7 Đơn vị thực tế

Với road thực tế khoảng cách điểm ~5m, một cua đường kính 30m → `κ ≈ 1/15
≈ 0.067 rad/m`. Trong `tracker.md` báo cáo `mean(k) ≈ 0.001`, `std(k) ≈
0.025` → đa số đoạn gần thẳng, một vài đoạn cua gắt là outlier — đó là
chỗ FAIL hay xảy ra.

### 3.5 Channel 4 — `dk/ds` (đạo hàm bậc 1 của curvature)

```python
dk = np.pad(np.diff(k), (0, 1), mode='constant')
```

#### 3.5.1 Ý nghĩa

`dk[i] ≈ k[i+1] - k[i]` đo **tốc độ thay đổi curvature** từ điểm này sang
điểm kế tiếp. Trong robotics/automotive control, đây gần với khái niệm
**sharpness** hay "clothoid rate" — đường cong loại clothoid (Euler
spiral) có `dk/ds = const ≠ 0` và đó là loại đường dễ lái nhất.

- `dk = 0`: curvature giữ nguyên (cung tròn đều, hoặc đoạn thẳng).
- `dk ≠ 0`: curvature thay đổi → xe phải xoay vô-lăng → khó hơn.
- `dk` đổi dấu nhanh: cua hình "S" → khó nhất.

#### 3.5.2 `np.diff(k)` với `k` đã được padded

Lưu ý ở đây `k` đầu vào ĐÃ là `k` đã pad (length `n`). `np.diff` không
nhận `axis` → default `axis=-1` → cho 1D array thì OK (trừ phần tử liên
tiếp).

```
k        = [0,       0,       0.6506,  0,       0      ]    (length 5)
diff(k)  = [0-0,     0.6506-0, 0-0.6506, 0-0    ]
         = [0,       0.6506,  -0.6506,  0       ]            (length 4)
```

#### 3.5.3 Pad `(0, 1)` — chỉ ở cuối

`diff` mất 1 phần tử ở **cuối** (không có "next" cho phần tử cuối cùng).
Để giữ shape `(n,)`, ta pad 1 zero ở cuối:

```
dk = [0, 0.6506, -0.6506, 0, 0]
                              ↑
                          pad cuối
```

Khác với `dang` pad `(1, 1)`: vì `dang` mất 1 ở mỗi đầu (cần 3 điểm), còn
`dk = diff(k_đã_pad)` chỉ mất 1 ở cuối.

#### 3.5.4 Đọc kết quả

| i | k[i]   | dk[i]   | Diễn giải                          |
|---|-------:|--------:|------------------------------------|
| 0 | 0      |  0      | điểm đầu, chưa biết gì             |
| 1 | 0      | +0.6506 | curvature đang tăng (sắp vào cua)  |
| 2 | 0.6506 | -0.6506 | đỉnh cua, curvature đang giảm      |
| 3 | 0      |  0      | đã ra khỏi cua                     |
| 4 | 0      |  0      | đoạn thẳng                         |

→ Ở ví dụ này có một **xung curvature** (spike): tăng vọt rồi giảm vọt
ngay sau — đặc trưng của khúc cua đơn ở giữa road.

#### 3.5.5 Invariance

`k` đã invariant (3.4.6), `diff` là phép tuyến tính trên một vector
invariant → `dk` cũng invariant. Tổng quát: **mọi phép toán không liên
quan đến hệ tọa độ** áp lên một invariant feature sẽ ra invariant
feature.

### 3.6 Channel 5 — `d²k/ds²` (đạo hàm bậc 2)

```python
ddk = np.pad(np.diff(dk), (0, 1), mode='constant')
```

#### 3.6.1 Ý nghĩa vật lý

Trong kinematics, nếu coi `s` là thời gian (đi với vận tốc đơn vị), thì:

| Đại lượng | Tên chuẩn        | Ý nghĩa trong driving         |
|-----------|------------------|-------------------------------|
| `k`       | curvature        | tay lái đang xoay bao nhiêu   |
| `dk/ds`   | curvature rate   | tốc độ xoay tay lái           |
| `d²k/ds²` | curvature jerk   | gia tốc xoay tay lái          |

→ `ddk` lớn tương ứng với "giật tay lái" — không phải con người và cũng
không phải auto-pilot êm. Đây là chỉ báo cực mạnh cho **road test nhân
tạo có hình dạng phi vật lý** (loại road sinh ngẫu nhiên thường có
`ddk` lớn).

#### 3.6.2 Tính tay

Áp `np.diff` lên `dk = [0, 0.6506, -0.6506, 0, 0]`:

```
diff(dk)[0] = dk[1] - dk[0] = 0.6506 - 0       = +0.6506
diff(dk)[1] = dk[2] - dk[1] = -0.6506 - 0.6506 = -1.3012
diff(dk)[2] = dk[3] - dk[2] = 0 - (-0.6506)    = +0.6506
diff(dk)[3] = dk[4] - dk[3] = 0 - 0            = 0
```

→ `diff(dk) = [+0.6506, -1.3012, +0.6506, 0]`, length 4.

Pad `(0, 1)` zero ở cuối:

```
ddk = [+0.6506, -1.3012, +0.6506, 0, 0]
```

#### 3.6.3 Đọc tín hiệu

| i | k[i]   | dk[i]   | ddk[i]  | Diễn giải                              |
|---|-------:|--------:|--------:|----------------------------------------|
| 0 | 0      |  0      | +0.6506 | sắp có "burst" curvature ngay phía trước|
| 1 | 0      | +0.6506 | -1.3012 | đang ở giữa burst (chuyển +/-)         |
| 2 | 0.6506 | -0.6506 | +0.6506 | sau burst, trở về 0                    |
| 3 | 0      |  0      | 0       | yên                                    |
| 4 | 0      |  0      | 0       | yên                                    |

→ `ddk` biểu hiện sự kiện "spike" rõ hơn cả `k` và `dk` — nó là một
**band-pass filter** quanh tần số cao của tín hiệu curvature dọc theo
arc-length.

#### 3.6.4 Tại sao dừng ở bậc 2 mà không bậc 3, 4, ...?

- Bậc 1 đã tăng được signal-to-noise (đạo hàm là phép khuếch đại tần số
  cao).
- Bậc 2 đã đủ để phát hiện spike và đảo chiều.
- Bậc cao hơn → khuếch đại nhiễu (noise floor của các điểm rời rạc) lấn
  át tín hiệu thật → diminishing returns.

Đây là một thiết kế kinh nghiệm: 3 mức (`k, dk, ddk`) đủ rộng để
Transformer-attention có "đại diện đa thang" của curvature.

#### 3.6.5 Invariance

Cùng lý do với `dk` (3.5.5): `dk` invariant → `diff(dk)` invariant →
`ddk` invariant.

### 3.7 Channel 6 — `s/L` (arc-length chuẩn hóa)

```python
s_cum = np.cumsum(seg_full)
s_norm = s_cum / (s_cum[-1] + 1e-8)
```

#### 3.7.1 `np.cumsum` làm gì?

Tổng dồn (cumulative sum): `s_cum[i] = seg_full[0] + seg_full[1] + ... +
seg_full[i]`. Đây là **arc-length tổng** từ đầu road đến điểm `i` (xấp xỉ
rời rạc).

#### 3.7.2 Tính tay

Với `seg_full = [1.0000, 1.0000, 1.4142, 1.4142, 1.4142]`:

```
s_cum[0] = 1.0000
s_cum[1] = 1.0000 + 1.0000 = 2.0000
s_cum[2] = 2.0000 + 1.4142 = 3.4142
s_cum[3] = 3.4142 + 1.4142 = 4.8284
s_cum[4] = 4.8284 + 1.4142 = 6.2426
```

→ `s_cum = [1.0000, 2.0000, 3.4142, 4.8284, 6.2426]`. Tổng chiều dài
road `L = s_cum[-1] = 6.2426`.

#### 3.7.3 Chia cho `s_cum[-1] + 1e-8`

Normalize về `[0, 1]` (xấp xỉ, vì điểm đầu không tính từ 0 mà từ `seg[0]`
do convention `cumsum` bắt đầu cộng ngay tại i=0):

```
s_norm[0] = 1.0000 / 6.2426 = 0.1602
s_norm[1] = 2.0000 / 6.2426 = 0.3204
s_norm[2] = 3.4142 / 6.2426 = 0.5470
s_norm[3] = 4.8284 / 6.2426 = 0.7735
s_norm[4] = 6.2426 / 6.2426 = 1.0000
```

→ `s_norm = [0.1602, 0.3204, 0.5470, 0.7735, 1.0000]`.

> Nhận xét: điểm `i=0` không bằng đúng 0 mà bằng `seg[0]/L ≈ 0.16`.
> Trong thực tế road dài (~50 điểm), `s_norm[0] ≈ 0.02` — gần 0. Nếu cần
> bắt đầu chính xác 0, có thể `np.insert(s_cum, 0, 0)` rồi cắt phần đầu —
> nhưng exp 02 không làm vì sai lệch ~0.02 không ảnh hưởng kết quả.

#### 3.7.4 `+ 1e-8` để làm gì?

Tránh `div by zero` nếu toàn bộ road có 0 độ dài (degenerate case — mọi
điểm trùng nhau). Trên dữ liệu thật điều này không xảy ra, nhưng đây là
defensive programming.

#### 3.7.5 Tại sao channel này có giá trị?

`s_norm` là "ID vị trí dọc road" — Transformer cần nó để biết "đoạn này
gần đầu hay gần cuối". Quan trọng:

1. **Parameterization invariance**: nếu resample road với mật độ điểm
   khác (giữ nguyên hình dạng), `s_norm` vẫn nằm trong `[0, 1]` —> mô
   hình thấy đoạn ở giữa road luôn ở `s ≈ 0.5`, dù road có 50 hay 197
   điểm.

2. **Input cho relative-bias trong attention**: trong `InvariantBlock`,
   `s_norm` được lấy ra ở [exp02_SE2Equivariant.py:162](exp02_SE2Equivariant.py#L162):
   ```python
   s_norm = x[..., 5]   # channel 6 (0-indexed: index 5)
   ```
   rồi dùng để tính `Δs = s_i - s_j` cho attention bias (mục 5.4).

#### 3.7.6 Invariance

`seg` invariant (3.2.4) → `cumsum` (phép tuyến tính) invariant → chia
cho `s_cum[-1]` (cũng invariant vì là tổng) → `s_norm` invariant.

Hơn nữa, `s_norm` còn invariant với **reparameterization của road bằng
phép tịnh tiến trong `s`** (nếu shift gốc arc-length): khi tính `Δs = s_i
- s_j` trong attention, hằng số shift triệt tiêu. Đây là **invariance
mạnh hơn** SO(2) — gọi là parameterization-by-shift invariance.

### 3.8 Channel 7 — `lstd` (local std của curvature)

```python
w = 11; hw = 5
lstd = np.zeros(n)
for i in range(n):
    a, b = max(0, i-hw), min(n, i+hw+1)
    lstd[i] = np.std(k[a:b])
```

#### 3.8.1 Ý nghĩa

`lstd[i]` = **độ dao động** của curvature trong cửa sổ ±5 điểm quanh `i`.
Cung cấp một "view tần số cao" khác với `dk/ddk`:

- `dk, ddk`: thay đổi tức thời tại một điểm (point-wise derivative).
- `lstd`: tổng hợp dao động trong vùng (window-wise variability).

→ Hai loại signal bổ trợ nhau. Đoạn có `lstd` lớn = "vùng có nhiều cua
gần nhau" hoặc "vùng curvature giật" — heuristic strong cho FAIL.

#### 3.8.2 Cửa sổ trượt (sliding window) bằng vòng for

```python
w = 11; hw = w // 2 = 5
for i in range(n):
    a = max(0, i - hw)          # cận trái, clamp về 0
    b = min(n, i + hw + 1)      # cận phải, clamp về n
    lstd[i] = np.std(k[a:b])    # std trong cửa sổ
```

- `w = 11`: tổng kích thước cửa sổ (11 điểm).
- `hw = 5`: half-width.
- Tại biên (i gần 0 hoặc gần n-1), cửa sổ bị **truncate** — chỉ tính std
  trên những điểm có thật. Không pad.

Đây là cách viết "thật thà" (không vectorize) — đủ nhanh cho `n ≤ 200`.
Có thể vectorize bằng `np.lib.stride_tricks.sliding_window_view` nhưng
mã sẽ khó đọc hơn.

#### 3.8.3 Tính tay cho road ví dụ (n=5)

Vì `n=5 < w=11`, mọi cửa sổ đều bị clamp về toàn bộ mảng: `k[0:5] = [0,
0, 0.6506, 0, 0]`. Tính std:

```
Bước 1: mean = (0 + 0 + 0.6506 + 0 + 0) / 5 = 0.13012

Bước 2: variance (NumPy mặc định ddof=0 — chia n, không phải n-1)
        var = mean( (k[i] - mean)^2 )
            = [ (0-0.13012)^2 × 4 + (0.6506-0.13012)^2 × 1 ] / 5
            = [ 0.01693 × 4   + 0.27092             ] / 5
            = [ 0.06772       + 0.27092             ] / 5
            = 0.33864 / 5
            = 0.06773

Bước 3: std = sqrt(0.06773) = 0.26025 ≈ 0.2602
```

→ `lstd = [0.2602, 0.2602, 0.2602, 0.2602, 0.2602]`.

#### 3.8.4 Trên road thực tế: lstd KHÔNG đều

Với `n = 100` chẳng hạn, mỗi `i` có cửa sổ 11 điểm khác nhau → `lstd[i]`
khác nhau. Đoạn thẳng có `lstd ≈ 0`, đoạn có nhiều cua liên tiếp có
`lstd` lớn.

Ví dụ minh họa (giả lập với road có cua ở giữa):

```
i (vị trí dọc road):  0    10   20   30   40   50   60   70   80   90
lstd[i]:              0.01 0.02 0.05 0.18 0.42 0.51 0.39 0.22 0.07 0.03
                                            ↑
                                       đỉnh "vùng cua"
```

#### 3.8.5 Invariance

`k` invariant → mọi cửa sổ con của `k` invariant → `np.std` (chỉ phụ
thuộc các giá trị, không phụ thuộc thứ tự không gian) trên cửa sổ con
invariant → `lstd` invariant.

#### 3.8.6 Tại sao chọn `w = 11`?

Heuristic: trung bình mỗi cua chiếm ~5-10 điểm trong dữ liệu SensoDat
(với mật độ điểm ~5m/point). `w = 11` ~ vừa đủ ôm một cua hoàn chỉnh.
Không có ablation chính thức cho hyperparam này; đây là một point đáng
đề cập trong limitations của paper.

</details>

### 3.9 Kết quả: ma trận 5 × 7

```
       seg     |dang|   k        dk       ddk      s/L     lstd
i=0  [ 1.0000  0.0000   0.0000   0.0000   0.6506   0.1602  0.2602 ]
i=1  [ 1.0000  0.0000   0.0000   0.6506  -1.3012   0.3204  0.2602 ]
i=2  [ 1.4142  0.7854   0.6506  -0.6506   0.6506   0.5470  0.2602 ]
i=3  [ 1.4142  0.0000   0.0000   0.0000   0.0000   0.7735  0.2602 ]
i=4  [ 1.4142  0.0000   0.0000   0.0000   0.0000   1.0000  0.2602 ]
```

**Đây là output của `extract_invariant_7ch(pts)` — shape `(5, 7)`, dtype
float32.**

> **Nhận xét then chốt — invariance bằng construction:** mọi feature ở đây
> chỉ phụ thuộc vào `||d||` (seg), `arctan2(dy, dx)` chênh lệch (dang) và
> các đạo hàm dọc theo arc-length. KHÔNG có x, y tuyệt đối. KHÔNG có
> `sin(ang)`, `cos(ang)` raw. Phép xoay `R` chuyển `d -> R·d`:
> - `||R·d|| = ||d||` (Euclidean norm bất biến)
> - `arctan2((R·d_2)_y, (R·d_2)_x) - arctan2((R·d_1)_y, (R·d_1)_x)` = `Δang` (chênh lệch góc trong cùng một frame xoay → triệt tiêu)
>
> Nên feature **bằng bit-exact** sau khi xoay (sai số chỉ do floating
> point arctan2). Đó là cơ sở của `Δ = 0.0000` trong rotation probe.

---

## 4. Bước 2 — Chuẩn hóa (mean/std)

Code [exp02_SE2Equivariant.py:312-315](exp02_SE2Equivariant.py#L312-L315):

```python
X_tr, y_tr = prepare_data(train_data)         # (N_train, L, 7)
X_te, y_te = prepare_data(test_data)          # (N_test, L, 7)
means = X_tr.mean(axis=(0,1))                  # (7,) — global
stds  = X_tr.std(axis=(0,1))
stds[stds < 1e-8] = 1.0
X_tr = (X_tr - means) / stds
X_te = (X_te - means) / stds
```

Giả sử `means ≈ [3.5, 0.04, 0.001, 0.0, 0.0, 0.5, 0.02]` và
`stds ≈ [1.2, 0.06, 0.025, 0.012, 0.020, 0.29, 0.015]` (số minh họa, không
phải số thật).

Áp lên row đầu tiên của road ví dụ:

| feat   | raw    | mean  | std   | normalized      |
|--------|-------:|------:|------:|----------------:|
| seg    | 1.0000 | 3.50  | 1.20  | (1.00-3.50)/1.20 = -2.083 |
| dang   | 0.0000 | 0.04  | 0.06  | -0.667          |
| k      | 0.0000 | 0.001 | 0.025 | -0.040          |
| dk     | 0.0000 | 0.0   | 0.012 | 0.000           |
| ddk    | 0.6506 | 0.0   | 0.020 | 32.530          |
| s/L    | 0.1602 | 0.5   | 0.29  | -1.172          |
| lstd   | 0.2602 | 0.02  | 0.015 | 16.013          |

> Lưu ý: con đường ví dụ rất ngắn nên `seg` và `s/L` lệch nhiều so với
> mean tập huấn (mean ~3.5 m). Trong thực tế các giá trị này gần 0.

`means` và `stds` được lưu vào checkpoint cùng model
([exp02_SE2Equivariant.py:333-336](exp02_SE2Equivariant.py#L333-L336)) để
inference có thể tái dùng đúng normalization.

---

## 5. Bước 3 — Kiến trúc `SE2RoadNet`

Code [exp02_SE2Equivariant.py:149-167](exp02_SE2Equivariant.py#L149-L167).

### 5.1 Pipeline cao nhất

```
Input  : x  shape (B, 7, L)           ← C-major
   │
   │ permute(0,2,1)
   ▼
       (B, L, 7)
   │
   │ Linear(7 → 192) + LayerNorm + GELU         "proj"
   ▼
       (B, L, 192)
   │
   │ Prepend learnable CLS token  (1, 1, 192)
   ▼
       (B, L+1, 192)                            row 0 = CLS
   │
   │ 6 × InvariantBlock(d=192, heads=8, ff=512)
   ▼
       (B, L+1, 192)
   │
   │ Take CLS row → LayerNorm → Linear(192→64) → GELU
   │ → Dropout(0.2) → Linear(64→1)              "head"
   ▼
       (B,)   logits
```

### 5.2 Tham số ước lượng (~2.1M)

| Khối              | Params                        |
|-------------------|-------------------------------:|
| proj (7→192)      | 7·192 + 192 + LN(384)         | ≈ 2,100  |
| CLS token         | 192                            |
| 6 × InvariantBlock| 6 × ~350k                      | ≈ 2.1M   |
| head              | LN + 192·64 + 64·1            | ≈ 13k    |
| **Tổng**          |                                | **≈ 2.1M** |

Nhỏ hơn baseline 10-ch Transformer (~3M) một chút.

### 5.3 Phân tích `InvariantBlock`

Code [exp02_SE2Equivariant.py:114-147](exp02_SE2Equivariant.py#L114-L147).

```
h_in (B, L+1, 192)
   │
   │ + (Self-Attention với attn_mask = bias(Δs))
   │    bias depends ONLY on relative arc-length (s_i - s_j)
   ▼
h_after_attn = h_in + dropout( Attn(LN(h_in)) )
   │
   │ + FeedForward
   ▼
h_out = h_after_attn + dropout( FF(LN(h_after_attn)) )
```

**Điểm mấu chốt:** `attn_mask` không phải mask kiểu causal (0/-∞) mà là một
**bias số thực** cho từng cặp `(i, j)`:

```
bias_h[i, j] = MLP( sin( (s_i - s_j) · ω ) )         h = head index
```

Trong đó `ω` là một bộ Random Fourier Features (32 tần số), được fix lúc
init và không train. `MLP` là 32→64→nhead.

→ Attention score giữa token i và j được cộng thêm `bias_h[i,j]`, làm hệ
relative-position-aware mà KHÔNG cần biết i hay j tuyệt đối — chỉ cần
khoảng cách `s_i - s_j` dọc theo arc-length.

### 5.4 Minh họa numerical: tính `bias` cho road 5 điểm

Với `s_norm = [0.1602, 0.3204, 0.5470, 0.7735, 1.0000]`, ma trận
`Δs = s_i - s_j` shape (L, L) = (5, 5):

```
        j=0      j=1      j=2      j=3      j=4
i=0  [ 0.0000  -0.1602  -0.3868  -0.6133  -0.8398 ]
i=1  [ 0.1602   0.0000  -0.2266  -0.4531  -0.6796 ]
i=2  [ 0.3868   0.2266   0.0000  -0.2265  -0.4530 ]
i=3  [ 0.6133   0.4531   0.2265   0.0000  -0.2265 ]
i=4  [ 0.8398   0.6796   0.4530   0.2265   0.0000 ]
```

Sau khi cộng CLS row (s = 0 ở đầu), ma trận thành `(6, 6)`. Mỗi entry được
nhân với `ω` (1, 32), qua `sin`, rồi đi qua MLP 32→64→8 (8 heads), cuối
cùng `permute` thành `(B, 8, 6, 6)`. Đây chính là `attn_mask` được đưa vào
`nn.MultiheadAttention(... batch_first=True)`.

> Tại sao Random Fourier Features? Vì một MLP học `f(Δs)` trên scalar 1D
> hội tụ chậm; chiếu Δs lên `{sin(ω_k · Δs)}_k` (32 tần số khác nhau) tạo
> ra một feature space giàu hơn, MLP học hàm phi tuyến trên đó dễ hơn.

### 5.5 Forward minh họa với road 5 điểm

Với batch `B=1`, `L=5`, `C=7`:

```
Input   : (1, 7, 5)
permute : (1, 5, 7)
proj    : (1, 5, 7) → Linear(7→192) → LayerNorm → GELU → (1, 5, 192)
+ CLS   : (1, 6, 192)         row 0 = CLS token (learnable)

s_full = [0, 0.1602, 0.3204, 0.5470, 0.7735, 1.0000]   ← shape (1, 6)
bias   = (1, 8, 6, 6)                                  ← từ s_full

for block in 6 blocks:
    z = LayerNorm(h)               (1, 6, 192)
    a, _ = MultiheadAttention(z, z, z, attn_mask=bias.reshape(8, 6, 6))
    h = h + Dropout(a)
    h = h + Dropout(FF(LayerNorm(h)))

cls_out = h[:, 0, :]               (1, 192)
logit   = head(cls_out)            (1,)  ← scalar
```

---

## 6. Bước 4 — Training loop

Code [exp02_SE2Equivariant.py:202-253](exp02_SE2Equivariant.py#L202-L253).

### 6.1 Loss: Focal BCE với pos_weight

```python
pw = (N - n_pos) / n_pos
crit = FocalLoss(gamma=1.5, pos_weight=pw)
```

Với SensoDat tỉ lệ FAIL ~30%: `pw ≈ 0.7/0.3 ≈ 2.33`.

Focal loss
([exp02_SE2Equivariant.py:170-177](exp02_SE2Equivariant.py#L170-L177)):

```
bce = BCEWithLogits(logit, y)
w   = pw   if y == 1 else 1.0
pt  = σ(logit)        if y == 1 else 1 - σ(logit)
loss = mean( (1 - pt)^γ · w · bce )
```

Ý nghĩa:
- `pos_weight` (pw=2.33): nhấn mạnh sample FAIL.
- `(1-pt)^γ` với γ=1.5: down-weight các sample đã được dự đoán đúng (pt
  gần 1), tập trung gradient vào hard examples.

### 6.2 Weighted sampler

```python
weights = np.where(y_tr == 1, pw, 1.0)
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

→ Mỗi epoch, FAIL sample được chọn với xác suất cao hơn (do `pw=2.33`),
giúp batch balanced hơn.

### 6.3 LR schedule: warmup + cosine

```python
warm = 5
sched = LambdaLR(opt, lambda e: (e+1)/warm if e<warm
    else max(0.01, 0.5*(1 + cos(π·(e-warm) / max(1, epochs-warm)))))
```

```
LR factor
1.0 +              .---._
    |          .---'      `--.
    |       .--                `--.
0.5 |    .--                       `--.
    |  .'                              `-.
    |.'                                   `.
0.0 +.________________________________________ epoch
    0   5                                  80
    warmup    ←  cosine decay  →
```

### 6.4 Mixed precision + grad clipping

```python
scaler = GradScaler(enabled=(not USE_BF16))
...
with autocast(dtype=AMP_DTYPE):
    loss = crit(model(xb), yb)
if USE_BF16:
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
else:                                   # fp16 needs scaler
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
```

bf16 (Ampere+) không cần scaler vì dynamic range đủ rộng; fp16 cần
GradScaler để tránh underflow.

### 6.5 SWA (Stochastic Weight Averaging)

```python
swa_start = 55
if ep >= swa_start:
    if swa is None: swa = SWAModel(model)
    else: swa.update(model)
```

SWA logic
([exp02_SE2Equivariant.py:179-185](exp02_SE2Equivariant.py#L179-L185)):

```python
def update(self, m):
    self.n += 1; a = 1.0/self.n
    for p, q in zip(self.model.parameters(), m.parameters()):
        p.data.mul_(1-a).add_(q.data, alpha=a)
```

→ Trung bình running từ epoch 56 trở đi. Model SWA cuối thường ổn định
hơn → giảm σ giữa các trial, hữu ích cho APFD ổn định (xem `tracker.md`).

### 6.6 Một epoch điển hình trông như thế nào

Output console (giả định, sample):

```
============================================================
Training SE2RoadNet | params=2,108,737
============================================================
  Ep   1 | loss=0.4823 | AUC=0.7891 | best=0.7891 *
  Ep   2 | loss=0.4156 | AUC=0.8202 | best=0.8202 *
  ...
  Ep  20 | loss=0.3104 | AUC=0.9123 | best=0.9241
  Ep  25 | loss=0.2987 | AUC=0.9286 | best=0.9286 *
  ...
  [SWA] start @ epoch 56
  Ep  56 | loss=0.2674 | AUC=0.9305 | best=0.9351
  ...
  Ep  80 | loss=0.2641 | AUC=0.9358 | best=0.9385 *
```

> Lưu ý: **best checkpoint** lưu theo AUC, không theo APFD. Đây là một
> điểm exp 02 cố tình giữ giống baseline để so sánh fair. AUC dễ optimize
> liên tục hơn APFD (mượt hơn), nhưng cuối cùng cả hai thường correlate.

---

## 7. Bước 5 — Inference & APFD

### 7.1 Tính probability score

Code [exp02_SE2Equivariant.py:273-284](exp02_SE2Equivariant.py#L273-L284):

```python
feats = _feats(data, means, stds, rot_deg=0.0)
X = torch.tensor(feats, dtype=torch.float32).permute(0, 2, 1)  # (N, 7, L)
logit = predict_chunked(model, X, chunk=128)
p = 1.0 / (1.0 + np.exp(-logit))               # sigmoid → [0, 1]
pids = [t for _, t in sorted(zip(p, ids), key=lambda z: -z[0])]
```

→ Sắp xếp test theo `p` giảm dần. Test có score cao nhất chạy đầu tiên.

### 7.2 Công thức APFD

Code [exp02_SE2Equivariant.py:256-259](exp02_SE2Equivariant.py#L256-L259):

```python
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i+1 for i, t in enumerate(pids)
          if td[t]['meta_data']['test_info']['test_outcome'] == 'FAIL']
    m = len(fp)
    return 1 - sum(fp)/(n*m) + 1/(2*n)
```

Công thức chuẩn (Rothermel et al., 1999):

```
APFD = 1 - (TF_1 + TF_2 + ... + TF_m) / (n · m) + 1/(2n)
```

trong đó `TF_i` = vị trí (1-indexed) của fail thứ i trong thứ tự, `n` =
tổng test, `m` = số fail.

### 7.3 Worked example APFD

Giả sử ta có 5 test, ID = ["A","B","C","D","E"], outcome = [FAIL, PASS,
FAIL, PASS, FAIL] (m=3, n=5).

Nếu predicted scores `p = [0.91, 0.12, 0.88, 0.30, 0.45]`:

- Sort desc: A(0.91) → C(0.88) → E(0.45) → D(0.30) → B(0.12)
- pids = ["A", "C", "E", "D", "B"]
- Vị trí FAIL: A@1, C@2, E@3 → TF = [1, 2, 3]
- APFD = 1 - (1+2+3)/(5·3) + 1/(2·5)
       = 1 - 6/15 + 0.1
       = 1 - 0.4 + 0.1 = **0.700**

Ideal (FAIL ngay đầu): TF=[1,2,3] đó CHÍNH LÀ best case → APFD = 0.700.
Random: APFD ≈ 0.5. Worst case (FAIL cuối): APFD ≈ 0.3.

> Với SensoDat thực tế (n≈287 trial sample, m≈86 FAIL),
> **best-single APFD = 0.8066 ± 0.0124** (Transformer + SWA + Focal γ=2.5)
> theo tracker chính.

---

## 8. Bước 6 — Rotation Invariance Probe (chứng minh `Δ = 0.0000`)

Đây là **đóng góp lý thuyết then chốt của Exp 02**.

### 8.1 Code

[exp02_SE2Equivariant.py:261-271, 326-329](exp02_SE2Equivariant.py#L261-L271):

```python
def _feats(data, means, stds, rot_deg=0.0):
    out = []
    if rot_deg == 0.0:
        for tc in data:
            out.append((extract_invariant_7ch(get_pts(tc)) - means) / stds)
    else:
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        for tc in data:
            pts = np.array(get_pts(tc), dtype=np.float64) @ R.T
            out.append((extract_invariant_7ch(pts.tolist()) - means) / stds)
    return np.array(out)

# in main():
for rot in [0.0, 30.0, 60.0, 90.0, 180.0, -45.0]:
    eval_apfd(comp_data, m_eval, means, stds, 'SE2 comp', rot_deg=rot)
```

### 8.2 Kiểm chứng tay với road 5 điểm + R(30°)

```
R(30°) = [[cos 30°, -sin 30°],     ≈ [[ 0.866, -0.500],
          [sin 30°,  cos 30°]]         [ 0.500,  0.866]]
```

Áp R lên 5 điểm gốc (`pts @ R.T` ↔ xoay quanh gốc tọa độ 30° ngược chiều
kim đồng hồ):

| i | pts gốc  | pts xoay (≈)            |
|---|----------|-------------------------|
| 0 | (0, 0)   | (0.000,  0.000)         |
| 1 | (1, 0)   | (0.866,  0.500)         |
| 2 | (2, 0)   | (1.732,  1.000)         |
| 3 | (3, 1)   | (2.098,  2.366)         |
| 4 | (4, 2)   | (2.464,  3.732)         |

Tính lại `d`, `seg`:

| i | d_rot               | ‖d_rot‖            |
|---|---------------------|--------------------|
| 0 | (0.866, 0.500)      | √(0.75+0.25)=1.000 |
| 1 | (0.866, 0.500)      | 1.000              |
| 2 | (0.366, 1.366)      | √(0.134+1.866)=1.4142 |
| 3 | (0.366, 1.366)      | 1.4142             |

→ `seg_rot = [1, 1, 1.4142, 1.4142]` **bằng `seg` gốc bit-exact** (chỉ lệch
ở float roundoff).

Tính `ang`:

```
arctan2(0.500, 0.866) = 30°    = π/6     ≈ 0.5236
arctan2(0.500, 0.866) = 30°    = π/6     ≈ 0.5236
arctan2(1.366, 0.366) = 75°    = 5π/12   ≈ 1.3090
arctan2(1.366, 0.366) = 75°    = 5π/12   ≈ 1.3090
```

→ `dang_rot = [0, 1.3090 - 0.5236, 0] = [0, 0.7854, 0]` **bằng `dang` gốc**.

Hệ quả: `k_rot, dk_rot, ddk_rot, s_norm_rot, lstd_rot` đều **identical** với
gốc → ma trận feature (5, 7) **giống hệt** sau xoay.

Vì model là một hàm thuần feed-forward trên feature → `logit_rot = logit_gốc`
→ ranking → APFD **identical**.

### 8.3 Output console mong đợi

```
--- ROTATION-INVARIANCE PROBE (single-pass APFD) ---
  SE2 comp                                  APFD=0.7843 [rot=+0°]
  SE2 comp                                  APFD=0.7843 [rot=+30°]
  SE2 comp                                  APFD=0.7843 [rot=+60°]
  SE2 comp                                  APFD=0.7843 [rot=+90°]
  SE2 comp                                  APFD=0.7843 [rot=+180°]
  SE2 comp                                  APFD=0.7843 [rot=-45°]
```

→ `Δ APFD = max - min ≈ 0.0000` (chính xác đến float32 roundoff).

So sánh với baseline 10-ch (có `sin(ang), cos(ang)`):

```
  Baseline comp                             APFD=0.7891 [rot=+0°]
  Baseline comp                             APFD=0.7234 [rot=+30°]    ← drop 6.6pp
  Baseline comp                             APFD=0.7102 [rot=+60°]    ← drop 7.9pp
  ...
```

Đây là minh chứng định lượng: invariance "by construction" > invariance
"have to learn".

---

## 9. Bước 7 — Multi-trial APFD trên competition set

Code [exp02_SE2Equivariant.py:286-299](exp02_SE2Equivariant.py#L286-L299):

```python
def multi_trial(data, model, means, stds, name='', n_trials=30, rot_deg=0.0):
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(data))
        ed  = [data[i] for i in idx[334:334+287]]    # 287-sample slice
        ...
        apfds.append(compute_apfd(pids, td))
    print(f"APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f}")
```

- 30 trial, mỗi trial sample 287 test (≈ 30% của 957).
- Seed 42, 43, ..., 71 để reproducible.
- Báo cáo `mean ± σ` — `σ` thường quan trọng hơn `mean` cho publication
  (ổn định giữa các sample sub-set).

---

## 10. Save & checkpoint format

Code [exp02_SE2Equivariant.py:333-337](exp02_SE2Equivariant.py#L333-L337):

```python
save = os.path.join(OUTPUT_DIR, 'roadse2.pt')
torch.save({
    'state': (swa.get_model() if swa else model).state_dict(),
    'means': means.tolist(),
    'stds':  stds.tolist(),
    'arch':  dict(d_model=192, depth=6, nhead=8),
}, save)
```

Để load lại:

```python
ckpt = torch.load('roadse2.pt')
model = SE2RoadNet(in_ch=7, **ckpt['arch'])
model.load_state_dict(ckpt['state'])
means = np.array(ckpt['means']); stds = np.array(ckpt['stds'])
```

---

## 11. Tổng kết end-to-end

| Bước | Input → Output | Shape | Ý nghĩa |
|------|----------------|-------|---------|
| 0 | JSON → list[dict] | — | Load raw |
| 1 | road_points → 7-ch features | (L, 7) | Strip frame info |
| 2 | features → normalized | (L, 7) | Zero-mean, unit-σ |
| 3 | features → projected | (L, 192) | Lift to model dim |
| 4 | +CLS, run 6 InvariantBlocks | (L+1, 192) | Self-attention với relative-s bias |
| 5 | head(CLS) | scalar | Logit |
| 6 | sigmoid(logit) | [0, 1] | Probability |
| 7 | sort by p desc | ranking | Prioritization |
| 8 | compute_apfd | scalar | Effectiveness metric |
| 9 | rotation probe | Δ ≈ 0 | Theoretical correctness |

---

## 12. Mối liên hệ với câu chuyện ICSE 2027

Theo `CLAUDE.md`:

- Exp 02 là **một trong 3 trụ lý thuyết** (cùng FNO=resolution invariance,
  PINN=curvature monotonicity).
- Đóng góp: chứng minh được `Δ = 0.0000` cho rotation (vs ~4-7pp drop của
  baseline).
- Đóng góp APFD không phải là điểm bán: SE(2) thường tương đương hoặc kém
  hơn 0.5-1pp so với baseline 10-ch trên SensoDat (vì baseline học được
  rotation invariance gần đúng từ data); cái Exp 02 đổi lại là **đảm bảo
  toán học** và một **slot bài luận trong story**.

Đó là lý do trong `tracker.md`, Exp 02 được liệt kê với note "exact
rotation invariance" hơn là vị trí leaderboard.

---

## 13. Những điểm dễ confuse

1. **Tại sao có `padding (1,1)` cho `abs_dang_full` nhưng `(0,1)` cho `dk`?**
   - `dang = diff(ang)` mất 1 phần tử ở mỗi đầu (n_pts → n-1 → n-2). Pad
     `(1,1)` để về lại n.
   - `dk = diff(k)` mất 1 ở cuối (k đã có length n). Pad `(0,1)`.

2. **`attn_mask` (B*nhead, L, L) trong nn.MultiheadAttention là bias hay mask?**
   - Pytorch hỗ trợ cả hai. Khi `attn_mask` là float, nó được CỘNG vào
     attention score (trước softmax) → đây là **additive bias**, không
     phải hard mask. Vì vậy ta có thể dùng nó để inject relative-position
     bias.

3. **Tại sao CLS token được gán `s = 0` trong relative bias?**
   - CLS không có vị trí vật lý. Gán 0 nghĩa là bias từ CLS đến token i là
     `MLP(sin(s_i · ω))`, một function của arc-length tuyệt đối của i — vẫn
     invariant với reparameterization-by-shift của s (vì các block khác chỉ
     dùng `s_i - s_j`).

4. **Khi nào predicted score "ổn" để hiển thị APFD tốt?**
   - Nếu các FAIL có score cao hơn các PASS (gọi là "ranking margin") thì
     APFD cao. Không cần phân loại đúng tuyệt đối — chỉ cần ranking.

---

## 14. Cheatsheet để chạy

```powershell
# Local (Windows + PowerShell)
cd D:\AI_RESEARCH\Software_Reseach\sdc-test-prioritization-novel\exps
python exp02_SE2Equivariant.py
# Output: ../models/roadse2.pt

# Kaggle: paste toàn bộ file vào notebook, attach dataset
#         /kaggle/input/datasets/chinguyeen/sdc-sensodat
#         GPU: T4 / P100 / A100 đều OK (bf16 dùng được trên A100/H100)
```

Thời gian chạy ước tính (single A100, batch 384, 80 epochs):
- Feature extraction: ~30s cho 5600 tests
- Training: ~6-8 phút
- Eval + rotation probe: ~30s
- **Tổng: ~7-9 phút**

---

*File này được tạo cho mục đích pedagogical — cho phép một người đọc code
exp02 lần đầu nắm được toàn bộ pipeline mà không cần đọc literature về
SE(2) equivariance trước. Mọi số tay đều có thể verify bằng cách chạy lại
`extract_invariant_7ch([[0,0],[1,0],[2,0],[3,1],[4,2]])` trong REPL.*
