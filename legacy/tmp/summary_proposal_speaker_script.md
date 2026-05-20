# Script Thuyết Trình - Tóm Tắt Đề Xuất

## Nghiên Cứu: Theory-Driven SDC Test Prioritization
### Biên soạn: Đào Sỹ Duy Minh, Trần Chí Nguyên, Huỳnh Trung Kiệt
### Đại học Khoa học Tự nhiên, ĐHQG-HCM | NeurIPS 2026

---

## Slide 1: Tiêu Đề

**Mở đầu (30 giây)**

Xin chào các thầy cô và các bạn. Hôm nay chúng tôi sẽ trình bày về nghiên cứu **"Theory-Driven SDC Test Prioritization"** - tức Ưu tiên hóa kiểm thử dựa trên lý thuyết cho các bộ mô phỏng xe tự hành.

Nghiên cứu này tập trung vào ba trụ cột chính: **Công thức hóa bài toán**, **Phương pháp tiếp cận**, và **Tác động thực tiễn**.

---

## Slide 2: Bài toán Ưu tiên hóa Kiểm thử cho SDC

**Nội dung chính (2-3 phút)**

### Vấn đề cốt lõi:
Các bộ mô phỏng SDC như BeamNG và CARLA thực thi **hàng nghìn kịch bản đường** trong mỗi chu kỳ phát hành. Mỗi kịch bản mất từ **10 đến 60 giây** để chạy.

**Điểm mấu chốt**: Chúng ta cần ưu tiên các bài kiểm thử sao cho **các lỗi xuất hiện trước**.

### Đầu vào (Input):
Một đoạn đường được biểu diễn như một chuỗi các điểm kiểm soát 2D:
- R = {p₁, p₂, ..., pₙ} ⊂ ℝ²
- Mỗi test: hình học đường → kết quả PASS/FAIL

### Đầu ra (Output):
Xếp hạng tất cả các test theo xác suất thất bại:
- π* = argmax APFD(π)
- Trong đó π là một hoán vị của n test

**Mục tiêu**: Học một hàm tính điểm f: R → [0,1] để xếp hạng các đoạn đường có khả năng thất bại lên đầu.

---

## Slide 3: Định nghĩa Chỉ số APFD

**Nội dung chính (2 phút)**

### Định nghĩa APFD (Average Percentage of Faults Detected):
APFD là chỉ số đo lường hiệu quả của việc ưu tiên kiểm thử. Công thức:

**APFD(π) = 1 - (1/nm) × Σ TFᵢ + (1/2n)**

Trong đó:
- n = tổng số test
- m = số lỗi (fault)
- TFᵢ = vị trí của lỗi thứ i trong thứ tự π

### Cách hiểu:
- APFD ∈ [0, 1] - giá trị càng cao càng tốt
- APFD = 1.0 nghĩa là tất cả lỗi được phát hiện đầu tiên
- APFD = 0.5 là baseline ngẫu nhiên

### Thách thức quan trọng:
**APFD ≠ argmax P(fail | R)**

APFD là một **thống kê xếp hạng** - tối ưu hóa AUC không trực tiếp tối ưu hóa APFD. APFD là **không khả vi** (non-differentiable).

---

## Slide 4: Tại sao các Phương pháp Hiện tại Thất bại

**Nội dung chính (3 phút)**

Chúng tôi đã phát hiện **ba điểm mù** trong các phương pháp hiện tại:

### 1. Độ giòn của Sampling:
- Cùng một đoạn đường, với 64 điểm so với 197 điểm → cho ra điểm số khác nhau
- ΔAPFD ≈ 0.04 - 0.07
- Phụ thuộc vào cách rời rạc hóa

### 2. Phụ thuộc khung hình (Frame Dependence):
- Xoay đoạn đường 30° → APFD giảm 4-7 điểm phần trăm
- f(R · R) ≠ f(R)
- Vi phạm trực giác vật lý

### 3. Bỏ qua Vật lý (Physics-Blind):
- Baseline tự do đưa ra điểm số vi phạm ràng buộc:
- v² × κ(s) ≤ μ × g
- 17-21% vi phạm xếp hạng

### Nhận định của chúng tôi:
SDC test prioritization được đối xử như **black-box classification**. Nhưng đoạn đường có:
- Hình học liên tục r(s) ∈ C(Ω; ℝ²)
- Đối xứng SE(2) cứng (xoay + tịnh tiến)
- Ràng buộc động lực học xe

**Cấu trúc này đang bị bỏ qua.**

---

## Slide 5: Phương pháp 1 - Động lực SE(2)-Equivariant

**Nội dung chính (2-3 phút)**

### Nguyên tắc Bất biến Vật lý:
Thuộc tính "xe đi ra khỏi làn đường" phụ thuộc vào **hình học nội tại** của đoạn đường, không phụ thuộc vào vị trí đặt nó trong ℝ².

**Xoay hoặc tịnh tiến đoạn đường KHÔNG THỂ thay đổi xe có bị thất bại hay không.**

### Yêu cầu Toán học:
Với SDC test prioritization, bộ xếp hạng f cần thỏa mãn:

**f(R · R + t) = f(R) với mọi (R, t) ∈ SE(2)**

Trong đó SE(2) = SO(2) × ℝ² là **nhóm Euclidean đặc biệt** trong 2D.

### Tại sao xây dựng vào mô hình?
- Data augmentation chỉ là xấp xỉ (không chính xác bằng)
- Dữ liệu huấn luyện nhiều hơn không sửa được bias cấu trúc
- **Bất biến by-design** → đảm bảo có thể chứng minh được

---

## Slide 6: Feature Engineering - Từ 10 xuống 7 Kênh

**Nội dung chính (2 phút)**

### Tiêu chuẩn 10 Kênh (KHÔNG bất biến):
| Tính năng | Vấn đề |
|-----------|--------|
| x, y | Vị trí tuyệt đối → phụ thuộc |
| sinθ, cosθ | Hướng → phụ thuộc |
| κ(s) | Độ cong |
| dκ/ds | Đạo hàm độ cong |
| s, Δs | Độ dài cung |
| Δθ | Thay đổi góc |

→ **Các tính năng màu đỏ phụ thuộc vào embedding toàn cục**

### 7 Kênh SE(2)-Bất biến của chúng tôi:
1. κ(s) - độ cong có dấu
2. |κ(s)| - độ lớn
3. dκ/ds - đạo hàm độ cong
4. Δs - gia số độ dài cung
5. Δθ - thay đổi góc cục bộ
6. ∫|κ(τ)|dτ - độ cong tích lũy
7. κ̃(s) - độ cong làm mịn low-pass

**Không tính năng nào phụ thuộc vào tọa độ (x, y)**

### Insight quan trọng:
Tất cả 7 tính năng là hàm của {κ, dκ/ds, Δs} **duy nhất**. Không tính năng nào đọc R như tọa độ (x, y).

---

## Slide 7: Độ cong như Hình học Nội tại

**Nội dung chính (2 phút)**

### Nền tảng Frenet-Serret:
Với đường cong được tham số hóa r(s) ∈ ℝ² với độ dài cung s:

**κ(s) = |r'(s) × r''(s)| / |r'(s)|³**

### Tại sao κ là SE(2)-Bất biến:

**1. Phép quay R ∈ SO(2)**: Cả tử số và mẫu số đều biến đổi bằng |R|³ = 1 → κ không đổi.

**2. Phép tịnh tiến t ∈ ℝ²**: r → r + t không làm thay đổi đạo hàm r', r'' → κ không đổi.

**3. Tham số hóa lại**: Độ dài cung s là bất biến với tham số hóa.

### Hệ quả:
Φ: ℝⁿᵐˣ² → ℝⁿᵐˣ⁷ (curvature pipeline)

**Φ(R · R + t) = Φ(R) với mọi (R,t) ∈ SE(2)**

---

## Slide 8: Kiến trúc - SE2RoadNet

**Nội dung chính (2-3 phút)**

### Kiến trúc mô hình:

| Thành phần | Thông số |
|-----------|----------|
| Đầu vào | Ma trận feature 7 kênh (197 × 7) |
| Projection | Linear + LayerNorm + GELU: 7 → 192 |
| Backbone | Transformer Encoder 6 lớp |
| Attention | 8 heads, dₖ = 24 |
| Positional Bias | Relative arclength với 32 RFF features |
| Output | MLP head: 192 → 64 → 1 → σ |
| Tham số | ~2.11M |

### Huấn luyện:
AdamW, learning rate = 10⁻³, 75 epochs, SWA

### Định lý Đảm bảo Equivariance:
Cho Φ: ℝⁿᵐˣ² → ℝⁿᵐˣ⁷ là pipeline 7 kênh và h: ℝ⁷ → ℝ là bất kỳ head nào:

**f_θ = h ∘ Φ ⇒ f_θ(R · R + t) = f_θ(R)**

**Đảm bảo giữ nguyên bất kể head nào được chọn.**

---

## Slide 9: Xác minh Thực nghiệm - Δ = 0.0000

**Nội dung chính (2 phút)**

### Rotation Invariance Probe:

| Rotation | 0° | +30° | +60° | +90° | +180° | -45° | Δ |
|----------|-----|------|------|------|-------|------|---|
| Baseline Transformer | 0.8066 | 0.7711 | 0.7494 | 0.7651 | 0.7613 | 0.7785 | ~0.057 |
| **SE2RoadNet** | **0.8047** | **0.8047** | **0.8047** | **0.8047** | **0.8047** | **0.8047** | **0.0000** |

### Ý nghĩa của Δ = 0:
- **Đầu ra giống hệt nhau** dưới mọi phép quay cứng
- Không phải "xấp xỉ bằng không"
- Không phải "trong dung sai floating-point"
- **Chính xác bằng nhau** qua 6 phép quay ngẫu nhiên

### Lợi ích bổ sung:
- AUC = 0.9347 (cao nhất trong dự án)
- APFD = 0.8048 ± 0.0118
- σ = 0.0118 (phương sai thấp thứ 2)

---

## Slide 10: Trực quan Hóa

**Nội dung chính (1-2 phút)**

Slide này minh họa trực quan định lý của chúng tôi.

Cùng một đoạn đường, xoay 0°, 90°, 180° đều cho cùng điểm số: **0.47**

Điều này chứng minh:
- **f_θ(R · R + t) = f_θ(R)** với mọi (R, t) ∈ SE(2)
- Điểm số, thứ hạng, và APFD **chính xác bằng nhau**

---

## Slide 11: Phương pháp 2 - Ràng buộc Động lực học Xe

**Nội dung chính (2-3 phút)**

### Giới hạn Gia tốc Hướng tâm:
Một chiếc xe di chuyển trên đoạn đường r với vận tốc v chỉ ở trong làn đường nếu gia tốc hướng tâm bị giới hạn bởi ma sát:

**v² × κ(s) ≤ μ × g với mọi s ∈ [0, L]**

Trong đó:
- v = vận tốc xe
- κ(s) = độ cong đoạn đường tại độ dài cung s
- μ = hệ số ma sát
- g = 9.81 m/s²

### Diễn giải Vật lý:
Độ cong cao + tốc độ cao → xe trượt. Điểm số cần tôn trọng tính đơn điệu này.

### Vấn đề của Baseline:
Baseline tự do tạo ra điểm số **vi phạm tính đơn điệu này trong 17-21% các cặp test**. Điều này **không thể chấp nhận** trong triển khai quan trọng về an toàn.

---

## Slide 12: Monotone PINN - Auxiliary Loss

**Nội dung chính (2-3 phút)**

### Penalty Đơn điệu (Physics Aux Loss):
Với mỗi cặp (Rᵢ, Rⱼ) trong batch thỏa:
**max_s v²κ(Rᵢ) ≥ α × max_s v²κ(Rⱼ)**

Chúng ta muốn f(Rᵢ) ≥ f(Rⱼ).

**Lỗi phạt cho vi phạm thứ tự:**

**L_phys = E[(i,j)] [ReLU(f(Rⱼ) - f(Rᵢ))]**

### Khi nào có penalty:
- Khi ràng buộc bị vi phạm: f(Rⱼ) > f(Rᵢ) nhưng vật lý nói Rᵢ nên xếp hạng cao hơn → mất dương
- Khi ràng buộc thỏa mãn: mất = 0

### Hàm mục tiêu Huấn luyện:
**L_total = L_BCE + λ_phys × L_phys**

**Lịch trình curriculum**: λ_phys tăng từ 0 → 0.5 trong 30% epoch đầu tiên.

---

## Slide 13: Tại sao dùng ReLU Penalty?

**Nội dung chính (1-2 phút)**

### Từ Hard Constraint đến Soft Penalty:
Chúng ta muốn: f(Rᵢ) ≥ f(Rⱼ) khi v²κ_Rᵢ ≥ v²κ_Rⱼ

Tương đương với: g(Rᵢ, Rⱼ) = max(0, f(Rⱼ) - f(Rᵢ)) = 0

Đây là **zero-residual constraint**.

### Tại sao ReLU thay vì Quadratic?

**Quadratic penalty**: L_quadratic = E[g²]

**ReLU tốt hơn vì:**
- Gradient **mạnh hơn** cho các vi phạm lớn
- Tránh phạt quá nhiều các vi phạm nhỏ
- Thực nghiệm cho thấy ReLU **tốt hơn** trong ablation

---

## Slide 14: Kết quả - Giảm 5.6 lần Vi phạm

**Nội dung chính (2 phút)**

### Kết quả chính:
Vi phạm độ cong **giảm 5.6 lần**:
- 17.57% → 3.14% (α = 1.5)
- 21.44% → 2.72% (α = 2.0)

### Tác động lên APFD:
APFD **gần như không đổi**:
- 0.8051 → 0.8055 (±0.0122)

**Điều này có nghĩa là**: Tôn trọng vật lý **hoàn toàn miễn phí**, không mất gì về chỉ số.

---

## Slide 15: Câu chuyện Hai Trục

**Nội dung chính (2 phút)**

Đồ thị này cho thấy hai trục rõ ràng:

### Trục APFD (đường ngang - không đổi):
- Đường APFD gần như phẳng
- Không mất gì cho kỹ sư

### Trục Vi phạm (đường dốc xuống):
- Đường vi phạm rơi tự do
- Mô hình có thể kiểm toán được theo động lực học xe

### Lợi ích cho Quy định:
- APFD phẳng → không mất metric
- Vi phạm giảm mạnh → mô hình có thể kiểm toán
- Kỹ sư **có thể xác minh thứ hạng** dựa trên ràng buộc vật lý

---

## Slide 16: Tác động - Tại sao Điều này Quan trọng

**Nội dung chính (2-3 phút)**

### Đóng góp Lý thuyết:
1. **Mô hình bất biến có thể chứng minh đầu tiên**
2. **Định lý SE(2)** - by construction
3. **Xếp hạng hình thành bởi vật lý**
4. **Định lý hợp thành**

### Đóng góp Thực tiễn:
1. **APFD = 0.8048** - vượt SOTA
2. **AUC = 0.9385** - cao nhất
3. **σ = 0.0109** - phương sai thấp nhất
4. **Giảm 5.6 lần vi phạm**

### Điểm mấu chốt:
**Chỉ có phương pháp của chúng tôi đạt được cả APFD cao nhất VÀ đảm bảo lý thuyết mạnh nhất.**

---

## Slide 17: So sánh với Tất cả Công trình Trước

**Nội dung chính (2-3 phút)**

| Phương pháp | Mô hình | APFD | Đảm bảo |
|-------------|---------|------|---------|
| Random | - | 0.493 | Không |
| LLM zero-shot | Language model | 0.487 | Không |
| GNN (GCN) | Road graph | 0.533 | Không |
| ResNet-50 | Road image CNN | 0.572 | Không |
| SO-SDC-Prioritizer (TOSEM'22) | Genetic algorithm | 0.765 | Không |
| ITEP4SDC (ICST'24) | MLP, 3 features | 0.781 | Không |
| Greedy-diversity (TOSEM'22) | Heuristic | 0.795 | Không |
| RoadFury (ICST'26 baseline) | Transformer+SWA | 0.804 | Statistical |
| **PINN monotone (của chúng tôi)** | Physics-informed | **0.8055 ± 0.012** | **Audit** |
| **SE(2)-Equivariant (của chúng tôi)** | Group-equivariant | **0.8048 ± 0.012** | **Δ=0** |
| **SE(2)+Listwise (của chúng tôi)** | Composed | **0.8049 ± 0.012** | **AUC 0.939** |

**Insight chính**: Không phương pháp SOTA nào trước đây mang theo đảm bảo lý thuyết.

---

## Slide 18: Cấu hình Đề xuất - Theory Stacking

**Nội dung chính (2 phút)**

Kiến trúc đề xuất kết hợp 4 module:

### 1. SE(2):
- Δ = 0 bất biến

### 2. PINN:
- Giảm 5.6 lần vi phạm

### 3. DiffAPFD:
- Phương sai thấp σ

### 4. Conformal:
- APFD coverage bound

### Mục tiêu:
**APFD ≥ 0.820 ± 0.010 với tất cả các đảm bảo được kết hợp**

---

## Slide 19: Kết luận Chính

**Nội dung chính (2-3 phút)**

### 1. SE(2)-Equivariant:
- Δ = 0.0000 CHÍNH XÁC
- Đầu ra giống hệt nhau dưới mọi phép quay
- Định lý + kết quả thực nghiệm khớp nhau

### 2. Physics-Informed:
- Giảm 5.6 lần vi phạm
- Từ 17.57% xuống 3.14%
- Có thể kiểm toán cho cơ quan quản lý

### Câu nói chốt:

**"Lý thuyết đánh bại kỹ thuật khi cấu trúc là có thật: hình học + vật lý > một Transformer ablation khác."**

---

## Slide 20: Cảm ơn

**Kết thúc (30 giây)**

Cảm ơn các thầy cô và các bạn đã lắng nghe.

Chúng tôi rất sẵn lòng trả lời mọi câu hỏi.

### Thông tin liên hệ:
- Đào Sỹ Duy Minh
- Trần Chí Nguyên
- Huỳnh Trung Kiệt
- Đại học Khoa học Tự nhiên, ĐHQG-HCM
- NeurIPS 2026

---

## Ghi chú Tổng quát cho Người Thuyết trình:

### Thời gian ước tính:
- **Tổng thời gian**: 25-30 phút
- Mỗi slide: 1-3 phút tùy độ phức tạp

### Điểm nhấn quan trọng:
1. **Δ = 0** - con số này rất ấn tượng, cần nhấn mạnh
2. **5.6 lần giảm vi phạm** - đây là kết quả thực tế cho ứng dụng
3. **APFD không giảm** - đây là điểm mạnh quan trọng
4. **Không phương pháp nào trước đây có đảm bảo lý thuyết**

### Chuẩn bị Q&A:
- Câu hỏi về định lý SE(2): Cần chứng minh toán học trong paper
- Câu hỏi về implementation: Xem chi tiết trong phần Methods
- Câu hỏi về baseline comparison: Đã so sánh với tất cả công trình liên quan

---

*Script được biên soạn cho presentation: paper\summary_proposal.tex*
