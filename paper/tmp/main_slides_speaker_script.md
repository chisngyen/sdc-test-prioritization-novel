# SE2-Equivariant SDC Test Prioritization
## Speaker Script - ICST 2026 Tool Competition

**Tổng thời gian ước tính:** 15-18 phút  
**Số lượng slides:** 18 slides chính + 2 backup

---

## SLIDE 1: Title Slide
**[Mở slide — Title Slide xuất hiện]**

---

### NỘI DUNG NÓI:

> "Xin chào tất cả mọi người. Hôm nay chúng tôi sẽ trình bày về **Theory-Driven Test Prioritization for Self-Driving Car Simulators** — một cách tiếp cận dựa trên lý thuyết nhóm Lie để ưu tiên các bài kiểm tra trong bộ mô phỏng xe tự lái.
>
> Tiêu đề phụ của chúng tôi là **'An SE(2)-Equivariant Approach'** — SE(2) là nhóm các phép biến đổi Euclidean trong mặt phẳng, bao gồm phép quay và phép tịnh tiến.
>
> Đây là công trình của nhóm chúng tôi từ Đại học Khoa học Tự nhiên, ĐHQG TP.HCM, tham gia ICST 2026 Tool Competition — Self-Driving Car Testing."

**[Đợi 2-3 giây để khán giả đọc title]**

---

## SLIDE 2: Team & Overview
**[Slide chuyển — Team & Project Overview xuất hiện]**

---

### NỘI DUNG NÓI:

> "Để tôi giới thiệu nhanh về dự án và nhóm chúng tôi.
>
> **Về dự án:** Đây là một framework dựa trên lý thuyết toán học để ưu tiên các bài kiểm tra trong bộ mô phỏng xe tự lái. Điểm khác biệt quan trọng nhất: chúng tôi thay thế các heuristic kỹ thuật bằng **tính bất biến có thể chứng minh** và **các ràng buộc vật lý**.
>
> **Về nhóm:**
>
> - **Đào Sỹ Duy Minh** — thành viên đầu tiên
> - **Trần Chí Nguyên** — thành viên thứ hai
> - **Huỳnh Trung Kiệt** — thành viên thứ ba
>
> Tất cả đều đến từ Khoa Công nghệ Thông tin, Trường ĐHKHTN, ĐHQG TP.HCM."

**[Gestures: Chỉ tay lần lượt vào từng thành viên nếu có màn hình phụ]**

---

## SLIDE 3: Nội dung chính
**[Slide chuyển — TOC xuất hiện]**

---

### NỘI DUNG NÓI:

> "Đây là roadmap của bài trình bày hôm nay.
>
> Đầu tiên, chúng tôi sẽ **giới thiệu bài toán** — SDC Test Prioritization là gì, tại sao nó quan trọng.
>
> Sau đó, chúng tôi sẽ **giải thích độ đo APFD** — metric chuẩn để đánh giá.
>
> Tiếp theo, chúng tôi sẽ **phân tích những điểm mù của baseline** hiện tại — tại sao các phương pháp state-of-the-art vẫn còn thiếu sót.
>
> Phần cốt lõi: **SE(2)-Equivariance** — lý thuyết nhóm mà chúng tôi xây dựng framework trên.
>
> Rồi chúng tôi sẽ **trình bày phương pháp** — kiến trúc SE2RoadNet và training pipeline.
>
> Cuối cùng: **kết quả thực nghiệm** — benchmark với SOTA và proof-of-invariance.
>
> Bắt đầu thôi!"

**[Gestures: Quay người nhìn khán giả, giọng hào hứng]**

---

## SLIDE 4: SDC Simulator Testing
**[Slide chuyển — SDC Simulator Testing xuất hiện]**

---

### NỘI DUNG NÓI:

> "Chúng ta bắt đầu với **bài toán gốc**: SDC Simulator Testing.
>
> **Thách thức thực tế:**
>
> Các bộ mô phỏng như **BeamNG** và **CARLA** thực thi hàng **nghìn kịch bản đường** mỗi đợt phát hành phần mềm. Mỗi kịch bản có thể mất hàng **nghìn giờ** để chạy toàn bộ. Điều này tạo ra một vấn đề rất thực tế: kỹ sư muốn **thất bại xuất hiện sớm nhất** — có nghĩa là, nếu một kịch bản sẽ fail, chúng ta muốn nó được chạy trước tiên.
>
> Đây chính là **Test Prioritization**.
>
> **Định nghĩa đơn giản:** Xếp hạng pool kịch bản sao cho các bài kiểm tra **THẤT BẠI** xuất hiện sớm trong thứ tự chạy. Metric chuẩn cho bài toán này là **APFD** — Average Percentage of Faults Detected.
>
> **Về cuộc thi:**
>
> ICST 2026 Tool Competition cung cấp **956 tests**, được chia thành **287-test sub-trials**, và chạy **30 trials**. Tất cả kết quả được đánh giá bằng APFD."

**[Gestures: Đưa tay ra đếm từng điểm, giọng rõ ràng, chậm rãi ở con số]**

---

## SLIDE 5: Định nghĩa APFD
**[Slide chuyển — APFD Formula xuất hiện]**

---

### NỘI DUNG NÓI:

> "Bây giờ chúng ta cùng xem chi tiết **độ đo APFD**.
>
> **Công thức:** Với thứ tự π của n bài kiểm tra có m lỗi, TF_i là vị trí của lỗi thứ i, APFD được tính như sau:
>
> APFD = 1 - (1/nm) × Σ TF_i + 1/(2n)
>
> **Đặc điểm quan trọng:**
> - APFD nằm trong khoảng **[0, 1]** — cao hơn là tốt hơn
> - Đây là **thống kê xếp hạng thuần túy**, không khả vi — không thể dùng gradient descent trực tiếp
> - Giao thức đánh giá là **multi-trial** trên SBFT 2026 split
>
> **Baseline để đánh bại:**
>
> Baseline của cuộc thi — sử dụng **Transformer + Stochastic Weight Averaging (SWA)** — đạt được:
> - **0.8066 ± 0.0124** (single run tốt nhất)
> - **0.8077 ± 0.0115** (5-config ensemble)
>
> Mục tiêu của chúng tôi là vượt qua con số này, đồng thời đạt được những tính chất lý thuyết mà baseline không có."

**[Gestures: Chỉ vào công thức, giải thích từng biến bằng ngón tay]**

---

## SLIDE 6: Những điểm mù của Baseline
**[Slide chuyển — Problems with Baseline xuất hiện]**

---

### NỘI DUNG NÓI:

> "Đây là phần quan trọng nhất để hiểu tại sao chúng tôi làm điều này.
>
> **Bốn vấn đề nghiêm trọng của baseline hiện tại:**
>
> **Thứ nhất — Độ giòn của sampling-rate:** Một đoạn đường được lấy mẫu ở 64 điểm so với 197 điểm cho kết quả dự đoán **khác nhau đáng kể**. Đây là instability.
>
> **Thứ hai — Frame-dependence:** Xoay đường đi 30 độ, APFD giảm từ 4 đến 7 điểm phần trăm. Điều này có nghĩa là model không có tính bất biến với phép quay — một tính chất mà về mặt vật lý, **PHẢI CÓ**.
>
> **Thứ ba — Physics-blind:** Model vi phạm ràng buộc vật lý cơ bản — ràng buộc gia tốc hướng tâm: v² × κ(s) ≤ μ × g. Một chiếc xe không thể đi qua đường cong mà gia tốc ly tâm vượt quá lực ma sát.
>
> **Thứ tư — Calibration không tương đương với Ranking:** AUC và APFD không nhất quán — model tối ưu cho AUC không nhất thiết tốt cho APFD.
>
> **Vấn đề sâu hơn:**
>
> SDC test prioritization đang bị đối xử như **black-box sequence classification**. Nhưng đường là gì?
>
> - Đường cong liên tục r(s) ∈ C(Ω; ℝ²)
> - Phụ thuộc đối xứng SE(2) cứng nhắc
> - Được chi phối bởi động lực học xe
>
> **Cấu trúc lý thuyết đang bị bỏ qua.** Và đó chính xác là những gì chúng tôi khắc phục."

**[Gestures: Đếm từng vấn đề bằng ngón tay, nhấn mạnh "PHẢI CÓ" ở vấn đề 2, giọng nghiêm túc]**

---

## SLIDE 7: Tại sao cần SE(2)-Equivariance?
**[Slide chuyển — Why SE(2)-Equivariance xuất hiện]**

---

### NỘI DUNG NÓI:

> "Bây giờ, hãy nói về lý thuyết cốt lõi: **SE(2)-Equivariance**.
>
> **Quan sát vật lý cơ bản:**
>
> Xoay hoặc tịnh tiến toàn bộ đường — **KHÔNG THỂ** thay đổi việc xe có bị thất bại hay không.
>
> Đây là một **bất biến vật lý hiển nhiên**. Nếu tôi xoay bản đồ thành phố 90 độ, chiếc xe vẫn đi được như cũ. Nếu tôi dịch điểm xuất phát sang bên trái 100 mét, kịch bản vẫn có cùng khả năng thất bại.
>
> **Vấn đề với Baseline:**
>
> Baseline sử dụng Transformer mã hóa **tọa độ tuyệt đối** và **góc heading**. Kết quả: xoay đường kiểm tra 30 độ **THAY ĐỔI** điểm dự đoán.
>
> **Claim của chúng tôi:**
>
> Với SDC test prioritization, ranker f nên thỏa mãn:
>
> f(R · r + t) = f(r) cho mọi (R, t) ∈ SE(2)
>
> Và quan trọng nhất: chúng tôi xây dựng model **thỏa mãn điều này BẰNG THIẾT KẾ**, không phải bằng data augmentation. Không cần tạo thêm dữ liệu xoay, không tăng training time — invariance được "hard-wire" vào kiến trúc."

**[Gestures: Vẽ hình xoay bằng tay, mô phỏng phép biến đổi SE(2)]**

---

## SLIDE 8: 7 kênh đặc trưng SE(2)-bất biến
**[Slide chuyển — SE(2)-Invariant Features xuất hiện]**

---

### NỘI DUNG NÓI:

> "Đây là cách chúng tôi implement SE(2)-invariance: thông qua **7 kênh đặc trưng bất biến**.
>
> **So sánh với tiếp cận cũ:**
>
> Baseline sử dụng 10 kênh, bao gồm:
> - x, y — vị trí tuyệt đối (**KHÔNG bất biến**)
> - sin θ, cos θ — heading (**KHÔNG bất biến**)
> - curvature κ, đạo hàm dκ/ds
> - arc-length s, Δs
> - local angular change Δθ
>
> **7 kênh SE(2)-bất biến của chúng tôi:**
>
> 1. **κ(s)** — độ cong có dấu, thay đổi theo hướng quay
> 2. **|κ(s)|** — độ lớn của độ cong, bất biến với hướng
> 3. **dκ/ds** — đạo hàm của độ cong theo arc-length
> 4. **Δs** — arc-length increment
> 5. **Δθ** — local angular change
> 6. **Cumulative |κ|** — tổng tích lũy độ cong tuyệt đối
> 7. **Smoothed κ** — low-pass filtered curvature
>
> **Điểm mấu chốt: KHÔNG có đặc trưng phụ thuộc tọa độ.** Không x, không y, không θ tuyệt đối.
>
> **Kiến trúc SE2RoadNet:**
>
> - d_model = 192
> - depth = 6 layers
> - 8 attention heads
> - Relative-arclength attention bias với 32 RFF features mỗi layer
> - Tổng cộng ~**2.11 triệu parameters**"

**[Gestures: So sánh hai cột bằng cách đưa hai tay sang hai bên, nhấn mạnh "KHÔNG bất biến" vs "bất biến"]**

---

## SLIDE 9: Pipeline huấn luyện SE2RoadNet
**[Slide chuyển — Training Pipeline xuất hiện]**

---

### NỘI DUNG NÓI:

> "Bây giờ hãy đi vào chi tiết **pipeline huấn luyện**.
>
> **Data Flow — 6 bước:**
>
> **Bước 1:** Input là Road Trajectory r = {(x_i, y_i, θ_i)} từ i=1 đến N. Đầu tiên, chúng tôi **normalize về origin** — đặt điểm trung tâm về (0, 0) để loại bỏ translation dependency.
>
> **Bước 2:** Tính **curvature κ(s)** theo arc-length parameterization. Đây là phép biến đổi toán học cốt lõi.
>
> **Bước 3:** Trích xuất **7 kênh SE(2)-bất biến** mà chúng ta vừa thảo luận.
>
> **Bước 4:** Đưa vào **SE2RoadNet encoder** với relative-arclength attention — attention mechanism sử dụng khoảng cách tương đối trên đường, không phải absolute position.
>
> **Bước 5:** **Listwise ranking head** xuất ra scores s_i cho mỗi test.
>
> **Bước 6:** Train với **ListMLE loss**.
>
> **Training Configuration chi tiết:**
>
> - Optimizer: **AdamW**
> - Learning rate: **3 × 10⁻⁴**
> - Batch size: **32 road batches**
> - Epochs: **100** với warmup 5 epochs
> - Weight decay: **0.01**
> - Scheduler: **Cosine annealing**
> - Early stopping: **patience = 15**
>
> Kiến trúc: d_model=192, 6 layers, 8 heads, 32 RFF features → 2.11M parameters."

**[Gestures: Đếm từng bước bằng ngón tay, thời gian chậm ở bước 2 và 3]**

---

## SLIDE 10: Tại sao dùng ListMLE loss?
**[Slide chuyển — ListMLE Explanation xuất hiện]**

---

### NỘI DUNG NÓI:

> "Một câu hỏi quan trọng: **tại sao chúng tôi chọn ListMLE loss**?
>
> **So sánh ba paradigm:**
>
> **Pointwise** (MSE, BCE): Dự đoán score tuyệt đối cho từng test. Nhược điểm: không tối ưu cho ranking — không quan tâm thứ tự tương đối.
>
> **Pairwise** (RankNet): So sánh từng cặp test. Nhược điểm: O(n²) pairs — cực kỳ chậm khi n lớn (956 tests).
>
> **Listwise** (ListMLE): Tối ưu trực tiếp trên permutation. Ưu điểm: O(n), gradient sparse, phù hợp với APFD.
>
> **Công thức ListMLE:**
>
> L = -Σ log P(π*(i) | π*(1..i-1))
>
> = -Σ log [exp(s_π*(i)) / Σ exp(s_π*(j))]
>
> **Điểm quan trọng:** APFD gradient được ước lượng **implicitly** — không cần gọi APFD trong forward pass. Điều này tránh được vấn đề non-differentiability.
>
> **Tại sao SE(2)-invariance giúp ListMLE?**
>
> - Feature extraction Φ giữ nguyên dưới SE(2) → gradient **ổn định**
> - Không cần data augmentation → **không tăng training time**
> - Invariance là **inductive bias**, không phải regularization — nó bắt buộc phải đúng, không phải "có thể đúng"."

**[Gestures: Vẽ ba thanh cho O(n), O(n²), nhấn mạnh sự khác biệt về độ phức tạp]**

---

## SLIDE 11: Kết quả chính — Δ = 0.0000 CHÍNH XÁC
**[Slide chuyển — Main Result Table xuất hiện]**

---

### NỘI DUNG NÓI:

> "Và đây là **KẾT QUẢ CHÍNH** — phần quan trọng nhất của bài trình bày.
>
> **Rotation-invariance probe: 956 tests, 30-trial APFD**
>
> Bảng này cho thấy APFD của cả hai phương pháp dưới 6 góc xoay khác nhau: 0°, +30°, +60°, +90°, +180°, -45°.
>
> **Baseline Transformer:**
>
> - 0°: 0.8066
> - +30°: 0.7711 → giảm 0.035
> - +60°: 0.7494 → giảm tiếp
> - +90°: 0.7651
> - +180°: 0.7613
> - -45°: 0.7785
> - **Δ = 0.057** — baseline thay đổi đến 5.7 điểm phần trăm!
>
> **SE2RoadNet của chúng tôi:**
>
> - 0°: **0.8047**
> - +30°: **0.8047**
> - +60°: **0.8047**
> - +90°: **0.8047**
> - +180°: **0.8047**
> - -45°: **0.8047**
> - **Δ = 0.0000** — **CHÍNH XÁC BẰNG NHAU**
>
> **Ý nghĩa:**
>
> Model của chúng tôi **HOÀN TOÀN GIỐNG NHAU bit-wise** dưới các phép xoay cứng nhắc của đường đầu vào. Không phải "trong phạm vi tolerance", không phải "cơ bản bằng 0", mà là **CHÍNH XÁC BẰNG NHAU** qua 6 phép xoay ngẫu nhiên.
>
> Đây là kết quả **'lý thuyết-xác-minh-bằng-thực-nghiệm'** sạch nhất mà chúng tôi có thể có."

**[Gestures: Chỉ vào bảng, đọc từng con số rõ ràng, dừng lại ở Δ = 0.0000, giọng đầy tự hào]**

---

## SLIDE 12: Visualizing the invariance
**[Slide chuyển — Visual xuất hiện]**

---

### NỘI DUNG NÓI:

> "Hình ảnh này minh họa trực quan cho kết quả vừa rồi.
>
> **Bên trái:** Đường gốc r — APFD = **0.8047**
>
> **Ở giữa:** Cùng đường đó, xoay 90° — vẫn là **R_90 · r** — APFD = **0.8047**
>
> **Bên phải:** Xoay 180° — **R_180 · r** — APFD = **0.8047**
>
> Ba đường trông khác nhau hoàn toàn, nhưng **APFD giống nhau CHÍNH XÁC**.
>
> **Định lý được kiểm chứng:**
>
> Với mọi (R, t) ∈ SE(2):
> f_θ(R · r + t) = f_θ(r)
>
> **Đây không phải approximation — đây là equality, bit-identical, exact.**
>
> Score, ranking, và APFD **hoàn toàn bằng nhau**."

**[Gestures: Quay người chỉ vào từng phần của hình, mô phỏng bằng tay phép quay]**

---

## SLIDE 13: So sánh với SOTA
**[Slide chuyển — SOTA Comparison xuất hiện]**

---

### NỘI DUNG NÓI:

> "Bây giờ chúng ta so sánh với toàn bộ các phương pháp state-of-the-art khác.
>
> **Multi-trial APFD trên Competition split (956 tests)**
>
> Từ dưới lên:
>
> - **Random** baseline: 0.493 — hoàn toàn ngẫu nhiên
> - **LLM zero-shot**: 0.487 ± 0.019 — kém cả random! — chênh lệch -0.318
> - **GNN (3-layer GCN)**: 0.533 ± 0.025 — dùng road graph — chênh lệch -0.272
> - **ResNet-50 visual**: 0.572 ± 0.013 — dùng road image — chênh lệch -0.233
> - **SO-SDC-Prioritizer (TOSEM'22)**: 0.765 — single-objective GA — chênh lệch -0.040
> - **ITEP4SDC (ICST'24)**: 0.781 — MLP với 3 aggregate stats — chênh lệch -0.024
> - **Greedy-diversity (TOSEM'22)**: 0.795 — diversity heuristic — chênh lệch -0.010
> - **RoadFury (baseline)**: 0.804 ± 0.012 — Transformer + SWA — đây là internal reference
>
> Và đây:
>
> - **SE(2)-Equivariant (ours)**: **0.8048 ± 0.0118** — **chiến thắng tuyệt đối**
>
> **Điểm nhấn:**
>
> Chúng tôi vượt qua LLM/Random khoảng 0.32 điểm, GNN 0.27, CNN 0.23, TOSEM'22 SO-SDC-Prioritizer 0.04, ITEP4SDC 0.024, Greedy-diversity 0.01.
>
> **VÀ** thêm tính bất biến quay có thể chứng minh — mà không method nào khác có được."

**[Gestures: Đọc từng hàng từ dưới lên, nhấn mạnh sự tăng dần, dừng ở con số 0.8048]**

---

## SLIDE 14: Thêm chiến thắng về AUC và Stability
**[Slide chuyển — AUC & Stability xuất hiện]**

---

### NỘI DUNG NÓI:

> "Nhưng chúng tôi không chỉ thắng ở APFD. Đây là **ba chiến thắng độc lập**:
>
> **Thứ nhất — APFD:** Vượt mọi SOTA như đã thấy.
>
> **Thứ hai — AUC:** Đạt **0.9347** — cao nhất trong dự án, cao hơn baseline 0.917 đến **+0.018**. Biểu đồ bar bên trái cho thấy rõ sự khác biệt.
>
> **Thứ ba — Stability:** Variance thấp — σ = 0.0118, thấp thứ 2 trong tất cả methods. Điều này có nghĩa model **ổn định** qua các trials khác nhau.
>
> **Điểm quan trọng:**
>
> SE(2)-equivariance là một **inductive bias hữu ích**, không chỉ là property về robustness.
>
> Điều này có nghĩa: khi bạn đưa vào cấu trúc toán học đúng, model không chỉ "bền vững hơn" — nó thực sự **tốt hơn** trên mọi metric."

**[Gestures: Đếm ba ngón tay, chỉ vào biểu đồ bar]**

---

## SLIDE 15: Tình trạng dự án
**[Slide chuyển — Project Status xuất hiện]**

---

### NỘI DUNG NÓI:

> "Để tôi tóm tắt **tình trạng dự án** hiện tại.
>
> **Đã hoàn thành:**
>
> - **FNO Roads** — Fourier Neural Operator cho road representation
> - **SE(2)-Equivariant** ⭐ (đây là main result của chúng tôi)
> - **Listwise Learning-to-Rank** — phương pháp huấn luyện
> - **Physics-informed (PINN)** — Physics-Informed Neural Networks
> - **Conformal Prediction** — uncertainty quantification
>
> **Đang tiến hành / Sắp tới:**
>
> - **SE(2) + PINN stack** — kết hợp triple guarantee: SE(2)-invariance + physics constraints + conformal prediction
> - Tiếp tục cải thiện APFD để submit các **top-tier conference** khác
>
> Dấu ⭐ là headline pick — phần quan trọng nhất, phần mà chúng tôi trình bày hôm nay."

**[Gestures: Quay người nhìn khán giả, nói rõ ràng từng mục]**

---

## SLIDE 16: Hai điểm chính
**[Slide chuyển — Key Takeaways xuất hiện]**

---

### NỘI DUNG NÓI:

> "Hãy tổng kết thành **hai điểm chính**.
>
> **Điểm thứ nhất: Δ = 0.0000 CHÍNH XÁC**
>
> SE(2)-equivariant RoadNet của chúng tôi hoàn toàn **bit-identical** dưới các phép xoay cứng nhắc. Đây là một **định lý được verify bằng thực nghiệm**.
>
> **Điểm thứ hai: Lý thuyết > Engineering**
>
> Khi cấu trúc là thật — khi đường thực sự có tính bất biến SE(2) theo vật lý — thì:
>
> **Hình học + Vật lý > Một Transformer ablation khác**
>
> Đây là cách tiếp cận đúng. Không phải cố gắng thêm regularization, không phải tăng data augmentation. Mà là **hiểu bản chất toán học** của bài toán và build nó vào kiến trúc từ đầu."

**[Gestures: Giơ hai ngón tay, nhấn mạnh từng điểm, giọng trang trọng ở câu cuối]**

---

## SLIDE 17: Thank You / Q&A
**[Slide chuyển — Closing slide xuất hiện]**

---

### NỘI DUNG NÓI:

> "Và đó là toàn bộ bài trình bày của chúng tôi.
>
> **Theory-Driven Test Prioritization for Self-Driving Car Simulators** — An SE(2)-Equivariant Approach.
>
> Đào Sỹ Duy Minh, Trần Chí Nguyên, Huỳnh Trung Kiệt.
>
> Đại học Khoa học Tự nhiên, ĐHQG TP.HCM.
>
> **Cảm ơn mọi người đã lắng nghe!**
>
> **Questions?**"

**[Mỉm cười, đợi câu hỏi]**

---

## BACKUP SLIDES

### Backup Slide 1: Kiến trúc SE2RoadNet
*Chi tiết kỹ thuật nếu có câu hỏi về kiến trúc*

### Backup Slide 2: So sánh Variance
*Chi tiết về stability metric*

---

## GHI CHÚ KỸ THUẬT

### Timing Guide (18 phút):
- **Slide 1-3 (Intro)**: 2 phút
- **Slide 4-5 (Problem & APFD)**: 2.5 phút
- **Slide 6-7 (Problems & SE(2))**: 3 phút
- **Slide 8-9 (Features & Pipeline)**: 3 phút
- **Slide 10 (ListMLE)**: 2 phút
- **Slide 11-12 (Results)**: 3 phút
- **Slide 13-14 (SOTA)**: 2 phút
- **Slide 15-17 (Status & Closing)**: 1.5 phút

### Gestures tổng quát:
- **Khi nói số liệu**: Chỉ vào slide hoặc đếm ngón tay
- **Khi giải thích công thức**: Vẽ bằng tay trong không khí
- **Khi nhấn mạnh điểm quan trọng**: Dừng lại 1-2 giây, giọng trầm hơn
- **Khi chuyển slide**: Nhìn khán giả, không nhìn máy tính

### Tránh:
- Đọc nguyên từng chữ trên slide
- Nói quá nhanh ở phần số liệu
- Quên pause ở điểm quan trọng

---

*Hết script*
