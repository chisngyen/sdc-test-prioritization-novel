# Script Thuyết Trình - ICST 2026

---

## SLIDE 1: Title

Chào mọi người, hôm nay chúng tôi sẽ trình bày về **Theory-Driven Test Prioritization for Self-Driving Car Simulators** -- tức là ưu tiên kiểm thử bộ mô phỏng xe tự lái dựa trên lý thuyết.

Đây là đề tài nghiên cứu của nhóm chúng tôi tại ICST 2026 Tool Competition, với cách tiếp cận **SE(2)-Equivariant** -- tức là bất biến đối xứng.

Nhóm gồm ba thành viên: Đào Sỹ Duy Minh, Trần Chí Nguyên, và Huỳnh Trung Kiệt, đến từ Đại học Khoa học Tự nhiên, ĐHQG-HCM.

---

## SLIDE 2: Nội dung

Trước tiên, chúng tôi sẽ giới thiệu bài toán. Sau đó, chúng tôi sẽ giải thích tại sao bài toán này khó với các phương pháp hiện tại. Rồi chúng tôi sẽ trình bày giải pháp của mình -- cách tiếp cận SE(2)-Equivariance. Cuối cùng là kết quả và kết luận.

---

## SLIDE 3: Bài toán SDC Test Prioritization

### Thách thức

Các bộ mô phỏng xe tự lái như BeamNG hay CARLA, mỗi đợt phát hành cần chạy hàng nghìn kịch bản đường. Mỗi kịch bản tốn khoảng 10 đến 60 giây mô phỏng. Tính ra, mỗi đợt release có thể tốn hàng nghìn giờ chạy test.

Vấn đề là: làm sao để **ưu tiên** các kịch bản này, sao cho những kịch bản gây **thất bại** xuất hiện **trước**? Để kỹ sư phát hiện lỗi sớm nhất có thể.

### Test Prioritization

Đây gọi là Test Prioritization -- tức là xếp hạng lại thứ tự test. Metric chuẩn để đánh giá là **APFD** -- Average Percentage of Faults Detected.

### Giới hạn cạnh tranh

Trong competition ICST 2026, chúng tôi có 956 tests, được chia thành 287-test sub-trials, và chạy 30 lần trials để đánh giá stability.

---

## SLIDE 4: Định nghĩa APFD

APFD là metric đo trung bình vị trí phần trăm mà các lỗi được phát hiện.

Công thức đây: APFD bằng 1 trừ tổng vị trí lỗi chia cho n nhân m, cộng 1 phần 2n.

APFD nằm trong khoảng 0 đến 1. Giá trị càng cao càng tốt.

Baseline mà chúng tôi cần đánh bại là Transformer + SWA, đạt **0.8066 ± 0.0124** -- đây là kết quả tốt nhất từng có.

---

## SLIDE 5: Những điểm mù của Baseline

### Vấn đề chính

Đây là những vấn đề mà baseline gặp phải:

**Thứ nhất**, độ giòn sampling-rate: cùng một đoạn đường, lấy mẫu 64 điểm hay 197 điểm sẽ cho kết quả khác nhau.

**Thứ hai**, frame-dependence: nếu xoay đường đi 30 độ, APFD giảm từ 4 đến 7 điểm. Điều này hoàn toàn không hợp lý về mặt vật lý.

**Thứ ba**, physics-blind: baseline không quan tâm đến ràng buộc vật lý như gia tốc hướng tâm.

**Thứ tư**, calibration không tương đương ranking: đường cong AUC và APFD không nhất quán với nhau.

### Vấn đề sâu hơn

Bản chất của vấn đề là: SDC test prioritization bị đối xử như bài toán black-box sequence classification.

Nhưng đường không phải là một chuỗi bytes. Đường là đường cong liên tục, phụ thuộc đối xứng SE(2), và được chi phối bởi động lực học xe.

**Cấu trúc lý thuyết đang bị bỏ qua hoàn toàn.**

---

## SLIDE 6: Tại sao cần SE(2)-Equivariance?

### Quan sát vật lý

Đây là quan sát quan trọng: "Xe rời làn đường" là tính chất của hình học **nội tại** của đường, không phải vị trí đặt trong không gian.

Nghĩa là: xoay hoặc tịnh tiến toàn bộ đường **KHÔNG THỂ** thay đổi xe có bị thất bại hay không.

### Baseline không có tính bất biến này

Baseline dùng Transformer mã hóa tọa độ tuyệt đối. Kết quả là: xoay đường kiểm tra 30 độ sẽ **thay đổi** điểm dự đoán.

### Claim của chúng tôi

Chúng tôi đề xuất rằng ranker f nên thỏa mãn: f của (R nhân r cộng t) bằng f của r, với mọi phép quay R và tịnh tiến t trong SE(2).

Điểm mấu chốt: chúng tôi xây dựng model thỏa mãn điều này **BẰNG THIẾT KẾ**, không phải bằng data augmentation.

---

## SLIDE 7: 7 kênh đặc trưng SE(2)-bất biến

### So sánh

Baseline dùng 10 kênh đặc trưng tiêu chuẩn: tọa độ x, y, góc heading sin theta cos theta, curvature, và các thứ khác.

**Vấn đề**: x, y, sin theta, cos theta đều phụ thuộc vào vị trí tuyệt đối -- không bất biến.

### 7 kênh của chúng tôi

Chúng tôi chỉ dùng 7 kênh hoàn toàn bất biến:

1. Độ cong có dấu κ(s)
2. Độ lớn của độ cong |κ(s)|
3. Đạo hàm dκ/ds
4. Arc-length increment Δs
5. Local angular change Δθ
6. Cumulative |κ| -- tổng tích lũy
7. Smoothed κ -- độ cong đã làm mượt

**Điểm quan trọng**: KHÔNG có đặc trưng nào phụ thuộc tọa độ.

### Kiến trúc SE2RoadNet

Chúng tôi xây dựng kiến trúc SE2RoadNet với d_model 192, depth 6, 8 heads, và relative-arclength attention bias. Tổng cộng khoảng 2.11 triệu parameters.

---

## SLIDE 8: Pipeline huấn luyện SE2RoadNet

### Data Flow

Quy trình như sau:

**Bước 1**: Đầu vào là road trajectory -- tập hợp các điểm x, y, theta. Chúng tôi normalize về gốc tọa độ.

**Bước 2**: Tính curvature κ(s) theo arc-length parameterization.

**Bước 3**: Trích xuất 7 kênh SE(2)-bất biến như vừa nói.

**Bước 4**: Cho qua SE2RoadNet encoder.

**Bước 5**: Listwise ranking head để ra scores.

**Bước 6**: Huấn luyện với ListMLE loss.

### Training Configuration

Chúng tôi dùng AdamW optimizer, learning rate 3e-4, batch size 32, 100 epochs với cosine annealing và early stopping patience 15.

---

## SLIDE 9: Tại sao dùng ListMLE loss?

### Ba cách tiếp cận

Trong learning to rank, có ba cách tiếp cận chính:

**Pointwise**: dự đoán score tuyệt đối. Nhược điểm: không tối ưu cho ranking.

**Pairwise**: so sánh từng cặp. Nhược điểm: O(n²) pairs, rất chậm.

**Listwise**: tối ưu trực tiếp trên permutation. Ưu điểm: O(n), gradient sparse.

### ListMLE formulation

ListMLE tối ưu xác suất của thứ tự đúng. Gradient của APFD được ước lượng implicitly -- không cần gọi APFD trong forward pass.

### Tại sao SE(2) giúp ListMLE?

Vì feature extraction giữ nguyên dưới SE(2), gradient ổn định hơn. Không cần data augmentation. Và invariance là **inductive bias**, không phải regularization.

---

## SLIDE 10: Kết quả APFD theo góc xoay -- MAIN RESULT

### Kết quả chính

Đây là kết quả quan trọng nhất của chúng tôi.

Bảng này show APFD của 956 tests dưới 6 phép xoay khác nhau: 0 độ, +30, +60, +90, +180, -45.

**Baseline Transformer**: các giá trị dao động từ 0.7613 đến 0.8066, Δ = 0.057.

**SE2RoadNet của chúng tôi**: tất cả đều bằng **0.8047**. Δ = **0.0000**.

### Ý nghĩa

Model hoàn toàn giống nhau bit-wise dưới các phép xoay cứng nhắc.

Không phải "trong phạm vi tolerance". Không phải "cơ bản bằng 0".

Mà là **CHÍNH XÁC BẰNG NHAU** qua 6 phép xoay ngẫu nhiên.

**Đây là kết quả lý thuyết-xác minh-bằng-thực nghiệm sạch nhất mà chúng tôi từng thấy.**

---

## SLIDE 11: Visualizing the Invariance

### Diagram

Hình này minh họa: cùng một đường, xoay 90 độ, xoay 180 độ, đều cho APFD = 0.8047.

### Theorem holds bit-identically

Với mọi phép biến đổi SE(2), model output **hoàn toàn giống nhau**.

---

## SLIDE 12: Định lý

### Phát biểu

Cho Φ là pipeline đặc trưng 7 kênh (chỉ curvature). Khi đó với mọi phép quay R và tịnh tiến t trong SE(2):

Φ của (R nhân r cộng t) bằng Φ của r.

Do đó, model hợp thành f bằng h nhân Φ cũng thỏa mãn tính bất biến này, bất kể head h được chọn.

### Sketch chứng minh

**Bước 1**: Curvature κ(s) là bất biến nội tại của đường cong -- đây là kết quả từ Frenet-Serret formulas.

**Bước 2**: Arc-length s bất biến khi reparameterize.

**Bước 3**: 7 đặc trưng chỉ là hàm của κ, dκ/ds, và Δs.

**Bước 4**: Không đặc trưng nào đọc r như tọa độ (x, y) trực tiếp.

### Xác nhận thực nghiệm

Kết quả Δ = 0 thực nghiệm xác nhận định lý này bằng số.

---

## SLIDE 13: So sánh với SOTA

### Kết quả

Bảng này so sánh với tất cả các phương pháp state-of-the-art:

- Random: 0.493
- LLM zero-shot: 0.487 ± 0.019
- GNN: 0.533 ± 0.025
- ResNet-50: 0.572 ± 0.013
- SO-SDC-Prioritizer (TOSEM'22): 0.765
- ITEP4SDC (ICST'24): 0.781
- Greedy-diversity (TOSEM'22): 0.795
- Baseline RoadFury: 0.804 ± 0.012
- **SE(2)-Equivariant của chúng tôi**: **0.8048 ± 0.0118**

### Điểm nhấn

Chúng tôi vượt qua LLM, Random khoảng 0.32 điểm. Vượt GNN 0.27 điểm. Vượt CNN 0.23 điểm.

Vượt cả TOSEM'22, ITEP4SDC, Greedy-diversity.

**VÀ thêm tính bất biến quay có thể chứng minh được -- mà không method nào khác có được.**

---

## SLIDE 14: AUC và Stability

### Ba chiến thắng độc lập

Chúng tôi có ba chiến thắng độc lập:

**Thứ nhất, APFD**: đánh bại mọi SOTA.

**Thứ hai, AUC**: đạt 0.9347, cao hơn baseline 0.018 điểm.

**Thứ ba, Stability**: variance σ = 0.0118, thấp thứ 2 trong bảng.

### Điểm nhấn

SE(2)-equivariance là inductive bias hữu ích, không chỉ là property về robustness.

Nó thực sự giúp model học tốt hơn, không chỉ ổn định hơn.

---

## SLIDE 15: Tình trạng dự án

### Đã hoàn thành

Chúng tôi đã hoàn thành: FNO Roads, SE(2)-Equivariant (đây là main result), Listwise Learning-to-Rank, Physics-informed (PINN), và Conformal Prediction.

### Đang tiến hành

Tiếp theo, chúng tôi sẽ kết hợp SE(2) với PINN để có triple guarantee. Và tiếp tục cải thiện APFD để submit các top tier conference khác.

---

## SLIDE 16: Hai điểm chính

### Điểm 1: Δ = 0.0000 CHÍNH XÁC

SE2RoadNet hoàn toàn bit-identical dưới các phép xoay cứng nhắc.

Định lý và kết quả thực nghiệm khớp nhau.

### Điểm 2: Lý thuyết > Engineering

Khi cấu trúc là thật, thì hình học và vật lý sẽ tốt hơn một Transformer ablation khác.

Đây là cách tiếp cận đúng.

---

## SLIDE 17: Cảm ơn

Cảm ơn mọi người đã lắng nghe!

Chúng tôi sẵn sàng trả lời câu hỏi.

---

## GHI CHÚ NHANH CHO NGƯỜI THUYẾT TRÌNH

### Điểm cần nhấn mạnh

1. **Slide 10**: "CHÍNH XÁC BẰNG NHAU" -- đây là điểm đột phá, không phải approximate
2. **Slide 12**: Giải thích định lý bằng ví dụ trực quan nếu có thể
3. **Slide 16**: Quote "Lý thuyết > Engineering" là điểm kết luận mạnh

### Timing

- Giới thiệu: 1-2 phút
- Bài toán & APFD: 2-3 phút
- SE(2)-Equivariance: 3-4 phút
- Phương pháp: 2-3 phút
- Kết quả: 4-5 phút
- Kết luận: 1-2 phút
- Q&A: 5-10 phút

**Tổng: ~18-24 phút**

### Khi bị hỏi về

- **Tại sao không dùng data augmentation?**: Vì invariance by design tốt hơn, gradient ổn định hơn, và không tăng training time.

- **So sánh với GNN?**: GNN vẫn phụ thuộc vào cách graph được constructed, và topology không capture được hết curvature information.

- **Làm sao tính curvature?**: Dùng Frenet-Serret formulas với finite differences, chi tiết trong paper.
