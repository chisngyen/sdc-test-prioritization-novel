# Paper Outline: SE2RoadNet for SOICT

## 1. Title & Narrative

- **Proposed Title**:
  *SE2RoadNet: A Geometry- and Physics-Grounded Transformer for Robust Test Prioritization in Self-Driving Cars*
  *(Hoặc: Preserving Geometry and Physics: Invariant Test Prioritization for Autonomous Driving via SE(2) Attention and Physics-Informed Regularization)*
- **Core Narrative (Câu chuyện xuyên suốt)**:
  - **Kế thừa & Xuất phát điểm**: Công cụ **RoadFury** (quán quân ICST 2026 Tool Competition) đã chứng minh việc bảo toàn chuỗi hình học (sequential geometry) với Pre-LN Transformer + SWA vượt trội hơn hẳn các phương pháp tóm tắt đặc trưng truyền thống (SDC-Scissor) hay LLM/GNN/CNN.
  - **Hạn chế cố hữu của Baseline**: Mặc dù đạt APFD cao (~0.807), RoadFury vẫn có 3 "điểm mù" nguy hiểm trong môi trường kiểm thử xe tự hành:
    1. *Phụ thuộc hệ tọa độ*: Sử dụng góc tuyệt đối ($\sin\theta, \cos\theta$) và absolute positional encoding khiến thứ tự ưu tiên bị trôi dạt khi xoay đường thử.
    2. *Nhạy cảm với bước lấy mẫu*: Độ phân giải lấy mẫu đường thay đổi (do cấu hình simulator) làm thay đổi điểm số.
    3. *Dự đoán phi vật lý*: Thiếu ràng buộc động lực học xe, dẫn đến việc xếp hạng sai các đoạn đường có độ cong gắt vượt quá giới hạn ma sát.
  - **Cải tiến vượt bậc (SE2RoadNet)**:
    - **7 kênh đặc trưng nội tại (Intrinsic coordinate-free features)**: Hoàn toàn độc lập với hệ tọa độ.
    - **Relative-Arclength Attention Bias (RFF)**: Bất biến với phép dời/xoay $SE(2)$ và điểm bắt đầu lấy mẫu.
    - **PINN Auxiliary Loss**: Ép hàm điểm số đơn điệu theo gia tốc hướng tâm $v^2 \kappa(s)$, giảm 5.6 lần tỷ lệ vi phạm vật lý.
    - **Listwise Ranking Loss (DiffAPFD)**: Tối ưu trực tiếp thứ hạng, giảm độ phân tán $\sigma$.

---

## 2. Research Questions (RQs)

- **RQ1 (Geometric & Discretization Invariance)**:
  *Liệu SE2RoadNet có đạt được tính bất biến quay $SE(2)$ và bất biến độ phân giải lấy mẫu một cách tuyệt đối về mặt kiến trúc so với các mô hình baseline không?*
  - **Minh chứng**:
    - Rotation probe (6 góc quay): $\Delta \text{APFD} = 0.0000$ (bit-identical) so với baseline bị tụt.
    - Resolution probe ($N \in \{64, 96, 128, 160, 197\}$): $\Delta \text{APFD} \le 0.0012$.
- **RQ2 (Physical Plausibility & Safety Audit)**:
  *Ràng buộc vật lý (PINN loss) có giúp triệt tiêu các dự đoán vi phạm động lực học xe mà không làm suy giảm hiệu năng sắp thứ tự không?*
  - **Minh chứng**: Tỷ lệ vi phạm curvature giảm từ $17.57\%$ xuống $3.14\%$ (giảm 5.6 lần) trong khi vẫn duy trì APFD đỉnh cao ($0.8055 \pm 0.0122$).
- **RQ3 (Test Prioritization Performance - APFD & AUC)**:
  *SE2RoadNet thể hiện hiệu năng ra sao so với các phương pháp tiếp cận đa dạng (Random, Search-based, Feature ML, Deep Learning) trên SensoDat?*
  - **Minh chứng**:
    - AUC đạt đỉnh cao nhất dự án: **0.9347** (SE2RoadNet) và **0.9385** (DiffAPFD on SE2).
    - APFD vượt qua tất cả các baseline truyền thống (SDC-Scissor, ITEP4SDC, GNN, LLM).
- **RQ4 (Cross-Benchmark Generalization)**:
  *Một mô hình duy nhất với cùng siêu tham số có khả năng tổng quát hóa trên các benchmark khác nhau (OOB, SDC-Scissor, Travel, DriverAI) mà không cần tìm kiếm siêu tham số riêng không?*
  - **Minh chứng**: Bảng kết quả tổng hợp trên 8 benchmark công khai và ma trận chuyển giao cross-threshold trên OOB.

---

## 3. Cấu trúc bài báo chi tiết (Section breakdown)

1. **Section 1: Introduction**
   - Đặt vấn đề kiểm thử xe tự hành (SDC) trong môi trường mô phỏng (BeamNG.tech) tốn kém tài nguyên.
   - Định nghĩa bài toán Test Prioritization và chỉ số APFD.
   - Động lực: Tại sao tính bất biến hình học ($SE(2)$) và tính hợp lý vật lý lại sống còn cho an toàn xe tự hành?
   - Giới thiệu SE2RoadNet và tóm tắt 4 đóng góp chính.
2. **Section 2: Related Work**
   - Search-based & diversity (SDC-Prioritizer).
   - Feature-based ML (SDC-Scissor).
   - Deep sequence models (ITS4SDC, RoadFury).
   - Equivariance theory & Invariance gap trong SDC testing.
3. **Section 3: Methodology (Kiến trúc SE2RoadNet)**
   - Biểu diễn đường thử liên tục và 7 kênh đặc trưng nội tại $SE(2)$-invariant.
   - Pre-LN Transformer với Relative-Arclength Attention Bias (Random Fourier Features).
   - Physics-Informed Regularization (PINN loss dựa trên $a_c = v^2 \kappa \le \mu g$).
   - Listwise Differentiable APFD và kỹ thuật SWA.
4. **Section 4: Experimental Setup**
   - Bộ dữ liệu: SensoDat (32.580 test case) và 8 public benchmarks.
   - Giao thức đánh giá: 30 independent trials, sub-trial sampling, APFD $\pm \sigma$, AUC.
   - Baselines đối sánh: Random, SO-SDC-Prioritizer, SDC-Scissor, ITEP4SDC, GNN, LLM, RoadFury baseline.
5. **Section 5: Empirical Results (Trả lời RQ1 - RQ4)**
   - RQ1: Kết quả Rotation probe và Resolution probe.
   - RQ2: Bảng phân tích vi phạm vật lý (Curvature violation rate).
   - RQ3: So sánh APFD & AUC đa mô hình trên SensoDat.
   - RQ4: Bảng kết quả tổng quát hóa đa benchmark.
6. **Section 6: Threats to Validity**
   - Internal, External, Construct validity.
7. **Section 7: Conclusion & Future Work**
   - Tổng kết và hướng mở rộng (closed-loop simulation, real-world transfer).
