# Speaker script — RoadFury → SE2RoadNet (deck mới, 37 trang)

> Deck: `presentation/slide/RoadFury-to-SE2RoadNet.pptx` (s00–s31 trình bày +
> s32–s36 backup cho Q&A).
> Giới hạn của thầy: **tối đa 15 phút trình bày + 10 phút Q&A**, trọng tâm
> **Method (§02–03)** và **Benchmark (§04)**.
> Ngân sách thời gian gợi ý bên dưới: phần Bối cảnh (§01) đi nhanh để dồn
> thời gian cho Method + Benchmark.
>
> Mỗi mục: **[thời lượng]** + lời nói (đọc tự nhiên, không đọc nguyên si) +
> dòng `VERIFY:` nếu trang đó có số liệu cần lưu ý.

---

## ⚠️ TRƯỚC KHI THUYẾT TRÌNH — 5 điểm kiểm chứng phải xử lý

Đối chiếu deck với dữ liệu gốc trong repo, có 5 chỗ **không khớp / không có
nguồn**. Đọc kỹ trước, vì thầy sẽ hỏi đúng những chỗ này.

1. ~~**"Cắt 50–80% thời gian/chi phí" (s03, s05, s35)**~~ — **✅ ĐÃ SỬA
   (2026-06-24).** Đã thay bằng phát biểu định tính suy ra được từ chính APFD
   của nhóm (không cần cite): "dồn lỗi lên đầu hàng đợi → chỉ cần chạy phần đầu
   đã lộ phần lớn lỗi → tiết kiệm chi phí mô phỏng". Lý do bỏ con số: grep toàn
   bộ `exps/` = 0 kết quả, "50–80%" không phải số nhóm đo. (Nếu muốn con số cụ
   thể của nhóm: Exp 05 cho APFD@K=50/287 ≈ 0.94 → chạy ~17% test đầu thu gần
   hết giá trị phát hiện lỗi.)

2. **Leaderboard RoadFury 0.804 < SE2RoadNet 0.805 (s07, s09, s23, s24)** —
   **ĐẢO NGƯỢC dữ liệu gốc.** Số "to beat" chính thức của dự án là RoadFury /
   best-single baseline = **0.8066 ± 0.0124**; SE2RoadNet = **0.8048 ± 0.0118**
   (`tracker.md` §3 Exp 02: "APFD-comp 0.8048 is 0.0018 BELOW best-single
   baseline 0.8066"). Tức trên APFD, SE2RoadNet **thấp hơn** RoadFury 0.0018 —
   nằm trong 1σ (≈0.012), nên đúng là **hòa trong sai số**, KHÔNG phải thắng.
   - Số 0.8066 làm tròn ra **0.807**, không phải 0.804. Con số 0.804 chỉ ứng
     với cấu hình γ=1.5 (0.8045), không phải best-single γ=2.5 (0.8066).
   - **Sửa cho trung thực:** để RoadFury ≈ 0.807 (hoặc ghi rõ "hòa trong σ"),
     SE2RoadNet 0.805. Giữ headline thật = **AUC ↑ (0.917→0.934) + Δ=0** =
     Pareto improvement. (Script Q&A cũ Q3 đã nói đúng tinh thần này: "Em không
     claim APFD tốt hơn" — phần hình ảnh phải khớp với câu đó.)
   → Đây là điểm dễ bị bắt nhất. Đừng để bar chart ngụ ý SE2RoadNet thắng APFD.

3. **Bảng xoay ITS4SDC (s13/s23): 0.7810→0.7240/0.7518/0.7396/0.7334/0.7627,
   Δ=0.057** — `tracker.md` Exp 02 chỉ ghi baseline "should drop 0.04–0.07" như
   **dự đoán** + TODO "Run the baseline rotation probe"; không tìm thấy bảng 6
   góc đã đo. **THÊM:** paper ITS4SDC định nghĩa góc là "góc giữa 2 đoạn kề"
   (turning angle, **vốn bất biến xoay**) → bản đúng-paper sẽ KHÔNG drift. Δ=0.057
   chỉ đúng nếu bản nhóm tái lập dùng **góc/heading tuyệt đối** (phụ thuộc khung).
   → Phải có log tái lập sẵn cho Q&A; nếu thầy biết ITS4SDC, đây là chỗ bị hỏi.

4. **Số baseline — ĐÃ TRA LITERATURE (2026-06-24):**
   - ✅ **Greedy-diversity 0.795** — KHỚP CHÍNH XÁC. Birchler et al., TOSEM
     2023 ([2]): greedy search average APFD = **79.5%**.
   - ✅ **SO-SDC-Prioritizer 0.765** — KHỚP CHÍNH XÁC. TOSEM 2023: SO-SDC
     average APFD = **76.5%**.
   - ⚠️ **Random 0.493** — literature ghi **49.9%** (TOSEM 2023). Lệch nhẹ
     (0.493 vs 0.499); 0.493 là số nhóm tái lập, nói "≈ random ~0.5" là an toàn.
   - ✅ **ITS4SDC (đã sửa tên 2026-06-24, trước ghi nhầm "ITS4SDC").** Güllü,
     Shah, Pfahl — ICST 2025 SDC Tool Competition (arXiv:2501.03881). Là
     **bi-LSTM** (220 cells) trên 2 đặc trưng chuỗi (góc đoạn + chiều dài), test
     SELECTION, paper công bố **F1=0.89** — **KHÔNG có APFD**. → APFD 0.781 +
     bảng xoay Δ=0.057 là **nhóm tái lập**, không phải số trong paper. Slide đã
     sửa: tên ITS4SDC + method LSTM (bỏ "MLP 3 đặc trưng"). Còn lại: drift
     Δ=0.057 cần log tái lập (xem ⚠️3) + sửa nhãn "ICST'25" → ref riêng cho
     ITS4SDC (hiện refs deck chưa có entry ITS4SDC — nên thêm).
   - ⚠️ **GNN 0.533, ResNet-50 0.572, LLM zero-shot 0.487** — không có nguồn
     công bố; là baseline nhóm tái lập trong harness. Cần sẵn log để chứng minh.

5. **Resolution probe Δ=0.0012 (s15 caption, s27, s34/B3)** — số này đo trên
   **mô hình FNO (Exp 01)**, KHÔNG phải SE2RoadNet, nhưng deck SE2RoadNet trình
   bày như thể của SE2RoadNet. Ngoài ra bar ở B3 (.8060/.8066/.8072/.8067)
   **không khớp** bảng FNO thật (.8051/.8060/.8063/.8060/.8062) — cùng ra
   range 0.0012 nhưng giá trị từng N khác. → Hoặc chạy probe trên SE2RoadNet,
   hoặc ghi rõ "FNO (Exp 01)", và dùng đúng số bảng thật.

**Số đã kiểm và ĐÚNG (cứ tự tin nói):** dataset 28804/7202/956 + 38.4/38.4/
36.9% ✓; SE2RoadNet 2.11M params (2,108,721) ✓, ~2.5× RoadFury ✓, train 24.2
phút ✓, Focal γ=1.5 ✓, d=192/6 lớp/8 heads/32 RFF ✓; RoadFury 829K params
(828,801) ✓, AUC 0.917 ✓; multi-trial 30 lần, mẫu 287/956 = max(50, 0.3·|test|)
✓, seed 42 ✓, SE2RoadNet 0.8048±0.0118 ✓; single-pass 0.8047 ✓; rotation
Δ=0.0000/6 góc ✓; AUC 0.9347 cao nhất dự án ✓ (lưu ý làm tròn ra **0.935**,
deck ghi 0.934 — sai số làm tròn, nên dùng 0.935 hoặc 0.9347); công thức APFD
+ ví dụ tay B1 (0.80 / 0.20) ✓; AUC/APFD phân kỳ qua 4 exp ✓; toàn bộ trang
Hạn chế (s27) khớp tracker ✓.

---

# PHẦN MỞ ĐẦU

## s00 · Title — [15s]
Em chào thầy và cả lớp. Nhóm em trình bày đề tài **ưu tiên kiểm thử xe tự lái**
(self-driving car test prioritization). Tên phương pháp là **RoadFury →
SE2RoadNet**: RoadFury là baseline nền tảng của nhóm, SE2RoadNet là phiên bản
cải tiến có **bất biến hình học SE(2)** — xoay hay dịch con đường thì điểm số
không đổi. Nhóm gồm 3 thành viên.

## s01 · Outline — [20s]
Bài có 5 phần: (1) bối cảnh và bài toán, (2) RoadFury — phương pháp nền,
(3) SE2RoadNet — đề xuất chính, (4) đánh giá thực nghiệm, (5) hạn chế và hướng
phát triển. Trọng tâm là phần **3 và 4**. Em có thêm 5 slide backup cho phần
hỏi đáp.

---

# §01 — BÀI TOÁN  (đi nhanh, ~2.5 phút tổng)

## s02 · Divider §01 — [5s]
Phần một: bài toán.

## s03 · Bối cảnh — [40s]
Quy trình test xe tự lái hiện nay chạy trong **mô phỏng** như BeamNG.tech: mỗi
build sinh ra hàng nghìn kịch bản đường, mỗi kịch bản chạy vật lý mất nhiều
giây tới một phút, cộng lại là **nhiều giờ CPU**. Phần lớn kịch bản đều PASS,
chỉ ~30–40% là FAIL. Nên thay vì chạy ngẫu nhiên, ta **xếp các kịch bản dễ FAIL
lên đầu** để lộ lỗi sớm.
> **VERIFY:** "cắt 50-80% thời gian" ở cuối card 3 — **không có nguồn** (xem
> điểm ⚠️1). Nói "cắt phần lớn chi phí khi chỉ chạy prefix ngắn" cho an toàn,
> hoặc dẫn số APFD@K của ta.

## s04 · Phát biểu bài toán — [40s]
Hình thức hóa: **đầu vào** là một con đường = chuỗi N điểm 2D (N từ 64 đến
197), chỉ có hình học, không có hành vi xe. **Scorer** f_θ ánh xạ con đường →
xác suất FAIL trong [0,1]. **Đầu ra** là hoán vị π xếp giảm dần theo điểm, mục
tiêu tối đa hóa **APFD**. Khó ở chỗ: chỉ có hình học đường, và tập thử là
out-of-distribution.
> **VERIFY:** N 64–197 ✓ khớp dataset. Metric APFD (chính) + AUC (phụ) ✓.

## s05 · Vì sao cần ưu tiên — [35s]
APFD chính là **diện tích dưới đường cong "đã phát hiện bao nhiêu lỗi sau khi
chạy x% test"**. Ranking tốt → lỗi dồn lên đầu → đường cong dốc đứng sớm.
Ranking kém → lỗi nằm cuối → tốn nhiều giờ mô phỏng. Với 956 kịch bản nhân mô
phỏng vật lý, chạy hết là rất đắt — nên bài toán không phải "chạy nhanh hơn" mà
là "chạy đúng thứ tự".
> **VERIFY:** takeaway "cắt 50-80% chi phí" — **cùng vấn đề ⚠️1**, xử lý như
> s03. "APFD ngẫu nhiên ~0.52" trong chú thích: random thật ~0.49–0.50; 0.52
> hơi cao nhưng chỉ minh họa, không sai bản chất.

## s06 · Dataset SensoDat — [45s]
Dữ liệu là **SensoDat** (Birchler 2024, MSR — ref [1]), kịch bản SDC sinh tự
động trên BeamNG, công khai. Ba split: **Train** 28.804 (FAIL 38,4%), **Test**
7.202 (cùng phân phối), **Competition** 956 — đây là tập **OOD**: FAIL 36,9%,
đường ngắn hơn hẳn (129–229m so với tới 454m ở train). Biểu đồ phải cho thấy
đường càng dài càng dễ FAIL. Khoảng độ dài hẹp hơn ở Competition chính là
**dịch phân phối** mà ta phải tổng quát hóa qua.
> **VERIFY:** mọi số khớp `best.md`: 11059/28804=38,4% ✓, 353/956=36,9% ✓,
> 2765/7202=38,4% ✓. Các thanh fail-rate theo độ dài là nhóm tự tính (không có
> trong tracker) — đúng xu hướng "dài→FAIL nhiều", nhưng nếu thầy hỏi giá trị
> từng bin thì nói "nhóm tính trực tiếp trên split".

## s07 · Ví dụ kịch bản — [35s]
Đây là dữ liệu thật từ SensoDat và BeamNG. Quan sát: PASS thường là cong mượt;
FAIL thường có **chicane** — cua zigzag, độ cong đổi dấu nhanh. Histogram bên
phải: FAIL lệch về phía "tổng độ đổi hướng" lớn hơn. Điểm mấu chốt: **không có
tọa độ (x,y) tuyệt đối nào quyết định nhãn — chỉ hình dạng**. Đây là gợi ý đầu
tiên cho ràng buộc bất biến.
> **VERIFY:** ảnh thật (SensoDat + BeamNG.tech) — đã có dẫn nguồn ở slide
> Sources ✓.

## s08 · Related Work — 3 hướng — [35s]
Ba hướng tiếp cận trước đây: (1) **Search-based/diversity** (GA, greedy) — chọn
test đa dạng, nhưng đa dạng ≠ lộ lỗi; (2) **Feature-based ML** (ITS4SDC: MLP
trên 3 đặc trưng) — rẻ, dễ giải thích, nhưng ít đặc trưng và phụ thuộc khung;
(3) **Deep learning** (GNN, ResNet, RoadFury) — APFD cao nhất nhưng **không bất
biến**, xoay/lấy mẫu là điểm trôi. Khoảng trống: chưa method nào **đảm bảo** bất
biến — tất cả dựa vào augmentation hoặc đặc trưng phụ thuộc khung.
> **VERIFY:** "RoadFury 0.804" — xem ⚠️2 (số chuẩn là 0.807/0.8066). SO-SDC,
> SO-SDC ([2], TOSEM'23) + Greedy đã verify khớp literature. ITS4SDC là tool
> ICST'25 (arXiv 2501.03881, bi-LSTM) nhưng **APFD 0.781 là nhóm tái lập** —
> paper chỉ công bố F1=0.89. Nên thêm 1 entry ref riêng cho ITS4SDC.

## s09 · Related Work — tổng hợp — [40s]
Bảng tổng hợp APFD trên 956 test OOD. Đọc xu hướng: từ Random 0.493 lên dần tới
các deep model ~0.80. Cột "Bất biến" thì **mọi method trước đều ✗**. SE2RoadNet
thêm một **trục đảm bảo mới** (Δ=0), không chỉ chạy đua APFD. Đây là thông điệp:
ta không cố tăng APFD thêm vài phần nghìn, mà thêm một chiều bảo chứng chưa ai
có.
> **VERIFY:** ⚠️2 (RoadFury 0.804 vs SE2RoadNet 0.805) + ⚠️4 (nguồn các baseline).
> Câu "thêm trục đảm bảo, không chỉ tăng APFD" là cách diễn đạt TRUNG THỰC nhất
> — nên nhấn câu này thay vì khoe APFD cao hơn.

---

# §02 — ROADFURY  (Method phần 1, ~2 phút)

## s10 · Divider §02 — [5s]
Phần hai: RoadFury — phương pháp nền tảng của nhóm.

## s11 · RoadFury pipeline — [30s]
RoadFury là sơ đồ TikZ này: chuẩn hóa đường về 197 điểm và **10 kênh đặc
trưng** → RoadTransformer → SWA → inference (chấm điểm, sắp giảm dần). Đây là
baseline rất mạnh, nhưng có 2 điểm mù mà em phân tích ở slide sau.

## s12 · RoadFury chi tiết — [55s]
Bên trái là 10 kênh đặc trưng: chiều dài đoạn, biến thiên góc, độ cong Menger,
jerk độ cong... nhưng **3 kênh f5–f7 (sinθ, cosθ, θ tuyệt đối) phụ thuộc khung
quy chiếu**. Bên phải: RoadTransformer 829K tham số — Linear 10→128, token
[CLS] + **PE vị trí tuyệt đối**, 4 lớp Transformer, pool → sigmoid. Kết quả:
APFD ≈ 0.804–0.807, AUC 0.917, tốt nhất competition. Nhưng 3 kênh phụ thuộc
khung + PE tuyệt đối = **2 điểm mù**: APFD cao mà không có bảo chứng bất biến.
> **VERIFY:** 829K params ✓ (828,801). AUC 0.917 ✓. APFD: deck ghi "0.804 ±
> 0.012" — số canonical của dự án là **0.8066 ± 0.0124** (best-single γ=2.5).
> Nói "khoảng 0.805–0.807" an toàn hơn "0.804". Xem ⚠️2.

## s13 · Bridge "Vá 2 điểm mù" — [30s]
Mỗi điểm mù của RoadFury → một thành phần của SE2RoadNet. Điểm mù 1 (nhạy xoay,
ITS4SDC Δ=0.057) → vá bằng **7 kênh bất biến SE(2)** (bỏ sinθ, cosθ, θ). Điểm
mù 2 (nhạy tần số lấy mẫu) → vá bằng **attention bias theo Δs**. Tinh thần:
chuyển từ **augmentation (xấp xỉ)** sang **ràng buộc kiến trúc (bảo chứng)** —
vá có chủ đích, không refactor mù.
> **VERIFY:** Δ=0.057 cho ITS4SDC — xem ⚠️3 (xác nhận đã đo thật). "N=64 vs
> 197 → điểm đổi" ✓ khớp tinh thần resolution probe.

---

# §03 — SE2ROADNET  (Method phần 2 — TRỌNG TÂM, ~4 phút)

## s14 · Divider §03 — [5s]
Phần ba: SE2RoadNet — đề xuất chính.

## s15 · SE2RoadNet overview — [30s]
Sơ đồ tổng: 7 kênh bất biến → **InvariantBlock × 6** (attention bias theo Δs) →
pool → Failure Score. Kết quả then chốt ngay đây: **rotation probe ΔAPFD =
0.0000** — xoay đường thì điểm không đổi đến từng bit. Em đi vào 4 bước.
> **VERIFY:** Δ=0.0000 ✓ (tracker Exp 02). Caption không nhắc resolution ở đây
> — tốt.

## s16 · Ý tưởng cốt lõi SE(2) — [45s]
Định lý mục tiêu: f_θ(R·R + t) = f_θ(R) **đến từng bit**, với mọi phép xoay R và
tịnh tiến t. Bốn panel minh họa cùng một con đường — gốc, xoay, tịnh tiến, đổi
gốc tọa độ — đều cho **Score = 0.8047**. Bất biến ở đây là **tính chất của kiến
trúc**, không phải của data augmentation. Đó là khác biệt cơ bản: augmentation
chỉ làm mô hình "quen" với phép xoay; kiến trúc của ta **không cho phép** điểm
số đổi.
> **VERIFY:** 0.8047 = single-pass thật ✓. "bit-identical" đúng về APFD ở
> precision báo cáo; ở mức logit float32 có sai số ~1e-7 do ma trận xoay R chứa
> sin/cos (xem Q&A Q6 backup). Nếu thầy bắt bẻ "bit", dùng câu trả lời Q6.

## s17 · Bước 1 — 7 kênh bất biến — [55s]
Bước 1: chỉ giữ **đại lượng nội tại** của đường: (1) Δs chiều dài đoạn,
(2) |Δθ| biến thiên góc, (3) κ độ cong, (4) dκ/ds jerk, (5) d²κ/ds², (6) s_norm
arclength chuẩn hóa, (7) σ_local độ lệch chuẩn cục bộ. **0 kênh phụ thuộc khung**
— đã bỏ sinθ, cosθ, θ. Cơ sở lý thuyết là **Frenet–Serret**: một đường cong
phẳng được xác định duy nhất đến phép xoay-tịnh tiến bởi hàm độ cong κ(s). Nên
bỏ (x,y) và hướng tuyệt đối **không mất thông tin** — chỉ bỏ phần dư thừa gây
nhạy xoay. Bất biến được đảm bảo **ngay từ khâu đặc trưng**, trước khi mô hình
học.
> **VERIFY:** 7 kênh khớp tracker ("7-channel SE(2)-invariant features") ✓.

## s18 · Bước 2 — kiến trúc InvariantBlock×6 — [45s]
Bước 2: kiến trúc. Mỗi điểm → token 192 chiều. Thêm token [CLS]. **6
InvariantBlock**, mỗi block = MHA 8 heads + FFN 512 + LayerNorm, kèm
**relative-arclength bias** (bước 3). Pool [CLS] → head 192→64→1 → xác suất
FAIL. Tổng **2.11M tham số** (~2.5× RoadFury) — vẫn nhẹ, train 24 phút. Một
config duy nhất, không tinh chỉnh theo từng dataset.
> **VERIFY:** 2.11M ✓ (2,108,721), 24.2 phút ✓, d=192/depth6/8heads ✓.

## s19 · Bước 3 — Attention bias Δs — [60s]  ⭐ phần lõi
Bước 3 là lõi kỹ thuật. **Vấn đề của PE tuyệt đối**: PE chuẩn dùng sin/cos theo
*index* của token, khóa mô hình vào vị trí tuyệt đối; lấy mẫu khác thì index
ứng arclength khác → vỡ bất biến. **Giải pháp**: thêm bias vào ma trận attention
chỉ phụ thuộc **hiệu số arclength** Δs_ij = s_i − s_j, qua MLP(sin(Δs·ω)) với ω
là 32 tần số Fourier ngẫu nhiên cố định (RFF). Vì Δs là đại lượng nội tại nên
bias **bất biến** xoay/tịnh tiến. Heatmap: attention giảm theo |Δs|, không bao
giờ phụ thuộc i, j riêng lẻ. Giá phải trả: chi phí O(B·L²·32) mỗi block — đây là
điểm tốn nhất, nên train 24 phút; hướng cải tiến là **RoPE-1D** trên s_norm.
> **VERIFY:** O(B·L²·32) ✓, RoPE-1D là hướng tương lai ✓ (tracker action item).

## s20 · Bước 4 — Huấn luyện — [45s]
Bước 4: huấn luyện. **Focal loss γ=1.5**: vì FAIL chỉ ~30–40%, BCE thuần khiến
mô hình "đoán PASS hết"; hệ số (1−p̂_t)^γ giảm trọng số ca dễ, tăng trọng số ca
khó. Cấu hình: AdamW, LR 5e-4 cosine + warmup, batch 384, 80 epoch (SWA từ 56),
bf16, WeightedRandomSampler. Train 24,2 phút, 2.11M tham số. Không đổi siêu
tham số giữa các bench.
> **VERIFY:** γ=1.5 ✓ (Exp 02 setup), SWA từ ep56 ✓, batch 384 ✓, 24.2 phút ✓.
> Lưu ý: RoadFury best-single dùng γ=2.5, SE2RoadNet dùng γ=1.5 — cả hai đều
> đúng trong ngữ cảnh riêng, đừng nhầm lẫn nếu thầy hỏi.

---

# §04 — ĐÁNH GIÁ  (Benchmark — TRỌNG TÂM, ~3.5 phút)

## s21 · Divider §04 — [5s]
Phần bốn: đánh giá thực nghiệm.

## s22 · APFD & giao thức — [50s]
**APFD** = 1 − (ΣTF_i)/(n·m) + 1/(2n), khoảng [0,1], càng cao càng tốt; ngẫu
nhiên ~0.5. Ta đánh giá 3 lớp: (1) **single-pass** trên cả 956 test → APFD =
0.8047; (2) **multi-trial 30 lần**, mỗi lần lấy ngẫu nhiên 287/956 → 0.8048 ±
0.0118, để loại "may rủi" do thứ tự cố định; (3) **rotation probe** — xoay cả
split bằng 6 góc rồi đo Δ = max − min APFD, để kiểm chứng bất biến.
> **VERIFY:** công thức ✓, 0.8047 ✓, 0.8048±0.0118 ✓, 287/956 ✓ (=max(50,
> 0.3·956)). Tất cả khớp tracker.

## s23 · Headline Δ=0 — [70s]  ⭐⭐ slide quan trọng nhất
Đây là kết quả headline. Bảng: 6 góc xoay (0, +30, +60, +90, +180, −45),
SE2RoadNet đều cho **0.8047** — Δ = **0.0000**. Để so sánh, ITS4SDC tụt từ
0.781 xuống tận 0.724 ở +30°, Δ = 0.057. Em nhấn: đây **không phải "trong sai
số 1e-6"**, mà bằng nhau **đến từng bit float** — vì pipeline 7 kênh trả vector
y hệt sau xoay, forward pass deterministic, nên logit y nguyên. Đây là một
trong số rất ít chỗ trong ML có thể nói "lý thuyết được xác minh bằng thực
nghiệm" mà không cần đính kèm sigma.
> **VERIFY:** Δ=0.0000 ✓✓ (tracker, kết quả sạch nhất dự án). **Bảng ITS4SDC
> 6 góc (0.781→0.724...) — xem ⚠️3**: xác nhận đã đo thật, không phải số minh
> họa trong dải dự đoán 0.04–0.07.

## s24 · Leaderboard — [60s]
Bảng xếp hạng APFD với 8 baseline. SE2RoadNet 0.805 — **ngang baseline tốt nhất
trong sai số**, nhưng: AUC tăng từ 0.917 lên 0.934, **và** thêm bảo chứng Δ=0 mà
không method nào có → cải thiện theo nghĩa **Pareto**: không tệ hơn ở chiều nào,
tốt hơn ở AUC và lý thuyết. Không đánh đổi: giữ APFD đỉnh, tăng AUC, là phương
pháp duy nhất bất biến SE(2) tuyệt đối.
> **VERIFY:** ⚠️2 — RoadFury phải là **0.807** (không 0.804), nên nói "ngang
> trong sai số" chứ KHÔNG nói "cao hơn". AUC 0.934 → thực ra 0.9347≈**0.935**.
> "+0.017" ✓ (0.9347−0.9170). Đây là slide thầy dễ soi nhất → bám đúng từ
> "Pareto / ngang trong σ", đừng nói "thắng APFD".

## s25 · AUC vs APFD — [45s]
Vì sao tách bạch AUC và APFD? **AUC** đo khả năng phân loại PASS/FAIL trên từng
cặp; **APFD** đo tốc độ lộ lỗi theo thứ tự chạy — đúng cái Competition tối ưu.
Trong gần như mọi thí nghiệm của nhóm, hai chỉ số này **phân kỳ**: SE2RoadNet
tăng AUC 0.917→0.934 nhưng APFD gần như phẳng ~0.805. Bài học: khi báo cáo phải
nói rõ **metric nào** đang tối ưu, không gộp thành "một điểm tốt hơn".
> **VERIFY:** Phân kỳ AUC/APFD ✓ (tracker xác nhận ở Exp 01,02,03,04). Đây là
> phần TRUNG THỰC giúp giải thích vì sao APFD không tăng — nên giữ và nói rõ.

---

# §05 — HẠN CHẾ & HƯỚNG PHÁT TRIỂN  (~1.5 phút)

## s26 · Divider §05 — [5s]
Phần năm: hạn chế và hướng phát triển.

## s27 · Hạn chế — [50s]
Em xin thẳng thắn. **Về chỉ số**: AUC và APFD phân kỳ; listwise loss chưa tăng
APFD trung bình (chỉ giảm σ → đóng góp "ổn định"); ở bench FAIL cao (vd 95% FAIL)
APFD ~0.52 là **trần**, không phải thất bại. **Về phương pháp**: conformal an
toàn còn dở (v1 valid-nhưng-vô nghĩa, v2 informative-nhưng-invalid, cần v3);
IRM/TENT chưa đóng được gap SensoDat→Competition (negative đã biết); bất biến
lấy mẫu mới là **xấp xỉ** (Δ≈0.0012), chưa exact như phép xoay. Tóm lại: bất
biến xoay là exact, các đảm bảo khác đang hoàn thiện — báo cáo trung thực thay
vì giấu.
> **VERIFY:** TẤT CẢ khớp tracker ✓ (listwise σ↓, conformal v1/v2, IRM/TENT
> negative, ceiling 0.52). Resolution Δ≈0.0012 — xem ⚠️5 (đó là số FNO).

## s28 · Hướng phát triển — [40s]
Bốn hướng: (1) tổng quát hóa **8+ benchmark** (OOB, SDC-Scissor, sdc-travel)
với một công thức, không tinh chỉnh; (2) **conformal v3** vừa valid vừa
non-vacuous; (3) **bất biến lấy mẫu exact** bằng RoPE-1D thay RFF; (4) **SSL
vật lý** (physics-informed) thay SSL hình học ngây thơ (đã chuyển kém). Mục
tiêu: từ một baseline bất biến cho SDC → công thức chung bất biến và
audit-readable.
> **VERIFY:** "8+ benchmark" đang là **hướng tương lai**, không phải kết quả đã
> có — slide ghi đúng là future work ✓. Đừng lỡ miệng nói "đã chạy 8 bench".

## s29 · Tham khảo — [15s]
Tài liệu tham khảo theo IEEE. Nguồn dữ liệu là SensoDat [1]; SO-SDC [2], giao
thức APFD [3][4]; nền tảng kỹ thuật: Attention [5], Focal [6], SWA [7],
equivariance [8], RFF [9], conformal [10]. RoadFury và SE2RoadNet là của nhóm.

## s30 · Video — [20s]
(Track Method 60%) Đây là video demo trình bày phương pháp; link Google Drive
và Facebook ở đây. *(Bấm vào nếu cần phát.)*
> **VERIFY:** thumbnail vẫn là placeholder — **thay ảnh thật trước khi nộp**.
> Link đã nhúng sẵn ✓.

## s31 · Cảm ơn — [10s]
Em cảm ơn thầy và cả lớp đã lắng nghe. Em xin nhận câu hỏi.

---

# BACKUP (s32–s36) — chỉ mở khi Q&A hỏi

## s32 · B1 — APFD ví dụ tay
n=5, m=2 FAIL. Xếp tốt (FAIL ở 1,2): APFD = 1 − 3/10 + 1/10 = **0.80**. Xếp kém
(FAIL ở 4,5): 1 − 9/10 + 1/10 = **0.20**. Cùng tập lỗi, chỉ khác thứ tự, APFD
chênh 4 lần — đó là toàn bộ giá trị của test prioritization.
> **VERIFY:** số học ✓ đúng tuyệt đối.

## s33 · B2 — Multi-trial
Vì sao 30 lần: single-pass nhạy với thứ tự cố định → dễ may rủi. Mỗi trial lấy
ngẫu nhiên max(50, 0.3·|test|) = 287/956, tính APFD độc lập; báo cáo mean ± σ
qua 30 trials, seed 42. σ thường quan trọng hơn mean (độ ổn định). SE2RoadNet
= 0.8048 ± 0.0118.
> **VERIFY:** mọi số ✓ khớp code `multi_trial` trong best.md.

## s34 · B3 — Resolution probe
Resample cùng đường ở N ∈ {64..197}, đo Δ = max − min APFD. Kết quả Δ ≈ 0.0012
— rất nhỏ nhưng **chưa exact** như phép xoay (Δ=0). Lý do: RFF bias theo Δs là
gần-bất-biến tại tham số hóa → hướng RoPE-1D.
> **VERIFY:** ⚠️5 — Δ=0.0012 đo trên **FNO (Exp 01)**, và bar (.8060/.8066/
> .8072/.8067) KHÔNG khớp bảng FNO thật (.8051/.8060/.8063/.8060/.8062). Nếu
> thầy hỏi: nói rõ "probe resolution chạy trên backbone FNO Exp 01; SE2RoadNet
> hiện chỉ có exact rotation, resolution đang là hướng".

## s35 · B4 — Chi phí quy mô lớn
Inference chỉ 1 forward pass/đường → chấm cả tập trong vài giây GPU. Chi phí
thật nằm ở **mô phỏng vật lý** của các test được chọn, không phải scorer. Train
một lần 24,2 phút, 2.11M tham số → tái lập rẻ.
> **VERIFY:** điểm 3 ("cắt 50–80% số kịch bản") — **cùng vấn đề ⚠️1**. 24.2
> phút ✓, 2.11M ✓.

## s36 · B5 — Chứng minh bất biến
SE(2): p → R·p + t. Δs bất biến (R trực giao bảo toàn khoảng cách, t triệt
tiêu); |Δθ| bất biến (R bảo toàn góc); κ và đạo hàm theo s là hàm của đại lượng
bất biến → bất biến. Bỏ sinθ, cosθ, θ là điều kiện cần. Hệ quả: F(R·R+t) = F(R)
pixel-identical → mọi tầng sau chỉ nhận đầu vào bất biến → output bất biến.
> **VERIFY:** lập luận toán đúng ✓. Đây là backup mạnh cho Q1.

---

# CHUẨN BỊ Q&A (mang sẵn trong đầu)

**Q: "Cắt 50–80% thời gian — số này ở đâu?"**
→ Trung thực: đây là khoảng quen thuộc của lĩnh vực regression/test
prioritization, không phải số nhóm đo trực tiếp. Số NHÓM đo được: với ranking
của SE2RoadNet, prefix K=50/287 (~17% test) đã đạt APFD@K ≈ 0.94 → phần lớn giá
trị phát hiện lỗi nằm ở một phần nhỏ test đầu. *(Nên sửa slide theo ⚠️1 để khỏi
bị hỏi.)*

**Q: "SE2RoadNet APFD thấp hơn baseline 0.0018, sao gọi là tốt hơn?"**
→ Em **không** claim APFD tốt hơn. Em claim **Pareto improvement**: APFD ngang
trong sai số (σ≈0.012), AUC cao hơn rõ rệt (0.917→0.934), **cộng** guarantee
Δ=0 mà không method nào có. Trong bối cảnh safety-critical, guarantee đáng giá
hơn 0.0018 APFD.

**Q: "Vì sao Δ=0 gọi là exact — mạng nơ-ron mà?"**
→ Bất biến đến từ **pipeline đặc trưng**, không từ tham số học được. 7 kênh
không chứa (x,y) hay sin/cos tuyệt đối, nên xoay đường rồi trích lại cho vector
y hệt; mạng đã eval (dropout tắt), deterministic → logit y hệt → APFD y hệt. Ở
mức logit float32 vẫn có ~1e-7 do ma trận xoay R chứa sin/cos không exact, nhưng
APFD ở precision báo cáo là bit-identical.

**Q: "O(L²) với L=197 thì scale thế nào?"**
→ Đang triển khai **RoPE-1D** trên s_norm thay RFF bias: giữ bất biến arclength
mà chi phí O(L), kỳ vọng giảm train từ 24 xuống 3–4 phút.

**Q: "Competition sao gọi là OOD?"**
→ (1) generator khác; (2) FAIL rate khác (36,9% vs 38,4%); (3) độ dài khác hẳn
(129–229m vs tới 454m). Bằng chứng: APFD SensoDat-test ~0.76 vs Competition
~0.80, chênh >0.04 → dịch phân phối rõ.

**Q: "Các baseline 0.487/0.533/... lấy ở đâu?"**
→ SO-SDC 0.765 + Greedy 0.795 khớp chính xác Birchler TOSEM'23 ([2]). ITS4SDC
là tool ICST'25 (bi-LSTM, arXiv 2501.03881) nhưng APFD 0.781 là nhóm tái lập
(paper chỉ có F1=0.89). Random/LLM/GNN/ResNet là nhóm tái lập trong cùng harness
(956 test OOD, 30 trials). *(Chuẩn bị sẵn log
tái lập — xem ⚠️4.)*
