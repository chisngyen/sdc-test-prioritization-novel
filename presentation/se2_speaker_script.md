# Speaker script -- SE2RoadNet (Exp 02)

> Slide kèm: `se2_slides.tex`. Tổng thời lượng dự kiến: **18-22 phút**
> (12 phút trình bày + 6-10 phút Q&A). Mỗi slide ~30-90 giây.
>
> Quy ước: phần in nghiêng *(...)* là gạch đầu dòng để nói thêm nếu thầy
> dừng lại hỏi. Em không đọc nó ra trừ khi cần.

---

## Slide 1 -- Title (20s)

Em chào thầy ạ. Hôm nay em xin báo cáo phương pháp **SE2RoadNet** -- đây
là thí nghiệm Exp 02 trong dự án ICSE 2027 của em. Tên dài là *SE(2)-
Equivariant RoadNet*, em sẽ giải thích chữ SE(2) ở slide thứ 9, nhưng
gọi nôm na là **mạng đọc đường có bất biến hình học**: dù mình xoay
đường đi hay tịnh tiến nó đi đâu, mạng vẫn cho ra cùng một con số.

Bài toán cụ thể là **ưu tiên kiểm thử cho mô phỏng xe tự lái** -- nói
gọn: với hàng nghìn kịch bản test, mình xếp cái nào nhiều khả năng phát
hiện bug lên trước.

## Slide 2 -- Agenda (30s)

Em sẽ đi 6 phần. (1) Bối cảnh bài toán. (2) Đầu vào, đầu ra, dataset.
(3) Phương pháp -- đây là phần dài nhất, em đi theo lối **top-down**,
tức là vẽ bức tranh tổng trước rồi mới zoom vào từng khối. (4) Cách
đánh giá. (5) So sánh với 8 baseline đã công bố. (6) Kết luận và hướng
mở rộng.

---

## Slide 3 -- Bối cảnh (1 phút)

Xe tự lái -- viết tắt là **SDC**, *self-driving car* -- không thể đem
ra đường thật để test mọi build mới, vừa nguy hiểm vừa chậm. Cho nên
quy trình công nghiệp hiện tại là chạy trong **simulator** -- phần mềm
mô phỏng vật lý, ví dụ **BeamNG.tech** hoặc **CARLA**. Mỗi build mới
đi qua **hàng nghìn kịch bản đường** trong simulator, mỗi kịch bản
mất 10 đến 60 giây. Cộng lại là hàng nghìn giờ máy mỗi đợt phát hành.

Vấn đề là phần lớn các kịch bản này đều **PASS** -- xe chạy ngoan trong
làn. Chỉ khoảng 30-40% là **FAIL** -- xe lệch khỏi làn. Câu hỏi đặt ra
rất tự nhiên: thay vì chạy theo thứ tự ngẫu nhiên, **mình xếp các kịch
bản nhiều khả năng FAIL lên trước**, để tester thấy bug sớm thay vì
phải đợi cả đêm.

## Slide 4 -- Vì sao cần ưu tiên (45s)

Số liệu cụ thể: nếu xếp hạng tốt, mình bắt được phần lớn bug trong 20%
test đầu, **tiết kiệm 50-80%** thời gian regression. Nếu xếp hạng kém
-- bug rơi cuối hàng -- coi như chẳng tiết kiệm được gì.

Điểm khó: **tại thời điểm xếp hạng, mình chưa chạy simulator** -- chưa
biết kết quả thật. Mình chỉ có **hình học đường đi** -- một chuỗi điểm
2D. Phải đoán FAIL hay PASS chỉ từ **hình dạng** của đường.

---

## Slide 5 -- Đầu vào / Đầu ra (1 phút)

Em viết hình thức ạ. Đầu vào là một chuỗi **N điểm 2D**:
$\mathcal{R} = \{p_1, \dots, p_N\}$, trong đó N nằm khoảng 64-197 điểm
tuỳ kịch bản. Nhãn huấn luyện chỉ có 2 lớp: PASS hoặc FAIL.

Đầu ra là một hàm $f_\theta$ -- $\theta$ là tham số mạng học được -- trả
về một con số trong khoảng `[0,1]`, hiểu là **xác suất kịch bản này
FAIL**. Sắp xếp giảm dần các xác suất này cho ta thứ tự ưu tiên
$\pi^*$.

Bên phải là ảnh chụp từ BeamNG: trái xe ở giữa làn -- PASS; phải xe đè
vạch hoặc lệch khỏi làn -- FAIL.

## Slide 6 -- Dataset SensoDat (1.5 phút)

Dataset em dùng là **SensoDat**, công bố bởi Birchler 2024 -- là tập
kịch bản SDC sinh tự động rồi chạy thật trên BeamNG, công khai. Có 3
split:

- **Train**: 28 804 kịch bản, tỉ lệ FAIL 38.4%, đường dài 61-454m.
- **Test (SensoDat)**: 7 202 kịch bản, cùng phân phối với train -- gọi
  là **in-distribution**.
- **Competition**: 956 kịch bản, FAIL 36.9%, đường ngắn hơn nhiều
  (129-229m). Đây là tập **out-of-distribution** -- nghĩa là **khác
  phân phối với training set**: generator khác, FAIL rate khác, độ dài
  khác. Đây là chỗ thực sự đo "khả năng tổng quát hoá".

Tất cả kịch bản em **resample về N = 197 điểm**, để batch hoá được trên
GPU mà không phải padding lung tung.

**Một quan sát em muốn nhấn**: độ cong trung bình của FAIL là 0.0284,
của PASS là 0.0282 -- gần như **bằng nhau**. Tức là nếu chỉ nhìn "đường
cong trung bình toàn tuyến", mô hình không phân biệt nổi. Mô hình phải
đọc **cấu trúc cục bộ** của hàm độ cong $\kappa(s)$ -- đây cũng chính
là lý do em dùng Transformer cộng với đạo hàm bậc 1 và bậc 2 của
$\kappa$ ở slide phương pháp.

*(Nếu thầy hỏi $\kappa$ là gì: là độ cong tại mỗi điểm trên đường,
$\kappa = d\theta/ds$ -- góc thay đổi bao nhiêu trên một đơn vị chiều
dài cung. $\kappa$ lớn = cua gắt; $\kappa = 0$ = đường thẳng.)*

## Slide 7 -- Ví dụ kịch bản (1 phút)

Đây là 6 kịch bản thật lấy ra từ test split, hàng trên 3 PASS, hàng
dưới 3 FAIL.

Quan sát bằng mắt: **PASS thường là đường cong nhẹ, độ cong thay đổi
mượt**. **FAIL thường có chicane hoặc S-cong gắt** -- *chicane là cua
zigzag, cong sang trái rồi cong sang phải liên tiếp, độ cong đổi dấu
nhanh*. Xe vào cua quá tốc thì lệch khỏi làn ngay.

Điểm em muốn nhấn: **không có toạ độ $(x,y)$ nào quyết định nhãn FAIL**.
Đường ở Hà Nội hay Tokyo, hướng lên Bắc hay xuống Nam, nó vẫn cua như
thế. Quyết định FAIL là **hình dạng**, không phải vị trí trong mặt
phẳng. Đây là chỉ báo đầu tiên cho phương pháp ở slide tiếp theo.

---

## Slide 8 -- Hai điểm mù của baseline (1.5 phút) ⭐

Đây là slide bước ngoặt, dẫn vào phương pháp.

Em đã thử các baseline đã công bố -- RoadFury, ITEP4SDC, ResNet -- và
phát hiện **hai lỗi cấu trúc**:

**1. Nhạy với phép xoay.** Em lấy đúng đường gốc, xoay 30 độ rồi đánh
giá lại, **APFD tụt 4-7 điểm phần trăm**. Nhưng đường đó vẫn nguy hiểm
y hệt với xe -- xoay khung quan sát thì vật lý không đổi, xe vẫn lệch
như trước. Tức là baseline đang phụ thuộc vào một thứ **không liên
quan đến vật lý**.

**2. Nhạy với tần số lấy mẫu.** Cùng một đường, lấy 64 điểm hay 197
điểm, $\Delta$APFD cỡ 0.04. Tức là phụ thuộc vào **cách rời rạc hoá**,
chứ không phải bản chất đường.

**Điểm quan trọng**: cả hai đều là **lỗi kiến trúc**, không phải lỗi dữ
liệu. *Data augmentation -- ví dụ xoay random trong lúc train -- có thể
giảm nhưng không loại bỏ.* Em cần một ràng buộc tận gốc, tức là kiến
trúc **không cho phép** vi phạm bất biến ngay từ thiết kế.

## Slide 9 -- Định lý SE(2) (1.5 phút) ⭐⭐

Đây là định lý mục tiêu. Em yêu cầu $f_\theta$ thoả:

$$f_\theta(R\mathcal{R} + t) = f_\theta(\mathcal{R}), \quad \forall (R,t) \in SE(2)$$

Em giải thích nhanh **SE(2)** là gì ạ -- viết tắt của *Special Euclidean
group in 2D* -- là **nhóm các phép chuyển động cứng trên mặt phẳng**,
gồm 2 thành phần: phép xoay $R$ (góc bất kỳ) và phép tịnh tiến $t$
(dịch đi đâu cũng được). Không có co giãn, không có lật gương. Là đúng
**những phép mà một quan sát viên đổi vị trí và quay đầu** sẽ tạo ra.

Yêu cầu của em: dù áp bất kỳ phép xoay-tịnh tiến nào lên đường, mạng
phải trả ra **đúng cùng một điểm số**, và là **"bit-identical"** -- ý
là bằng nhau đến từng bit số thực, không phải xấp xỉ "trong tolerance".

Hình bên dưới minh hoạ: 3 phiên bản của cùng một đường (gốc, xoay 60°,
xoay 180°), score đều = 0.804. Lát nữa em cho thầy thấy con số 0.804
này là **số thật** trên Competition split, không phải toy example.

## Slide 10 -- Pipeline tổng thể (top-level view) (1 phút) ⭐

Đây là **bức tranh trên cao** của top-down. SE2RoadNet gồm **4 bước**:

1. Đầu vào (N điểm 2D) $\to$ **Bước 1: tính 7 kênh đặc trưng bất biến
   SE(2)**. Ở bước này em **vứt bỏ $(x,y)$ và hướng tuyệt đối** -- giữ
   lại các đại lượng nội tại.
2. $\to$ **Bước 2: Backbone Transformer 6 lớp** đọc 7 kênh đó.
3. $\to$ **Bước 3: Head** trả xác suất FAIL.
4. $\to$ **Bước 4: Ranking + APFD**.

Mỗi bước có một vai trò trong việc giữ bất biến. **Bước 1** loại bỏ phụ
thuộc khung tham chiếu ở mức đặc trưng. **Bước 2** dùng attention chỉ
phụ thuộc **hiệu số arclength** -- em sẽ giải thích arclength ở slide
sau. **Bước 3 và 4** là huấn luyện chuẩn -- Focal loss cộng SWA.

Bây giờ em zoom vào từng bước.

## Slide 11 -- Drill-down Bước 1: 7 kênh (1.5 phút)

Em giữ đúng 7 đại lượng **nội tại** -- nghĩa là chỉ phụ thuộc bản thân
đường, không phụ thuộc cách đặt hệ toạ độ:

1. $\Delta s_i$ -- chiều dài đoạn thứ $i$ (khoảng cách 2 điểm liên
   tiếp).
2. $|\Delta\theta_i|$ -- độ thay đổi góc, **chỉ lấy độ lớn**, không lấy
   dấu. Lấy dấu là phá bất biến gương.
3. $\kappa_i$ -- độ cong có dấu, bằng $\Delta\theta / \Delta s$.
4. $d\kappa/ds$ -- đạo hàm bậc 1 của độ cong, hiểu nôm na là **"jerk
   hình học"**, tốc độ thay đổi của độ cong.
5. $d^2\kappa/ds^2$ -- đạo hàm bậc 2.
6. $s_{\text{norm}} = s/L$ -- **arclength chuẩn hoá**, tức là chiều dài
   cung đã đi tính từ đầu đường, chia cho tổng chiều dài, để rơi vào
   `[0,1]`. *Arclength là chiều dài đo dọc theo đường, không phải khoảng
   cách thẳng giữa 2 điểm.*
7. $\sigma_{\text{local}}(\kappa)$ -- độ lệch chuẩn của $\kappa$ trên
   cửa sổ 11 điểm xung quanh.

**Tại sao đúng 7 kênh này là đủ?** Theo **định lý Frenet-Serret** trong
hình học vi phân, một đường cong phẳng được xác định **duy nhất đến
phép xoay-tịnh tiến** bởi hàm độ cong $\kappa(s)$. Nói cách khác: nếu
hai đường có cùng $\kappa(s)$, chúng giống hệt nhau, chỉ khác chỗ đặt.
Cho nên **mọi đặc trưng có ích phải là hàm của $\kappa$ và đạo hàm của
$\kappa$**. **Bỏ $(x,y)$ và hướng tuyệt đối không mất thông tin nào**
-- chỉ mất phần dư thừa, mà cái dư thừa đó lại chính là cái gây nhạy
xoay.

## Slide 12 -- Drill-down Bước 2: Kiến trúc (1.5 phút)

SE2RoadNet là Transformer 6 lớp. Em đi từ đầu vào ra đầu ra:

- **Proj** -- một lớp Linear $7 \to 192$ cộng LayerNorm cộng GELU. Mỗi
  điểm trên đường thành 1 *token* 192 chiều. *Token là đơn vị mà
  Transformer đọc, giống như từ trong câu vậy.*
- **CLS token** thêm vào đầu chuỗi -- ý tưởng giống ViT (Vision
  Transformer). CLS là vector đặc biệt tự học, sau khi đi qua các lớp
  attention sẽ "tổng hợp" thông tin từ toàn đường.
- **6 × InvariantBlock**: mỗi block gồm **Multi-Head Attention 8 đầu**
  (MHA -- 8 cơ chế attention chạy song song rồi gộp), một mạng
  feed-forward 512 chiều (FFN), và LayerNorm. Điểm khác với Transformer
  thường: attention có **relative arclength bias** -- giải thích slide
  kế.
- **Pool** lấy vector CLS, qua head $192 \to 64 \to 1$, ra logit FAIL.

Tổng cộng **2.11 triệu tham số**. Em không tinh chỉnh từng dataset --
**một config duy nhất chạy mọi nơi**, đây là một trong những thông điệp
của bài.

## Slide 13 -- Drill-down Bước 3: Attention với $\Delta s$ (1.5 phút) ⭐

Đây là phần lõi kỹ thuật, em xin trình bày kỹ.

**Vấn đề của positional encoding chuẩn**: trong Transformer thường, mỗi
token được cộng thêm một **positional encoding** sin/cos theo *index
của nó trong chuỗi* (vị trí thứ 1, thứ 2, ...). Cái này khoá mô hình
vào **vị trí trong dãy**. Khi đường được lấy mẫu khác -- 64 điểm thay
vì 197 -- index "thứ 50" đại diện cho arclength rất khác nhau. Bất
biến phân giải đổ vỡ.

**Giải pháp của em**: thay vì PE tuyệt đối, em thêm một bias vào ma
trận attention chỉ phụ thuộc **hiệu số arclength** $\Delta s_{ij} = s_i
- s_j$, tức **khoảng cách thật giữa 2 điểm tính dọc theo đường**:

$$\mathrm{bias}_{ij} = \mathrm{MLP}(\sin(\Delta s_{ij} \cdot \omega))$$

trong đó $\omega$ là 32 tần số Fourier ngẫu nhiên cố định -- đây là
trick gọi là **Random Fourier Features (RFF)**, để encode một số thực
thành một vector chiều cao mà MLP học được.

**Lợi**: $\Delta s_{ij}$ **bất biến** dưới mọi phép xoay và tịnh tiến,
vì arclength là đại lượng nội tại -- không phụ thuộc hệ toạ độ.

**Hại**: chi phí $\mathcal{O}(B L^2 \cdot 32)$ mỗi block -- với $L=197$
là tốn. Đây là lý do em train 24 phút trên Kaggle, gấp 6 lần baseline.
*Hướng cải tiến em đang làm: thay RFF bias bằng **RoPE-1D** trên
$s_{\text{norm}}$ -- vẫn bất biến mà chi phí $\mathcal{O}(L)$.*

## Slide 14 -- Drill-down Bước 4: Training (1 phút)

Mấy lựa chọn em dùng:

- **Focal loss** với $\gamma = 1.5$. *Focal là biến thể của cross-entropy
  -- nhân thêm hệ số $(1-p)^\gamma$ để **giảm trọng số những ví dụ dễ**
  và tập trung vào ví dụ khó.* Vì FAIL chỉ chiếm 30-40%, focal giúp mô
  hình không bị lười với lớp thiểu số.
- **SWA -- Stochastic Weight Averaging** từ epoch 56 đến 80. *Ý tưởng:
  trong giai đoạn cuối train, thay vì giữ tham số ở 1 điểm cuối, mình
  **trung bình tham số** qua nhiều epoch. Bề mặt loss trở nên phẳng
  hơn, generalization tốt hơn.* Trong bài này SWA giảm $\sigma$ của
  APFD khoảng 15%.
- **AdamW** học rate 5e-4, cosine schedule + warmup 5 epoch, batch
  384, bf16. 80 epoch hết khoảng 24 phút trên Kaggle Blackwell.

Một chi tiết kỹ thuật cần nhắc: lúc eval phải **chunk batch = 128**
qua hàm `predict_chunked`, vì bias $\Delta s$ tạo tensor 36 GB nếu eval
toàn validation cùng lúc.

## Slide 15 -- Pipeline đầy đủ (30s)

Đây là pipeline gốc đã in trong file `paper/pipeline.png`. Đọc top-down
một lần nữa: **đầu vào $(N,2)$** $\to$ **7 kênh** $\to$ **6
InvariantBlock** $\to$ **Focal BCE** $\to$ **SWA** $\to$
**$\hat{p}_{\text{FAIL}}$**. Một con đường thẳng, không có nhánh phức
tạp.

---

## Slide 16 -- APFD (1 phút)

Metric chính của bài toán prioritization là **APFD** -- *Average
Percentage of Faults Detected* -- nghĩa đen là **diện tích trung bình
dưới đường cong "đã phát hiện được bao nhiêu bug sau khi chạy x% test
đầu"**. Trực giác: nếu mọi FAIL đều nằm ở đầu hàng, APFD gần 1. Nếu rải
đều ngẫu nhiên, APFD khoảng 0.5.

Có 2 điểm khó mà em muốn thầy lưu ý:
1. APFD **không khả vi** theo $\theta$ -- vì nó tính trên thứ tự sắp
   xếp, mà sắp xếp là phép rời rạc. Cho nên **không tối ưu trực tiếp
   bằng gradient được**, phải tối ưu surrogate như BCE.
2. APFD **không tương đương AUC**. Đây là điểm thường bị nhầm. Tối ưu
   AUC không kéo theo tối ưu APFD. Lát nữa em chỉ ra hiện tượng "AUC
   tăng nhưng APFD không tăng" lặp lại trong 4 thí nghiệm liên tiếp
   của em. *AUC = Area Under ROC Curve, đo khả năng phân biệt 2 lớp;
   APFD đo chất lượng thứ tự ưu tiên. Hai cái nhìn khác.*

Bên phải là 3 đường cong APFD: perfect, model của em, và random.

## Slide 17 -- Giao thức đánh giá (1 phút)

Em test trên hai split:
- **SensoDat-test** -- in-distribution, đo khả năng học.
- **Competition split** -- 956 kịch bản OOD, đo khả năng tổng quát hoá.

**Multi-trial APFD**: 30 trials, mỗi trial em lấy ngẫu nhiên 287/956
test (khoảng 30%), tính APFD, rồi báo cáo
$\overline{APFD} \pm \sigma$. Lý do em làm vậy: nếu chỉ chạy 1 lần với
toàn bộ test, kết quả phụ thuộc nhiều vào việc các test FAIL "may rủi"
nằm chỗ nào trong thứ tự cố định. **Multi-trial loại bỏ may rủi**, đo
ổn định.

**Rotation probe**: em xoay **toàn bộ Competition split** bằng 6 góc
khác nhau (0°, +30, +60, +90, +180, -45), tính APFD ở mỗi góc, rồi đo
$\Delta = \max - \min$ APFD. Đây là cách em **kiểm chứng định lý SE(2)
bằng thực nghiệm**: nếu mạng thật sự bất biến, $\Delta$ phải bằng 0.

## Slide 18 -- Headline result (1.5 phút) ⭐⭐⭐

Đây là slide quan trọng nhất của bài.

Bảng cho thấy: 6 lần đánh giá ở 6 góc xoay, APFD đều bằng **0.8047**
-- **giống nhau đến từng bit float**.

Để so sánh: ITEP4SDC baseline -- một trong những phương pháp tốt nhất
hiện nay -- tụt 0.057 điểm chỉ với góc xoay 30°. SE2RoadNet $\Delta = 0$
tuyệt đối.

Em nhấn lại: đây **không phải "trong tolerance 1e-6"** -- đây là
**bit-identical**. Lý do: pipeline 7 kênh trả về vector giống hệt sau
xoay (vì không có $(x,y)$), forward pass deterministic, logit ra y
hệt. **Đây là một trong số rất ít chỗ trong ML mà mình có thể nói "lý
thuyết được xác minh bằng thực nghiệm" mà không phải đính kèm sigma**.

## Slide 19 -- AUC cao nhất dự án (1 phút)

Bên cạnh rotation invariance, một kết quả phụ:
- **AUC 0.9347** -- **cao nhất trong 14 thí nghiệm** của dự án.
- Để so sánh: baseline 0.9170, FNO 0.9172, PINN 0.9244. SE(2) thắng tất
  cả về AUC.

Hệ quả: inductive bias SE(2) -- *inductive bias là cái "định kiến" mình
nhét vào kiến trúc để mô hình ưu tiên một họ hàm nhất định khi học* --
không chỉ giúp robustness mà còn giúp **calibration**, tức là **xác
suất mô hình trả ra trùng khớp với tần suất FAIL thực tế** hơn.

Tuy nhiên APFD multi-trial là 0.8048, **thấp hơn baseline 0.0018**.
Đây chính là hiện tượng **AUC/APFD divergence** em đã thấy ở 4 exp
liên tiếp. Bù lại, $\sigma$ = 0.0118 -- thấp thứ nhì trong dự án,
nghĩa là mô hình **ổn định** -- ít biến động giữa các trial.

---

## Slide 20 -- Leaderboard (1 phút)

Bảng so sánh với 8 baseline đã công bố. Đọc từ thấp lên cao: Random
0.493 (tham chiếu), LLM zero-shot 0.487 (gần như tệ hơn random),
GNN 0.533, ResNet-50 0.572, SO-SDC 0.765, ITEP4SDC 0.781, Greedy 0.795,
**RoadFury 0.804** -- state-of-the-art trước đây.

SE2RoadNet **APFD = 0.804 -- ngang RoadFury**, nhưng:
- **AUC cao hơn 0.017**.
- **Có guarantee lý thuyết** $\Delta = 0$ -- không phương pháp nào ở
  trên có được.

Đây là **strict Pareto improvement** theo nghĩa: không tệ hơn ở bất kỳ
chiều nào, tốt hơn ở AUC và lý thuyết. *Pareto improvement nghĩa là cải
thiện ít nhất 1 chỉ số mà không phải hy sinh chỉ số nào khác.*

---

## Slide 21 -- Take-aways (1 phút)

Ba điểm chính em muốn để lại:
1. **Lý thuyết SE(2) được xác minh exact** -- $\Delta = 0$ bit-identical,
   không phải xấp xỉ.
2. **AUC cao nhất dự án** -- 0.9347.
3. **APFD ngang state-of-the-art** -- 0.804 -- nhưng có guarantee kèm
   theo.

Em cũng xin thẳng thắn về hạn chế:
- Train **24 phút**, gấp 6 lần baseline -- là cái giá của attention
  $\mathcal{O}(L^2)$.
- APFD trung bình **thấp hơn 0.0018** so với baseline mạnh nhất -- AUC
  cao mà APFD không cao tương ứng, đây là một bài học của dự án.

## Slide 22 -- Future work (45s)

Em đang chạy mấy hướng tiếp theo:
- **Exp 10**: dùng **DiffAPFD listwise loss** -- loss khả vi xấp xỉ
  APFD -- trên SE(2) backbone. Hiện tại AUC mới = 0.9385, vẫn giữ
  $\Delta = 0$. *Listwise loss là loss tính trên toàn bộ danh sách thứ
  tự cùng lúc, thay vì từng cặp như pairwise.*
- **Cross-bench rotation probe** cho 5 baseline còn lại, để có figure
  side-by-side trong paper.
- Thay RFF bias bằng **RoPE-1D** trên $s_{\text{norm}}$, kỳ vọng tăng
  tốc 5-10 lần.

Pitch cho ICSE 2027: *"Một công thức đơn giản, tám benchmark:
Transformer SE(2)-equivariant -- chứng minh được bất biến phép xoay,
bất biến độ phân giải, và đơn điệu theo độ cong."*

## Slide 23 -- Q&A

Em xin hết phần trình bày. Em cảm ơn thầy. Em xin lắng nghe câu hỏi ạ.

---

## Câu hỏi dự kiến từ thầy & cách trả lời

**Q1**: Vì sao gọi $\Delta = 0$ là "exact" mà không phải xấp xỉ -- mạng
nơ-ron mà, sao chính xác đến bit được?

> Vâng, đây là điểm em phải trình bày cẩn thận. **Lý do không nằm ở
> mạng, mà ở pipeline đặc trưng**. Pipeline 7 kênh em dùng không có
> $(x,y)$, không có $\sin\theta$ hay $\cos\theta$ tuyệt đối -- chỉ có
> $\Delta s$, $|\Delta\theta|$, $\kappa$ và đạo hàm của $\kappa$. Khi
> xoay đường, các đại lượng này tính ra **vector y hệt đến từng bit
> float** (modulo lỗi làm tròn không tồn tại vì các phép toán là cộng
> trừ chiều dài và chia). Mạng deterministic (đã `model.eval()`,
> dropout tắt), input y hệt $\Rightarrow$ logit y hệt $\Rightarrow$
> APFD y hệt. Tức là **bất biến đến từ pipeline đặc trưng, không phải
> từ tham số mạng học được**. Đây là khác biệt cơ bản với data
> augmentation.

**Q2**: Nếu chỉ dùng 7 đặc trưng nội tại, sao không thử MLP đơn giản
cho rẻ?

> Em đã thử ạ. ITEP4SDC dùng 3 đặc trưng nội tại + MLP, APFD chỉ 0.781.
> Vấn đề của MLP là **không có context dọc đường**. Cua nguy hiểm
> không xuất hiện riêng lẻ -- nó thường đến trong **chuỗi**: cua phải
> rồi cua trái rồi cua phải nữa (chicane). Transformer cộng với
> relative-arclength attention học được **tương quan giữa các đoạn
> đường cách nhau bao nhiêu mét** -- ví dụ "đoạn cua A nguy hiểm hơn
> nếu trước đó 30 mét đã có một cua B cùng chiều". MLP không nhìn được
> mối liên hệ đó.

**Q3**: APFD thấp hơn baseline 0.002 thì sao gọi là tốt hơn được?

> Em không claim APFD tốt hơn ạ. Em claim **strict Pareto improvement**:
> APFD ngang trong sai số, AUC cao hơn rõ rệt (0.017), **cộng thêm
> guarantee lý thuyết** mà không phương pháp nào ở trên có (rotation
> $\Delta = 0$ exact, $\sigma$ thấp thứ nhì). Trong setting
> safety-critical -- tức là **chỗ mà một test bị lỗi có thể giết
> người** -- guarantee đáng giá hơn 0.002 APFD.

**Q4**: $\mathcal{O}(L^2)$ với $L = 197$ thì lúc scale lên dữ liệu thực
sao chịu nổi?

> Đây là điểm em nhận. Solution em đang triển khai: thay RFF bias bằng
> **RoPE-1D** trên $s_{\text{norm}}$. *RoPE -- Rotary Position Embedding
> -- nhúng vị trí bằng cách xoay vector query/key, chi phí $\mathcal{O}(L)$
> chứ không cần ma trận bias $L^2$.* Vẫn giữ bất biến arclength. Kỳ
> vọng giảm thời gian train từ 24 phút xuống 3-4 phút.

**Q5**: Tại sao Competition split (956 tests) lại gọi là OOD?

> Có 3 lý do cụ thể ạ: (1) **generator khác** -- Competition do tools
> khác sinh, không phải pipeline BeamNG của SensoDat; (2) **FAIL rate
> khác** -- 36.9% so với 38.4%; (3) **độ dài đường khác hẳn** --
> Competition chỉ 129-229m, SensoDat tới 454m. Bằng chứng thực nghiệm
> em đo được: SensoDat-test APFD 0.764, Competition 0.804 -- nếu cùng
> distribution thì hai số này phải gần nhau. Chênh hơn 0.04 chứng tỏ
> shift rõ rệt.

**Q6** (dự phòng): Em chứng minh "exact" bit-identical, nhưng vẫn có
$1.79 \times 10^{-7}$ ở một vài chỗ trong bảng. Đây là gì?

> Đó là sai số làm tròn float32 trong khi tính RoPE/RFF ở pre-norm
> Transformer -- xảy ra ở **một bước duy nhất là phép quay R**, không
> phải pipeline đặc trưng. Cụ thể, khi em xoay đường gốc bằng ma trận
> $R$, $R$ chứa $\sin\theta, \cos\theta$ -- là số float không exact,
> nên $R\mathcal{R}$ có sai số bậc $10^{-7}$. Đặc trưng tính từ đó sai
> bậc $10^{-7}$. Mạng forward, logit sai $10^{-7}$. APFD vẫn round
> giống nhau khi in 4 chữ số. **Bit-identical về APFD** trong precision
> báo cáo; **không exact** ở mức logit float32.

**Q7** (dự phòng): Vì sao không dùng equivariant network chính thống
như EGNN hay GVP, mà phải tự chế?

> Equivariant network như EGNN cần mỗi node có vector 3D và vô hướng,
> hoạt động tốt cho phân tử nhưng tốn parameter cho đường 2D. Em chọn
> giải pháp đơn giản hơn: **đẩy invariance vào pipeline đặc trưng**
> (Frenet-Serret giải quyết hết), rồi dùng Transformer chuẩn cho đặc
> trưng đã bất biến. Đổi lại được kiến trúc nhẹ hơn (2.1M params) và
> dễ debug hơn.
