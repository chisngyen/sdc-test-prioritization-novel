title: "Sổ tay công thức đầy đủ — RoadFury → SE2RoadNet"
subtitle: "Toàn bộ công thức trong slide + đào sâu method + ví dụ tính tay chi tiết"
lang: vi
---

> **Cách dùng tài liệu này (bản ASCII-safe cho mọi trình đọc).** Mỗi công thức trình bày theo 4 bước:
> **Công thức → Giải nghĩa ký hiệu → Ví dụ tính tay → Ý nghĩa/Lưu ý.**
> Mọi ví dụ tính tay dùng **cùng một con đường 5 điểm** (Phần 0) nên số liệu
> nối liền mạch: đặc trưng → chuẩn hoá → attention → logit → APFD. Tất cả số
> đã được kiểm lại bằng chạy code thực. Phần 7 là "bẫy câu hỏi" — đọc kỹ trước
> khi lên bảng.

---

# Phần 0 — Ký hiệu & Con đường ví dụ chuẩn

## 0.1 Bảng ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{R}=\{p_1,\dots,p_N\}\subset\mathbb{R}^2$ | một con đường = chuỗi điểm 2D (một test SDC) |
| $N$ | số điểm thô của road (biến thiên 64–197) |
| $L=197$ | số điểm sau resample (cố định) |
| $n$ | số test trong một lần xếp hạng (ranking) |
| $m$ | số test FAIL trong $n$ test |
| $p=m/n$ | tỉ lệ FAIL (prior) của split đánh giá |
| $f_\theta$ | mô hình prioritizer, $f_\theta:\mathcal{R}\mapsto[0,1]$ |
| $\pi$ | một hoán vị (thứ tự chạy) của $n$ test; $\pi^*$ là thứ tự tối ưu |
| $\theta_i$ | góc heading của đoạn $i$: $\theta_i=\operatorname{atan2}(\Delta y_i,\Delta x_i)$ |
| $s$ | độ dài cung (arc-length); $s/L$ là arc-length chuẩn hoá |
| $\kappa$ | độ cong (curvature) |
| $B$ | batch size; $d$ = số chiều model ($d{=}128$ RoadFury, $192$ SE2); $h{=}8$ số head; $d_k=d/h$ |
| $\sigma(\cdot)$ | hàm sigmoid (khi là hàm); $\sigma$ cũng dùng cho độ lệch chuẩn (khi là số) |
| $R,t$ | ma trận xoay $R\in SO(2)$, vector tịnh tiến $t\in\mathbb{R}^2$ |
| $\omega$ | tần số Fourier ngẫu nhiên (RFF) của attention bias |
| $\gamma,\alpha$ | tham số focal loss (focusing $\gamma$, cân bằng $\alpha$) |

## 0.2 Con đường ví dụ (dùng xuyên suốt)

$$
\text{pts}=\big[(0,0),\,(1,0),\,(2,0),\,(3,1),\,(4,2)\big],\qquad n=5.
$$

Hình dạng: 2 đoạn đầu đi thẳng về hướng Đông, đến điểm 2 **bẻ trái $45^\circ$**,
rồi 2 đoạn cuối đi xiên Đông–Bắc.

Vector đoạn $d_i=p_{i+1}-p_i$:

$$
d=\big[(1,0),\,(1,0),\,(1,1),\,(1,1)\big].
$$

Đây là điểm khởi đầu của mọi ví dụ bên dưới.

---

# Phần 1 — Bài toán & Chỉ số đánh giá

> **Lưu ý về chữ $n$.** Trong Phần 0, $n=5$ là **5 điểm của MỘT con đường**. Từ
> đây trở đi (APFD, AUC, multi-trial) $n$ là **5 test (5 con đường) trong một
> lần xếp hạng**. Đừng lẫn hai nghĩa: đặc trưng tính *trong* một road; APFD tính
> *giữa* các road.

## 1.1 Phát biểu bài toán (slide s-04)

$$
\text{Input: } \mathcal{R}=\{p_1,\dots,p_N\}\subset\mathbb{R}^2,\quad
\text{Scorer: } f_\theta:\mathcal{R}\mapsto[0,1],\quad
\text{Output: } \pi^*=\arg\max_{\pi}\ \operatorname{APFD}(\pi).
$$

- $f_\theta(\mathcal{R})$ = xác suất road làm xe **đi chệch lane (FAIL)**.
- Xếp test theo score giảm dần → hoán vị $\pi$. Mục tiêu: FAIL bị lộ **càng sớm càng tốt**.
- Nhãn train $y\in\{0,1\}$ (PASS/FAIL). Metric chính = **APFD**, phụ = **AUC**, **Rotation $\Delta$**.

#### Giải thích chi tiết: $f_\theta$, hoán vị $\pi$, và $\arg\max$

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Đọc là | Ở đâu ra / tính từ cái gì |
|---|---|---|
| $\mathcal{R}=\{p_1,\dots,p_N\}$ | một **con đường** (một test SDC) | chuỗi $N$ điểm 2D thô của road, $N\in[64,197]$ |
| $f_\theta$ | **scorer** (mô hình prioritizer) | mạng Transformer; $\theta$ = toàn bộ trọng số học được |
| $f_\theta(\mathcal{R})\in[0,1]$ | **điểm rủi ro** của road | qua sigmoid ở đầu ra → xác suất road này FAIL |
| $\pi$ | một **hoán vị** = thứ tự chạy $n$ test | do sắp xếp $n$ score giảm dần mà ra |
| $\pi^*$ | thứ tự **tối ưu** | hoán vị đạt APFD cao nhất trong mọi cách xếp |
| $\arg\max_\pi$ | "**lấy $\pi$ làm cực đại**" | trả về *đối số* $\pi$ khiến $\operatorname{APFD}(\pi)$ lớn nhất |

- $\theta$ (chỉ số dưới của $f$) là **tham số mạng**, khác hẳn $\theta_i$ (góc heading của đoạn $i$) ở Phần 2. Cùng chữ cái, hai nghĩa — đọc theo ngữ cảnh.
- **Hoán vị $\pi$**: một cách đánh số lại $\{1,\dots,n\}$, tức "chạy test nào trước, test nào sau". Có $n!$ hoán vị; ví dụ $n=5$ có $5!=120$ thứ tự.
- **$\arg\max$ vs $\max$**: $\max_\pi \operatorname{APFD}$ trả về *giá trị* APFD lớn nhất (một con số); $\arg\max_\pi \operatorname{APFD}$ trả về *thứ tự* $\pi^*$ đạt giá trị đó. Ta cần cái thứ tự để đem đi chạy test, nên bài toán viết theo $\arg\max$.

**(2) Vì sao phát biểu như vậy.** Ta không thể **duyệt cả $n!$ hoán vị** để tìm $\pi^*$ (bùng nổ tổ hợp). Mẹo: APFD **chỉ phụ thuộc vị trí của các FAIL** (mục 1.2), nên **xếp giảm dần theo score** $f_\theta$ là cách *tham lam* để tiệm cận $\pi^*$. Nếu model hoàn hảo (mọi FAIL có score cao hơn mọi PASS) thì thứ tự sort **chính là** $\pi^*$. Vậy học $f_\theta$ tốt = đẩy score FAIL lên cao = sort ra thứ tự gần $\pi^*$. Bài toán "xếp hạng" được quy về "học một hàm chấm điểm".

```
   road R  -->  f_theta(R)  -->  score in [0,1]  -->  sort giam dan  -->  pi
 (N diem 2D)     (Transformer)      (rui ro FAIL)      (n test)         (thu tu chay)
                                                                          |
                                                              danh gia bang APFD(pi)
```

**(3) Tính tay (mini).** Giả sử 5 test có score $[0.9,\,0.2,\,0.8,\,0.3,\,0.1]$ và nhãn thật $[\text{FAIL},\text{PASS},\text{FAIL},\text{PASS},\text{PASS}]$. Sort giảm dần theo score → thứ tự chạy là test#1 ($0.9$), test#3 ($0.8$), test#4 ($0.3$), test#2 ($0.2$), test#5 ($0.1$). Hai FAIL (test#1, test#3) rơi vào **rank 1 và 2** → $\pi$ này rất tốt (ta tính APFD của nó ở 1.2 = $0.80$).

**(4) Ý nghĩa.** Tóm 1 câu: bài toán là **học một hàm chấm rủi ro $f_\theta$ để khi sort giảm dần thì FAIL trồi lên đầu**, và $\pi^*=\arg\max_\pi\operatorname{APFD}$ chỉ nói "hãy chọn thứ tự đưa fault ra sớm nhất".

## 1.2 APFD — Average Percentage of Faults Detected (slide s-05, s-22, s-32)

**Công thức** (Rothermel et al., TSE 2001):

$$
\boxed{\ \operatorname{APFD}(\pi)=1-\frac{\sum_{i=1}^{m} TF_i}{n\cdot m}+\frac{1}{2n}\ }
$$

**Ký hiệu.** $n$ = tổng số test; $m$ = số test FAIL; $TF_i$ = **vị trí (rank, đánh
số từ 1)** của ca FAIL thứ $i$ trong thứ tự $\pi$. Miền giá trị $[0,1]$: ngẫu
nhiên $\approx 0.5$, lý tưởng (mọi FAIL lên đầu) $\to 1.0$.

**Ví dụ tính tay** ($n=5$, $m=2$):

- **Xếp tốt** — 2 FAIL ở vị trí 1 và 2:
$$
\operatorname{APFD}=1-\frac{1+2}{5\cdot 2}+\frac{1}{2\cdot 5}=1-\frac{3}{10}+\frac{1}{10}=\mathbf{0.80}.
$$
- **Xếp kém** — 2 FAIL ở vị trí 4 và 5:
$$
\operatorname{APFD}=1-\frac{4+5}{10}+\frac{1}{10}=1-0.9+0.1=\mathbf{0.20}.
$$
- **Xếp trung bình** — FAIL ở vị trí 1 và 3:
$$
\operatorname{APFD}=1-\frac{1+3}{10}+\frac{1}{10}=1-0.4+0.1=\mathbf{0.70}.
$$

**Ý nghĩa.** APFD chỉ phụ thuộc **thứ hạng** của các FAIL, không cần phân loại
đúng tuyệt đối — chỉ cần FAIL có score cao hơn PASS. Số hạng $\frac{1}{2n}$ là
hiệu chỉnh biên (để trường hợp lý tưởng tiệm cận 1).

### 1.2.1 Giải thích chi tiết: mổ xẻ TỪNG số hạng

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Ý nghĩa | Ví dụ ($n=5$, FAIL ở 1,3) |
|---|---|---|
| $n$ | tổng số test trong lần xếp hạng | $5$ |
| $m$ | số test FAIL (có fault) | $2$ |
| $TF_i$ | **rank** (vị trí, đếm từ 1) của FAIL thứ $i$ trong $\pi$ | $TF_1=1,\ TF_2=3$ |
| $\sum_i TF_i$ | tổng vị trí của mọi FAIL | $1+3=4$ |

**(2) Vì sao công thức như vậy — bóc từng khối.**

*Khối $\sum_i TF_i$ (tổng vị trí các FAIL): vì sao lại CỘNG các rank?*
Rank nhỏ = phát hiện sớm. Ta muốn **mọi** FAIL đều sớm, nên phạt theo **tổng quãng chờ**: cộng vị trí của tất cả FAIL. Tổng càng nhỏ → phát hiện càng sớm → (vì đứng sau dấu trừ) APFD càng lớn. Một FAIL nằm cuối bảng (rank lớn) làm tổng phình ra, kéo tụt điểm — đúng tinh thần "đừng để sót fault ở cuối".

*Khối $\dfrac{\sum TF_i}{n\cdot m}$ (chia cho $n\cdot m$): vì sao chia $n$, vì sao chia $m$?*
- Chia cho $n$: đổi **rank** (số nguyên $1..n$) thành **tỉ lệ suite đã chạy** ($TF_i/n\in(0,1]$). "Bắt được fault sau khi đã chạy $TF_i/n$ phần của bộ test." Nhờ vậy metric **so sánh được giữa các suite kích thước khác nhau**.
- Chia cho $m$: **lấy trung bình trên $m$ fault**. Chữ "Average" trong APFD là ở đây — trung bình tỉ lệ-suite-đã-chạy tại thời điểm bắt từng fault. Không chia $m$ thì suite nhiều fault bị phạt nặng vô lý.

*Khối $+\dfrac{1}{2n}$ (hiệu chỉnh biên): vì sao có nó?*
APFD nguyên bản là **diện tích dưới đường "phần trăm fault đã bắt vs phần trăm test đã chạy"**. Đường này là bậc thang; tính diện tích bằng hình thang thì mỗi bậc dôi ra **nửa bề rộng một ô** $=\frac{1}{2n}$. Cộng $\frac{1}{2n}$ chính là bù nửa-ô đó, để **trường hợp lý tưởng chạm sát 1** thay vì thiếu một chút. Không có nó, ngay cả xếp hoàn hảo cũng không đạt trần đẹp.

**(3) Suy ra chặn tốt nhất / xấu nhất (tính tay tổng quát).**

*Tốt nhất* — mọi FAIL lên đầu, rank $=1,2,\dots,m$:
$$
\sum TF_i=\frac{m(m+1)}{2}\ \Rightarrow\
\operatorname{APFD}_{\max}=1-\frac{m(m+1)/2}{nm}+\frac1{2n}
=1-\frac{m+1}{2n}+\frac{1}{2n}=1-\frac{m}{2n}=1-\frac{p}{2}.
$$

*Xấu nhất* — mọi FAIL xuống đáy, rank $=n{-}m{+}1,\dots,n$:
$$
\sum TF_i=mn-\frac{m(m-1)}{2}\ \Rightarrow\
\operatorname{APFD}_{\min}=1-\frac{mn-m(m-1)/2}{nm}+\frac1{2n}
=\frac{m}{2n}=\frac{p}{2}.
$$

| Trường hợp | Vị trí các FAIL | Công thức | Giá trị ($n{=}5,m{=}2,p{=}0.4$) |
|---|---|---|---|
| Tốt nhất | $1,2,\dots,m$ | $1-\dfrac{p}{2}$ | $1-0.2=\mathbf{0.80}$ |
| Ngẫu nhiên | rải đều | $\approx 0.5$ | $\approx 0.50$ |
| Xấu nhất | $n{-}m{+}1,\dots,n$ | $\dfrac{p}{2}$ | $0.2=\mathbf{0.20}$ |

Miền $[\,p/2,\ 1-p/2\,]$ **đối xứng quanh $0.5$**. Nhận xét quan trọng: hai ví dụ tính tay ở trên **chính là hai cực trị** — "xếp tốt" $\{1,2\}=0.80$ đúng bằng $1-p/2$ (trần), "xếp kém" $\{4,5\}=0.20$ đúng bằng $p/2$ (sàn). Với $n=5,m=2$ thì trần chỉ là $0.80$ chứ **không thể đạt $1.0$**: dù xếp hoàn hảo, vẫn phải "chạy tới" vị trí 2 mới lộ hết 2 fault. Trần $=1.0$ chỉ đạt tiệm cận khi $m$ rất nhỏ so với $n$ ($p\to0$).

**(4) Ý nghĩa.** Tóm 1 câu: APFD $=1-$ (tỉ lệ trung bình suite phải chạy để bắt fault) $+$ nửa-ô hiệu chỉnh; nó **chỉ nhìn thứ hạng FAIL**, thưởng "sớm", phạt "muộn", và bị **kẹp trong $[p/2,\,1-p/2]$** bởi chính tỉ lệ FAIL $p$.

## 1.3 Giao thức Multi-trial (slide s-22, s-33)

$$
\text{sample size}=\max\!\big(50,\ 0.3\times|\text{test}|\big)=\max(50,\,0.3\cdot 956)=\mathbf{287}\ \text{test}.
$$

- Mỗi trial: lấy ngẫu nhiên 287/956 test (seed cố định $42,43,\dots$), tính APFD.
- Lặp **30 trial**, báo cáo $\text{mean}\pm\sigma$. Ví dụ SE2RoadNet: $0.8048\pm0.0118$.
- **$\sigma$ (độ ổn định) thường là con số publication quan trọng hơn mean.**

#### Giải thích chi tiết: vì sao 287, vì sao 30 trial, vì sao seed cố định

**(1) Từng ký hiệu.** $|\text{test}|$ = cỡ tập test $=956$ (SensoDat). $0.3$ = tỉ lệ lấy mẫu mỗi trial (30%). $50$ = sàn cứng để trial không bị bé quá. $\max(\cdot)$ chọn con lớn hơn giữa "sàn 50" và "30% cỡ tập".

**(2) Vì sao các con số này.**
- **Vì sao 287?** $0.3\times956=286.8$, làm tròn lên $\lceil 286.8\rceil=\mathbf{287}$. Sàn $50$ chỉ có tác dụng khi tập test rất nhỏ ($0.3\cdot|\text{test}|<50$, tức $|\text{test}|<167$); ở đây $286.8>50$ nên $\max$ chọn $287$.
- **Vì sao 30% chứ không lấy cả 956?** Mỗi trial dùng một **tập con khác nhau** để ước lượng APFD "sẽ ra sao trên một mẻ test mới". Lấy cả 956 mỗi lần thì mọi trial giống hệt → $\sigma=0$ giả tạo, không đo được độ dao động. Bootstrap 30% tạo biến thiên để đo **độ ổn định thật**.
- **Vì sao 30 trial?** Đủ để trung bình cộng và độ lệch chuẩn hội tụ (sai số chuẩn của mean $\sim \sigma/\sqrt{30}$) mà vẫn rẻ về thời gian. 30 là ngưỡng "đủ mẫu" quy ước, cho $\pm\sigma$ đáng tin mà không tốn nhiều lần chạy.
- **Vì sao seed cố định ($42,43,\dots$)?** Để **tái lập bit-for-bit**: cùng seed → cùng 30 tập con → cùng dãy APFD → cùng $\text{mean}\pm\sigma$. Quan trọng khi **so hai model công bằng**: RoadFury và SE2RoadNet phải gặp **đúng cùng 30 mẻ test**, nếu không chênh lệch có thể chỉ do may rủi chia mẫu. Seed = "khoá" sự ngẫu nhiên để khác biệt còn lại là do model, không do dữ liệu.

**(3) Tính tay (đọc kết quả).** Sample size $=\lceil 0.3\cdot956\rceil=287$. Chạy 30 lần → 30 giá trị APFD; ví dụ SE2RoadNet cho $\text{mean}=0.8048$, $\sigma=0.0118$, viết gọn $0.8048\pm0.0118$.

**(4) Ý nghĩa.** Tóm 1 câu: multi-trial biến "một điểm APFD" thành "**một phân phối APFD**" ($\text{mean}\pm\sigma$) để báo cả **độ tốt** lẫn **độ ổn định**, với seed cố định để mọi so sánh đều **lặp lại được và công bằng**.

## 1.4 Rotation probe & Resolution probe (slide s-22, s-23, s-34)

$$
\Delta_{\text{rot}}=\max_{\phi} \operatorname{APFD}(\phi)-\min_{\phi}\operatorname{APFD}(\phi),\quad
\phi\in\{0^\circ,30^\circ,60^\circ,90^\circ,180^\circ,-45^\circ\}.
$$

$$
\Delta_{\text{res}}=\max_{N}\operatorname{APFD}(N)-\min_{N}\operatorname{APFD}(N),\quad
N\in\{64,96,128,160,197\}.
$$

- SE2RoadNet: $\Delta_{\text{rot}}=\mathbf{0.0000}$ (exact, đến từng bit float).
- Resolution: $\Delta_{\text{res}}\approx 0.0012$ — **là số của mô hình FNO (Exp 01)**, xem Phần 7.

#### Giải thích chi tiết: $\Delta$ là gì, đo cái gì

**(1) Từng ký hiệu.**
- $\phi$ = **góc xoay** áp lên toàn bộ con đường trước khi đưa vào model (xoay cả road quanh gốc). Tập 6 góc $\{0,30,60,90,180,-45\}$ độ là 6 phép "chụp lại con đường ở tư thế khác".
- $N$ = **số điểm lấy mẫu** (độ phân giải) của road, từ thưa (64) đến dày (197).
- $\operatorname{APFD}(\phi)$ / $\operatorname{APFD}(N)$ = APFD của **cùng model, cùng tập test**, chỉ khác góc xoay (hoặc độ phân giải).
- $\Delta$ = **biên độ dao động** = (giá trị lớn nhất) $-$ (giá trị nhỏ nhất) qua các phép biến đổi.

**(2) Vì sao dùng $\Delta=\max-\min$.** $\Delta$ đo trực tiếp câu hỏi: "**xoay/lấy mẫu lại con đường có làm điểm số đổi không?**" Nếu model **bất biến** (invariant) thì mọi $\phi$ (hay mọi $N$) cho **cùng một APFD** → $\max=\min$ → $\Delta=0$. $\Delta$ càng lớn = model càng **nhạy khung/độ phân giải** = càng dễ gãy khi road bị xoay hay đổi sampling-rate. Đây là "điểm mù" mà SE2RoadNet ra đời để bịt.

**(3) Tính tay (đọc số).**
- **Rotation:** SE2RoadNet cho APFD **y hệt** ở cả 6 góc (đặc trưng chỉ dùng đại lượng nội tại: chiều dài, hiệu góc, curvature — bất biến xoay), nên $\Delta_{\text{rot}}=\max-\min=\mathbf{0.0000}$, exact đến từng bit float. (Chi tiết residual float $1.79\times10^{-7}$ ở mức logit: Phần 7.9 — không đổi ranking nên APFD vẫn bằng nhau.)
- **Resolution:** ở 5 độ phân giải $N\in\{64,\dots,197\}$, APFD chênh nhau nhỏ, $\Delta_{\text{res}}\approx\mathbf{0.0012}$. **Cẩn thận:** con số $0.0012$ này là của **mô hình FNO (Exp 01)**, KHÔNG phải của SE2RoadNet (SE2 mới chỉ bảo chứng bất biến *xoay* exact, bất biến *độ phân giải* còn là xấp xỉ — xem Phần 7.3).

**(4) Ý nghĩa.** Tóm 1 câu: $\Delta$ là **thước đo "model có bất biến không"** — $\Delta=0$ nghĩa là xoay road (rot) hoặc đổi độ phân giải (res) **không đổi kết quả**, đúng thứ ta cần cho một baseline "rotation-invariant, resolution-invariant".

## 1.5 AUC và Đẳng thức AUC – APFD (slide s-24, s-25)

**AUC** (ROC) = xác suất một FAIL ngẫu nhiên có score cao hơn một PASS ngẫu nhiên
(thống kê Mann–Whitney U):

$$
\operatorname{AUC}=\frac{\#\{(i,j):\ y_i=1,\,y_j=0,\ \text{score}_i>\text{score}_j\}}{m\,(n-m)}.
$$

**Đẳng thức mấu chốt** (không có ties):

$$
\boxed{\ \operatorname{APFD}=(1-p)\,\operatorname{AUC}+\frac{p}{2}\ },\qquad p=\frac{m}{n}.
$$

Suy ra AUC hiệu chỉnh theo prior: $\operatorname{AUC}^*=\dfrac{\operatorname{APFD}-p/2}{1-p}$.

**Ví dụ tính tay.** $n=5$, nhãn $[\text{FAIL},\text{PASS},\text{FAIL},\text{PASS},\text{PASS}]$,
$m=2$, $p=0.4$. Xếp theo score: FAIL ở rank 1 và 3.

- APFD $=1-\frac{1+3}{10}+\frac{1}{10}=0.70$.
- AUC: 6 cặp (FAIL,PASS). FAIL\@1 thắng cả 3 PASS; FAIL\@3 thắng 2 PASS (đứng sau nó) → $5/6\approx0.8333$.
- Kiểm đẳng thức: $(1-0.4)\cdot0.8333+\frac{0.4}{2}=0.5+0.2=\mathbf{0.70}$ (OK).

**Ý nghĩa.** Giải thích hiện tượng "**AUC tăng nhưng APFD phẳng**": AUC đo trên
SensoDat (prior $p$ khác), APFD đo trên Competition (prior khác) → chênh lệch là
**hiệu ứng prior**, không phải mâu thuẫn. Đây là câu trả lời chuẩn cho câu hỏi
về slide s-25.

### 1.5.1 Giải thích chi tiết: ROC / Mann–Whitney, đếm cặp thắng, và CHỨNG MINH đẳng thức

#### (a) ROC và AUC là gì

**ROC curve** (Receiver Operating Characteristic): quét ngưỡng $t$ từ cao xuống thấp; ở mỗi $t$, coi "score $\ge t$ → dự đoán FAIL" rồi chấm hai trục:
- **TPR** (true positive rate) $=$ tỉ lệ FAIL bị bắt đúng (trục tung).
- **FPR** (false positive rate) $=$ tỉ lệ PASS bị báo nhầm (trục hoành).

**AUC** = **diện tích dưới đường ROC** đó $\in[0,1]$. Điểm mấu chốt: AUC **chỉ phụ thuộc THỨ HẠNG** của score (đổi $0.9\to0.99$ không đổi AUC nếu thứ tự giữ nguyên), **không** phụ thuộc giá trị tuyệt đối hay ngưỡng.

**Đẳng thức Mann–Whitney U.** Có một định lý cổ điển: diện tích dưới ROC **đúng bằng** xác suất một FAIL ngẫu nhiên có score cao hơn một PASS ngẫu nhiên:
$$
\operatorname{AUC}=\Pr[\text{score}(\text{FAIL})>\text{score}(\text{PASS})]
=\frac{\#\{\text{cặp (FAIL,PASS) mà FAIL thắng}\}}{m\,(n-m)}.
$$
Mẫu số $m(n-m)$ = **tổng số cặp** ghép một FAIL với một PASS ($m$ FAIL $\times$ $(n{-}m)$ PASS). Tử số = số cặp mà FAIL **xếp trước** PASS. Vậy AUC = "tỉ lệ trận thắng" trong giải đấu FAIL-đấu-PASS.

#### (b) Đếm 6 cặp thắng — tính tay chi tiết

Nhãn theo **thứ tự rank** (rank 1 = score cao nhất): rank1=FAIL, rank2=PASS, rank3=FAIL, rank4=PASS, rank5=PASS. Có $m=2$ FAIL, $n-m=3$ PASS → $2\times3=6$ cặp. FAIL "thắng" khi đứng **trước** (rank nhỏ hơn) PASS:

| Cặp (FAIL, PASS) | rank FAIL | rank PASS | FAIL trước? | Kết quả |
|---|---|---|---|---|
| (FAIL@1, PASS@2) | 1 | 2 | có | thắng |
| (FAIL@1, PASS@4) | 1 | 4 | có | thắng |
| (FAIL@1, PASS@5) | 1 | 5 | có | thắng |
| (FAIL@3, PASS@2) | 3 | 2 | không | **thua** |
| (FAIL@3, PASS@4) | 3 | 4 | có | thắng |
| (FAIL@3, PASS@5) | 3 | 5 | có | thắng |

Thắng $5$, thua $1$ → $\operatorname{AUC}=\dfrac{5}{6}=0.8333$. (FAIL@1 quét sạch 3 PASS; FAIL@3 chỉ thua PASS@2 vì PASS đó xếp trước nó.)

#### (c) Chứng minh đẳng thức $\operatorname{APFD}=(1-p)\operatorname{AUC}+\frac p2$ (từng bước)

Gọi rank của $m$ FAIL là $r_1<r_2<\dots<r_m$ (vị trí trong $\pi$).

**Bước 1 — đếm cặp thắng của một FAIL.** FAIL thứ $i$ (rank $r_i$) có $n-r_i$ phần tử đứng **sau** nó. Trong đó số FAIL đứng sau $=m-i$ (vì $r_i$ là rank FAIL nhỏ thứ $i$). Nên số PASS mà FAIL này thắng $=(n-r_i)-(m-i)$.

**Bước 2 — tổng cặp thắng.**
$$
\text{thắng}=\sum_{i=1}^m\big[(n-r_i)-(m-i)\big]
=nm-\sum_i r_i-\underbrace{\sum_{i=1}^m(m-i)}_{=\,m(m-1)/2}
= nm-\sum_i r_i-\frac{m(m-1)}{2}.
$$

**Bước 3 — viết AUC.** Chia cho $m(n-m)$:
$$
\operatorname{AUC}=\frac{nm-\sum_i r_i-\tfrac{m(m-1)}2}{m(n-m)}.
$$

**Bước 4 — rút $\sum_i r_i$ theo AUC.**
$$
\sum_i r_i=nm-\frac{m(m-1)}2-\operatorname{AUC}\cdot m(n-m).
$$

**Bước 5 — thay vào APFD** ($\sum TF_i=\sum_i r_i$):
$$
\operatorname{APFD}=1-\frac{\sum_i r_i}{nm}+\frac1{2n}
=1-\frac{nm-\tfrac{m(m-1)}2-\operatorname{AUC}\,m(n-m)}{nm}+\frac1{2n}.
$$
Tách từng hạng tử ($\dfrac{nm}{nm}=1$):
$$
=1-1+\frac{m(m-1)/2}{nm}+\operatorname{AUC}\frac{m(n-m)}{nm}+\frac1{2n}
=\frac{m-1}{2n}+\operatorname{AUC}\frac{n-m}{n}+\frac1{2n}.
$$

**Bước 6 — gộp lại.** $\dfrac{m-1}{2n}+\dfrac1{2n}=\dfrac{m}{2n}=\dfrac p2$, và $\dfrac{n-m}{n}=1-p$:
$$
\boxed{\ \operatorname{APFD}=(1-p)\operatorname{AUC}+\frac p2\ }.
$$

**Kiểm lại bằng ví dụ (b):** $p=0.4$, $\operatorname{AUC}=5/6$ → $(1-0.4)\cdot\frac56+\frac{0.4}2=0.6\cdot0.8333+0.2=0.5+0.2=\mathbf{0.70}$, khớp APFD tính trực tiếp. Đảo lại: $\operatorname{AUC}^*=\dfrac{\operatorname{APFD}-p/2}{1-p}=\dfrac{0.7-0.2}{0.6}=\dfrac{0.5}{0.6}=0.8333$. (Cũng thấy ngay: đẳng thức tốt nhất/xấu nhất ở 1.2.1 là hệ quả — AUC$=1$ cho APFD$=1-p/2$; AUC$=0$ cho APFD$=p/2$.)

#### (d) Ý nghĩa "AUC tăng nhưng APFD phẳng"

Đẳng thức cho thấy APFD là **hàm afin của AUC** với **hệ số góc $(1-p)$** và **dịch $+p/2$**:
$$
\Delta\operatorname{APFD}=(1-p)\cdot\Delta\operatorname{AUC}.
$$
Hai lý do khiến AUC nhích lên mà APFD gần như đứng yên:

1. **Hiệu ứng prior (câu trả lời chuẩn cho s-25).** AUC thường báo trên **SensoDat** (prior $p$ này), còn APFD báo trên **Competition** (prior $p$ khác). Cùng một model nhưng **đo ở hai prior khác nhau** thì hai số không buộc phải đi cùng chiều — chênh lệch là **hiệu ứng prior**, KHÔNG phải mâu thuẫn nội tại.
2. **Hệ số $(1-p)$ nén AUC.** Khi FAIL nhiều ($p$ lớn), $(1-p)$ nhỏ, nên một cải thiện AUC $\Delta\operatorname{AUC}$ chỉ chuyển thành $(1-p)\Delta\operatorname{AUC}$ rất bé ở APFD → APFD "phẳng". Ví dụ $p=0.4$: tăng AUC $0.02$ chỉ đẩy APFD lên $0.6\cdot0.02=0.012$.

Tóm 1 câu: APFD và AUC **cùng đo thứ hạng nhưng khác thang** — APFD $=(1-p)\operatorname{AUC}+p/2$; vậy "AUC lên, APFD phẳng" là do **prior khác nhau** và **hệ số nén $(1-p)$**, hoàn toàn nhất quán chứ không nghịch lý.

---

# Phần 2 — Đặc trưng RoadFury: 10 kênh (slide s-12, s-38, s-39, s-40)

**Pipeline:** `road_points` $(N,2)$ → 10 kênh → resample về $L=197$ → z-norm →
ma trận $(197\times 10)$. Ví dụ dưới đây tính trực tiếp trên 5 điểm (chưa resample).

## 2.1 f0 — Chiều dài đoạn (segment length)

$$
\text{seg}_i=\lVert p_{i+1}-p_i\rVert_2=\sqrt{\Delta x_i^2+\Delta y_i^2},\quad
\text{pad "edge" 1 phần tử cuối}.
$$

**Ví dụ:** $d=[(1,0),(1,0),(1,1),(1,1)]$ →
$\text{seg}=[\sqrt1,\sqrt1,\sqrt2,\sqrt2]=[1,1,1.4142,1.4142]$, pad cuối →
$$\text{f0}=[1.0000,\,1.0000,\,1.4142,\,1.4142,\,1.4142].$$

#### Giải thích chi tiết: f0 (norm Euclid + pad edge)

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Nghĩa | Tính từ cái gì |
|---|---|---|
| $p_i$ | điểm thứ $i$ của road, toạ độ $(x_i,y_i)$ | dữ liệu thô của test |
| $p_{i+1}-p_i$ | vector đoạn $d_i$ (mũi tên nối 2 điểm liên tiếp) | hiệu 2 điểm |
| $\Delta x_i,\ \Delta y_i$ | thành phần ngang/dọc của $d_i$ | $\Delta x_i=x_{i+1}-x_i$, $\Delta y_i=y_{i+1}-y_i$ |
| $\lVert\cdot\rVert_2$ | **chuẩn Euclid (L2)** = độ dài hình học của vector | định lý Pythagoras $\sqrt{\Delta x^2+\Delta y^2}$ |
| $\text{seg}_i$ | **chiều dài đoạn** thứ $i$ (mét) | chính là $\lVert d_i\rVert_2$ |
| pad "edge" | lặp lại **giá trị cuối** thêm 1 lần | bù cho việc thiếu 1 phần tử |

**(2) Vì sao công thức như vậy.** Chuẩn Euclid chính là *chiều dài đường thẳng*
nối hai điểm — cạnh huyền của tam giác vuông có 2 cạnh góc vuông $\Delta x,\Delta y$
(Pythagoras). Đây là đại lượng **nội tại**: xoay hay tịnh tiến con đường thì độ dài
mỗi đoạn không đổi (đó là lý do seg xuất hiện lại nguyên vẹn trong SE2RoadNet).

Vì sao cần **pad**? Có $n$ điểm nhưng chỉ có $n-1$ đoạn (đoạn nối điểm cuối với... không
có gì). Kênh phải dài đúng $n$ để ghép cùng 9 kênh kia thành ma trận $(n\times10)$,
nên phải bù thêm 1 phần tử. Ta pad kiểu **"edge" (lặp giá trị cuối)** chứ **không pad 0**:
pad 0 sẽ tạo ra một "đoạn dài 0" giả ở cuối (như thể xe đứng im), làm lệch thống kê;
lặp giá trị cuối giữ đúng "nhịp" chiều dài của khúc cuối.

**(3) Tính tay từng bước** trên con đường ví dụ chuẩn $d=[(1,0),(1,0),(1,1),(1,1)]$:

| Đoạn | $\Delta x$ | $\Delta y$ | $\sqrt{\Delta x^2+\Delta y^2}$ | seg |
|---|---|---|---|---|
| $d_0$ | 1 | 0 | $\sqrt{1+0}$ | 1.0000 |
| $d_1$ | 1 | 0 | $\sqrt{1+0}$ | 1.0000 |
| $d_2$ | 1 | 1 | $\sqrt{1+1}=\sqrt2$ | 1.4142 |
| $d_3$ | 1 | 1 | $\sqrt{1+1}=\sqrt2$ | 1.4142 |
| pad edge | — | — | lặp lại $d_3$ | 1.4142 |

→ $\text{f0}=[1.0000,\,1.0000,\,1.4142,\,1.4142,\,1.4142]$.

**(4)** Tóm 1 câu: f0 = *độ dài mỗi bước đi* (chuẩn Euclid của vector đoạn), pad "edge"
1 phần tử để đủ $n$ điểm; đây là đại lượng bất biến xoay/tịnh tiến.

## 2.2 f1 — Biến thiên góc tuyệt đối $|\Delta\theta|$

$$
\theta_i=\operatorname{atan2}(\Delta y_i,\Delta x_i),\quad
\Delta\theta_i=\operatorname{wrap}(\theta_{i+1}-\theta_i),\quad
\operatorname{wrap}(a)=((a+\pi)\bmod 2\pi)-\pi\in(-\pi,\pi].
$$

Kênh $= |\Delta\theta_i|$, pad 0 hai đầu.

**Ví dụ:** $\theta=[0^\circ,0^\circ,45^\circ,45^\circ]=[0,0,0.7854,0.7854]$ rad;
$\Delta\theta=[0,\,0.7854,\,0]$; lấy trị tuyệt đối, pad $(1,1)$ →
$$\text{f1}=[0,\,0,\,0.7854,\,0,\,0].$$
(Chỉ điểm 2 có cua $45^\circ=0.7854$ rad.)

**Lưu ý.** `wrap` xử lý nhảy $\pm180^\circ$: ví dụ $170^\circ\to-170^\circ$ thực ra
chỉ cua $20^\circ$.

### 2.2.1 Giải thích chi tiết: θ (heading), `atan2`, và `wrap`

#### (a) θ — "góc hướng" — ở đâu ra?

θ là **hướng đi của mỗi đoạn đường**, tính từ **vector đoạn** $d_i=p_{i+1}-p_i$
(mũi tên nối điểm này sang điểm kế), **không** phải từ tọa độ điểm.

| đoạn | $d_i=p_{i+1}-p_i$ | ý nghĩa |
|---|---|---|
| $d_0$ | $(1,0)$ | đi sang phải (Đông) |
| $d_1$ | $(1,0)$ | đi sang phải (Đông) |
| $d_2$ | $(1,1)$ | đi xiên lên (Đông–Bắc) |
| $d_3$ | $(1,1)$ | đi xiên lên (Đông–Bắc) |

$$
\theta_i=\operatorname{atan2}(\Delta y_i,\Delta x_i)=\operatorname{atan2}(d_i.y,\ d_i.x)
\ \to\ \theta=[0^\circ,\,0^\circ,\,45^\circ,\,45^\circ]=[0,\,0,\,0.7854,\,0.7854]\ \text{rad}.
$$

θ trả lời câu hỏi: "*xe đang chạy về hướng nào?*" tại mỗi đoạn.

#### (b) `atan2(y, x)` là gì, tính số độ ra sao?

`atan2(y, x)` = **góc của vector $(x,y)$** so với trục Ox dương, trong khoảng $(-\pi,\pi]$.

**Vì sao không dùng `arctan(y/x)`?** Vì `arctan` chỉ cho góc trong $(-90^\circ,90^\circ)$
nên **mất thông tin phần tư**: vector $(1,1)$ và $(-1,-1)$ đều có $y/x=1$ → `arctan`
ra cùng $45^\circ$, dù chúng **ngược hướng nhau**. `atan2` xét **dấu riêng của $x$
và $y$** nên biết đúng phần tư.

**Quy ước dấu góc.** Mốc $0^\circ$ = hướng Đông ($+x$). Quay **ngược chiều kim đồng
hồ** (hất lên) → góc **dương** $(+)$; quay **cùng chiều kim đồng hồ** (hạ xuống) →
góc **âm** $(-)$:

```
                 90 deg (Bac, +y)
                   |
       +135 deg    |    +45 deg
             \     |     /
               \   |   /
 180 deg ----------+---------- 0 deg  (Dong, +x)
 (Tay)         /   |   \
             /     |     \
       -135 deg    |    -45 deg
                   |
                 -90 deg (Nam, -y)
```

> **Mẹo nhớ:** *dấu* của góc = *dấu của $y$* (thành phần đứng): $y>0$ hướng lên →
> $+$, $y<0$ hướng xuống → $-$. *Độ lớn* liên quan $x$: $x>0$ (bên Đông) →
> $|\text{góc}|<90^\circ$, $x<0$ (bên Tây) → $|\text{góc}|>90^\circ$.

**Cách tính.** Đặt góc cơ sở $\beta=\arctan\dfrac{|y|}{|x|}\in[0^\circ,90^\circ]$, rồi tùy phần tư:

| $x$ | $y$ | Phần tư | Công thức | Khoảng góc |
|---|---|---|---|---|
| $+$ | $+$ | I (Đông–Bắc) | $\theta=+\beta$ | $0^\circ \to +90^\circ$ |
| $-$ | $+$ | II (Tây–Bắc) | $\theta=180^\circ-\beta$ | $+90^\circ \to +180^\circ$ |
| $-$ | $-$ | III (Tây–Nam) | $\theta=-(180^\circ-\beta)$ | $-90^\circ \to -180^\circ$ |
| $+$ | $-$ | IV (Đông–Nam) | $\theta=-\beta$ | $0^\circ \to -90^\circ$ |

**Ví dụ 4 vector cùng độ lớn** ($|x|=|y|=1$ nên $\beta=\arctan 1=45^\circ$ — chỉ **dấu** làm khác):

| Vector | $x,y$ | Phần tư | Tính | Kết quả |
|---|---|---|---|---|
| $(1,1)$ | $+,+$ | I | $+\beta$ | $\mathbf{+45^\circ}$ |
| $(-1,1)$ | $-,+$ | II | $180^\circ-45^\circ$ | $\mathbf{+135^\circ}$ |
| $(-1,-1)$ | $-,-$ | III | $-(180^\circ-45^\circ)$ | $\mathbf{-135^\circ}$ |
| $(1,-1)$ | $+,-$ | IV | $-45^\circ$ | $\mathbf{-45^\circ}$ |

→ Tính tay `atan2(1,1)`: $x{+},\,y{+}$ → phần tư I → $\beta=\arctan\frac{1}{1}=45^\circ=0.7854$ rad.

#### (c) `wrap` là gì, tính ra sao?

$$
\operatorname{wrap}(a)=((a+\pi)\bmod 2\pi)-\pi\ \in(-\pi,\pi].
$$

Ép **một góc bất kỳ về khoảng** $(-180^\circ,180^\circ]$: cộng $\pi$ → lấy dư cho $2\pi$ → trừ lại $\pi$.

**Vì sao cần?** Khi lấy **hiệu hai góc** $\theta_{i+1}-\theta_i$, kết quả có thể "lố"
ra ngoài khoảng và cho **góc cua sai** — nhất là khi xe băng qua mốc $\pm180^\circ$:

- Đoạn trước $\theta=170^\circ$, đoạn sau $\theta=-170^\circ$. Hiệu thô $=-170^\circ-170^\circ=-340^\circ$ (nghe như xe quay gần trọn vòng).
- Thực tế xe **chỉ cua $20^\circ$**. Tính từng bước:
$$
\operatorname{wrap}(-340^\circ):\quad
-340^\circ+180^\circ=-160^\circ;\quad
-160^\circ \bmod 360^\circ=200^\circ;\quad
200^\circ-180^\circ=\mathbf{+20^\circ}.
$$

Trong road ví dụ, các hiệu góc là $0$ và $0.7854$ (đã nằm trong khoảng) nên **wrap
không đổi gì** — nó chỉ "ra tay" khi có cú băng mốc $\pm180^\circ$.

#### (d) Ghép lại → ra f1

$$
\theta=[0,0,0.7854,0.7854]\ \to\ \Delta\theta=[\,0,\ 0.7854,\ 0\,]\ \to\ |\Delta\theta|\ \to\ \text{pad}(1,1)\ \to\ \text{f1}=[0,0,0.7854,0,0].
$$

Lấy $|\Delta\theta|$ rồi **pad 0 hai đầu** (điểm đầu và cuối không có "khúc cua" vì
cần 3 điểm mới xác định một cua) → chỉ **điểm 2** có cua $0.7854$ rad $=45^\circ$,
đúng chỗ con đường bẻ trái.

> **Tóm 1 câu:** θ = *hướng đi* (từ `atan2` của vector đoạn); dấu $+/-$ của θ do
> *hướng lên/xuống* (dấu $y$); Δθ = *góc cua* (hiệu 2 hướng liên tiếp, đã `wrap`
> để không sai ở mốc $180^\circ$); f1 = *độ lớn cua* $|\Delta\theta|$.

## 2.3 f2 — Độ cong Menger $\kappa$

$$
a,b,c=\text{3 cạnh tam giác 3 điểm liên tiếp};\quad
s=\tfrac{a+b+c}{2};\quad
\text{Area}=\sqrt{s(s-a)(s-b)(s-c)}\ \text{(Heron)};
$$
$$
\boxed{\ \kappa=\frac{1}{R}=\frac{4\cdot\text{Area}}{a\,b\,c}\ },\qquad R=\frac{a\,b\,c}{4\cdot\text{Area}}\ (\text{bán kính đường tròn ngoại tiếp}).
$$
Kênh lấy $|\kappa|$ (bỏ dấu), pad 0 hai đầu.

**Ví dụ** — tam giác điểm $(1,0),(2,0),(3,1)$ (curvature gán cho điểm giữa = điểm 2):

$$
a=1,\quad b=\sqrt2=1.4142,\quad c=\sqrt5=2.2361,
$$
$$
s=\tfrac{1+1.4142+2.2361}{2}=2.3251,\quad
\text{Area}=\sqrt{2.3251\cdot1.3251\cdot0.9109\cdot0.0891}=0.5000,
$$
$$
R=\frac{1\cdot1.4142\cdot2.2361}{4\cdot0.5}=\frac{3.1623}{2}=1.5811,\quad
\kappa=\frac{1}{1.5811}=\mathbf{0.6325}.
$$

Điểm 1 và 3 thẳng hàng → Area $=0$ → $\kappa=0$. Vậy
$$\text{f2}=[0,\,0,\,0.6325,\,0,\,0].$$

**Lưu ý.** Menger dùng bán kính đường tròn ngoại tiếp → **bỏ dấu**. So sánh: kênh
curvature của SE2RoadNet (Phần 3) là **có dấu** và bằng $0.6506$ (khác cách tính).

#### Giải thích chi tiết: f2 (độ cong Menger)

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Nghĩa | Ở đâu ra |
|---|---|---|
| 3 điểm | $p_{i-1},p_i,p_{i+1}$ (điểm giữa là điểm gán κ) | 3 điểm liên tiếp của road |
| $a,b,c$ | **3 cạnh** tam giác do 3 điểm tạo ra | $a=\lVert p_i-p_{i-1}\rVert$, $b=\lVert p_{i+1}-p_i\rVert$, $c=\lVert p_{i+1}-p_{i-1}\rVert$ (cạnh nối 2 đầu, "dây cung") |
| $s$ | **nửa chu vi** (semi-perimeter) | $s=(a+b+c)/2$ |
| Area | **diện tích tam giác** | công thức Heron $\sqrt{s(s-a)(s-b)(s-c)}$ |
| $R$ | **bán kính đường tròn ngoại tiếp** 3 điểm | $R=abc/(4\,\text{Area})$ |
| $\kappa=1/R$ | **độ cong Menger** | nghịch đảo bán kính |

**(2) Vì sao công thức như vậy.**

- *Tam giác 3 điểm ở đâu ra:* 3 điểm không thẳng hàng luôn xác định **duy nhất một
  đường tròn** đi qua cả 3 (đường tròn ngoại tiếp). Menger định nghĩa độ cong rời rạc
  tại điểm giữa = độ cong của đường tròn đó. Đây là cách "đo cua" chỉ từ hình học,
  **không cần đạo hàm** — hợp với dữ liệu điểm rời rạc.
- *Vì sao $\kappa=1/R$:* đường tròn to (bán kính lớn) trông gần **thẳng** → cua ít →
  $\kappa$ nhỏ; đường tròn nhỏ → cua gắt → $\kappa$ lớn. Đường thẳng = đường tròn bán
  kính $\infty$ → $\kappa=0$. Vậy độ cong tỉ lệ nghịch với bán kính.
- *Heron ở đâu ra:* để tính $R$ ta cần **diện tích** tam giác, nhưng ta chỉ có 3 **cạnh**
  (không có chiều cao, không cần toạ độ). Công thức Heron $\text{Area}=\sqrt{s(s-a)(s-b)(s-c)}$
  cho diện tích **chỉ từ 3 độ dài cạnh** — mà 3 cạnh lại là chuẩn Euclid nên **bất biến
  xoay/tịnh tiến**. Từ đó $R=\dfrac{abc}{4\,\text{Area}}$ là công thức chuẩn của bán kính
  đường tròn ngoại tiếp, và $\kappa=1/R=\dfrac{4\,\text{Area}}{abc}$.
- *Đơn vị:* $\kappa$ có đơn vị **1/mét** (nghịch đảo chiều dài). Diễn giải kiểu $d\theta/ds$
  thì là **rad/m** (rad không thứ nguyên nên vẫn là 1/m): mỗi mét đi được thì hướng
  xe đổi bao nhiêu radian.

**(3) Tính TAY từng bước** — tam giác 3 điểm liên tiếp $(1,0),(2,0),(3,1)$ (κ gán cho điểm 2):

| Đại lượng | Phép tính | Kết quả |
|---|---|---|
| $a$ | $\lVert(2,0)-(1,0)\rVert=\sqrt{1^2+0^2}$ | $1$ |
| $b$ | $\lVert(3,1)-(2,0)\rVert=\sqrt{1^2+1^2}=\sqrt2$ | $1.4142$ |
| $c$ (dây cung) | $\lVert(3,1)-(1,0)\rVert=\sqrt{2^2+1^2}=\sqrt5$ | $2.2361$ |
| $s$ | $(1+1.4142+2.2361)/2$ | $2.3251$ |
| $s-a$ | $2.3251-1$ | $1.3251$ |
| $s-b$ | $2.3251-1.4142$ | $0.9109$ |
| $s-c$ | $2.3251-2.2361$ | $0.0891$ |
| Area | $\sqrt{2.3251\cdot1.3251\cdot0.9109\cdot0.0891}$ | $0.5000$ |
| $R$ | $\dfrac{1\cdot1.4142\cdot2.2361}{4\cdot0.5}=\dfrac{3.1623}{2}$ | $1.5811$ |
| $\kappa$ | $1/1.5811$ | $\mathbf{0.6325}$ |

Kiểm nhanh: diện tích tam giác này cũng bằng $\tfrac12\lvert$đáy $\times$ cao$\rvert
=\tfrac12\cdot2\cdot1=0.5$ (đáy từ $(1,0)$ đến $(3,1)$... thực ra đáy $= a=1$ nằm ngang,
cao $=1$ → $\tfrac12\cdot1\cdot1=0.5$) — khớp Heron. Với điểm 1 và 3: bộ 3 điểm chứa
chúng thẳng hàng nên Area $=0\Rightarrow\kappa=0$.

$$\text{f2}=[0,\,0,\,0.6325,\,0,\,0].$$

**(4)** Tóm 1 câu: f2 = *độ gắt của khúc cua* = $1/$bán kính đường tròn qua 3 điểm liên
tiếp (Heron cho diện tích chỉ từ 3 cạnh nên bất biến xoay); đơn vị 1/m, **bỏ dấu**.

## 2.4 f3 — Biến thiên độ cong (jerk hình học) $\Delta\kappa$

$$
\Delta\kappa_i=\kappa_{i+1}-\kappa_i,\quad\text{pad 0 cuối}.
$$
**Ví dụ:** $\text{diff}([0,0,0.6325,0,0])=[0,0.6325,-0.6325,0]$, pad cuối →
$$\text{f3}=[0,\,+0.6325,\,-0.6325,\,0,\,0].$$

#### Giải thích chi tiết: f3 (sai phân độ cong, jerk hình học)

**(1) Ký hiệu.** $\kappa_i$ = độ cong Menger tại điểm $i$ (từ f2); $\Delta\kappa_i$ =
**sai phân bậc 1** (finite difference) = độ cong điểm sau trừ điểm hiện tại;
"pad 0 cuối" = thêm 1 số 0 ở đuôi cho đủ $n$ phần tử (vì diff của $n$ số cho $n-1$ số).

**(2) Vì sao công thức như vậy.** f2 cho biết *đang cua gắt cỡ nào*, nhưng chưa cho
biết *độ gắt đang thay đổi nhanh hay chậm*. Sai phân $\Delta\kappa$ chính là xấp xỉ
rời rạc của đạo hàm $d\kappa/ds$ — gọi là **jerk hình học** (nhịp đổi độ cong). Trong
lái xe, jerk lớn = đổi độ cong đột ngột (vào/ra cua gấp), là dấu hiệu đoạn đường khó,
dễ FAIL. Dấu cho biết đang **vào cua** ($+$, độ cong tăng) hay **ra cua** ($-$, giảm).

**(3) Tính TAY.** Từ $\kappa=[0,0,0.6325,0,0]$:

| $i$ | $\kappa_{i+1}-\kappa_i$ | $\Delta\kappa$ |
|---|---|---|
| 0 | $0-0$ | $0$ |
| 1 | $0.6325-0$ | $+0.6325$ (vào cua) |
| 2 | $0-0.6325$ | $-0.6325$ (ra cua) |
| 3 | $0-0$ | $0$ |
| pad | — | $0$ |

→ $\text{f3}=[0,\,+0.6325,\,-0.6325,\,0,\,0]$. Cặp $+/-$ liền nhau = "một spike cua đơn".

**(4)** Tóm 1 câu: f3 = *tốc độ đổi độ cong* (jerk hình học, sai phân bậc 1 của κ);
$+$ = đang vào cua, $-$ = đang ra cua.

## 2.5 f4 — Chiều dài cung tích luỹ chuẩn hoá $s/L$

$$
s_i=\sum_{j\le i}\text{seg}_j,\qquad \frac{s}{L}=\frac{s_i}{s_{n-1}+10^{-8}}\in[0,1].
$$
**Ví dụ:** cumsum$([1,1,1.4142,1.4142,1.4142])=[1,2,3.4142,4.8284,6.2426]$, $L=6.2426$ →
$$\text{f4}=[0.1602,\,0.3204,\,0.5469,\,0.7735,\,1.0000].$$

#### Giải thích chi tiết: f4 (arc-length chuẩn hoá s/L)

**(1) Từng ký hiệu.**

| Ký hiệu | Nghĩa | Ở đâu ra |
|---|---|---|
| $\text{seg}_j$ | chiều dài đoạn $j$ | kênh f0 |
| $s_i=\sum_{j\le i}\text{seg}_j$ | **cung tích luỹ (cumulative sum)** đến điểm $i$ = tổng chiều dài đã đi | cộng dồn f0 |
| $L=s_{n-1}$ | **tổng chiều dài** con đường (phần tử cuối của cumsum) | $s$ tại điểm cuối |
| $10^{-8}$ | epsilon chống chia 0 (road dài 0) | hằng số kỹ thuật |
| $s/L\in[0,1]$ | vị trí **theo quãng đường** đã đi, chuẩn hoá | $s_i/L$ |

*Cumsum là gì:* thay vì "độ dài từng đoạn", ta cộng dồn để biết "đã đi được bao xa
tính từ đầu". Ví dụ đi 1m rồi 1m rồi 1.41m → mốc tích luỹ là 1, 2, 3.41m.

**(2) Vì sao chuẩn hoá (chia cho $L$).** Các con đường **dài ngắn khác nhau** (road này
6m, road kia 200m). Nếu để $s$ thô thì model thấy thang đo khác nhau, không so sánh
được. Chia cho $L$ đưa mọi road về **cùng thang $[0,1]$**: 0 = đầu đường, 1 = cuối
đường, 0.5 = đúng giữa quãng đường. Nhờ vậy $s/L$ **bất biến độ phân giải** (lấy 64
hay 197 điểm vẫn cùng vị trí tương đối) và **bất biến xoay/tịnh tiến** (vì seg bất biến).
Chính $s/L$ này (không phải f7) được đưa vào **attention bias $\Delta s$** ở Phần 4.7.

**(3) Tính TAY từng bước** (seg từ f0 $=[1,1,1.4142,1.4142,1.4142]$):

| $i$ | cumsum $s_i$ | $s_i/L$ ($L=6.2426$) |
|---|---|---|
| 0 | $1$ | $1/6.2426=0.1602$ |
| 1 | $1+1=2$ | $2/6.2426=0.3204$ |
| 2 | $2+1.4142=3.4142$ | $3.4142/6.2426=0.5469$ |
| 3 | $3.4142+1.4142=4.8284$ | $4.8284/6.2426=0.7735$ |
| 4 | $4.8284+1.4142=6.2426=L$ | $6.2426/6.2426=1.0000$ |

→ $\text{f4}=[0.1602,\,0.3204,\,0.5469,\,0.7735,\,1.0000]$.

**(4)** Tóm 1 câu: f4 = *đã đi được bao nhiêu phần trăm con đường* (cung tích luỹ chia
tổng chiều dài); bất biến độ phân giải và xoay, là "toạ độ dọc đường" đưa vào attention.

## 2.6 f5, f6 — $\sin\theta$, $\cos\theta$ ((!) phụ thuộc khung)

$$
\text{f5}=\sin\theta_i,\qquad \text{f6}=\cos\theta_i.
$$
**Ví dụ:** $\theta=[0,0,45^\circ,45^\circ,45^\circ]$ (pad edge) →
$$\text{f5}=[0,\,0,\,0.7071,\,0.7071,\,0.7071],\quad
\text{f6}=[1,\,1,\,0.7071,\,0.7071,\,0.7071].$$

**(!) Lưu ý cực quan trọng.** Đây là 2 trong các kênh **phụ thuộc hệ quy chiếu**:
xoay con đường $30^\circ$ thì $\sin\theta,\cos\theta$ **đổi số** → baseline không
bất biến. Chính điểm mù này là lý do ra đời SE2RoadNet (bỏ f5, f6).

#### Giải thích chi tiết: f5/f6 (sin/cos của heading — VÌ SAO phụ thuộc khung)

**(1) Ký hiệu.** $\theta_i$ = góc heading (từ `atan2`, mục 2.2); $\sin\theta_i,\cos\theta_i$
= 2 thành phần **đơn vị hướng** của đoạn (vector hướng chuẩn hoá $(\cos\theta,\sin\theta)$).
Ở đây pad **edge** (lặp giá trị cuối) để đủ $n$ điểm, khác f1 pad 0 hai đầu.

**(2) Vì sao dùng sin/cos thay vì θ trực tiếp.** Góc $\theta$ có **điểm gãy** ở
$\pm180^\circ$ ($+179^\circ$ và $-179^\circ$ rất gần nhau về hướng nhưng cách xa về số).
Cặp $(\cos\theta,\sin\theta)$ **liên tục, tuần hoàn trơn** và luôn nằm trên vòng tròn
đơn vị → mã hoá hướng "mượt" cho mạng học. *Nhưng* cả hai đều **đo hướng tuyệt đối so
với trục Đông ($+x$)** — tức là **so với khung toạ độ của dữ liệu**.

**Vì sao phụ thuộc khung?** Nếu xoay cả con đường đi một góc $\phi$, thì **mọi heading
cộng thêm $\phi$**: $\theta\to\theta+\phi$. Khi đó $\sin,\cos$ đổi theo:
$\sin(\theta+\phi),\cos(\theta+\phi)$ là **số khác hẳn** — dù hình dạng con đường (các
khúc cua) **y hệt**. Model học trên feature này sẽ cho ranking khác khi cùng một con
đường bị vẽ theo hướng khác → không bất biến xoay (Rotation $\Delta\ne0$).

**(3) Tính TAY — ví dụ xoay $30^\circ$ đổi số thế nào.** Con đường gốc có
$\theta=[0^\circ,0^\circ,45^\circ,45^\circ,45^\circ]$. Xoay toàn bộ road $+30^\circ$ →
$\theta'=\theta+30^\circ=[30^\circ,30^\circ,75^\circ,75^\circ,75^\circ]$:

| Điểm | $\theta$ gốc | f5 $=\sin\theta$ | f6 $=\cos\theta$ | $\theta'$ (+30°) | f5' $=\sin\theta'$ | f6' $=\cos\theta'$ |
|---|---|---|---|---|---|---|
| 0 | $0^\circ$ | $0.0000$ | $1.0000$ | $30^\circ$ | $0.5000$ | $0.8660$ |
| 1 | $0^\circ$ | $0.0000$ | $1.0000$ | $30^\circ$ | $0.5000$ | $0.8660$ |
| 2 | $45^\circ$ | $0.7071$ | $0.7071$ | $75^\circ$ | $0.9659$ | $0.2588$ |
| 3 | $45^\circ$ | $0.7071$ | $0.7071$ | $75^\circ$ | $0.9659$ | $0.2588$ |
| 4 | $45^\circ$ | $0.7071$ | $0.7071$ | $75^\circ$ | $0.9659$ | $0.2588$ |

Ví dụ điểm 0: $\sin0^\circ=0\to\sin30^\circ=0.5$; $\cos0^\circ=1\to\cos30^\circ=0.8660$.
**Toàn bộ 2 cột đổi số** dù con đường chỉ bị xoay, hình dạng không đổi. Ngược lại, các
kênh nội tại (f0 seg, f1 $|\Delta\theta|$, f2 $\kappa$) giữ **nguyên** sau xoay — vì
chúng chỉ phụ thuộc **hiệu** hướng, không phụ thuộc hướng tuyệt đối.

**(4)** Tóm 1 câu: f5/f6 = *hướng tuyệt đối* của đoạn (sin/cos của heading), mượt hơn
θ thô nhưng **đổi số khi xoay road** → điểm mù nhạy-xoay mà SE2RoadNet loại bỏ.

## 2.7 f7 — Vị trí tương đối theo chỉ số $\text{rel\_pos}$

$$
\text{rel\_pos}_i=\frac{i}{n-1}\in[0,1]\quad(\texttt{np.linspace}(0,1,n)).
$$
**Ví dụ:** $\text{f7}=[0,\,0.25,\,0.50,\,0.75,\,1.00]$.

**(!) Đừng nhầm với f4.** f7 chia đều **theo chỉ số điểm** (index), còn f4 $=s/L$
chia **theo độ dài cung** (arc-length). Trên code, kênh 7 là `rel_pos = i/(n−1)`,
**không phải** "hướng tuyệt đối $\theta$" (xem Phần 7, điểm vênh s-12 vs s-38).

#### Giải thích chi tiết: f7 (rel_pos — chỉ số điểm, khác s/L)

**(1) Ký hiệu.** $i$ = chỉ số điểm (0-based); $n$ = số điểm; $\text{rel\_pos}_i=i/(n-1)$
= vị trí **theo thứ tự điểm**, chuẩn hoá về $[0,1]$. `np.linspace(0,1,n)` sinh ra đúng
dãy chia đều $n$ mốc từ 0 tới 1.

**(2) Vì sao có f7 và khác f4 thế nào.** f7 cho model một "trục thời gian điểm" đơn
giản: điểm thứ nhất = 0, điểm cuối = 1, chia **đều theo số điểm** bất kể khoảng cách
hình học giữa chúng. So với f4 $=s/L$ chia theo **quãng đường thực**:

- f7 phụ thuộc **cách lấy mẫu** (nhiều điểm dồn ở một khúc → khúc đó "chiếm" nhiều mốc index).
- f4 phụ thuộc **độ dài cung thực** → bất biến độ phân giải; chỉ f4 vào attention bias.

Với con đường **các đoạn dài đều nhau** thì f7 và f4 gần trùng; khi đoạn dài không đều
chúng **tách nhau** (xem bảng dưới).

**(3) Tính TAY** ($n=5$): $\text{rel\_pos}=[\tfrac{0}{4},\tfrac{1}{4},\tfrac{2}{4},\tfrac{3}{4},\tfrac{4}{4}]=[0,0.25,0.50,0.75,1.00]$.

So sánh trực tiếp f7 vs f4 trên chính con đường ví dụ (2 đoạn đầu dài 1, hai đoạn sau dài 1.4142):

| Điểm | f7 $=i/(n-1)$ | f4 $=s/L$ | Lệch |
|---|---|---|---|
| 0 | 0.0000 | 0.1602 | index và cung khác nhau |
| 1 | 0.2500 | 0.3204 | |
| 2 | 0.5000 | 0.5469 | |
| 3 | 0.7500 | 0.7735 | |
| 4 | 1.0000 | 1.0000 | chỉ trùng ở 2 đầu |

Hai cột **khác nhau** vì các đoạn cuối dài hơn → theo quãng đường (f4) điểm bị "đẩy"
xa hơn so với theo chỉ số (f7).

**(4)** Tóm 1 câu: f7 = *vị trí theo thứ tự điểm* (linspace $i/(n-1)$), khác f4 = *vị
trí theo quãng đường*; f7 KHÔNG phải $\theta$ (bẫy s-12 vs s-38) và không vào attention.

## 2.8 f8 — Độ lệch chuẩn cục bộ của curvature $\sigma_{\text{loc}}(\kappa)$

$$
\sigma_{\text{loc}}(\kappa)_i=\operatorname{std}\big(\kappa_{[\,i-5\,:\,i+5\,]}\big),\quad\text{cửa sổ } w=11.
$$
**Ví dụ:** $n=5<11$ nên mọi cửa sổ = toàn bộ $\kappa=[0,0,0.6325,0,0]$.
$\text{mean}=0.6325/5=0.1265$;
$\operatorname{var}=\frac{4(0-0.1265)^2+(0.6325-0.1265)^2}{5}=0.0640$;
$\operatorname{std}=\sqrt{0.0640}=\mathbf{0.2530}$ →
$$\text{f8}=[0.2530,\,0.2530,\,0.2530,\,0.2530,\,0.2530].$$
(Trên road thực $n\approx100$, cửa sổ mỗi điểm khác nhau → f8 thay đổi dọc road.)

#### Giải thích chi tiết: f8 (local std của curvature — cửa sổ trượt)

**(1) Từng ký hiệu.**

| Ký hiệu | Nghĩa |
|---|---|
| $\kappa$ | dãy curvature Menger (kênh f2) |
| $[\,i-5:i+5\,]$ | **cửa sổ trượt** (sliding window) $w=11$ điểm quanh điểm $i$ (5 trái + chính nó + 5 phải) |
| $\operatorname{std}(\cdot)$ | độ lệch chuẩn của các curvature trong cửa sổ |
| $\sigma_{\text{loc}}(\kappa)_i$ | "độ dao động cục bộ" của curvature quanh điểm $i$ |

*Cửa sổ trượt là gì:* với mỗi điểm $i$, ta cắt ra một đoạn con gồm 11 điểm lân cận rồi
tính thống kê **chỉ trên đoạn đó**; trượt cửa sổ dọc road sẽ ra một dãy cùng độ dài.
Ở biên (đầu/cuối road) cửa sổ bị cắt ngắn lại.

**(2) Vì sao dùng std cục bộ.** f2 nói *đang cua gắt cỡ nào*, nhưng đường "cua đều một
cung tròn lớn" và đường "ngoằn ngoèo liên tục" có thể cùng độ cong trung bình. **Std
cục bộ** phân biệt hai loại đó: đoạn ngoằn ngoèo → curvature dao động mạnh → std lớn;
đoạn trơn → std nhỏ. Đây là đặc trưng "**độ gồ ghề / bất ổn hình học**" của khúc đường,
dự báo mức khó lái.

**(3) Tính TAY từng bước.** Con đường ví dụ có $n=5<w=11$ nên mọi cửa sổ = **toàn bộ**
$\kappa=[0,0,0.6325,0,0]$. Std dùng công thức (population, chia $n$):

- **Bước 1 — mean:** $\mu=\dfrac{0+0+0.6325+0+0}{5}=\dfrac{0.6325}{5}=0.1265$.
- **Bước 2 — độ lệch bình phương từng phần tử:**
  - bốn số $0$: $(0-0.1265)^2=0.016002$ mỗi số → $4\times0.016002=0.064006$.
  - số $0.6325$: $(0.6325-0.1265)^2=(0.5060)^2=0.256036$.
- **Bước 3 — variance:** $\operatorname{var}=\dfrac{0.064006+0.256036}{5}=\dfrac{0.320042}{5}=0.064008\approx0.0640$.
- **Bước 4 — std:** $\operatorname{std}=\sqrt{0.0640}=\mathbf{0.2530}$.

Vì mọi điểm dùng chung cửa sổ (cả road) nên
$\text{f8}=[0.2530,0.2530,0.2530,0.2530,0.2530]$. Trên road thực ($n\approx100$) mỗi
điểm có cửa sổ 11 riêng → f8 **biến thiên dọc road**.

**(4)** Tóm 1 câu: f8 = *độ dao động curvature trong cửa sổ 11 điểm* (std cục bộ),
tách "cua trơn đều" khỏi "ngoằn ngoèo"; road ví dụ quá ngắn nên cửa sổ = cả đường → cùng 0.2530.

## 2.9 f9 — Gia tốc curvature $\Delta^2\kappa$

$$
\Delta^2\kappa_i=\Delta\kappa_{i+1}-\Delta\kappa_i,\quad\text{pad 0 cuối}.
$$
**Ví dụ:** $\text{diff}([0,0.6325,-0.6325,0,0])=[0.6325,-1.2649,0.6325,0]$, pad cuối →
$$\text{f9}=[+0.6325,\,-1.2649,\,+0.6325,\,0,\,0].$$

#### Giải thích chi tiết: f9 (sai phân bậc 2 của curvature)

**(1) Ký hiệu.** $\Delta\kappa$ = sai phân bậc 1 (kênh f3); $\Delta^2\kappa_i=\Delta\kappa_{i+1}-\Delta\kappa_i$
= **sai phân bậc 2** (diff của diff) = xấp xỉ đạo hàm bậc hai $d^2\kappa/ds^2$;
"pad 0 cuối" bù 1 phần tử.

**(2) Vì sao công thức như vậy.** Nếu f3 (bậc 1) đo *tốc độ đổi độ cong* thì f9 (bậc 2)
đo **gia tốc của độ cong** — độ cong đang tăng/giảm **nhanh dần hay chậm dần**. Về vật
lý lái xe, đây gắn với *jerk/độ giật* của vô-lăng: giá trị lớn = chuyển tiếp vào/ra cua
rất đột ngột (không có đoạn "clothoid" chuyển tiếp mượt), là dấu hiệu đoạn đường dễ gây
lệch lane. Sai phân bậc 2 cũng làm nổi bật **spike đơn**: một cú cua đơn cho mẫu dấu
$+/-/+$ rất đặc trưng.

**(3) Tính TAY** — lấy diff của f3 $=[0,0.6325,-0.6325,0,0]$:

| $i$ | $\Delta\kappa_{i+1}-\Delta\kappa_i$ | $\Delta^2\kappa$ |
|---|---|---|
| 0 | $0.6325-0$ | $+0.6325$ |
| 1 | $-0.6325-0.6325$ | $-1.2649$ |
| 2 | $0-(-0.6325)$ | $+0.6325$ |
| 3 | $0-0$ | $0$ |
| pad | — | $0$ |

→ $\text{f9}=[+0.6325,\,-1.2649,\,+0.6325,\,0,\,0]$. Mẫu $+,\,-,\,+$ (với đỉnh âm gấp
đôi) chính là chữ ký của "một spike cua đơn" ở điểm 2.

**(4)** Tóm 1 câu: f9 = *gia tốc độ cong* (sai phân bậc 2 của κ), đo độ đột ngột khi
vào/ra cua; spike đơn cho mẫu dấu $+/-/+$.

## 2.10 Ma trận đặc trưng $(5\times 10)$ (trước z-norm)

| pt | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0 | 0 | 0 | 0.1602 | 0 | 1 | 0 | 0.2530 | +0.6325 |
| 1 | 1.0000 | 0 | 0 | +0.6325 | 0.3204 | 0 | 1 | 0.25 | 0.2530 | −1.2649 |
| 2 | 1.4142 | 0.7854 | 0.6325 | −0.6325 | 0.5469 | 0.7071 | 0.7071 | 0.50 | 0.2530 | +0.6325 |
| 3 | 1.4142 | 0 | 0 | 0 | 0.7735 | 0.7071 | 0.7071 | 0.75 | 0.2530 | 0 |
| 4 | 1.4142 | 0 | 0 | 0 | 1.0000 | 0.7071 | 0.7071 | 1.00 | 0.2530 | 0 |

> Xoay road $30^\circ$ → **cột f5, f6 đổi số** (và PE tuyệt đối cũng đổi) → điểm mù nhạy-xoay.

## 2.11 Chuẩn hoá z-score

$$
x_{\text{norm}}=\frac{x-\mu}{\sigma},\quad
\mu=\text{mean theo kênh trên tập train},\ \sigma=\text{std theo kênh},\ \sigma[\sigma<10^{-8}]\leftarrow1.
$$
$\mu,\sigma$ được lưu cùng checkpoint để inference tái dùng đúng chuẩn hoá.

**Ví dụ minh hoạ** (giá trị $\mu,\sigma$ *chỉ để minh hoạ*, số thật nằm trong
checkpoint). Với $\mu_{\text{f0}}=3.5,\ \sigma_{\text{f0}}=1.2$: kênh f0 điểm 0
$\to (1.0-3.5)/1.2=-2.083$. (Road ví dụ rất ngắn nên lệch xa mean tập train.)

---

# Phần 3 — Đặc trưng SE2RoadNet: 7 kênh bất biến (slide s-17, s-41, s-42, s-43)

**Ý tưởng.** Bỏ mọi kênh phụ thuộc khung ($\sin\theta,\cos\theta,\theta$), chỉ giữ
đại lượng **nội tại** (chiều dài, hiệu góc, curvature, arc-length). Kết quả:
rotation $\Delta=0.0000$ (exact).

#### Vì sao đúng **7** kênh này bất biến SE(2)? (bảng tổng)

Một phép biến đổi SE(2) là $p\mapsto R\,p+t$ với $R\in SO(2)$ (xoay) và
$t\in\mathbb{R}^2$ (tịnh tiến). Mọi kênh dưới đây **chỉ** phụ thuộc **vector đoạn**
$d_i=p_{i+1}-p_i$ hoặc **hiệu góc** — không dùng toạ độ tuyệt đối, không dùng hướng
tuyệt đối. Đó là "bí kíp" giữ bất biến:

| Kênh | Đại lượng | Vì sao bất biến SE(2) |
|---|---|---|
| c1 | $\text{seg}=\lVert d_i\rVert$ | $t$ triệt tiêu trong hiệu $p_{i+1}-p_i$; $\lVert R d_i\rVert=\lVert d_i\rVert$ vì $R^\top R=I$ |
| c2 | $\lvert\Delta\text{ang}\rvert$ | xoay cộng **cùng** hằng $\phi$ vào mọi góc → hiệu **khử** $\phi$ |
| c3 | $k=\Delta\text{ang}/\Delta s$ | tử (hiệu góc) và mẫu (chiều dài) đều bất biến → thương bất biến |
| c4 | $dk=\Delta k$ | hiệu của đại lượng bất biến vẫn bất biến |
| c5 | $ddk=\Delta^2 k$ | hiệu bậc 2 của đại lượng bất biến vẫn bất biến |
| c6 | $s/L$ | tổng $\text{seg}$ bất biến; chia $L$ → thêm bất biến **độ phân giải** |
| c7 | $\operatorname{std}(k)$ | hàm thống kê của $k$ (bất biến) → bất biến |

So với RoadFury (Phần 2): SE2RoadNet **bỏ hẳn** f5 $=\sin\theta$, f6 $=\cos\theta$
(hai kênh đổi số khi xoay) và **không** có PE tuyệt đối. Đây chính là "điều kiện
cần" để đạt $\Delta_{\text{rot}}=0$ (xem chứng minh Phần 4.1).

## 3.1 c1 — Chiều dài đoạn $\text{seg}$

Giống f0: $\text{seg}_i=\lVert p_{i+1}-p_i\rVert$, pad edge →
$[1,1,1.4142,1.4142,1.4142]$.

#### Giải thích chi tiết

1. **Ký hiệu.** $p_i\in\mathbb{R}^2$ = toạ độ điểm thứ $i$ của road; $d_i=p_{i+1}-p_i$
   = **vector đoạn** (mũi tên nối điểm này sang điểm kế); $\lVert\cdot\rVert$ = độ dài
   Euclid. "pad edge" = lặp giá trị cuối để dãy đủ $n=5$ phần tử (vì $n$ điểm chỉ có
   $n-1$ đoạn). Đây **đúng là kênh f0** của RoadFury (Phần 2.1) — cùng công thức, cùng số.

2. **Vì sao thế.** Chiều dài đoạn là đại lượng hình học **nội tại** đầu tiên: nó cho
   model biết road được lấy mẫu "thưa hay dày" ở từng khúc, và là **mẫu số** để chuẩn
   hoá curvature (c3) và arc-length (c6). Nó bất biến SE(2) một cách hiển nhiên: tịnh
   tiến $t$ **triệt tiêu** trong hiệu $p_{i+1}-p_i$; xoay $R$ **bảo toàn độ dài** vì
   $\lVert R d_i\rVert=\sqrt{d_i^\top R^\top R\,d_i}=\lVert d_i\rVert$ ($R^\top R=I$).

3. **Tính tay** trên con đường ví dụ ($d=[(1,0),(1,0),(1,1),(1,1)]$):
   $$
   \text{seg}=[\sqrt{1},\ \sqrt{1},\ \sqrt{2},\ \sqrt{2}]=[1,\,1,\,1.4142,\,1.4142]
   \ \to\ (\text{pad edge})\ \to\ [1,\,1,\,1.4142,\,1.4142,\,1.4142].
   $$
   (Xoay con đường $90^\circ$: $d\to[(0,1),(0,1),(-1,1),(-1,1)]$, độ dài vẫn
   $[1,1,\sqrt2,\sqrt2]$ — **y hệt**.)

> **Tóm 1 câu:** c1 = độ dài mỗi bước đi, bất biến vì xoay/tịnh tiến không đổi khoảng cách.

## 3.2 c2 — $|\Delta\text{ang}|$ (biến thiên góc)

$$
|\Delta\text{ang}_i|=\big|\operatorname{wrap}(\text{ang}_{i+1}-\text{ang}_i)\big|,\quad\text{pad 0 hai đầu}.
$$
**Ví dụ:** $\text{ang}=[0,0,0.7854,0.7854]$; $\Delta\text{ang}=[0,0.7854,0]$ →
$\text{c2}=[0,\,0,\,0.7854,\,0,\,0]$.

#### Giải thích chi tiết

1. **Ký hiệu.** $\text{ang}_i=\operatorname{atan2}(\Delta y_i,\Delta x_i)$ = **hướng đi**
   của đoạn $i$ (giống hệt $\theta_i$ ở Phần 2.2 — cùng `atan2`, cách tính dấu và
   `wrap` xem lại mục 2.2.1). $\Delta\text{ang}_i=\operatorname{wrap}(\text{ang}_{i+1}-\text{ang}_i)$
   = **góc cua** tại đỉnh (hiệu 2 hướng liên tiếp, đã `wrap` về $(-\pi,\pi]$ để không
   sai ở mốc $\pm180^\circ$). "pad 0 hai đầu" vì điểm đầu/cuối không có khúc cua (cần 3
   điểm mới xác định 1 cua).

2. **Điểm KHÁC RoadFury (quan trọng).** SE2RoadNet chỉ giữ **hiệu góc** $|\Delta\text{ang}|$,
   **KHÔNG** giữ hướng tuyệt đối. RoadFury còn có f5 $=\sin\theta$, f6 $=\cos\theta$ —
   hai kênh này **đổi số khi xoay road**. Ở đây, dưới phép xoay $R$ góc $\phi$, mọi
   hướng bị **cộng cùng một hằng** $\phi$: $\text{ang}_i\mapsto\text{ang}_i+\phi$. Khi
   lấy hiệu $\text{ang}_{i+1}-\text{ang}_i$, hằng $\phi$ **tự triệt tiêu** → $\Delta\text{ang}$
   không đổi. Đó là lý do dùng **ang tương đối** thay vì $\theta$ tuyệt đối: cua trái
   $45^\circ$ vẫn là cua trái $45^\circ$ dù ta cầm bản đồ xoay theo hướng nào.

3. **Tính tay.** $\text{ang}=[0^\circ,0^\circ,45^\circ,45^\circ]=[0,0,0.7854,0.7854]$ rad.
   Hiệu liên tiếp: $[\,0-0,\ 0.7854-0,\ 0.7854-0.7854\,]=[0,\,0.7854,\,0]$ (đều nằm
   trong $(-\pi,\pi]$ nên `wrap` không đổi gì); lấy trị tuyệt đối rồi pad $(1,1)$ →
   $\text{c2}=[0,\,0,\,0.7854,\,0,\,0]$. Chỉ **điểm 2** có cua $0.7854$ rad $=45^\circ$.

> **Tóm 1 câu:** c2 = độ lớn góc cua ($|\Delta\text{ang}|$); dùng **hiệu góc** (không phải hướng tuyệt đối) nên xoay road không đổi nó.

## 3.3 c3 — Curvature **có dấu** $k$

$$
\boxed{\ k_i=\frac{\Delta\text{ang}_i}{\tfrac12(\text{seg}_i+\text{seg}_{i+1})+10^{-8}}\ }
\quad(\text{xấp xỉ } d\theta/ds),\qquad\text{pad 0 hai đầu}.
$$
**Ký hiệu.** Tử = hiệu góc tại đỉnh; mẫu = trung bình 2 đoạn quanh đỉnh. Dấu
cho biết cua **trái ($+$) / phải ($-$)**.

**Ví dụ:** $\text{seg}[:-1]=[1,1,1.4142]$, $\text{seg}[1:]=[1,1.4142,1.4142]$ →
$\text{denom}=[1.0000,1.2071,1.4142]$;
$$
k_{\text{mid}}=\frac{0.7854}{1.2071}=\mathbf{0.6506}\ \Rightarrow\ \text{c3}=[0,\,0,\,+0.6506,\,0,\,0].
$$

**Lưu ý.** Menger (RoadFury) cho $0.6325$ nhưng **bỏ dấu**; SE2 dùng
$k=\Delta\theta/\Delta s$ **giữ dấu** = $0.6506$. Hai cách tính khác nhau → hai số
khác nhau (Phần 7).

### 3.3.1 Giải thích chi tiết: $k$ CÓ DẤU ($d\theta/ds$)

Đây là kênh **quan trọng nhất** của SE2RoadNet — vừa mang thông tin cua, vừa **giữ
dấu** (trái/phải), vừa bất biến SE(2). Bốn câu hỏi cần trả lời trọn vẹn:

#### (a) Từng ký hiệu ở đâu ra?

| Ký hiệu | Là gì | Tính từ |
|---|---|---|
| $\Delta\text{ang}_i$ | góc cua **có dấu** tại đỉnh $i$ (rad) | $\operatorname{wrap}(\text{ang}_{i+1}-\text{ang}_i)$, **không** lấy trị tuyệt đối |
| $\text{seg}_i,\text{seg}_{i+1}$ | độ dài 2 đoạn kề đỉnh | c1 |
| $\tfrac12(\text{seg}_i+\text{seg}_{i+1})$ | $\Delta s$ = arc-length "quanh đỉnh" | trung bình 2 đoạn |
| $10^{-8}$ | epsilon chống chia 0 | hằng số |
| $k_i$ | curvature có dấu (đơn vị **rad / đơn-vị-dài** = $1/\text{length}$) | thương trên |

Chú ý khác c2: c2 lấy $|\Delta\text{ang}|$ (bỏ dấu) để đo **độ lớn** cua; c3 giữ
$\Delta\text{ang}$ (**có dấu**) trên tử số để đo **hướng** cua.

#### (b) Vì sao $\Delta\theta/\Delta s$ thay vì Menger? Vì sao mẫu là **trung bình 2 đoạn**? Vì sao **có dấu**?

- **Vì sao $\Delta\theta/\Delta s$ (không dùng Menger).** Curvature "sách giáo khoa"
  là $\kappa=d\theta/ds$ = **tốc độ đổi hướng theo quãng đường**. Công thức
  $k=\Delta\text{ang}/\Delta s$ là bản **rời rạc trực tiếp** của định nghĩa đó: tử là
  "đổi hướng" $\Delta\theta$, mẫu là "quãng đường" $\Delta s$. Ưu điểm so với Menger
  (bán kính đường tròn ngoại tiếp 3 điểm): (i) **giữ được dấu** một cách tự nhiên (Menger
  qua $1/R$ luôn $\ge 0$); (ii) là **thương của 2 đại lượng bất biến SE(2)** nên bản
  thân bất biến; (iii) mở rộng thẳng sang đạo hàm bậc cao $dk,ddk$ (c4, c5).

- **Vì sao mẫu = trung bình 2 đoạn.** Góc cua $\Delta\text{ang}_i$ "thuộc về" **đỉnh** $i$,
  nằm giữa đoạn vào ($\text{seg}_i$) và đoạn ra ($\text{seg}_{i+1}$). Quãng đường "gánh"
  cú cua đó trải **nửa đoạn vào + nửa đoạn ra**:
  $$
  \Delta s_i \approx \tfrac12\,\text{seg}_i+\tfrac12\,\text{seg}_{i+1}=\tfrac12(\text{seg}_i+\text{seg}_{i+1}).
  $$
  Đây là arc-length **canh giữa đỉnh** (central difference) → thương $\Delta\theta/\Delta s$
  chính là ước lượng $d\theta/ds$ tại đúng đỉnh đó. Nếu chỉ chia cho 1 đoạn thì ước
  lượng bị lệch về đoạn vào hoặc đoạn ra.

- **Vì sao có dấu, dấu nghĩa là gì.** Dấu của $k$ = dấu của $\Delta\text{ang}$ = **chiều
  cua**. Theo quy ước `atan2` (mục 2.2.1): hất **lên/ngược chiều kim đồng hồ** → $\Delta\text{ang}>0$
  → $k>0$ = **cua TRÁI**; hạ **xuống/cùng chiều kim đồng hồ** → $\Delta\text{ang}<0$ →
  $k<0$ = **cua PHẢI**. Menger bỏ mất thông tin này (khúc cua trái và phải cùng độ gắt
  cho cùng một số) — với SDC, phân biệt trái/phải là tín hiệu vật lý thật (lực ngang
  đổi chiều), nên SE2RoadNet **giữ dấu**.

  ```
  Con duong vi du: di Dong roi be TRAI len Dong-Bac tai diem 2

              (4,2)
             /
          (3,1)
           /
  (0,0)--(1,0)--(2,0)      Dang = +45 deg (hat len) -> k = +0.6506 (cua TRAI)

  Neu be xuong (Dong-Nam) thi Dang < 0 -> k < 0 (cua PHAI)
  ```

#### (c) Tính tay từng bước

Bước 1 — hai dãy seg lệch nhau 1 nhịp (để ghép cặp đoạn kề đỉnh):
$$
\text{seg}[:-1]=[\,1,\ 1,\ 1.4142\,],\qquad \text{seg}[1:]=[\,1,\ 1.4142,\ 1.4142\,].
$$
Bước 2 — mẫu số = trung bình từng cặp:
$$
\text{denom}=\Big[\tfrac{1+1}{2},\ \tfrac{1+1.4142}{2},\ \tfrac{1.4142+1.4142}{2}\Big]=[\,1.0000,\ 1.2071,\ 1.4142\,].
$$
Bước 3 — tử số = hiệu góc có dấu tại 3 đỉnh trong: $\Delta\text{ang}=[\,0,\ 0.7854,\ 0\,]$.
Bước 4 — chia từng phần tử:
$$
k=\Big[\tfrac{0}{1.0000},\ \tfrac{0.7854}{1.2071},\ \tfrac{0}{1.4142}\Big]=[\,0,\ 0.6506,\ 0\,].
$$
Bước 5 — pad 0 hai đầu → $\text{c3}=[\,0,\ 0,\ +0.6506,\ 0,\ 0\,]$.

**Đối chiếu Menger (Phần 2.3):** cùng khúc cua đó Menger cho $\kappa=0.6325$. Chênh
$0.6506$ vs $0.6325$ là do **hai cách rời rạc hoá khác nhau**, KHÔNG phải lỗi (xem
bẫy 7.2): Menger = $1/R$ đường tròn ngoại tiếp 3 điểm; $k$ = $\Delta\theta/\Delta s$.
Trên đường cong liên tục cả hai tiến về cùng một $\kappa$; trên dữ liệu rời rạc chúng
lệch nhẹ.

**Bất biến SE(2):** tử ($\Delta\text{ang}$, bất biến vì hằng $\phi$ khử — mục 3.2) chia
mẫu ($\Delta s$ = tổ hợp seg, bất biến — mục 3.1) → $k$ bất biến. Xoay road $90^\circ$:
$\Delta\text{ang}$ và các seg đều không đổi → $k=+0.6506$ **y hệt**.

> **Tóm 1 câu:** c3 = $d\theta/ds$ rời rạc (đổi-hướng chia quãng-đường-canh-giữa-đỉnh); **giữ dấu** để phân biệt cua trái ($+$) / phải ($-$); bất biến vì là thương của hai đại lượng bất biến.

## 3.4 c4, c5 — $dk$, $ddk$ (đạo hàm bậc 1, 2 của curvature)

$$
dk_i=k_{i+1}-k_i,\qquad ddk_i=dk_{i+1}-dk_i,\quad\text{pad 0 cuối}.
$$
**Ví dụ:**
$$
\text{c4}=dk=[0,\,+0.6506,\,-0.6506,\,0,\,0],\qquad
\text{c5}=ddk=[+0.6506,\,-1.3013,\,+0.6506,\,0,\,0].
$$
Cụm 3 dấu xen kẽ $+/-/+$ = đặc trưng "một spike cua đơn".

#### Giải thích chi tiết

1. **Ký hiệu.** $dk_i=k_{i+1}-k_i$ = **sai phân bậc 1** của curvature dọc road
   ("curvature đang tăng hay giảm khi đi tới"). $ddk_i=dk_{i+1}-dk_i$ = **sai phân bậc 2**
   ("gia tốc của curvature"). Cả hai dùng `np.diff` thuần theo **chỉ số điểm** (KHÔNG
   chia lại cho $\Delta s$), rồi **pad 0 ở cuối** cho đủ $n$ phần tử. Đây là chuỗi đạo
   hàm nối tiếp c3: $k\to dk\to ddk$.

2. **Vì sao thế.** Curvature $k$ nói "khúc này cong bao nhiêu"; $dk$ nói "đang **vào**
   hay **ra** khỏi khúc cua" (dấu $+$ = cong dần lên, $-$ = duỗi thẳng dần); $ddk$ bắt
   "cua **đột ngột** hay **mượt**" (jerk hình học). Ba mức này cho Transformer đọc được
   **hình thái** cú cua chứ không chỉ độ gắt — một cú "giật" ($ddk$ lớn) nguy hiểm hơn
   cùng độ cong nhưng vào ra mượt. Bất biến SE(2) là **hệ quả tự động**: hiệu (và hiệu
   của hiệu) của đại lượng bất biến $k$ vẫn bất biến.

3. **Tính tay** (từ $k=[0,0,0.6506,0,0]$):
   - **c4** $=\operatorname{diff}(k)$: $[\,0-0,\ 0.6506-0,\ 0-0.6506,\ 0-0\,]=[0,\,0.6506,\,-0.6506,\,0]$,
     pad cuối → $[0,\,+0.6506,\,-0.6506,\,0,\,0]$.
   - **c5** $=\operatorname{diff}(dk)$ với $dk=[0,0.6506,-0.6506,0,0]$:
     $[\,0.6506-0,\ -0.6506-0.6506,\ 0-(-0.6506),\ 0-0\,]=[0.6506,\,-1.3013,\,0.6506,\,0]$,
     pad cuối → $[+0.6506,\,-1.3013,\,+0.6506,\,0,\,0]$.
     ($-0.6506-0.6506=-1.3013$; đúng bằng $-2\times0.6506$ ở độ chính xác cao.)

   | mẫu hình | dãy | đọc |
   |---|---|---|
   | $k$ | $[0,0,{+}0.6506,0,0]$ | 1 đỉnh cong dương |
   | $dk$ | $[0,{+}0.6506,{-}0.6506,0,0]$ | lên rồi xuống (vào/ra 1 khúc) |
   | $ddk$ | $[{+}0.6506,{-}1.3013,{+}0.6506,0,0]$ | dấu **+ / - / +** = spike đơn |

> **Tóm 1 câu:** c4/c5 = đạo hàm bậc 1 và 2 của curvature theo dọc đường; chúng mô tả road **vào/ra** và **giật** khúc cua ra sao, bất biến vì chỉ là hiệu của $k$ (đã bất biến).

## 3.5 c6 — Arc-length chuẩn hoá $s/L$

Giống f4: $s/L=[0.1602,0.3204,0.5469,0.7735,1.0000]$. **Kênh này được đọc lại
bên trong model** để tính attention bias $\Delta s$ (Phần 4.7).

#### Giải thích chi tiết

1. **Ký hiệu.** $s_i=\sum_{j\le i}\text{seg}_j$ = **cung tích luỹ** (đã đi bao xa tính
   từ đầu road); $L=s_{n-1}$ = tổng chiều dài road; $s/L\in[0,1]$ = **vị trí theo quãng
   đường**. Đây đúng là kênh f4 của RoadFury (Phần 2.5) — cùng công thức, cùng số.

2. **Vì sao thế + hai lớp bất biến.** (i) **Bất biến SE(2):** mọi $\text{seg}_j$ bất
   biến (mục 3.1) → tổng tích luỹ $s_i$ bất biến. (ii) **Bất biến độ phân giải:** chia
   cho $L$ đưa về thang $[0,1]$, nên road lấy mẫu $N=64$ hay $N=197$ điểm đều cho cùng
   một "toạ độ cung chuẩn hoá" ở cùng vị trí vật lý. Chính vì vậy c6 được **đọc lại
   trong model** để sinh attention bias $\Delta s_{ij}=s_i-s_j$ (Phần 4.7): dùng **hiệu**
   cung nên bias cũng bất biến cả xoay lẫn độ phân giải. Lưu ý phân biệt với `rel_pos`
   $=i/(n-1)$ (chia đều theo **chỉ số**, không phải theo cung) — SE2RoadNet dùng $s/L$,
   không dùng rel_pos cho bias (bẫy 7.7).

3. **Tính tay.** cumsum$([1,1,1.4142,1.4142,1.4142])=[1,\,2,\,3.4142,\,4.8284,\,6.2426]$,
   $L=6.2426$ →
   $$
   s/L=[\tfrac{1}{6.2426},\tfrac{2}{6.2426},\tfrac{3.4142}{6.2426},\tfrac{4.8284}{6.2426},\tfrac{6.2426}{6.2426}]=[0.1602,\,0.3204,\,0.5469,\,0.7735,\,1.0000].
   $$

> **Tóm 1 câu:** c6 = "đã đi bao nhiêu phần trăm chiều dài road"; bất biến SE(2) **và** độ phân giải, là nguồn cho attention bias $\Delta s$.

## 3.6 c7 — Local std của $k$ ($\text{lstd}$)

$$
\text{lstd}_i=\operatorname{std}\big(k_{[\,i-5:i+5\,]}\big).
$$
**Ví dụ:** $k=[0,0,0.6506,0,0]$; $\text{mean}=0.1301$;
$\operatorname{var}=\frac{4(0.1301)^2+(0.6506-0.1301)^2}{5}=0.0678$;
$\operatorname{std}=\sqrt{0.0678}=\mathbf{0.2603}$ → $\text{c7}=[0.2603,\dots]$.

### 3.6.1 Giải thích chi tiết: $\text{lstd}$ (độ lệch chuẩn cửa sổ)

1. **Ký hiệu.** $k_{[\,i-5:i+5\,]}$ = **cửa sổ trượt 11 phần tử** của curvature quanh
   điểm $i$ (5 bên trái, chính nó, 5 bên phải); $\operatorname{std}$ = độ lệch chuẩn
   (population std, chia $n$) của cửa sổ đó. Kết quả gán cho điểm $i$. Đây là bản "curvature"
   của kênh f8 RoadFury (Phần 2.8), nhưng tính trên $k$ **có dấu** thay vì $|\kappa|$ Menger.

2. **Vì sao thế.** lstd đo **độ "gồ ghề" cục bộ**: vùng nhiều khúc ngoằn ngoèo → $k$ dao
   động mạnh → std lớn; đường thẳng dài → std $\approx 0$. Đây là tín hiệu texture bổ
   sung cho $k$ tức thời (một điểm có $k=0$ nhưng nằm giữa vùng zig-zag vẫn "nguy hiểm").
   Bất biến SE(2): std là **hàm của $k$** (đã bất biến — mục 3.3), nên lstd bất biến theo.

3. **Tính tay** (road ví dụ $n=5<11$ nên **mọi cửa sổ = toàn bộ** $k=[0,0,0.6506,0,0]$):
   - **Mean:** $\dfrac{0+0+0.6506+0+0}{5}=\dfrac{0.6506}{5}=0.1301$.
   - **Var** (chia $n=5$): 4 phần tử bằng $0$ (lệch $0-0.1301=-0.1301$) và 1 phần tử
     bằng $0.6506$ (lệch $0.6506-0.1301=0.5205$):
     $$
     \operatorname{var}=\frac{4\,(0.1301)^2+(0.5205)^2}{5}=\frac{4(0.016926)+0.270920}{5}=\frac{0.338624}{5}=0.0678.
     $$
   - **Std:** $\sqrt{0.0678}=0.2603$ → $\text{c7}=[0.2603,\,0.2603,\,0.2603,\,0.2603,\,0.2603]$
     (5 điểm bằng nhau vì cùng chung một cửa sổ).
   - (Road thực $n\approx100>11$: mỗi điểm có cửa sổ 11 riêng → c7 biến thiên dọc road.)

> **Tóm 1 câu:** c7 = độ dao động cục bộ của curvature (std cửa sổ 11 của $k$); lớn ở vùng ngoằn ngoèo, bằng 0 trên đoạn thẳng, bất biến vì là thống kê của $k$.

## 3.7 Ma trận đặc trưng $(5\times 7)$ (trước z-norm)

| pt | c1 seg | c2 \|Δang\| | c3 k | c4 dk | c5 ddk | c6 s/L | c7 lstd |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0 | 0 | 0 | +0.6506 | 0.1602 | 0.2603 |
| 1 | 1.0000 | 0 | 0 | +0.6506 | −1.3013 | 0.3204 | 0.2603 |
| 2 | 1.4142 | 0.7854 | +0.6506 | −0.6506 | +0.6506 | 0.5469 | 0.2603 |
| 3 | 1.4142 | 0 | 0 | 0 | 0 | 0.7735 | 0.2603 |
| 4 | 1.4142 | 0 | 0 | 0 | 0 | 1.0000 | 0.2603 |

> Xoay road $30^\circ$ hay $90^\circ$ → ma trận này **y hệt** ($\Delta=0.0000$, exact).

#### Cách đọc bảng (bất biến từng cột)

Mỗi cột trong bảng chỉ dựa vào $d_i=p_{i+1}-p_i$ hoặc hiệu góc, **không** cột nào
dùng toạ độ/hướng tuyệt đối — nên xoay/tịnh tiến con đường cho **đúng từng con số**
(khác hẳn bảng $(5\times10)$ RoadFury ở mục 2.10, nơi cột f5/f6 đổi số khi xoay). Cụm
cua đơn ở **điểm 2** hiện rõ: c2/c3 có 1 spike, c4 đảo dấu quanh nó ($+/-$), c5 cho
mẫu $+/-/+$ — chữ ký hình học của "một khúc cua trái".

## 3.8 Cơ sở lý thuyết: Frenet–Serret

Một đường cong phẳng được xác định **duy nhất tới một phép dời hình (rigid motion)**
bởi **hàm độ cong $\kappa(s)$** của nó. Do đó chỉ cần $\kappa(s)$ (và $s$) là đủ
thông tin hình dạng → hợp lý khi bỏ toạ độ/hướng tuyệt đối.

#### Giải thích chi tiết: định lý cơ bản nối với 7 kênh

1. **Phát biểu.** Định lý cơ bản của đường cong phẳng: cho trước hàm $\kappa(s)$ (curvature
   theo arc-length) và một điều kiện đầu (một điểm + một hướng), tồn tại **duy nhất** một
   đường cong có đúng $\kappa(s)$ đó, **sai khác một phép dời hình** $SE(2)$. Nói cách khác:
   *hình dạng* $\equiv$ *cặp* $(\kappa(s),\,s)$; còn *vị trí/hướng đặt nó trong mặt phẳng*
   chính là phần $SE(2)$ mà ta **muốn vứt bỏ**.

2. **Nối với 7 kênh.** SE2RoadNet mã hoá đúng cặp bất biến đó: $s$ qua **c6** ($s/L$),
   và $\kappa(s)$ qua **c3** ($k$) cùng đạo hàm **c4/c5** ($dk,ddk$) và thống kê **c7**
   (lstd); c1 (seg) là bước rời rạc của $ds$, c2 ($|\Delta\text{ang}|$) là dạng chưa
   chuẩn hoá của cua. Vì bộ này = "toàn bộ thông tin hình dạng trừ đi $SE(2)$", model
   không thể "nhìn thấy" phép xoay/tịnh tiến → $\Delta_{\text{rot}}=0.0000$ là **hệ quả
   toán học**, không phải may mắn huấn luyện. (Chi tiết chứng minh bit-identical: Phần 4.1.)

> **Tóm 1 câu:** Frenet–Serret bảo *hình dạng đường cong = $(\kappa(s),s)$ tới sai khác $SE(2)$*; 7 kênh chính là bản rời rạc của $(\kappa(s),s)$, nên bỏ đúng phần khung mà vẫn giữ trọn hình dạng.

---

# Phần 4 — Kiến trúc & Công thức method

## 4.1 Định lý bất biến SE(2) (slide s-16, s-36)

$$
\boxed{\ f_\theta(R\,\mathcal{R}+t)=f_\theta(\mathcal{R})\quad \forall (R,t)\in SE(2)\ }
\qquad(R\in SO(2),\ t\in\mathbb{R}^2),
$$
yêu cầu **bit-identical**, không phải "trong sai số".

**Chứng minh ngắn** (theo s-36). Dưới $p\mapsto R\,p+t$:

1. $d_i=p_{i+1}-p_i\mapsto R\,d_i$ (số hạng $t$ triệt tiêu).
2. $\lVert R\,d_i\rVert=\sqrt{d_i^\top R^\top R\,d_i}=\lVert d_i\rVert$ vì $R^\top R=I$ → **seg bất biến**.
3. $\text{ang}_i\mapsto \text{ang}_i+\phi$ (mọi góc cộng cùng hằng $\phi$) → hiệu $\Delta\text{ang}$ **triệt tiêu hằng** → bất biến; $|\Delta\theta|$ bất biến.
4. $k=\Delta\text{ang}/\Delta s$ và các đạo hàm theo $s$ là hàm của đại lượng bất biến → **bất biến**.
5. Loại $\sin\theta,\cos\theta,\theta$ (phụ thuộc $R$) là **điều kiện cần**.

**Hệ quả:** feature $(197\times7)$ giống hệt sau biến đổi → logit giống → ranking
giống → APFD giống. (Sai số duy nhất: roundoff float của `atan2`, cỡ $1.79\times10^{-7}$
ở mức logit — xem Phần 7.)

### 4.1.1 Giải thích chi tiết: đọc từng ký hiệu và chứng minh từng bước

#### (a) Từng ký hiệu là gì / ở đâu ra

| Ký hiệu | Là gì | Ở đâu ra |
|---|---|---|
| $\mathcal{R}=\{p_1,\dots,p_N\}$ | **con đường** = chuỗi điểm 2D | dữ liệu test SDC |
| $R\in SO(2)$ | **ma trận xoay** $2\times2$, xoay quanh gốc góc $\phi$ | phép biến đổi khung |
| $t\in\mathbb{R}^2$ | **vector tịnh tiến** (dời cả road đi một đoạn) | phép biến đổi khung |
| $SE(2)$ | nhóm **dời hình phẳng** = xoay + tịnh tiến (không co giãn, không lật) | $\{(R,t)\}$ |
| $f_\theta$ | **mô hình** (đọc road, ra score FAIL) | mạng đã train |
| $R\mathcal{R}+t$ | road **sau khi** xoay $R$ rồi dời $t$ (áp cho từng điểm $p\mapsto Rp+t$) | ảnh của road |

$R$ có dạng
$$
R=\begin{bmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{bmatrix},
$$
đây là ma trận **trực giao**: $R^\top R=I$ (chứng minh (b)). $SO(2)$ nghĩa là
"xoay thuần" ($\det R=+1$, không có lật gương).

**Đẳng thức muốn chứng minh nói gì?** "Xoay con đường đi $\phi$ độ và/hoặc dời nó
sang chỗ khác thì **score FAIL không đổi một chút nào**." Một con đường nguy hiểm
vẫn nguy hiểm dù ta vẽ nó quay theo hướng nào trên bản đồ.

#### (b) Vì sao đúng — chứng minh từng bước

**Bước 1 — vector đoạn: $t$ triệt tiêu.** Điểm biến đổi $p_i\mapsto Rp_i+t$. Vector đoạn:
$$
d_i=p_{i+1}-p_i\ \mapsto\ (Rp_{i+1}+t)-(Rp_i+t)=R(p_{i+1}-p_i)+\underbrace{(t-t)}_{=0}=R\,d_i.
$$
Tịnh tiến $t$ **luôn tự huỷ** khi lấy hiệu hai điểm — đó là lý do model chỉ dùng
$d_i$ (hiệu) chứ không dùng toạ độ thô $p_i$.

**Bước 2 — chiều dài đoạn (seg) bất biến vì $R^\top R=I$.** Trước hết:
$$
R^\top R=
\begin{bmatrix}\cos\phi & \sin\phi\\ -\sin\phi & \cos\phi\end{bmatrix}
\begin{bmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{bmatrix}
=\begin{bmatrix}\cos^2\phi+\sin^2\phi & 0\\ 0 & \sin^2\phi+\cos^2\phi\end{bmatrix}
=\begin{bmatrix}1&0\\0&1\end{bmatrix}=I.
$$
(hai phần tử ngoài đường chéo: $-\cos\phi\sin\phi+\sin\phi\cos\phi=0$). Do đó
$$
\lVert R\,d_i\rVert^2=(R d_i)^\top(R d_i)=d_i^\top R^\top R\,d_i=d_i^\top I\,d_i=d_i^\top d_i=\lVert d_i\rVert^2.
$$
Xoay **giữ nguyên độ dài** → kênh `seg` (f0 / c1) bất biến.

**Bước 3 — góc heading cộng thêm cùng một hằng $\phi$.** Nếu $d_i$ có góc $\text{ang}_i$
thì $R d_i$ (xoay đi $\phi$) có góc $\text{ang}_i+\phi$. Khi lấy **hiệu góc** hai đoạn kề:
$$
\Delta\text{ang}_i=\text{ang}_{i+1}-\text{ang}_i\ \mapsto\ (\text{ang}_{i+1}+\phi)-(\text{ang}_i+\phi)=\text{ang}_{i+1}-\text{ang}_i.
$$
Hằng $\phi$ **triệt tiêu** → $\Delta\text{ang}$ (và $|\Delta\theta|$) bất biến. Đây chính
là lý do dùng **hiệu góc** thay vì góc tuyệt đối.

**Bước 4 — curvature và đạo hàm theo $s$ bất biến.** $k=\Delta\text{ang}/\Delta s$
là **thương của hai đại lượng đã bất biến** (tử $\Delta\text{ang}$ ở bước 3, mẫu là
trung bình `seg` ở bước 2) → bất biến. Các đạo hàm $dk,ddk$ (hiệu của $k$) và
$s/L$ (tổng tích luỹ của `seg` chuẩn hoá) cũng vậy.

**Bước 5 — điều kiện cần.** Ngược lại, $\sin\theta,\cos\theta,\theta$ **đổi số** khi
xoay ($\theta\mapsto\theta+\phi$) nên nếu để chúng trong feature thì bất biến **hỏng**.
Vì thế SE2RoadNet **bắt buộc bỏ** f5, f6 và PE tuyệt đối.

#### (c) Tính TAY: xoay $30^\circ$ + dời $t=(5,5)$ con đường ví dụ

Lấy $\phi=30^\circ$ ($\cos30^\circ=0.8660,\ \sin30^\circ=0.5$), $t=(5,5)$. Biến đổi
$p\mapsto Rp+t$ cho 5 điểm:

| Điểm gốc | $Rp$ | $Rp+t$ |
|---|---|---|
| $(0,0)$ | $(0,0)$ | $(5.000,\,5.000)$ |
| $(1,0)$ | $(0.866,\,0.500)$ | $(5.866,\,5.500)$ |
| $(2,0)$ | $(1.732,\,1.000)$ | $(6.732,\,6.000)$ |
| $(3,1)$ | $(2.098,\,2.366)$ | $(7.098,\,7.366)$ |
| $(4,2)$ | $(2.464,\,3.732)$ | $(7.464,\,8.732)$ |

Tính lại vector đoạn, seg và heading trên road **đã xoay+dời**:

| $i$ | $d_i$ (mới) | $\lVert d_i\rVert$ | $\text{ang}_i$ |
|---|---|---|---|
| 0 | $(0.866,\,0.500)$ | $\sqrt{0.75+0.25}=1.0000$ | $30^\circ$ |
| 1 | $(0.866,\,0.500)$ | $1.0000$ | $30^\circ$ |
| 2 | $(0.366,\,1.366)$ | $\sqrt{0.134+1.866}=1.4142$ | $75^\circ$ |
| 3 | $(0.366,\,1.366)$ | $1.4142$ | $75^\circ$ |

- **seg** $=[1,1,1.4142,1.4142]$ — **y hệt f0 gốc**. (mọi $\lVert d_i\rVert$ không đổi).
- **heading** đều cộng $+30^\circ$: $[0,0,45,45]\to[30,30,75,75]$.
- **hiệu góc** $\Delta\text{ang}=[30{-}30,\,75{-}30,\,75{-}75]=[0,\,45^\circ,\,0]=[0,\,0.7854,\,0]$ — **y hệt** khi chưa xoay → $|\Delta\theta|,\ \kappa,\ k$ đều **không đổi**.

Vậy ma trận $(5\times7)$ của SE2RoadNet **trùng khít từng số** trước và sau biến đổi
→ logit trùng → ranking trùng → $\Delta_{\text{rot}}=0.0000$ (exact). (Ngược lại,
f5$=\sin\theta$ đổi từ $[0,0,0.7071,\dots]$ sang $\sin[30,30,75,75]=[0.5,0.5,0.966,\dots]$
→ RoadFury **không** bất biến.)

> **Tóm 1 câu:** dùng **hiệu điểm** ($t$ tự huỷ), **độ dài** ($R^\top R=I$ giữ nguyên)
> và **hiệu góc** (hằng xoay $\phi$ triệt tiêu) làm feature → mọi số sống sót nguyên
> vẹn qua xoay+tịnh tiến, nên $f_\theta(R\mathcal R+t)=f_\theta(\mathcal R)$ đúng đến từng bit.

## 4.2 Lớp Linear (fully-connected)

$$
y=x\,W^\top+b,\qquad W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},\ b\in\mathbb{R}^{d_{\text{out}}}.
$$
Áp lên trục cuối, độc lập theo từng token. Ví dụ `proj`: $7\to192$ (SE2) hoặc $10\to128$ (RoadFury).

#### Giải thích chi tiết

**(1) Từng ký hiệu là gì.**

| Ký hiệu | Là gì | Kích thước |
|---|---|---|
| $x$ | vector đặc trưng **vào** của một token | $d_{\text{in}}$ |
| $W$ | **ma trận trọng số** (học được), hàng $k$ = "công thức" tạo kênh ra thứ $k$ | $d_{\text{out}}\times d_{\text{in}}$ |
| $W^\top$ | chuyển vị của $W$ (để nhân bên phải $x$) | $d_{\text{in}}\times d_{\text{out}}$ |
| $b$ | **bias** (học được), dịch mỗi kênh ra một hằng | $d_{\text{out}}$ |
| $y$ | vector đặc trưng **ra** | $d_{\text{out}}$ |

Công thức từng phần tử: $y_k=\sum_{j=1}^{d_{\text{in}}} x_j\,W_{kj}+b_k$ — mỗi kênh ra
là một **tổ hợp tuyến tính** của mọi kênh vào, cộng một offset.

**(2) Vì sao như vậy.** Đây là phép **trộn kênh** cơ bản nhất: `proj` đầu vào nâng
7 (hoặc 10) đặc trưng hình học lên $d$ chiều để Transformer có "không gian" biểu diễn
rộng; các Linear trong FFN/attention trộn lại thông tin sau mỗi block. Vì áp **độc lập
theo token** (cùng $W,b$ cho mọi vị trí) nên nó **không** trộn giữa các điểm — việc
trộn giữa điểm là nhiệm vụ của attention (4.6).

**(3) Tính TAY** (ví dụ nhỏ minh hoạ, $d_{\text{in}}=2\to d_{\text{out}}=3$):
$$
x=[1,\,2],\quad
W=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
b=[0,\,0,\,1].
$$
Nhân từng hàng của $W$ với $x$ rồi cộng bias:
$$
y_0=(1,0)\cdot(1,2)+0=1,\quad
y_1=(0,1)\cdot(1,2)+0=2,\quad
y_2=(1,1)\cdot(1,2)+1=3+1=4,
$$
$$
\Rightarrow\ y=[1,\,2,\,4].
$$

> **Tóm 1 câu:** Linear = "mỗi kênh ra là một trung bình có trọng số của các kênh vào,
> cộng bias" ($y_k=\sum_j x_j W_{kj}+b_k$), làm riêng cho từng token.

## 4.3 LayerNorm

$$
\operatorname{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}\odot\gamma+\beta,\quad
\mu=\frac1d\sum_i x_i,\ \sigma^2=\frac1d\sum_i (x_i-\mu)^2,\ \varepsilon=10^{-5}.
$$
Chuẩn hoá **mỗi vector token** trên $d$ chiều; $\gamma,\beta$ học được. Dùng
**pre-LN** (`norm_first=True`): chuẩn hoá **trước** attention/FFN.

**Ví dụ:** $x=[1,2,3]$ → $\mu=2$, $\sigma^2=\frac{1+0+1}{3}=0.6667$,
$\sigma=0.8165$ → $\hat x=[-1.2247,\,0,\,1.2247]$ (khi $\gamma{=}1,\beta{=}0,\varepsilon{\approx}0$).

#### Giải thích chi tiết

**(1) Từng ký hiệu là gì.**

| Ký hiệu | Là gì |
|---|---|
| $x$ | vector token $d$ chiều (một điểm road sau `proj`) |
| $\mu$ | **trung bình** $d$ thành phần của **chính token đó** |
| $\sigma^2$ | **phương sai** $d$ thành phần của chính token đó |
| $\varepsilon=10^{-5}$ | hằng nhỏ chống chia 0 (khi token gần như phẳng) |
| $\gamma$ | **scale** học được (mỗi chiều một hệ số) — cho phép "phóng to lại" nếu cần |
| $\beta$ | **shift** học được — cho phép dịch lại trung tâm |
| $\odot$ | nhân **từng phần tử** (elementwise) |

**(2) Vì sao chuẩn hoá theo token (không theo batch).** LayerNorm tính $\mu,\sigma$
**trong nội bộ một token**, nên hoàn toàn **không phụ thuộc batch** hay các token khác
→ ổn định khi batch nhỏ, khi inference từng mẫu, và không rò rỉ thông tin giữa các
road. Nó ép mỗi token về "cùng thang" (mean 0, var 1) để attention/FFN phía sau không
bị vài kênh biên độ lớn lấn át. $\gamma,\beta$ trả lại cho model quyền **học lại** biên
độ/độ dịch nếu việc chuẩn hoá cứng là quá mạnh. Dùng **pre-LN** (chuẩn hoá *trước* mỗi
sub-layer) giúp gradient chảy mượt, train sâu ổn định hơn post-LN.

**(3) Tính TAY** với $x=[1,2,3]$ (đặt $\gamma{=}1,\beta{=}0,\varepsilon\approx0$):

| Bước | Phép tính | Kết quả |
|---|---|---|
| $\mu$ | $(1+2+3)/3$ | $2$ |
| $\sigma^2$ | $\big[(1{-}2)^2+(2{-}2)^2+(3{-}2)^2\big]/3=(1+0+1)/3$ | $0.6667$ |
| $\sigma$ | $\sqrt{0.6667}$ | $0.8165$ |
| $\hat x_0$ | $(1-2)/0.8165$ | $-1.2247$ |
| $\hat x_1$ | $(2-2)/0.8165$ | $0$ |
| $\hat x_2$ | $(3-2)/0.8165$ | $+1.2247$ |

→ $\operatorname{LN}([1,2,3])=[-1.2247,\,0,\,+1.2247]$ (mean 0, độ lớn cân xứng).

> **Tóm 1 câu:** LayerNorm kéo **từng token** về mean 0 / var 1 rồi cho model
> "nắn lại" bằng $\gamma,\beta$ — chuẩn hoá theo chiều đặc trưng nên độc lập batch.

## 4.4 GELU (Gaussian Error Linear Unit)

**Dạng chính xác (erf) — code dùng dạng này:**
$$
\boxed{\ \operatorname{GELU}(x)=x\,\Phi(x)=x\cdot\tfrac12\Big(1+\operatorname{erf}\!\big(x/\sqrt2\big)\Big)\ }
$$
với $\Phi$ = CDF chuẩn tắc. **Dạng xấp xỉ tanh** (chỉ để tham khảo, code KHÔNG dùng):
$$
\operatorname{GELU}(x)\approx0.5x\Big(1+\tanh\!\big[\sqrt{2/\pi}\,(x+0.044715x^3)\big]\Big).
$$

**Ví dụ tính tay:**
$$
\operatorname{GELU}(1)=1\cdot\tfrac12(1+\operatorname{erf}(0.7071))=0.5(1+0.6827)=\mathbf{0.8413},
$$
$$
\operatorname{GELU}(-1)=-1\cdot\tfrac12(1+\operatorname{erf}(-0.7071))=0.5\cdot(-1)(1-0.6827)=\mathbf{-0.1587}.
$$
(Dạng tanh cho $0.8412$ — sai khác $\sim10^{-4}$.)

**Ý nghĩa.** GELU "mượt" hơn ReLU: cổng theo xác suất $\Phi(x)$ thay vì ngưỡng cứng.

#### Giải thích chi tiết: $\Phi$, $\operatorname{erf}$ là gì và vì sao mượt hơn ReLU

**(1) $\Phi$ và $\operatorname{erf}$ là gì.**
- $\Phi(x)$ = **CDF của phân phối chuẩn tắc** $\mathcal N(0,1)$ = "xác suất một biến
  ngẫu nhiên chuẩn nhỏ hơn $x$" $=P(Z\le x)$. Nó tăng từ $0$ (ở $-\infty$) lên $1$
  (ở $+\infty$), qua $0.5$ tại $x=0$.
- $\operatorname{erf}(u)=\dfrac{2}{\sqrt\pi}\displaystyle\int_0^u e^{-s^2}\,ds$ là
  **hàm sai số** (error function), hàm lẻ ($\operatorname{erf}(-u)=-\operatorname{erf}(u)$),
  chạy từ $-1$ đến $+1$. Liên hệ: $\Phi(x)=\tfrac12\big(1+\operatorname{erf}(x/\sqrt2)\big)$
  — chia $\sqrt2$ vì $\mathcal N(0,1)$ dùng $e^{-x^2/2}$ còn $\operatorname{erf}$ dùng $e^{-s^2}$.

Vậy $\operatorname{GELU}(x)=x\cdot\Phi(x)$ đọc là: "**giữ lại $x$ nhân với xác suất
$x$ vượt một ngưỡng chuẩn ngẫu nhiên**" — một cái **cổng mềm** (soft gate) trong $[0,1]$.

**(2) Vì sao mượt hơn ReLU.** ReLU$(x)=\max(0,x)$ là **cổng cứng**: nhân với 0 nếu
$x<0$, nhân với 1 nếu $x>0$, **gãy** tại 0 (đạo hàm nhảy $0\to1$, và $x<0$ có gradient
$=0$ → "dying ReLU"). GELU thay ngưỡng cứng bằng $\Phi(x)$ **trơn**: quanh 0 nó cong
mềm, cho $x$ âm nhỏ vẫn **rò một chút tín hiệu âm** (nên có gradient khác 0), giúp train
ổn định. Cụ thể GELU **âm nhẹ** quanh $x\in(-2,0)$ rồi tiến về 0.

**(3) Tính TAY** (dùng $\operatorname{erf}(0.7071)=0.6827$, $\operatorname{erf}(1.4142)=0.9545$):

| $x$ | $x/\sqrt2$ | $\operatorname{erf}(x/\sqrt2)$ | $\Phi(x)=\tfrac12(1+\cdot)$ | $\operatorname{GELU}=x\Phi(x)$ | ReLU |
|---|---|---|---|---|---|
| $-2$ | $-1.4142$ | $-0.9545$ | $0.0228$ | $-0.0455$ | $0$ |
| $-1$ | $-0.7071$ | $-0.6827$ | $0.1587$ | $-0.1587$ | $0$ |
| $0$ | $0$ | $0$ | $0.5$ | $0$ | $0$ |
| $0.5$ | $0.3536$ | $0.3829$ | $0.6915$ | $0.3457$ | $0.5$ |
| $1$ | $0.7071$ | $0.6827$ | $0.8413$ | $0.8413$ | $1$ |
| $2$ | $1.4142$ | $0.9545$ | $0.9772$ | $1.9545$ | $2$ |

Chi tiết hai mốc chính:
$$
\operatorname{GELU}(1)=1\cdot\tfrac12(1+0.6827)=0.5\cdot1.6827=0.8413,
$$
$$
\operatorname{GELU}(-1)=-1\cdot\tfrac12(1-0.6827)=-1\cdot0.5\cdot0.3173=-0.1587.
$$

Sơ đồ so hình dạng (ReLU gãy tại 0; GELU cong, thò xuống âm nhẹ bên trái):

```
 out |                         ReLU  /
     |                             / .
     |                           /  .  GELU (gan trung nhau khi x lon)
     |                         /  .
   0 +----------------+------/---------- x
     |          . . . |    /
     |      GELU am nhe khi x < 0 (ReLU = 0)
```

> **Tóm 1 câu:** GELU $=x\cdot\Phi(x)$ = "$x$ nhân xác suất chuẩn tắc $\le x$",
> một cổng **mềm** thay cho ngưỡng cứng của ReLU nên trơn, còn gradient ở vùng âm.

## 4.5 CLS token & Positional Encoding (3 kiểu)

- **RoadFury (PE học được):** $x=[\text{CLS};x]+\text{PE}$, với
  $\text{PE}=\texttt{Parameter}(1,L{+}1,d)$ khởi tạo $\mathcal N(0,0.02^2)$. CLS cũng học được.
- **SE2RoadNet:** **KHÔNG** có PE tuyệt đối — vị trí vào model **chỉ** qua bias $\Delta s$ (4.7).
- **Sinusoidal (ablation Vaswani 2017):**
$$
PE_{\text{pos},2i}=\sin\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big),\quad
PE_{\text{pos},2i+1}=\cos\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big).
$$

#### Giải thích chi tiết: CLS để làm gì, vì sao cần vị trí

**(1) CLS token là gì / để làm gì.** CLS ("classification") là **một token nhân tạo
thêm vào đầu chuỗi**, không ứng với điểm road nào. Qua các block attention, nó **đọc
và gom** thông tin từ mọi điểm (nó "hỏi" mọi key). Cuối cùng **chỉ vector CLS** được đưa
vào head (4.9) để ra logit → CLS đóng vai "bản tóm tắt toàn con đường". Đây là ý tưởng
mượn từ BERT: thay vì pooling trung bình, để một token học được cách tự tổng hợp.

Kích thước: chuỗi vào có $L$ điểm → thêm CLS thành $L+1$ token ($197\to198$).

**(2) Vì sao cần Positional Encoding (PE).** Self-attention là phép trên **tập hợp**:
nếu ta xáo trộn thứ tự các token, attention (không có PE) cho **kết quả y hệt** (hoán vị
đầu ra theo cùng hoán vị đầu vào). Nhưng con đường **có thứ tự** (điểm 1 → 2 → 3 …) và
thứ tự đó mang nghĩa (điểm nào cua trước, đoạn nào nối đoạn nào). Nếu không "tiêm" thông
tin vị trí, model sẽ coi con đường như một **túi điểm không thứ tự** → mất cấu trúc. PE
là cách **tiêm thứ tự** vào.

**(3) Ba kiểu PE — so sánh.**

| Kiểu | Vị trí mã hoá theo | Bất biến độ phân giải? | Bất biến xoay? | Dùng ở |
|---|---|---|---|---|
| **PE học được** | **chỉ số token** $0..L$ (learnable) | Không (đổi khi resample $N$) | Không (là kênh cộng thêm) | RoadFury |
| **Sinusoidal** | chỉ số token qua $\sin/\cos$ nhiều tần số | Không | Không | ablation (Vaswani 2017) |
| **Bias $\Delta s$** | **hiệu arc-length** $s_i-s_j$ (4.7) | **Có** ($s/L\in[0,1]$) | **Có** ($s$ nội tại) | SE2RoadNet |

Điểm mấu chốt: PE **tuyệt đối** đánh theo **chỉ số** ("đây là token số 37") nên khi
lấy mẫu lại road ($N$ đổi) thì "token số 37" trỏ vào chỗ khác → **không** bất biến độ
phân giải, và là một kênh cộng thêm nên **không** bất biến xoay. SE2RoadNet vì thế
**bỏ hẳn** PE tuyệt đối, chỉ đưa vị trí vào qua **hiệu cung** $\Delta s$ (mục 4.7) —
thứ vừa bất biến xoay vừa bất biến độ phân giải.

Sinusoidal (tham khảo): mỗi cặp chiều $(2i,2i{+}1)$ là một "kim đồng hồ" quay với tần số
giảm dần theo $i$ (số hạng $10000^{2i/d}$ ở mẫu càng lớn → quay càng chậm), giúp mã hoá
vị trí ở nhiều "độ phân giải" khác nhau.

> **Tóm 1 câu:** CLS = token "tóm tắt cả road" mà head đọc; PE = cách **tiêm thứ tự**
> vào attention vốn coi input như tập hợp — và SE2 chọn kiểu PE **theo cung $\Delta s$**
> để không mất bất biến.

## 4.6 Multi-Head Self-Attention (slide s-19, s-44)

$$
Q=zW_Q,\ K=zW_K,\ V=zW_V\quad(\text{mỗi cái } d\to d);\qquad
\text{tách } h=8 \text{ head, } d_k=d/h.
$$
$$
\boxed{\ \operatorname{head}=\operatorname{softmax}\!\Big(\frac{Q\,K^\top}{\sqrt{d_k}}+B\Big)V\ }
\qquad d_k=192/8=24\ (\text{SE2}),\ \sqrt{d_k}=4.899.
$$
Softmax theo trục **key $j$**. Gộp $h$ head → Linear ra $d\to d$. ($B$ = bias $\Delta s$ ở 4.7; RoadFury dùng $B{=}0$, chỉ có PE.)

**Softmax:** $\operatorname{softmax}(v)_j=\dfrac{e^{v_j}}{\sum_k e^{v_k}}$.

#### Giải thích chi tiết: Q/K/V, vì sao chia $\sqrt{d_k}$, softmax làm gì

**(1) Q, K, V nghĩa là gì (ẩn dụ tra cứu).** Mỗi token chiếu qua 3 ma trận học được
thành ba vai:

| Vai | Tên | Ẩn dụ | Vai trò |
|---|---|---|---|
| $Q$ | **Query** (truy vấn) | "tôi đang **cần** thông tin gì?" | token đang xét dùng để so khớp |
| $K$ | **Key** (khoá) | "tôi **chứa** thông tin gì?" | mọi token quảng cáo nội dung |
| $V$ | **Value** (giá trị) | "nội dung tôi sẽ **trao đi**" | thứ được lấy ra khi khớp |

Điểm số $Q_iK_j^\top$ = "độ hợp" giữa nhu cầu của token $i$ và nội dung token $j$
(tích vô hướng: càng cùng hướng càng lớn). Softmax biến các điểm này thành **trọng số**,
rồi token $i$ nhận **tổng có trọng số của $V$** → "đọc nhiều từ token mà nó thấy liên
quan". Có $h=8$ **head**: chạy song song 8 phép attention trên các không gian con
$d_k=d/h$ chiều, mỗi head học một kiểu quan hệ (gần theo cung, cùng độ cong, …), rồi
ghép lại.

**(2) Vì sao chia $\sqrt{d_k}$.** Tích vô hướng $Q_i\cdot K_j$ là tổng của $d_k$ số
hạng; nếu mỗi thành phần cỡ phương sai 1 thì tổng có **phương sai $\sim d_k$**, tức
biên độ $\sim\sqrt{d_k}$. Điểm số quá lớn đẩy softmax vào vùng **bão hoà** (một trọng
số $\approx1$, còn lại $\approx0$) → gradient teo, học kém. Chia $\sqrt{d_k}$ kéo phương
sai về $\sim1$, giữ softmax "mềm".

*Minh hoạ số* ($d_k=24,\ \sqrt{d_k}=4.899$): giả sử điểm thô $Q_i\cdot K=[10,\,0]$.
$$
\text{không chia: }\operatorname{softmax}([10,0])=\Big[\tfrac{e^{10}}{e^{10}+1},\ \tfrac{1}{e^{10}+1}\Big]\approx[0.99995,\ 0.00005]\ (\text{gần cứng});
$$
$$
\text{chia }\sqrt{d_k}: [10,0]/4.899=[2.041,\,0]\Rightarrow\operatorname{softmax}=[0.885,\ 0.115]\ (\text{mềm hơn hẳn}).
$$

**(3) Softmax làm gì — tính TAY.** Softmax biến vector điểm bất kỳ thành **phân phối
xác suất** (dương, tổng 1), nhấn mạnh giá trị lớn nhưng vẫn trơn.

*Ví dụ đều* (không có gì nổi trội): $v=[1,1,1]$ →
$$
\operatorname{softmax}([1,1,1])=\Big[\tfrac{e^1}{3e^1},\dots\Big]=[0.333,\,0.333,\,0.333]
$$
(bằng nhau → attention rải đều).

*Ví dụ có chênh lệch* (đã chia $\sqrt{d_k}$, điểm $=[2.041,\,1.0,\,0]$):
$$
e^{2.041}=7.70,\ e^{1.0}=2.718,\ e^{0}=1;\quad \text{tổng}=11.418;
$$
$$
\operatorname{softmax}=[7.70,\,2.718,\,1]/11.418=[0.674,\,0.238,\,0.088].
$$
→ key mạnh nhất chiếm $\approx67\%$ trọng số nhưng hai key kia vẫn được nghe.

> **Tóm 1 câu:** attention = "token **Query** so khớp mọi **Key**, softmax (đã chia
> $\sqrt{d_k}$ để không bão hoà) đổi độ khớp thành trọng số, rồi lấy trung bình có trọng
> số các **Value**"; 8 head học 8 kiểu quan hệ song song.

## 4.7 (*) Attention bias theo hiệu cung $\Delta s$ (RFF) — slide s-19, s-44, s-45

**Công thức (5 bước):**
$$
\text{(1)}\quad \Delta s_{ij}=s_i-s_j\quad(\text{CLS gán } s=0);
$$
$$
\text{(2)}\quad \varphi=\sin(\Delta s_{ij}\cdot\omega),\quad \omega\in\mathbb{R}^{32}\ \text{cố định (frozen)},\ \omega\sim\mathcal N(0,2^2);
$$
$$
\text{(3)}\quad B^{(h)}_{ij}=\big[\operatorname{MLP}_{32\to64\to8}(\varphi)\big]_h
=\big[W_2\,\operatorname{GELU}(W_1\varphi+b_1)+b_2\big]_h\in\mathbb{R}^{8};
$$
$$
\text{(4)}\quad \text{score}_{ij}=\frac{q_i\cdot k_j}{\sqrt{d}}+B_{ij};\qquad
\text{(5)}\quad \text{attn}=\operatorname{softmax}_j(\text{score}),\ \ \text{out}_i=\sum_j \text{attn}_{ij}\,v_j.
$$

**Vì sao bất biến (4 lý do):**
- Dùng **hiệu** $\Delta s$: dời gốc cung ($s\to s+c$) → $\Delta s$ không đổi.
- $s/L\in[0,1]$: bất biến **độ phân giải** (N=64 hay 197 vẫn cùng thang).
- $s$ là cung **nội tại**: bất biến xoay/tịnh tiến SE(2).
- $\sin$ là hàm **lẻ**: $\Delta s_{ji}=-\Delta s_{ij}$ → bias **bất đối xứng** (phân biệt trước/sau).

**Ví dụ tính tay** (slide s-45). $s_{\text{full}}$ (CLS + 5 điểm) $=[0,\,0.160,\,0.320,\,0.547,\,0.774,\,1.000]$.
Query = điểm 2 ($s=0.547$):
$$
\Delta s_{2j}=0.547-s_j=[\,+0.547,\,+0.387,\,+0.227,\,0.000,\,-0.227,\,-0.453\,].
$$
RFF cho cặp $(2,4)$, $\Delta s=-0.453$, minh hoạ 3/32 tần số $\omega=\{1,2,4\}$:
$$
\varphi=\sin([-0.453,-0.906,-1.812])=[\,-0.438,\,-0.787,\,-0.971\,]\ \to\ \operatorname{MLP}\to B_{2,4}\in\mathbb{R}^8.
$$

**Ví dụ softmax trước/sau bias** (toy 3 token, score thô $=1$ đều, bias $=[+0.8,+1.0,-0.5]$):
$$
\text{trước: } [0.333,\,0.333,\,0.333]\ \Rightarrow\ \text{sau: } [0.401,\,0.490,\,0.109].
$$
→ bias kéo attention về token gần theo cung. Chi phí $\mathcal O(B\cdot L^2\cdot 32)$ mỗi block.

**So sánh PE:** PE tuyệt đối đánh theo **chỉ số token** → đổi khi lấy mẫu lại; bias $\Delta s$ theo **cung** → không đổi.

### 4.7.1 Giải thích chi tiết: vì sao RFF, vì sao $\sin$, vì sao MLP

#### (a) Từng ký hiệu là gì / ở đâu ra

| Ký hiệu | Là gì | Nguồn |
|---|---|---|
| $s_i$ | **arc-length chuẩn hoá** của token $i$ ($s/L\in[0,1]$) | kênh c6 (Phần 3.5), đọc lại trong model |
| $\Delta s_{ij}=s_i-s_j$ | **khoảng cách theo cung** (có dấu) giữa query $i$ và key $j$ | tính trong block |
| $\omega\in\mathbb R^{32}$ | 32 **tần số Fourier ngẫu nhiên**, đóng băng, lấy từ $\mathcal N(0,2^2)$ | khởi tạo 1 lần, không train |
| $\varphi=\sin(\Delta s\cdot\omega)$ | **đặc trưng RFF** (32 chiều) mã hoá $\Delta s$ | bước (2) |
| $\operatorname{MLP}_{32\to64\to8}$ | mạng nhỏ $W_1(32{\to}64)$, GELU, $W_2(64{\to}8)$ | học được |
| $B^{(h)}_{ij}$ | **bias cộng vào điểm attention** của head $h$ | bước (3) |

CLS gán $s=0$ nên $\Delta s$ của CLS với điểm $j$ là $-s_j$ (CLS "đứng ở gốc cung").

#### (b) Vì sao thiết kế như vậy

**Vì sao RFF (random Fourier features)?** Ta muốn bias là một **hàm trơn bất kỳ của
khoảng cách** $\Delta s$ — "gần thì thiên vị nhiều, xa thì ít", nhưng không biết trước
hình dạng hàm đó. Định lý Rahimi–Recht: chiếu $\Delta s$ lên một bó $\sin$ với **tần số
ngẫu nhiên** rồi lấy tổ hợp tuyến tính thì **xấp xỉ được mọi kernel dịch-bất-biến**
(shift-invariant). Nên thay vì học trực tiếp một hàm của số vô hướng $\Delta s$ (rất
khó, dễ overfit), ta "nở" $\Delta s$ thành 32 toạ độ $\varphi$ giàu thông tin rồi để
MLP nhỏ tổ hợp. $\omega$ **đóng băng** ($\sim\mathcal N(0,2^2)$): phương sai $2^2$ chọn
dải tần hợp với thang $s\in[0,1]$ (không quá mịn, không quá thô); giữ cố định để phần
"cơ sở Fourier" ổn định, chỉ MLP học.

**Vì sao $\sin$ (không $\cos$)?** $\sin$ là hàm **lẻ**: $\sin(-a)=-\sin(a)$. Do đó khi
đổi vai query/key, $\Delta s_{ji}=-\Delta s_{ij}$ kéo theo $\varphi$ đổi dấu → bias
$B_{ji}\neq B_{ij}$ nói chung → attention **bất đối xứng**, phân biệt được **"đứng trước"
và "đứng sau"** dọc con đường (chiều đi có nghĩa). Nếu dùng $\cos$ (chẵn) thì $B_{ij}=B_{ji}$,
mất chiều.

**Vì sao MLP (không dùng thẳng $\varphi$)?** Cần **8 con số** bias khác nhau (mỗi head
một profile khoảng cách riêng: head này "chuộng lân cận sát", head kia "nhìn xa"). MLP
$32\to64\to8$ trộn phi tuyến $\varphi$ (nhờ GELU) rồi xuất 8 kênh → mỗi head có hàm bias
riêng theo $\Delta s$, học được.

#### (c) Tính TAY

**Ma trận $\Delta s$ (6×6, antisymmetric).** Với $s_{\text{full}}=[0,0.160,0.320,0.547,0.774,1.000]$,
$\Delta s_{ij}=s_i-s_j$ (hàng = query $i$, cột = key $j$; thứ tự CLS, pt0..pt4):

| $i\backslash j$ | CLS | pt0 | pt1 | pt2 | pt3 | pt4 |
|---|---|---|---|---|---|---|
| **CLS** | 0 | -0.160 | -0.320 | -0.547 | -0.774 | -1.000 |
| **pt0** | +0.160 | 0 | -0.160 | -0.387 | -0.614 | -0.840 |
| **pt1** | +0.320 | +0.160 | 0 | -0.227 | -0.454 | -0.680 |
| **pt2** | +0.547 | +0.387 | +0.227 | 0 | -0.227 | -0.453 |
| **pt3** | +0.774 | +0.614 | +0.454 | +0.227 | 0 | -0.226 |
| **pt4** | +1.000 | +0.840 | +0.680 | +0.453 | +0.226 | 0 |

Hàng **pt2** chính là $\Delta s_{2j}=[+0.547,+0.387,+0.227,0,-0.227,-0.453]$ (khớp s-45).
Chú ý **đối xứng lệch dấu**: ô $(\text{pt2},\text{pt4})=-0.453$ còn $(\text{pt4},\text{pt2})=+0.453$.

**RFF cho cặp (pt2, pt4)**, $\Delta s=-0.453$, lấy 3/32 tần số $\omega=\{1,2,4\}$:
$$
\Delta s\cdot\omega=[-0.453,\,-0.906,\,-1.812]\Rightarrow
\varphi=\sin(\cdot)=[-0.438,\,-0.787,\,-0.971].
$$
Cặp ngược (pt4, pt2), $\Delta s=+0.453$: $\varphi=[+0.438,+0.787,+0.971]$ — **đổi dấu
toàn bộ** (minh hoạ tính lẻ của $\sin$ → bias bất đối xứng). MLP ăn 32 số $\varphi$ →
ra $B\in\mathbb R^8$ (một số cho mỗi head).

**Softmax trước/sau bias** (toy 3 key, điểm thô $q\cdot k=[1,1,1]$ đều, bias sau MLP giả
định $=[+0.8,+1.0,-0.5]$):
$$
\text{trước: } \operatorname{softmax}([1,1,1])=[0.333,0.333,0.333];
$$
$$
\text{điểm+bias}=[1.8,\,2.0,\,0.5]\Rightarrow e^{[\cdot]}=[6.05,\,7.39,\,1.65],\ \text{tổng}=15.09;
$$
$$
\text{sau: } [6.05,7.39,1.65]/15.09=[0.401,\,0.490,\,0.109].
$$
Bias **dương** (2 key đầu, gần theo cung) kéo trọng số lên; bias **âm** (key 3, xa)
đẩy xuống → attention "chuộng hàng xóm theo cung".

#### (d) Bốn lý do bất biến — giải thích

1. **Dùng hiệu $\Delta s$:** nếu chọn gốc đo cung ở chỗ khác ($s\to s+c$) thì
   $\Delta s_{ij}=(s_i+c)-(s_j+c)=s_i-s_j$ **không đổi** → bias không đổi.
2. **$s/L\in[0,1]$ (chuẩn hoá):** dù road lấy $N=64$ hay $197$ điểm, cung vẫn trải trên
   $[0,1]$ cùng thang → **bất biến độ phân giải** (khác hẳn PE theo chỉ số).
3. **$s$ là cung nội tại:** độ dài cung không đổi khi xoay/tịnh tiến (đã chứng minh
   `seg` bất biến ở 4.1) → **bất biến SE(2)**.
4. **$\sin$ lẻ:** $\Delta s_{ji}=-\Delta s_{ij}$ → $B$ bất đối xứng → giữ được **chiều
   trước/sau** dọc đường (thông tin có nghĩa, không nên vứt).

> **Tóm 1 câu:** thay PE tuyệt đối bằng **bias theo hiệu cung** $\Delta s$ nở qua RFF
> ($\sin$ lẻ để có chiều) rồi MLP xuất 8 bias/head → attention "biết xa-gần theo cung"
> mà vẫn **bất biến xoay + độ phân giải**.

## 4.8 Sigmoid (đầu ra xác suất)

$$
p=\sigma(\text{logit})=\frac{1}{1+e^{-\text{logit}}}\in[0,1].
$$
**Ví dụ:** $\sigma(0)=0.5$, $\sigma(1)=0.7311$, $\sigma(-1)=0.2689$, $\sigma(2.2)=0.9002$.

#### Giải thích chi tiết

**(1) Ký hiệu.** `logit` $=z\in\mathbb R$ = điểm số thô (chưa chuẩn hoá) do head xuất
ra; $\sigma$ ép $z$ về xác suất $p\in(0,1)$ = "khả năng road này FAIL".

**(2) Vì sao dùng sigmoid.** Ta cần biến một số thực bất kỳ thành **xác suất**. Sigmoid
đơn điệu tăng, $\sigma(0)=0.5$ (ngưỡng trung tính), đối xứng $\sigma(-z)=1-\sigma(z)$,
và là **nghịch đảo của logit** $z=\ln\frac{p}{1-p}$ — khớp tự nhiên với BCE/Focal loss.
Với **ranking APFD**, sigmoid **đơn điệu** nên **không đổi thứ tự**: sort theo $z$ hay
theo $p=\sigma(z)$ cho **cùng một ranking** (sigmoid chỉ để đọc ra xác suất, không ảnh
hưởng APFD).

**(3) Tính TAY.**

| $z$ | $e^{-z}$ | $\sigma(z)=1/(1+e^{-z})$ |
|---|---|---|
| $0$ | $1$ | $1/2=0.5$ |
| $1$ | $0.3679$ | $1/1.3679=0.7311$ |
| $-1$ | $2.7183$ | $1/3.7183=0.2689$ |
| $2.2$ | $0.1108$ | $1/1.1108=0.9002$ |

Kiểm đối xứng: $\sigma(1)+\sigma(-1)=0.7311+0.2689=1.0000$ (OK).

> **Tóm 1 câu:** sigmoid ép logit thành xác suất FAIL trong $(0,1)$, đơn điệu nên
> **giữ nguyên ranking** — chỉ đổi "thang đọc", không đổi APFD.

## 4.9 Đầu ra (Head) & Forward pass tổng thể

Head: $\text{CLS}\to \operatorname{LN}\to \text{Linear}(d\to64)\to\operatorname{GELU}\to\text{Dropout}(0.2)\to\text{Linear}(64\to1)\to$ squeeze $\to$ logit.

**Bảng shape (SE2RoadNet, $B$ batch, $L{=}197$, $d{=}192$):**

| Bước | Phép toán | Output shape |
|---|---|---|
| A | Trích 7 kênh + z-norm | $(B,7,197)$ |
| B | permute + Linear$7\to192$ + LN + GELU | $(B,197,192)$ |
| B | Prepend CLS | $(B,198,192)$ |
| D | $\times6$ InvariantBlock (MHA + bias $\Delta s$, FFN 512) | $(B,198,192)$ |
| F | Lấy CLS + head $192\to64\to1$ | $(B,)$ logit |
| G | sigmoid → sort giảm dần → APFD | scalar |

**Số tham số:** RoadFury $\approx$ **828,801** ($\sim$829K); SE2RoadNet $\approx$ **2,108,721** ($\sim$2.11M).

#### Giải thích chi tiết: head làm gì và luồng forward end-to-end

**(1) Head = "đầu đọc CLS".** Sau 6 InvariantBlock, token **CLS** đã gom cả con đường.
Head chỉ lấy **một** vector CLS ($d=192$) và ép dần về **1 logit**:
$$
\text{CLS} \;\to\; \operatorname{LN} \;\to\; \text{Linear}(192\to64) \;\to\; \operatorname{GELU} \;\to\; \text{Dropout}(0.2) \;\to\; \text{Linear}(64\to1) \;\to\; z.
$$
LN ổn định đầu vào head; Linear$192\to64$ + GELU là một tầng phi tuyến nén; Dropout 0.2
regularize (chỉ khi train); Linear$64\to1$ ra **một số** = logit. `squeeze` bỏ chiều
thừa để ra shape $(B,)$.

**(2) Forward pass end-to-end** (bám bảng A→G): 7 kênh hình học → z-norm → `proj`
$7\to192$ + LN + GELU (bước B) → ghép CLS thành $198$ token → **6 block** attention (mỗi
block có MHA + **bias $\Delta s$** ở 4.7 + FFN ẩn 512) trộn thông tin giữa các điểm →
lấy CLS → **head** ra logit → **sigmoid** → **sort giảm dần** → **APFD**.

**(3) Chạy tay một mini-batch 5 road** (nối mạch từ 4.8). Giả sử head xuất logit cho 5
test là $z=[2.2,\,-1,\,1,\,-1,\,0]$; qua sigmoid (số ở 4.8):
$$
p=\sigma(z)=[0.9002,\,0.2689,\,0.7311,\,0.2689,\,0.5000].
$$
Sort **giảm dần** theo $p$ (hoán vị $\pi$):
$$
\underbrace{0.9002}_{A}\ >\ \underbrace{0.7311}_{C}\ >\ \underbrace{0.5000}_{E}\ >\ \underbrace{0.2689}_{B}\ =\ \underbrace{0.2689}_{D}.
$$
Nếu nhãn thật là **FAIL ở A và C** ($m=2$, $n=5$), hai FAIL rơi vào **rank 1 và 2**:
$$
\operatorname{APFD}=1-\frac{1+2}{5\cdot2}+\frac{1}{2\cdot5}=1-0.3+0.1=\mathbf{0.80}.
$$
Đây đúng là "xếp tốt" trong Phần 1.2 — mạch số chạy liền: feature (Phần 3) → block
(4.6–4.7) → head (4.9) → logit → sigmoid (4.8) → sort → APFD (Phần 1.2).

**(4) Vì sao SE2 nhiều tham số hơn.** RoadFury $d{=}128$; SE2RoadNet $d{=}192$ (rộng
hơn) + module RFF/MLP cho bias $\Delta s$ ở **mỗi** block → $\sim2.11$M so với $\sim829$K.
Đổi lại: bất biến xoay **exact** ($\Delta{=}0$) và AUC cao hơn (xem Phần 7).

> **Tóm 1 câu:** head đọc **riêng token CLS** ($192\to64\to1$) ra logit; forward =
> "7 kênh → proj → +CLS → 6 block (MHA + bias $\Delta s$) → CLS → head → sigmoid →
> sort → APFD", và trên 5 road ví dụ cho APFD $=0.80$.

---

# Phần 5 — Hàm mất mát & Huấn luyện (slide s-20)

## 5.1 BCE with logits (nền tảng)

$$
\operatorname{BCE}(z,y)=-\big[y\log\sigma(z)+(1-y)\log(1-\sigma(z))\big].
$$

#### Giải thích chi tiết: BCE từ đâu ra, vì sao lại là dạng $-[y\log p+(1-y)\log(1-p)]$

**(a) Từng ký hiệu là gì.**

| Ký hiệu | Là gì | Ở đâu ra |
|---|---|---|
| $z$ | **logit** (đầu ra thô của head, chưa qua sigmoid) | Phần 4.9, bước F |
| $p=\sigma(z)$ | xác suất model gán cho lớp FAIL | sigmoid Phần 4.8 |
| $y\in\{0,1\}$ | nhãn thật: $1$=FAIL, $0$=PASS | dữ liệu train |
| $\log$ | logarit tự nhiên ($\ln$) | quy ước ML |

Chữ **"with logits"** nghĩa là hàm nhận thẳng $z$ (không nhận $p$): nó tự tính
$\sigma(z)$ **bên trong** theo dạng ổn định số học, tránh tràn khi $z$ lớn.

**(b) Vì sao công thức như vậy (suy từ likelihood).**

Xem mỗi test là một phép thử Bernoulli: model tin FAIL với xác suất $p$, PASS với
xác suất $1-p$. Xác suất model "giải thích đúng" một nhãn $y$ quan sát được:

$$
P(y\mid p)=p^{\,y}\,(1-p)^{\,1-y}
=\begin{cases}p & y=1\\ 1-p & y=0\end{cases}.
$$

Lấy $\log$ (biến tích thành tổng) rồi **đổi dấu** (vì ta *cực tiểu hoá* loss thay
vì *cực đại hoá* likelihood):

$$
-\log P(y\mid p)=-\big[y\log p+(1-y)\log(1-p)\big]=\operatorname{BCE}.
$$

Nói cách khác BCE chính là **negative log-likelihood** của phân phối Bernoulli, cũng
đồng nhất với **cross-entropy** giữa phân phối thật $(y,\,1-y)$ và phân phối đoán
$(p,\,1-p)$. Cơ chế phạt: khi model **tự tin nhưng sai** ($p\to0$ mà $y=1$), số hạng
$-\log p\to+\infty$ → phạt rất nặng; khi **đúng và tự tin** ($p\to1$, $y=1$) →
$-\log p\to0$.

Dạng ổn định số học (code dùng, tương đương về giá trị):
$$
\operatorname{BCE}(z,y)=\max(z,0)-z\,y+\log\!\big(1+e^{-|z|}\big).
$$

**(c) Tính TAY từng bước.** Dùng các giá trị sigmoid đã có ở Phần 4.8
($\sigma(1)=0.7311$, $\sigma(-1)=0.2689$):

| Trường hợp | $y$ | $p=\sigma(z)$ | Phép tính | BCE |
|---|---|---|---|---|
| FAIL đoán đúng, tự tin | 1 | $\sigma(1)=0.7311$ | $-\ln 0.7311$ | $\mathbf{0.3132}$ |
| FAIL đoán lệch | 1 | $0.3$ | $-\ln 0.3$ | $\mathbf{1.2040}$ |
| FAIL đoán **rất sai** | 1 | $\sigma(-1)=0.2689$ | $-\ln 0.2689$ | $\mathbf{1.3134}$ |
| PASS đoán đúng | 0 | $\sigma(-1)=0.2689$ | $-\ln(1-0.2689)=-\ln 0.7311$ | $\mathbf{0.3132}$ |

Kiểm dạng ổn định cho dòng 1 ($z=1,y=1$):
$$
\max(1,0)-1\cdot1+\log(1+e^{-1})=0+\log(1.3679)=\mathbf{0.3133}\ (\approx0.3132,\ \text{khớp}).
$$

Đọc bảng: cùng là ca FAIL nhưng đoán **rất sai** ($p=0.2689$) bị phạt $1.3134$, gấp
hơn $4\times$ so với đoán đúng ($0.3132$) — độ phạt tăng **phi tuyến** theo mức độ tự
tin sai.

> **Tóm 1 câu:** BCE = negative log-likelihood của Bernoulli; nó là "độ ngạc nhiên"
> $-\log(\text{xác suất model gán cho nhãn đúng})$, phạt càng nặng khi model tự tin mà sai.

## 5.2 Focal Loss

$$
p=\sigma(z),\quad
p_t=\begin{cases}p & y=1\\ 1-p & y=0\end{cases},\quad
w=\begin{cases}w_+ & y=1\\ 1 & y=0\end{cases};
$$
$$
\boxed{\ \mathcal L=\operatorname{mean}\Big[\alpha\,(1-p_t)^{\gamma}\cdot w\cdot \operatorname{BCE}(z,y)\Big]\ }
$$

**Ký hiệu.** $\gamma$ (focusing) down-weight ca dễ ($p_t\to1$), tập trung ca khó;
$\alpha=1$; $w_+=n_{\text{neg}}/n_{\text{pos}}$ (pos_weight); $\gamma=0$ → về BCE có trọng số.

**Ví dụ tính tay** ($y=1$, $p=0.3$, $\alpha=1$, $w_+=1$):

| $\gamma$ | $(1-p_t)^\gamma=(0.7)^\gamma$ | BCE $=-\ln0.3$ | term |
|---|---|---|---|
| 0 | 1.0000 | 1.2040 | 1.2040 |
| 1.0 | 0.7000 | 1.2040 | 0.8428 |
| 1.5 | 0.5857 | 1.2040 | 0.7051 |
| 2.0 | 0.4900 | 1.2040 | 0.5899 |
| **2.5** | 0.4100 | 1.2040 | **0.4936** |

→ $\gamma$ càng lớn, ca "đã đúng khá chắc" ($p=0.3$ cho FAIL vẫn còn sai) bị giảm trọng số càng mạnh.

**Cài đặt $\gamma$** (đọc kỹ Phần 7): slide s-20 hiện dùng $\gamma=1.0$ (APFD $0.8095$);
script cũ dùng $\gamma=1.5$; RoadFury tốt nhất $\gamma=2.5$ (APFD $0.8066$).

#### Giải thích chi tiết: $p_t$, $w$, và vì sao $(1-p_t)^\gamma$ "làm mờ" ca dễ

**(a) $p_t$ và $w$ là gì.**

- $p_t$ = **xác suất model gán cho lớp ĐÚNG** của mẫu đó. Nếu $y=1$ (FAIL) thì
  $p_t=p$; nếu $y=0$ (PASS) thì $p_t=1-p$ (vì "gán cho PASS" $=1-p$). Nhờ vậy chỉ cần
  một biến $p_t$ là đo được "model đã đúng tới mức nào", bất kể nhãn là 0 hay 1.
  - $p_t\to1$: model **vừa đúng vừa tự tin** → **ca dễ**.
  - $p_t\to0$: model **tự tin nhưng sai** → **ca khó** (đáng học nhất).
- $w$ = **trọng số theo lớp**: $w_+=n_{\text{neg}}/n_{\text{pos}}$ cho FAIL, $1$ cho
  PASS (xem 5.3). $\alpha$ = hệ số vô hướng chung (ở đây $\alpha=1$).
- $\operatorname{BCE}(z,y)=-\log p_t$ (viết gọn), nên focal chỉ là **BCE nhân thêm
  hai bộ điều tiết**: $(1-p_t)^\gamma$ (theo độ khó) và $w$ (theo lớp).

**(b) Vì sao dùng $(1-p_t)^\gamma$.** Đây là **modulating factor** (Lin et al.,
2017). Đọc hai đầu:

- Ca dễ $p_t\to1$: $(1-p_t)^\gamma\to0$ → gần như **xoá** đóng góp của mẫu đó vào loss.
- Ca khó $p_t\to0$: $(1-p_t)^\gamma\to1$ → **giữ nguyên** gần như toàn bộ BCE.

Mũ $\gamma$ điều chỉnh "gắt tới đâu": $\gamma=0$ → hệ số $=1$ → về đúng BCE (có trọng
số $w$); $\gamma$ lớn → ca dễ bị dập mạnh hơn nữa. Ý nghĩa vật lý: dữ liệu SDC có rất
nhiều PASS "rõ ràng dễ" — nếu để BCE thường, tổng loss bị **hàng loạt ca dễ lấn át**,
gradient toàn hướng về việc "đoán PASS cho chắc". Focal **hạ tiếng nói của ca dễ** để
model dành sức cho ranh giới FAIL/PASS khó.

**(c) Giải thích từng dòng bảng $\gamma$** (đều ở $y=1,\ p=0.3$ nên $p_t=0.3$,
$1-p_t=0.7$):

- $\gamma=0$: $0.7^0=1.0$ → term $=1.0\times1.2040=1.2040$ (đúng bằng BCE, không đổi).
- $\gamma=1$: $0.7^1=0.7$ → giảm $30\%$ → $0.7\times1.2040=0.8428$.
- $\gamma=1.5$: $0.7^{1.5}=0.7\cdot\sqrt{0.7}=0.7\cdot0.8367=0.5857$ → $0.7051$.
- $\gamma=2$: $0.7^2=0.49$ → còn dưới nửa → $0.5899$.
- $\gamma=2.5$: $0.7^{2.5}=0.49\cdot0.8367=0.4100$ → $0.4936$.

**(d) Ca dễ bị dập mạnh cỡ nào?** So một ca dễ với ca khó ở cùng $\gamma=2$, dùng
$\sigma(2.2)=0.9002$ (Phần 4.8) cho ca dễ:

| Mẫu | $p=\sigma(z)$ | $p_t$ | $(1-p_t)^2$ | $\operatorname{BCE}=-\ln p_t$ | term |
|---|---|---|---|---|---|
| FAIL **dễ** (tự tin đúng) | $\sigma(2.2)=0.9002$ | $0.9002$ | $(0.0998)^2=0.00996$ | $0.1051$ | $\mathbf{0.0010}$ |
| FAIL **khó** (đoán lệch) | $0.3$ | $0.3$ | $(0.7)^2=0.49$ | $1.2040$ | $\mathbf{0.5899}$ |

Hệ số điều tiết của ca khó lớn gấp $0.49/0.00996\approx\mathbf{49\times}$ ca dễ: focal
làm mẫu dễ gần như "tàng hình" trong gradient, còn mẫu khó vẫn nguyên trọng lượng.

**(e) Ba cài đặt $\gamma$ đã chạy** (chi tiết ở Phần 7.1):

| Nguồn | $\gamma$ | APFD |
|---|---|---|
| Deck s-20 (hiện dùng) | $1.0$ | $0.8095$ |
| Script cũ | $1.5$ | $0.8047$ |
| RoadFury canonical | $2.5$ | $0.8066$ |

Cả ba chồng nhau trong $\pm\sigma$; chọn $\gamma=1.0$ vì cao nhất và robust.

> **Tóm 1 câu:** Focal = BCE nhân $(1-p_t)^\gamma$ (làm mờ ca đã đoán đúng chắc) và
> nhân $w$ (đền cho lớp FAIL hiếm), nhờ đó gradient tập trung vào ranh giới khó thay
> vì bị biển ca dễ nhấn chìm.

## 5.3 Cân bằng lớp

$$
w_+=\text{pos\_weight}=\frac{n_{\text{neg}}}{n_{\text{pos}}}\ (\approx 2.33\ \text{khi FAIL }30\%),\quad
\text{WeightedRandomSampler}(w).
$$

#### Giải thích chi tiết: vì sao cân bằng lớp, và hai cách làm

**(a) Vấn đề.** SDC test có **FAIL là thiểu số** (khoảng $30\%$). Nếu để BCE thường,
model có thể đạt loss thấp bằng chiêu tầm thường: **đoán PASS cho mọi test**. Khi đó
nó đúng $70\%$ nhưng **vô dụng để xếp hạng** (không FAIL nào được đẩy lên đầu). Ta cần
ép model coi trọng lớp FAIL. Hai cách tấn công cùng một vấn đề từ hai phía:

**(b) Cách 1 — pos_weight (phía loss).** Nhân riêng số hạng của FAIL trong BCE với
$w_+$, để **một lỗi trên FAIL bị phạt nặng bằng nhiều lỗi trên PASS**. Chọn
$$
w_+=\frac{n_{\text{neg}}}{n_{\text{pos}}}
$$
để **tổng trọng số hai lớp bằng nhau**: $n_{\text{pos}}\cdot w_+ = n_{\text{neg}}\cdot1$.

Tính tay (FAIL $=30\%$): với ví dụ $N=956$ test → $n_{\text{pos}}=0.3\times956\approx287$
FAIL, $n_{\text{neg}}=956-287=669$ PASS:
$$
w_+=\frac{n_{\text{neg}}}{n_{\text{pos}}}=\frac{669}{287}=2.331\ \approx\ \frac{0.7}{0.3}=\mathbf{2.33}.
$$
Nghĩa là mỗi ca FAIL "đáng giá" $\approx2.33$ ca PASS trong hàm mất mát.

**(c) Cách 2 — WeightedRandomSampler (phía dữ liệu).** Thay vì đổi loss, ta **đổi tần
suất lấy mẫu**: gán mỗi test một trọng số lấy mẫu tỉ lệ nghịch với kích thước lớp của
nó,
$$
w_i \propto \frac{1}{\text{số mẫu cùng lớp với } i}
\quad\Rightarrow\quad
w_i=\begin{cases}1/n_{\text{pos}} & y_i=1\\ 1/n_{\text{neg}} & y_i=0\end{cases}.
$$
Khi đó xác suất rút trúng một FAIL bằng xác suất rút trúng một PASS
($n_{\text{pos}}\cdot\frac{1}{n_{\text{pos}}}=n_{\text{neg}}\cdot\frac{1}{n_{\text{neg}}}=1$),
nên **mỗi mini-batch xấp xỉ 50% FAIL / 50% PASS** dù dữ liệu gốc lệch $30/70$. FAIL
được **over-sample** (lặp lại), PASS bị **under-sample**.

```
Du lieu goc (lech):   F F P P P P P P P P   (30% FAIL)
                       |
   WeightedRandomSampler (over-sample F)
                       v
Mini-batch (can bang): F P F P P F F P F P   (~50% FAIL)
```

**(d) Lưu ý.** Hai cách nhắm cùng đích (làm FAIL "nặng cân" hơn) nên khi dùng đồng
thời cần vừa phải để không **đền bù kép**; ở đây $\alpha=1$ giữ tổng scale hợp lý.

> **Tóm 1 câu:** FAIL hiếm ($30\%$) nên phải "đền" cho nó — pos_weight $=n_{\text{neg}}/n_{\text{pos}}\approx2.33$
> làm nặng lỗi FAIL trong loss, còn WeightedRandomSampler kéo mỗi batch về cân bằng
> $50/50$; hai đòn cùng chống thiên lệch "đoán PASS cho chắc".

## 5.4 Lịch học (LR): warmup + cosine

$$
\text{factor}(e)=
\begin{cases}
\dfrac{e+1}{\text{warm}} & e<\text{warm}\ (=5)\\[2mm]
\max\!\Big(0.01,\ 0.5\big(1+\cos\frac{\pi(e-\text{warm})}{\text{epochs}-\text{warm}}\big)\Big) & e\ge\text{warm}
\end{cases}
$$
**Ví dụ** (warm=5, epochs=80): $e{=}0\to0.20$; $e{=}4\to1.00$; $e{=}42\to0.51$; $e{=}79\to0.01$.
Optimizer **AdamW** (lr $5\times10^{-4}$, weight decay $10^{-3}$), grad-clip $1.0$, precision **bf16**.

#### Giải thích chi tiết: vì sao warmup, vì sao cosine, và tính tay từng epoch

**(a) Từng ký hiệu.** $e$ = chỉ số epoch (đếm từ 0); $\text{warm}=5$ = số epoch hâm
nóng; $\text{epochs}=80$ = tổng epoch; $\text{factor}(e)\in[0.01,1]$ = **hệ số nhân**
lên learning rate gốc $5\times10^{-4}$. LR thực tại epoch $e$ là
$\text{lr}(e)=5\times10^{-4}\cdot\text{factor}(e)$.

**(b) Vì sao có warmup (đoạn tuyến tính đầu).** Lúc mới khởi tạo, trọng số ngẫu nhiên
→ gradient **lớn và nhiễu**, hơn nữa các moment $m,v$ của Adam chưa "định hình". Nếu
bổ ngay LR đầy đủ, bước cập nhật đầu tiên dễ **văng model ra vùng xấu** (mất ổn định,
NaN). Warmup **tăng tuyến tính** LR từ nhỏ ($0.2$) lên đầy ($1.0$) trong $5$ epoch để
những bước đầu **nhẹ nhàng**, cho thống kê Adam kịp ổn định.

**(c) Vì sao cosine decay (đoạn sau).** Về cuối ta muốn **bước nhỏ dần** để lắng vào
đáy cực tiểu thay vì nhảy qua nhảy lại quanh nó. Cosine cho đường **giảm mượt** từ
$1.0$ về $\approx0$, không có cú tụt đột ngột (khác step-decay): dành **nhiều thời
gian ở LR cao lúc đầu** (khám phá rộng) rồi **hạ dịu ở cuối** (tinh chỉnh). Sàn
$\max(0.01,\cdot)$ giữ LR không về đúng $0$ để model vẫn nhích được ở epoch chót.

**(d) Tính TAY từng epoch.**

Pha warmup ($e<5$): $\text{factor}=(e+1)/5$.

| $e$ | $(e+1)/5$ | factor |
|---|---|---|
| 0 | $1/5$ | $\mathbf{0.20}$ |
| 1 | $2/5$ | $0.40$ |
| 2 | $3/5$ | $0.60$ |
| 3 | $4/5$ | $0.80$ |
| 4 | $5/5$ | $\mathbf{1.00}$ |

Pha cosine ($e\ge5$): $\text{factor}=\max\!\big(0.01,\ 0.5(1+\cos\frac{\pi(e-5)}{75})\big)$.

- $e=5$: $\cos(0)=1$ → $0.5(1+1)=\mathbf{1.00}$ (nối liền đỉnh warmup).
- $e=42$: $\frac{\pi(42-5)}{75}=\frac{37\pi}{75}=1.5498$ rad; $\cos(1.5498)=0.0210$
  (gần $\pi/2$) → $0.5(1+0.0210)=\mathbf{0.51}$ (đúng nửa đường → nửa LR, hợp lý).
- $e=79$: $\frac{\pi(79-5)}{75}=\frac{74\pi}{75}=3.0995$ rad; $\cos(3.0995)=-0.9991$
  → $0.5(1-0.9991)=0.00044$; sàn kích hoạt → $\max(0.01,\,0.00044)=\mathbf{0.01}$.

```
factor
 1.0 |      /\__
     |     /    \___
     |    /         \__
 0.5 |   /  (warmup)   \__  <- e=42 ~ 0.51 (giua duong)
     |  /               \__
0.01 | /                   \____ <- e=79 -> san 0.01
     +--+-------------------+---- epoch
     0  5                  79
        |cosine decay ------|
```

> **Tóm 1 câu:** LR tăng tuyến tính $0.20\to1.0$ trong $5$ epoch đầu để không "sốc"
> lúc trọng số còn ngẫu nhiên, rồi cosine hạ mượt $1.0\to0.01$ để lắng êm vào cực
> tiểu; giữa đường ($e=42$) đúng bằng nửa LR.

## 5.5 SWA (Stochastic Weight Averaging)

$$
\boxed{\ \theta^{\text{SWA}}_n=\Big(1-\frac1n\Big)\theta^{\text{SWA}}_{n-1}+\frac1n\,\theta_n
\ \Longrightarrow\ \theta^{\text{SWA}}_N=\frac1N\sum_{k=1}^{N}\theta_k\ }
$$
Trung bình **đều** các snapshot trọng số. **Lịch:** thu 1 snapshot **mỗi epoch** khi
$e\ge\text{swa\_start}$ (RoadFury: 75 epoch, start 50 → 24 snapshot; SE2: 80, start 55).

**Ví dụ:** 3 snapshot $\theta_1,\theta_2,\theta_3$ → $n{=}1$: $\theta_1$; $n{=}2$:
$\frac12\theta_1+\frac12\theta_2$; $n{=}3$: $\frac23(\cdot)+\frac13\theta_3=\frac{\theta_1+\theta_2+\theta_3}{3}$.

**Ý nghĩa.** SWA tìm cực tiểu "phẳng" → **tăng APFD** và **giảm $\sigma$** (dù đôi khi AUC giảm nhẹ).

#### Giải thích chi tiết: vì sao trung bình trọng số, và running-mean suy ra trung bình đều

**(a) Từng ký hiệu.** $\theta_k$ = **bộ trọng số** (toàn bộ tham số model) chụp lại ở
snapshot thứ $k$ (mỗi epoch một lần khi $e\ge\text{swa\_start}$); $\theta^{\text{SWA}}_n$
= trung bình chạy sau $n$ snapshot; $N$ = tổng số snapshot; $\frac1n$ = trọng số của
snapshot mới. Lưu ý đây là **trung bình trong không gian trọng số**, không phải trung
bình dự đoán (ensemble) — chỉ tốn bộ nhớ của **một** model.

**(b) Vì sao trung bình trọng số cho cực tiểu "phẳng".** Ở cuối huấn luyện với LR còn
đủ lớn, SGD **không đứng yên ở đáy** mà **lượn quanh vành** một lòng chảo loss rộng,
mỗi epoch rơi vào một điểm khác trên vành. Lấy trung bình các điểm trên vành → điểm
**gần tâm lòng chảo** hơn bất kỳ snapshot lẻ nào. Tâm của một lòng chảo **rộng &
phẳng** là nghiệm **ít nhạy với nhiễu trọng số** → khoảng cách train–test nhỏ hơn →
**tổng quát hoá tốt hơn** (Izmailov et al., 2018). Với bài toán này, hệ quả là **APFD
tăng và $\sigma$ giảm** (ổn định hơn), đôi khi đánh đổi AUC giảm nhẹ.

```
loss
   \        o   o        o = snapshot theta_k (luon tren vanh)
    \      /     \       * = trung binh SWA (gan tam)
     \    o   *   o
      \____\ /___/____   long chao rong, phang -> tong quat tot
            V
```

**(c) Vì sao công thức running-mean cho ra trung bình ĐỀU.** Ta muốn
$\theta^{\text{SWA}}_N=\frac1N\sum_k\theta_k$ nhưng **không muốn giữ hết** snapshot
trong RAM. Cập nhật đệ quy chỉ cần **một bản** trọng số. Chứng minh bằng quy nạp rằng
$\theta^{\text{SWA}}_n=\frac1n\sum_{k=1}^n\theta_k$:

- Cơ sở $n=1$: $\theta^{\text{SWA}}_1=\theta_1=\frac11\sum_{k=1}^1\theta_k$. Đúng.
- Bước quy nạp: giả sử $\theta^{\text{SWA}}_{n-1}=\frac1{n-1}\sum_{k=1}^{n-1}\theta_k$. Thay vào:
$$
\theta^{\text{SWA}}_n=\Big(1-\tfrac1n\Big)\theta^{\text{SWA}}_{n-1}+\tfrac1n\theta_n
=\frac{n-1}{n}\cdot\frac{1}{n-1}\sum_{k=1}^{n-1}\theta_k+\frac1n\theta_n
=\frac1n\sum_{k=1}^{n-1}\theta_k+\frac1n\theta_n=\frac1n\sum_{k=1}^{n}\theta_k.
$$

Vậy hệ số $(1-\frac1n)$ và $\frac1n$ **tự động** cho trọng số **đều** $\frac1N$ cho
mọi snapshot — không snapshot nào bị ưu ái.

**(d) Tính TAY 3 snapshot** $\theta_1,\theta_2,\theta_3$:

| $n$ | Cập nhật running-mean | Kết quả |
|---|---|---|
| 1 | khởi tạo $=\theta_1$ | $\theta_1$ |
| 2 | $(1-\tfrac12)\theta_1+\tfrac12\theta_2$ | $\dfrac{\theta_1+\theta_2}{2}$ |
| 3 | $(1-\tfrac13)\dfrac{\theta_1+\theta_2}{2}+\tfrac13\theta_3=\tfrac23\cdot\dfrac{\theta_1+\theta_2}{2}+\tfrac13\theta_3$ | $\dfrac{\theta_1+\theta_2+\theta_3}{3}$ |

Kiểm bước $n=3$: $\tfrac23\cdot\tfrac{\theta_1+\theta_2}{2}=\tfrac{\theta_1+\theta_2}{3}$;
cộng $\tfrac13\theta_3$ → $\tfrac{\theta_1+\theta_2+\theta_3}{3}$ (trung bình đều, đúng).

**(e) Lịch snapshot.** Thu 1 snapshot mỗi epoch khi $e\ge\text{swa\_start}$: RoadFury
huấn luyện $75$ epoch, start $50$ → khoảng $24$ snapshot; SE2RoadNet $80$ epoch, start
$55$. Vì kiến trúc dùng **LayerNorm** (Phần 4.3), **không có** running-stats kiểu
BatchNorm nên **không cần** pha `update_bn` sau khi trung bình — dùng ngay
$\theta^{\text{SWA}}$ để inference.

> **Tóm 1 câu:** SWA lấy trung bình đều các bộ trọng số cuối-huấn-luyện (running-mean
> $(1-\frac1n)\theta_{n-1}+\frac1n\theta_n$ chứng minh được bằng $\frac1N\sum\theta_k$)
> để đáp vào **tâm lòng chảo phẳng**, nhờ đó APFD nhích lên và $\sigma$ co lại.

---

# Phần 6 — Công thức lý thuyết nâng cao (slide s-13, s-27, s-28)

> Bốn công thức trong phần này là **đóng góp lý thuyết** của dự án, mỗi cái
> gắn với một experiment: FNO (Exp 01, bất biến độ phân giải), PINN (Exp 04,
> đơn điệu curvature), DiffAPFD (Exp 03, loss xếp hạng khả vi), Conformal
> (Exp 05, chặn dưới APFD). Mục tiêu ở đây là **khái niệm + trực giác + một
> ví dụ/hình dung** cho mỗi công thức, không sa vào chứng minh dài.

## 6.1 FNO — Bất biến độ phân giải (Exp 01)

Toán tử phổ **SpectralConv1d**:
$$
(\mathcal K x)=\mathcal F^{-1}\big[\,R_\theta\cdot\operatorname{trunc}_m(\mathcal F x)\,\big],\quad m=32\ \text{mode thấp nhất},\ R_\theta\in\mathbb{C}^{C_{\text{in}}\times C_{\text{out}}\times32}.
$$
Tính chất: $\big\lVert G_\theta(\rho_N x)-G_\theta(x)\big\rVert\to0$ khi $N\to\infty$
($\rho_N$ = rời rạc hoá $N$ điểm). Cắt mode → độc lập độ phân giải. **Số:** resolution
$\Delta\approx0.0012$ (đây là số của FNO, xem Phần 7).

#### Giải thích chi tiết: biến đổi Fourier, "mode", và vì sao cắt mode → bất biến độ phân giải

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Ý nghĩa | Từ đâu ra |
|---|---|---|
| $x$ | tín hiệu **một kênh** dọc con đường (ví dụ $\kappa(s)$), coi như hàm theo arc-length | 1 cột của ma trận đặc trưng $(L\times C)$ |
| $\mathcal F$ | biến đổi Fourier rời rạc (FFT) | phân tích $x$ thành tổng sóng sin/cos |
| $\mathcal F^{-1}$ | biến đổi Fourier ngược | ghép các sóng lại thành tín hiệu |
| mode $k$ | thành phần tần số thứ $k$: sóng lặp $k$ lần dọc road | hệ số thứ $k$ của $\mathcal F x$ |
| $\operatorname{trunc}_m$ | chỉ giữ $m{=}32$ mode **thấp nhất**, vứt hết mode cao | phép cắt phổ |
| $R_\theta$ | trọng số **phức** học được, mỗi mode một ma trận $C_{\text{in}}\times C_{\text{out}}$ trộn kênh | tham số duy nhất của lớp |
| $\rho_N$ | phép rời rạc hoá con đường thành $N$ điểm | resample road về lưới $N$ |
| $G_\theta$ | cả mô hình FNO | chồng nhiều lớp $\mathcal K$ |

"**Mode**" là câu hỏi cốt lõi. Nghĩ về nó như thế này: mọi tín hiệu tuần hoàn
đều **ghép được từ các sóng cơ bản** có tần số tăng dần:

```
mode 0 :  _____________     gia tri khong doi (trung binh / DC)
mode 1 :     /----\         mot buou tron doc ca con duong (song cham nhat)
            /      \
mode 2 :   /\    /\          hai buou (nhanh gap doi)
          /  \  /  \
mode k :  /\/\/\/\/\/\       dao dong nhanh -> chi tiet vun, nhieu, aliasing
```

Mode **thấp** = hình dạng thô, mượt, quy mô lớn (con đường cong về đâu). Mode
**cao** = dao động nhanh, chi tiết vụn, phần lớn là **nhiễu và aliasing** do lấy
mẫu.

**(2) Vì sao công thức như vậy — vì sao CẮT mode → bất biến độ phân giải.**

- Trong không gian Fourier, **tích chập trở thành phép nhân**: học $R_\theta$ trên
  từng mode chính là học một kernel tích chập **toàn cục** mà không phải trượt cửa
  sổ. Đó là lý do FNO nhân $\mathcal F x$ với $R_\theta$ rồi biến đổi ngược.
- Chìa khoá bất biến: **nội dung tần số thấp của một đường cong mượt gần như không
  phụ thuộc lấy mẫu bao nhiêu điểm**. Lấy 64 điểm hay 197 điểm, mode 0..31 gần như
  **y hệt**; chỉ mode cao (≥32) mới đổi (thêm/bớt chi tiết vụn). Cắt bỏ mode cao
  bằng $\operatorname{trunc}_{32}$ → toán tử chỉ còn "nhìn thấy" phần bất biến theo
  độ phân giải → $G_\theta(\rho_N x)\approx G_\theta(x)$ với mọi $N$.
- Cắt mode cao còn là **regularization**: bớt tham số, bỏ nhiễu tần số cao vốn
  không ổn định giữa các độ phân giải.

**(3) Ví dụ tính tay — "mode" là gì cho cụ thể.** Lấy một tín hiệu 4 mẫu đơn giản
(ví dụ phụ để thấy mode) $x=[1,1,-1,-1]$ (một bậc thang). DFT
$X_k=\sum_{n=0}^{3}x_n e^{-2\pi i kn/4}$:

$$
X_0=1+1-1-1=0\quad(\text{mode 0} = \text{trung bình} = 0),
$$
$$
X_1=1-i+1-i=2-2i\ \Rightarrow\ |X_1|=\sqrt{8}=2.828\quad(\text{mode 1 chiếm hết năng lượng}),
$$
$$
X_2=1-1-1+1=0,\qquad X_3=2+2i\ (\text{liên hợp của } X_1).
$$

→ Một tín hiệu "một cú chuyển" tái tạo được **chỉ bằng mode 1** ($X_2{=}0$). Con
đường ví dụ (một cú bẻ trái $45^\circ$ mượt) cũng vậy: gần như toàn bộ năng lượng
nằm ở mode 0 và mode 1; resample $5\to197$ điểm chỉ thêm mode cao ≈ 0 → đầu ra FNO
đổi cỡ $\Delta\approx0.0012$ (số của Exp 01).

**(4)** Tóm 1 câu: FNO học kernel **trong không gian tần số** và chỉ giữ 32 mode
thấp (phần "hình dạng thô" bất biến theo lấy mẫu), nên đổi độ phân giải $64\to197$
gần như không đổi đầu ra ($\Delta\approx0.0012$).

## 6.2 PINN — Đơn điệu theo curvature (Exp 04)

Động cơ vật lý: lực hướng tâm $v^2\lvert\kappa\rvert>\mu g$ → FAIL. Ràng buộc đơn điệu
(khuếch đại curvature không được **giảm** xác suất FAIL):
$$
\mathcal L_{\text{phys}}=\frac1{|A|}\sum_{\alpha\in A}\mathbb E_x\Big[\max\big(0,\ \sigma(s(x))-\sigma(s(x_\alpha))\big)^2\Big],\quad
x_\alpha=\text{amplify}(x,\alpha),\ A=\{1.25,1.5\};
$$
$$
\mathcal L_{\text{sob}}=\mathbb E\big[(\sigma(s(x))-\sigma(s(x+\varepsilon)))^2\big],\ \varepsilon\sim\mathcal N(0,10^{-2})\ \text{trên kênh curvature}.
$$
Tổng: $\mathcal L=\mathcal L_{\text{data}}+\text{ramp}\cdot(\lambda_{\text{phys}}\mathcal L_{\text{phys}}+\lambda_{\text{sob}}\mathcal L_{\text{sob}})$,
$\lambda_{\text{phys}}{=}0.5,\lambda_{\text{sob}}{=}0.1$. **Kết quả:** violation rate
$17.57\%\to3.14\%$ ($\alpha{=}1.5$) — **giảm 5.6×** ở cùng APFD.

#### Giải thích chi tiết: suy luận lực hướng tâm, phạt đơn điệu, và Sobolev

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Ý nghĩa |
|---|---|
| $s(x)$ | **logit** model xuất cho input $x$ (đây là hàm scorer, KHÔNG phải arc-length $s$ — trùng ký hiệu) |
| $\sigma(s(x))$ | xác suất FAIL (đưa logit qua sigmoid) |
| $x_\alpha=\operatorname{amplify}(x,\alpha)$ | bản của $x$ với **kênh curvature nhân $\alpha$ lần** (làm cua gắt hơn) |
| $A=\{1.25,1.5\}$ | hai hệ số khuếch đại curvature dùng để kiểm tra đơn điệu |
| $\max(0,\ \cdot)^2$ | chỉ phạt khi vi phạm, phạt theo bình phương |
| $\varepsilon\sim\mathcal N(0,10^{-2})$ | nhiễu nhỏ thêm vào kênh curvature (cho $\mathcal L_{\text{sob}}$) |
| ramp | lịch tăng dần trọng số vật lý theo epoch (warmup cho ràng buộc) |
| $\lambda_{\text{phys}}{=}0.5,\ \lambda_{\text{sob}}{=}0.1$ | trọng số hai số hạng vật lý |

**(2) Vì sao công thức như vậy.**

- **Suy luận lực hướng tâm.** Xe khối lượng $M$, tốc độ $v$, vào cua bán kính
  $R=1/|\kappa|$. Gia tốc hướng tâm cần có: $a=\dfrac{v^2}{R}=v^2|\kappa|$. Ma sát
  giữ xe tối đa cung cấp $a_{\max}=\mu g$. Điều kiện giữ được lane:
  $$
  v^2|\kappa|\le \mu g\quad\Rightarrow\quad v^2|\kappa|>\mu g\ \Rightarrow\ \text{trượt khỏi lane (FAIL)}.
  $$
  Hệ quả trực giác: **giữ nguyên $v$, tăng $|\kappa|$ (cua gắt hơn) thì càng dễ vượt
  ngưỡng** → P(FAIL) **không được giảm**. Đó chính là "đơn điệu theo curvature".
- **Vì sao dạng $\max(0,\ \sigma(s(x))-\sigma(s(x_\alpha)))^2$.** Ta muốn
  $p(x_\alpha)\ge p(x)$ (bản cua-gắt-hơn phải FAIL ít nhất bằng bản gốc). Viết lại:
  vi phạm khi $p(x)-p(x_\alpha)>0$. Hàm $\max(0,\cdot)$ **chỉ phạt đúng chiều sai**
  (chiều đúng $p(x_\alpha)\ge p(x)$ cho $\max(0,\text{âm}){=}0$, miễn phí). Bình
  phương → gradient mượt và phạt vi phạm lớn nặng hơn. Đây là "hinge một phía, bình
  phương" — cùng ý tưởng với $(1-p_t)^\gamma$ ở focal: nhắm đúng phần cần sửa.
- **Sobolev là gì.** Chuẩn Sobolev đo một hàm **cùng với đạo hàm** của nó; phạt
  Sobolev = phạt đạo hàm lớn → **ép hàm mượt (Lipschitz)**. Ở đây
  $\mathcal L_{\text{sob}}$ phạt việc đầu ra đổi mạnh khi curvature bị nhiễu nhẹ
  $\varepsilon$ → mặt quyết định **biến thiên trơn theo curvature**, bổ trợ cho đơn
  điệu (vừa *không giảm* vừa *không giật cục*).

**(3) Tính tay trên con đường ví dụ.** Kênh curvature $\kappa=[0,0,0.6325,0,0]$.
Amplify $\alpha{=}1.5$: nhân kênh curvature $\to$
$$
\kappa_\alpha=1.5\times[0,0,0.6325,0,0]=[0,\,0,\,0.9487,\,0,\,0]\quad(0.6325\times1.5=0.9487).
$$
Đưa hai bản qua model. Giả sử (số minh hoạ) model cho $p(x)=\sigma(s(x))=0.60$:

| Trường hợp | $p(x_\alpha)$ | $p(x)-p(x_\alpha)$ | $\max(0,\cdot)^2$ | Nhận xét |
|---|---|---|---|---|
| Vi phạm đơn điệu | 0.55 | $+0.05$ | $0.05^2=0.0025$ | cua gắt hơn mà FAIL **giảm** → bị phạt |
| Thoả đơn điệu | 0.68 | $-0.08$ | $\max(0,-0.08)^2=0$ | cua gắt hơn FAIL **tăng** → miễn phí |

Ngưỡng vật lý minh hoạ: $\mu{=}0.8,\ g{=}9.81\Rightarrow\mu g=7.85$; ở $v{=}10$ m/s,
$\kappa^{*}=\mu g/v^2=7.85/100=0.0785$ ($R^{*}\approx12.7$ m) — cua gắt hơn mức này ở
tốc độ đó thì trượt. **Kết quả huấn luyện thật:** violation rate $17.57\%\to3.14\%$
($\alpha{=}1.5$), giảm $17.57/3.14\approx\mathbf{5.6\times}$, **ở cùng APFD**.

**(4)** Tóm 1 câu: vật lý bảo "cua gắt hơn không được làm FAIL ít đi"
($v^2|\kappa|>\mu g$); $\mathcal L_{\text{phys}}$ phạt một-phía đúng khi model vi
phạm điều đó, $\mathcal L_{\text{sob}}$ ép mặt quyết định mượt theo curvature → tỉ
lệ vi phạm giảm 5.6× mà APFD không đổi.

## 6.3 DiffAPFD — Loss xếp hạng khả vi (Exp 03)

**Plackett–Luce (pairwise):**
$$
\mathcal L_{\text{PL}}=\operatorname{mean}_{i\in\text{FAIL},\,j\in\text{PASS}}\big[-\log\sigma(s_i-s_j)\big].
$$
**NeuralSort (Grover 2019):** hoán vị mềm $\hat P=\operatorname{softmax}\!\big((B\,s^\top-C)/\tau\big)$
với $B_k=n{+}1{-}2k$, $C=A\mathbf 1$, $A_{ij}=|s_i-s_j|$; rank mềm $r=\hat P[1..n]^\top$;
$\widehat{\operatorname{APFD}}_\tau=1-\frac{\sum_i r_i y_i}{nm}+\frac1{2n}$; loss $=-\widehat{\operatorname{APFD}}_\tau$
($\tau\to0$ → APFD chính xác). **Tổng:** $w_{\text{PL}}\mathcal L_{\text{PL}}+w_{\text{NS}}(-\widehat{\operatorname{APFD}})+w_{\text{BCE}}\mathcal L_{\text{BCE}}$.

**Lưu ý (Phần 7):** listwise **không tăng** mean APFD, chỉ **giảm $\sigma$**.

#### Giải thích chi tiết: vì sao APFD không khả vi, và cách làm mềm (PL & NeuralSort)

**(1) Vì sao APFD KHÔNG khả vi — cần công thức thay thế.** APFD phụ thuộc các
**rank** $TF_i$, mà rank đến từ **sort/argsort** các score. Sort là **hàm bậc
thang** của score: nhích score một FAIL lên từ từ thì rank của nó **đứng yên** cho
tới khi vượt qua score một test khác, rồi **nhảy nguyên một nấc**. Nên APFD nhìn
như hàm của score là **hằng-từng-khúc**: đạo hàm $=0$ gần khắp nơi và **vô định**
ngay chỗ nhảy → **không backprop được**.

```
APFD
 |            ______________
 |           |
 |      _____|
 |     |
 |_____|
 +--------------------------> score cua mot FAIL tang dan
   (bac thang: rank nhay tung nac -> grad = 0 hoac vo dinh)
```

Vì thế cần một **thay thế khả vi** (surrogate) trơn để có gradient.

**(2) Plackett–Luce làm mềm ra sao.**

| Ký hiệu | Ý nghĩa |
|---|---|
| $s_i$ | score (logit) của test $i$ |
| $i\in\text{FAIL},\ j\in\text{PASS}$ | duyệt mọi **cặp** FAIL–PASS |
| $\sigma(s_i-s_j)$ | xác suất "FAIL $i$ xếp trên PASS $j$" (sigmoid của **hiệu** score) |
| $-\log\sigma(\cdot)$ | phạt khi cặp bị xếp sai (softplus) |

Ý tưởng: đếm số cặp FAIL–PASS **đúng thứ tự** chính là **AUC** (Phần 1.5) — nhưng
"đếm" là bậc thang. PL thay "đếm cứng" bằng **xác suất mềm** $\sigma(s_i-s_j)$ khả
vi. Đẩy $\mathcal L_{\text{PL}}$ xuống → đẩy mọi FAIL lên trên mọi PASS → tăng AUC
→ (qua đẳng thức) tăng APFD.

**Tính tay một số hạng** (dùng $\sigma$ đã verify):

| Margin $s_i-s_j$ | $\sigma$ | $-\log\sigma$ | Ý nghĩa |
|---|---|---|---|
| $+1$ (FAIL trên PASS) | $0.7311$ | $-\ln0.7311=\mathbf{0.3133}$ | xếp đúng → phạt nhẹ |
| $0$ (hoà) | $0.5000$ | $-\ln0.5=\mathbf{0.6931}$ | mập mờ → phạt vừa |
| $-1$ (FAIL dưới PASS) | $0.2689$ | $-\ln0.2689=\mathbf{1.3133}$ | xếp sai → phạt nặng |

Đường phạt trơn, giảm dần theo margin → có gradient đẩy FAIL vượt PASS.

**(3) NeuralSort làm mềm phép sort ra sao.**

| Ký hiệu | Ý nghĩa |
|---|---|
| $A_{ij}=|s_i-s_j|$ | ma trận khoảng cách score |
| $C=A\mathbf 1$ | tổng theo hàng của $A$ (một vector) |
| $B_k=n{+}1{-}2k$ | hệ số cho **hàng thứ $k$** của ma trận hoán vị (hạng $k$) |
| $\tau$ | nhiệt độ; $\tau$ lớn → mờ (khả vi), $\tau\to0$ → sắc (sort thật) |
| $\hat P$ | ma trận hoán vị **mềm** (mỗi hàng là softmax, tổng $=1$) |
| $r=\hat P[1..n]^\top y$ | rank mềm, thế vào công thức APFD |

Thay `argsort` cứng bằng **ma trận gán mềm** điều khiển bởi $\tau$: cao thì nhoè
(khả vi), hạ $\tau\to0$ thì sắc lại thành sort thật, khi đó $\widehat{\operatorname{APFD}}_\tau$
→ APFD chính xác. Nhờ đó **cả APFD trở nên khả vi** đầu-cuối.

**Tính tay một hàng NeuralSort** (ví dụ phụ 3 test, $s=[2,1,0]$, $\tau{=}1$):

$$
C_j=\sum_i|s_i-s_j|:\quad C=\big[\,(0{+}1{+}2),\ (1{+}0{+}1),\ (2{+}1{+}0)\,\big]=[3,\,2,\,3].
$$
Hàng hạng-1 (chọn lớn nhất), hệ số $B_1=n{+}1{-}2=3{+}1{-}2=2$:
$$
\text{logit}_j=B_1 s_j-C_j=[\,2\cdot2-3,\ 2\cdot1-2,\ 2\cdot0-3\,]=[\,1,\,0,\,-3\,],
$$
$$
\hat P[1,:]=\operatorname{softmax}([1,0,-3])=\frac{[e^1,e^0,e^{-3}]}{e^1+e^0+e^{-3}}=\frac{[2.718,\,1,\,0.050]}{3.768}=[\,\mathbf{0.721},\,0.265,\,0.013\,].
$$
Hàng 1 dồn $0.721$ vào phần tử score-$2$ → **nhận đúng phần tử lớn nhất**; hạ
$\tau\to0$ thì $\to[1,0,0]$ (sort cứng).

**(4)** Tóm 1 câu: APFD "bậc thang" vì đi qua argsort nên đạo hàm bằng 0; PL làm
mềm **theo cặp** (thành AUC mềm) và NeuralSort làm mềm **cả phép sort** (nhiệt độ
$\tau$), cho gradient — nhưng thực nghiệm chỉ **giảm $\sigma$**, không tăng mean APFD.

## 6.4 Conformal — Chặn dưới APFD (Exp 05, v1/v2/v3)

Điểm non-conformity $e_i=-y_i\cdot\text{logit}_i$; quantile
$\hat q=\lceil(n+1)(1-\alpha)\rceil/n$. Số FAIL bảo đảm trong top-$K$: $r=\max(0,\,m+K-n)$.
$$
\operatorname{APFD}^{\text{LB}}_K=1-\frac{\sum_{i=1}^{r} i}{K\,m}+\frac1{2K}
=1-\frac{r(r+1)}{2Km}+\frac1{2K}.
$$
**Định lý:** $P\big(\text{prefix-APFD}(\hat\pi,X_{1..K})\ge L_\alpha\big)\ge 1-\alpha$ (dưới exchangeability).
- **v1**: valid nhưng **vacuous** (LB $=0$).
- **v2 (CRC)**: informative nhưng **invalid** (coverage $0$).
- **v3 (abstention)**: điểm $c_t=|s_t-0.5|$, bỏ phiếu nếu $c_t<\tau_\alpha$ → mục tiêu vừa valid vừa non-vacuous.

#### Giải thích chi tiết: exchangeability, quantile bảo đảm, suy ra chặn dưới, v1/v2/v3

**(1) Từng ký hiệu là gì / ở đâu ra.**

| Ký hiệu | Ý nghĩa |
|---|---|
| $e_i=-y_i\cdot\text{logit}_i$ | điểm **non-conformity**: FAIL ($y{=}1$) mà logit cao (tự tin FAIL) → $e_i$ rất âm (hợp); logit thấp → $e_i$ lớn (lệch/bất thường) |
| $\alpha$ | mức rủi ro mong muốn (ví dụ $0.1$ = chấp nhận sai $10\%$) |
| $\hat q=\lceil(n+1)(1-\alpha)\rceil/n$ | **hạng quantile**: lấy score bất thường lớn thứ $\lceil(n+1)(1-\alpha)\rceil$ làm ngưỡng |
| $n,m,K$ | $n$ = tổng test, $m$ = số FAIL, $K$ = độ dài prefix (top-$K$ chạy trước) |
| $r=\max(0,m+K-n)$ | số FAIL **bảo đảm** rơi vào top-$K$ (pigeonhole) |
| $L_\alpha$ | cận dưới hợp lệ với xác suất $\ge 1-\alpha$ |

**(2) Vì sao công thức như vậy.**

- **Exchangeability là gì.** Phân phối chung của (điểm calibration + điểm test)
  **bất biến khi hoán vị thứ tự** — tráo chỗ chúng cho nhau không đổi xác suất.
  Đây là điều kiện **yếu hơn i.i.d.** nhưng vẫn đủ. Hình dung: bỏ $n$ score
  calibration và $1$ score test vào chung một cỗ bài rồi xáo — mọi thứ tự đều **đồng
  khả năng**, nên **hạng của điểm test là ĐỀU** trên $\{1,\dots,n+1\}$.
- **Vì sao quantile đó bảo đảm coverage.** Vì hạng test đều, xác suất score test
  **không vượt** score bất thường lớn thứ $\lceil(n+1)(1-\alpha)\rceil$ là
  $\ge 1-\alpha$. Số hạng $(n+1)$ (thay vì $n$) là **hiệu chỉnh mẫu hữu hạn** để bất
  đẳng thức đúng chính xác, không chỉ tiệm cận.
- **Vì sao $r=\max(0,m+K-n)$ (pigeonhole).** Chọn $K$ trong $n$ test; đối thủ xấu
  nhất nhét hết FAIL ra sau. Nhưng chỉ có $n-K$ chỗ "ngoài top-$K$"; nếu
  $m>n-K$ thì **dư $m-(n-K)=m+K-n$ FAIL buộc phải lọt vào top-$K$**. Nếu
  $m\le n-K$ thì có thể nhét hết ra sau → bảo đảm $0$.
- **Vì sao chặn dưới có dạng đó.** Thế $r$ FAIL được bảo đảm vào công thức
  prefix-APFD (dạng như Phần 1.2 nhưng chuẩn hoá theo $K$): $\sum_{i=1}^r i=r(r+1)/2$.

**(3) Tính tay.**

*Quantile* (chọn $n{=}20,\ \alpha{=}0.1$): $\lceil(20{+}1)(1-0.1)\rceil=\lceil21\cdot0.9\rceil=\lceil18.9\rceil=19$
→ ngưỡng = score bất thường **lớn thứ 19 trong 20** ($\hat q=19/20=0.95$), coverage
$\ge 1-\alpha=90\%$. (Nếu $n$ quá nhỏ so với $1/\alpha$, ví dụ $n{=}5,\alpha{=}0.1$
cho $\lceil6\cdot0.9\rceil=6>5$ → **không có ngưỡng hữu hạn** → chính là kiểu vacuous.)

*Số FAIL bảo đảm* $r=\max(0,m+K-n)$ với $n{=}5,\ m{=}2$:

| $K$ | $m+K-n$ | $r$ | Ghi chú |
|---|---|---|---|
| 1 | $-2$ | 0 | prefix ngắn → không bảo đảm FAIL nào |
| 2 | $-1$ | 0 | vacuous |
| 3 | $0$ | 0 | vacuous |
| 4 | $+1$ | 1 | bắt đầu bảo đảm |
| 5 | $+2$ | 2 | cả 2 FAIL chắc chắn trong top-5 |

*Chặn dưới* tại $K{=}5,\ r{=}2$ ($n{=}5,m{=}2$):
$$
\operatorname{APFD}^{\text{LB}}_5=1-\frac{r(r+1)}{2Km}+\frac1{2K}=1-\frac{2\cdot3}{2\cdot5\cdot2}+\frac1{10}=1-\frac{6}{20}+0.1=\mathbf{0.80},
$$
đúng bằng APFD "xếp tốt" ở Phần 1.2. Ngược lại, muốn prefix ngắn (K nhỏ) để phát
hiện **sớm** thì $r{=}0$ → cận dưới rỗng → đó là gốc rễ **v1 vacuous**.

**(4) Phân biệt v1 / v2 / v3.**

| Bản | Cơ chế | Valid? | Non-vacuous? | Vì sao |
|---|---|---|---|---|
| **v1** | split-conformal chuẩn | Có | **Không** (LB $=0$) | prefix ngắn / non-conformity → $r{=}0$, ngưỡng vô hạn |
| **v2** | CRC (Conformal Risk Control) | **Không** (coverage $0$) | Có | chỉnh để LB dương nhưng giả thiết risk-control bị vi phạm |
| **v3** | abstention theo độ tin | mục tiêu: Có | mục tiêu: Có | dùng $c_t=|s_t-0.5|$, **bỏ phiếu** nếu $c_t<\tau_\alpha$ (quá mập mờ), chỉ chứng nhận ca chắc chắn |

$c_t=|s_t-0.5|$ = **khoảng cách xác suất tới ranh giới quyết định $0.5$**: gần $0$ =
mập mờ, lớn = tự tin. v3 chỉ "bỏ phiếu" khi đủ tự tin → nhắm **vừa valid vừa
non-vacuous** (đang hoàn thiện, xem Phần 7).

**(5)** Tóm 1 câu: nhờ **exchangeability** (hạng điểm test là đều), quantile
$\lceil(n+1)(1-\alpha)\rceil$ cho **coverage $\ge 1-\alpha$**; pigeonhole bảo đảm
$r=\max(0,m+K-n)$ FAIL trong top-$K$ → chặn dưới APFD; v1 valid-mà-rỗng, v2
thông-tin-mà-sai, v3 dùng abstention $c_t=|s_t-0.5|$ để nhắm cả hai.

---

# Phần 7 — Bẫy câu hỏi & Đối chiếu số liệu (đọc kỹ trước khi lên bảng)

> Đây là những chỗ số liệu **vênh nhau giữa các slide/script/code** mà thầy dễ vặn.
> Mỗi mục kèm **câu trả lời an toàn**.

### 7.1 Focal $\gamma$: 1.0 vs 1.5 vs 2.5
- Slide s-20 (bản deck hiện tại): $\gamma=1.0$ → APFD $0.8095\pm0.0121$ (cao nhất trong sweep $\gamma\in[1,5]$).
- Script cũ / một số số leaderboard: $\gamma=1.5$ → APFD $0.8047$.
- RoadFury canonical (SensoDat): $\gamma=2.5$ → APFD $0.8066\pm0.0124$.
- **Trả lời:** "Sweep $\gamma\in[1,5]$ cho APFD $0.807$–$0.810$, **chồng nhau trong $\pm\sigma$**; chọn $\gamma{=}1.0$ vì cao nhất và robust. Các con số cũ $\gamma{=}1.5$ là run trước, vẫn trong sai số."

### 7.2 Menger $\kappa=0.633$ vs signed $k=0.651$
- **Trả lời:** hai công thức curvature **khác nhau**. RoadFury dùng **Menger** (bán kính đường tròn ngoại tiếp, **bỏ dấu**) → $0.6325$. SE2RoadNet dùng **sai phân góc** $k=\Delta\theta/\Delta s$ (**có dấu**) → $0.6506$. Khác nhau là **đúng như kỳ vọng**, không phải lỗi.

### 7.3 Resolution $\Delta=0.0012$ là của FNO, KHÔNG phải SE2RoadNet
- Slide s-27, s-34 ghi $\Delta_{\text{res}}=0.0012$. Số này đo trên **mô hình FNO (Exp 01)**.
- **Trả lời:** "Bất biến **xoay** của SE2RoadNet là **exact** ($\Delta{=}0$). Bất biến **độ phân giải** hiện là **xấp xỉ**; con số $0.0012$ là từ nhánh FNO. Hướng phát triển: RoPE-1D trên $s_{\text{norm}}$ để đạt resolution-$\Delta=0$."

### 7.4 APFD 0.804 vs 0.8066; SE2RoadNet 0.805 KHÔNG "beat" RoadFury
- **Trả lời:** con số "$0.804$" trên slide là bản trình bày; số canonical để so là **$0.8066\pm0.0124$**. SE2RoadNet $0.8048$ **thấp hơn $0.0018$** (tie trong $\sigma$). **Đóng góp là "Pareto"**: cùng APFD nhưng **thêm bảo chứng $\Delta{=}0$** và **AUC cao hơn** ($0.917\to0.934$) — KHÔNG bán như "thắng APFD".

### 7.5 Kênh f7: s-12 vs s-38 (vênh nội bộ slide)
- s-12 ghi f7 = "hướng tuyệt đối $\theta_i$" và nói 3 kênh f5/f6/f7 phụ thuộc khung.
- s-38 (phụ lục, **khớp code**) ghi f7 = $\text{rel\_pos}=i/(n-1)$ và chỉ f5/f6 phụ thuộc khung.
- **Trả lời (dùng bản code):** "Code thực (`extract_sequence_10ch`) có f5$=\sin\theta$, f6$=\cos\theta$, **f7$=$rel\_pos $i/(n{-}1)$**; **không có** kênh $\theta$ thô. Ba kênh phụ thuộc khung thực chất là **f5, f6 và PE tuyệt đối**. Nhãn 'θ' trên s-12 là viết gọn nhầm."

### 7.6 Ký hiệu $R$ bị overload trong $f_\theta(R\,\mathcal R+t)$
- **Trả lời:** $R$ ở đây là **ma trận xoay**, còn $\mathcal R$ là **con đường**. Bản sạch trên s-36 viết $F(R\cdot\mathcal R+t)=F(\mathcal R)$.

### 7.7 rel_pos (chỉ số) vs $s/L$ (arc-length)
- **Trả lời:** f7 chia đều **theo chỉ số điểm**; f4/c6 chia **theo độ dài cung**. Chỉ $s/L$ mới vào attention bias $\Delta s$; rel_pos không.

### 7.8 "AUC tăng nhưng APFD phẳng"
- **Trả lời:** dùng đẳng thức $\operatorname{APFD}=(1-p)\operatorname{AUC}+p/2$ (Phần 1.5). AUC đo ở prior $p$ này, APFD ở prior khác → chênh là **hiệu ứng prior**, không mâu thuẫn.

### 7.9 Rotation $\Delta$ "=0" nhưng có residual float $1.79\times10^{-7}$
- **Trả lời:** bất biến là **exact về mặt toán**; residual $1.79\times10^{-7}$ ở mức logit chỉ do ma trận xoay $R$ chứa $\sin\theta,\cos\theta$ (roundoff float32), **không** đổi ranking → APFD bằng nhau đến từng bit.

---

# Phần 8 — Bảng tra nhanh (cheat sheet)

| Tên | Công thức |
|---|---|
| APFD | $1-\frac{\sum TF_i}{n m}+\frac{1}{2n}$ |
| AUC–APFD | $\operatorname{APFD}=(1-p)\operatorname{AUC}+\frac p2,\ p=\frac mn$ |
| seg | $\lVert p_{i+1}-p_i\rVert$ |
| $\lvert\Delta\theta\rvert$ | $\lvert\operatorname{wrap}(\theta_{i+1}-\theta_i)\rvert,\ \theta=\operatorname{atan2}(\Delta y,\Delta x)$ |
| Menger $\kappa$ | $\frac{4\,\text{Area}}{abc}=\frac1R$, Area $=\sqrt{s(s{-}a)(s{-}b)(s{-}c)}$ |
| signed $k$ | $\Delta\theta/\big(\tfrac12(\text{seg}_i+\text{seg}_{i+1})\big)$ |
| $s/L$ | $\big(\sum_{j\le i}\text{seg}_j\big)/L$ |
| z-norm | $(x-\mu)/\sigma$ |
| Linear | $y=xW^\top+b$ |
| LayerNorm | $\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}\odot\gamma+\beta$ |
| GELU | $x\cdot\tfrac12(1+\operatorname{erf}(x/\sqrt2))$ |
| Attention | $\operatorname{softmax}\!\big(\frac{QK^\top}{\sqrt{d_k}}+B\big)V$ |
| Bias $\Delta s$ | $B_{ij}=\operatorname{MLP}(\sin((s_i-s_j)\omega)),\ \omega\in\mathbb R^{32}$ |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ |
| BCE | $-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$ |
| Focal | $\operatorname{mean}[\alpha(1-p_t)^\gamma w\,\text{BCE}]$ |
| pos_weight | $n_{\text{neg}}/n_{\text{pos}}$ |
| LR cosine | $0.5(1+\cos\frac{\pi(e-\text{warm})}{\text{epochs}-\text{warm}})$ |
| SWA | $\theta^{\text{SWA}}=\frac1N\sum_k\theta_k$ |
| SE(2) inv. | $f_\theta(R\mathcal R+t)=f_\theta(\mathcal R)$ |
| FNO | $\mathcal F^{-1}[R_\theta\operatorname{trunc}_m(\mathcal F x)]$ |
| PINN | $\operatorname{mean}[\max(0,\sigma(s(x))-\sigma(s(x_\alpha)))^2]$ |
| Plackett–Luce | $\operatorname{mean}[-\log\sigma(s_i-s_j)]$ |
| Conformal LB | $1-\frac{r(r+1)}{2Km}+\frac1{2K},\ r=\max(0,m{+}K{-}n)$ |

---

*Tài liệu tổng hợp từ: bộ 46 slide `RoadFury → SE2RoadNet`, `exps/best.md`,
`exps/exp02_walkthrough.md`, `exps/se2roadnet_forward_pass.md`, `exps/tracker.md`
và các script `exp01/02/03/04/05`. Mọi ví dụ tính tay đã được kiểm bằng chạy code
thực trên con đường 5 điểm ở Phần 0.*
