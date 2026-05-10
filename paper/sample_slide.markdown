### MEMRES & CGAR: Agentic Python Dependency Resolution

###### Trần Chí Nguyên Đào Sỹ Duy Minh Huỳnh Trung Kiệt

University of Science, VNU-HCM

ML Course Project – 2026


## Machine Learning Course Project

#### MEMRES & Constraint-Guided Agentic Resolution (CGAR)

##### Project Overview

An agentic framework for Python dependency

resolution, leveraging LLMs and Constraint

Satisfaction Problems (CSP) to optimize

environment stability and speed.

##### Thành viên nhóm

```
▶ Trần Chí Nguyên – 23102244
```
```
▶ Huỳnh Trung Kiệt – 23122039
```
```
▶ Đào Sỹ Duy Minh – 23122041
```
```
University of Science, VNU-HCM
Department of Computer Science
```

#### Nội dung chính

##### ▶ Phát biểu Bài toán

##### ▶ Nghiên cứu trước: MEMRES

##### ▶ Proposed Extension: CGAR

##### ▶ Formulation

##### ▶ Đánh giá Thực nghiệm


### Phát biểu Bài toán


#### FSE-AIWare 2026 – Bài toán Cuộc thi

##### Input

```
Một đoạn code Python cô lập từ GitHub gist cũ (không metadata, không requirements).
Code sử dụng các thư viện chưa xác định và yêu cầu một bản Python chưa biết.
```
##### Output

```
Một môi trường chạy được đoạn code:
```
```
▶ Phiên bản Python (2.7, 3.6, 3.7,... )
```
```
▶ Danh sách các package cài qua pip, với phiên bản cụ thể
```
```
▶ Tùy chọn: Thư viện hệ thống (apt-get) cho C-extensions
```
##### Success Criterion

```
Đoạn code phải chạy qua mọi lệnh import trong Docker mà không gặp ImportError, ModuleNotFoundError,
or SyntaxError.
```

#### Requirements & Metrics

##### Hard requirements

```
▶ Mọi thao tác phải diễn ra bên trong Docker
(không cài máy host)
```
```
▶ Tối đa 10 GB VRAM cho LLM
```
```
▶ LLM phải là mã nguồn mở (không GPT-4 /
Claude API)
```
```
▶ Mô hình tham chiếu: Gemma-2 9B qua Ollama
```
##### Evaluation budget

```
▶ Tối đa 10 lần thử build (retry) mỗi đoạn code
```
```
▶ 180s cho mỗi lần build
```
```
▶ Tổng timeout 500s mỗi đoạn code
```
```
▶ Đánh giá tuần tự (1 worker)
```
##### Metrics

```
▶ Tỷ lệ thành công (Tiêu chí chính)
```
```
▶ Thời gian trung bình mỗi đoạn code
```
```
▶ Số lần gọi LLM mỗi đoạn code
```

#### Tập dữ liệu – HG2.9K & GitChameleon

##### HG2.9K (In-distribution)

```
2,891 đoạn code Python cực khó (Tập dataset chính
thức).
```
```
▶ Đặc điểm: Cú pháp Python 2/3 lẫn lộn, thư
viện ML/AI lỗi thời, phụ thuộc thư viện
C/C++.
```
```
▶ Mục tiêu: Đánh giá khả năng "cứu sống" đoạn
code cũ chạy mượt mà không lỗi.
```
##### GitChameleon (Out-of-distribution)

```
328 bài toán lập trình (Tập kiểm tra tổng quát).
```
```
▶ Đặc điểm: Chứa Unit Test ẩn, giấu kín thông
tin phiên bản gốc (ground-truth).
```
```
▶ Mục tiêu: Đo lường tính đúng đắn – tool phải
suy luận chính xác phiên bản để pass test, tránh
việc cài bừa bản mới nhất.
```

#### The Dependency Gap

```
from scipy.misc import imread # removed in scipy >= 1.
import sklearn.cross_validation # removed in sklearn >= 0.
import cv2 # PyPI name: opencv -python
```
##### ▶ Tên import gây hiểu lầm: cv2 → opencv-python

##### ▶ API biến mất: scipy.misc.imread chỉ tồn tại ở scipy ≤ 1. 1

##### ▶ Hiệu ứng domino: scipy ≤ 1. 1 đòi Python ≤ 3. 7 đòi numpy cũ...

##### ▶ Không có wheel: nhiều bản cũ không có wheel cho Linux/glibc

##### Không gian tìm kiếm

##### ∼500K package PyPI × chục phiên bản × nhiều bản Python ⇒ Bùng nổ tổ hợp.


### Nghiên cứu trước: MEMRES


#### MEMRES & Những hạn chế

##### Kiến trúc MEMRES (FSE-AIWare ́26)

```
Pipeline 4 bước thay thế vòng lặp thử sai LLM ngây thơ:
```
1. Knowledge Oracle: Tái sử dụng các giải pháp đã biết
2. Hybrid Eval: Phân tích tĩnh + Import ngữ nghĩa
3. Module Clean: KB Lỗi + Bộ nhớ gợi ý
4. Confidence Cascade: 6 cấp độ chọn phiên bản (tra cứu O(1)
    trước LLM)

```
Kết quả: Pass rate 86.3%, giảm 60%+ số lần gọi LLM so với PLLM.
```
##### Hạn chế (Vì sao cần CGAR)

```
▶ 12.8% Lỗi: Thất bại do API bị xóa & wheel
source-only.
▶ Chi phí lỗi cao: Vẫn mất ∼335s/đoạn code.
▶ Không tỉa nhánh: Lỗi Docker (VD
CouldNotBuildWheels) bị xem là hộp đen thay
vì ràng buộc logic.
```

### Proposed Extension: CGAR


#### Proposed Extension: CGAR Architecture

##### Cốt lõi: Phân giải phụ thuộc là bài toán CSP

```
Biến lỗi thực nghiệm (lỗi Docker) thành các ràng buộc logic.
Tìm kiếm không gian đã tỉa nhánh bằng thuật toán backtracking thay vì LLM thử mù.
```
```
▶ Bước 1: Candidate Graph Builder
Lấy metadata PyPI. Dùng Wheel Filter loại bỏ
source-only, tránh kẹt 3 phút build.
▶ Bước 2: Constraint Solver
Duyệt DFS quay lui, bị giới hạn bởi HARD, SOFT, và
UPPER BOUND.
```
```
▶ Bước 3: Failure Injector
Phân tích lỗi Docker (VD cannot import name X) →
ub(pkg)=v. Không hardcode.
▶ Bước 4: Counterfactual Retry
Giải lại CSP tức thì. Ràng buộc lưu trong
Session-scoped store để học chéo.
```

#### What we tried that did NOT work

```
Cách thất bại Why? it failed Cách khắc phục
```
```
Hardcode bảng API-removal
(scipy: 1.2 cho imread)
```
```
Rất dễ gãy; sập mỗi khi có API mới bị
xóa
```
```
Dùng regex inject_api_removed()
(không hardcode)
Reset store ràng buộc sau mỗi
snippet
```
```
Lãng phí khả năng học chéo Session-scoped store ⇒ thêm 19.7%
```
```
Chỉ dùng ràng buộc HARD Vô tình cấm phiên bản do lỗi ngẫu
nhiên
```
```
Thêm mức SOFT (đếm ≥ 2 )
```
```
Không lọc wheel (eval ban
đầu)
```
```
Build source tốn 3 phút; rescue kẹt ở
10%
```
```
_has_linux_wheel()⇒ cứu 17.9%
(∼ 2 ×)
DFS Tham lam (không giới
hạn)
```
```
Bị kẹt ở không gian rộng > 1000 ứng
viên
```
```
Giới hạn 50 lần thử + Python pivot
```
##### Lesson

```
Each failure mode taught us a structural principle: learn from errors, don’t hardcode them.
```

#### Lan truyền Ràng buộc

```
Không gian đầu D(scipy):1.10.1 1.9.3 1.7.3 1.5.4 1.4.1 1.2.0 1.1.0 1.0.
```
```
Sau ub(scipy)=1.2.0:1.10.1 1.9.3 1.7.3 1.5.4 1.4.1 1.2.0 1.1.0 1.0.0 bị loại
khả thi
```
```
Solver chọn: 1.1.
```
##### Effect

```
One Docker error ⇒ one regex parse ⇒ one upper bound ⇒ 6 of 8 versions bị loại mà không cần thử build
thêm.
```

### Formulation


#### CGAR as a Constraint Satisfaction Problem

```
We define the resolution task as a CSP tuple P =⟨X,D,C⟩:
```
```
Variables (X): The set of required packages {P 1 ,...,Pn} and Python version π.
```
```
Domains (D):
D(Pi) ={v ∈ versions(Pi)| req_py(Pi,v)|= π∧ has_wheel(Pi,v)}
```
```
Sorted by wheel-first preference and descending semantic versioning.
```
```
Constraints (C):
```
```
▶ Hard Upper-Bounds: vi< ub(Pi) inferred from previous build failures.
```
```
▶ Combinatorial: ((Pi,vi), (Pj,vj)) /∈Ccombo(known package conflicts).
```
```
▶ Feasibility: DockerBuild(π,{vi}) = SUCCESS.
```
```
Goal: Find an assignment A ={π∗,v 1 ∗,...,v∗n} that satisfies all constraints in C within a search budget of
k = 50 backtracking attempts.
```

#### Backtracking algorithm

```
def solve(remaining , assigned , pi, store , depth =0):
if depth > MAX_ATTEMPTS: return None # 50 cap
if not remaining: return assigned
```
```
P = remaining [0]
domain = build_graph(P, pi) # Stage 1 (metadata + wheel)
domain = [v for v in domain
if v < store.upper_bound(P)
and not store.is_hard(P, v)
and store.soft_count(P, v) < 2]
```
```
for v in domain: # wheel -first
if store.violates_combo(P, v, assigned): continue
result = solve(remaining [1:], assigned | {(P, v)},
pi, store , depth + 1)
if result is not None: return result
return None
```
```
Outer loop: try π ∈ Π in detector-suggested order; on full exhaustion, fall back to MEMRES cascade.
```

### Đánh giá Thực nghiệm


#### Overall Performance & Speed Insights

##### Pass Rates (All Tools)

```
Tool HG2.9K GitCham.
```
```
PLLM (FSE’25) 44.8% 65.5%
GPT-4.1 + RAG – 58.5%
o1 (đóng) – 51.2%
MEMRES (nhóm) 86.3% 81.7%
CGAR 87.1% 83.2%
```
##### Điểm nhấn

```
CGAR tăng độ chính xác (pass rate cao nhất) đồng
thời tăng tốc khủng khiếp nhờ việc tránh build
Docker vô ích.
```
##### Speed Insights

##### PLLMMEMRES

##### 0

##### 200

##### 400 369.^6335. 3

##### 22. 3

###### Avg seconds/snippet

```
Fail / pass time ratio: PLLM ( 2. 20 ×), MEMRES
( 1. 12 ×), CGAR ( 1. 31 ×).
```

#### Errors Elimination & Ablation Study

##### 4 Categories Eliminated

```
CGAR giảm các lỗi sau của MEMRES xuống 0 :
```
```
▶ SyntaxError (nhận diện Py2)
▶ NoMatchingDist (kéo từ PyPI)
▶ CouldNotBuild (lọc wheel)
▶ AttributeError (ràng buộc)
```
##### The Hard Floor

```
10.7% lỗi còn lại là do thiếu thư viện OS, Python 2 quá cũ,
hoặc package không hề tồn tại trên PyPI. Đây là giới hạn
nền tảng.
```
##### Ablation Study (CGAR)

```
Configuration Rescued Impact
```
```
Full CGAR 71 –
w/o wheel filter 40 ↓ 43.7%
w/o upper bound 23 ↓ 67.6%
w/o session store 56 ↓ 21.1%
```
```
Điểm nhấn: Upper bound infer-
ence đóng vai trò cốt lõi nhất.
```

## Thank you!

# Thankyou!

##### Questions & Discussion

###### github.com/chisngyen/fse-aiware-python-dependencies


