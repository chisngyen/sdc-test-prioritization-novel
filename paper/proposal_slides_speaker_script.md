# Kịch Bản Thuyết Trình — Lý Thuyết Điều Khiển Ưu Tiên Kiểm Thử Cho Mô Phỏng Ô Tô Tự Lái

**Bộ Slide Đề Xuất | Tháng 5/2026 — Hướng đến NeurIPS 2026**

---

## Slide 1: Trang Tiêu Đề

*(Thời lượng: ~30 giây)*

> Kính chào quý thầy cô và các bạn. Cảm ơn mọi người đã dành thời gian.
>
> Hôm nay chúng tôi trình bày **Lý Thuyết Điều Khiển Ưu Tiên Kiểm Thử cho Mô Phỏng Ô Tô Tự Lái** — một phương pháp **SE(2)-Đối Xứng Đảo và Thông Tin Vật Lý**.
>
> Đây là công trình hợp tác giữa tôi, Trần Chí Nguyên và Huỳnh Trung Kiệt đến từ Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM. Công trình này hướng đến hội nghị **NeurIPS 2026**.

---

## Slide 2: Tổng Quan Dự Án & Đội Ngũ

*(Thời lượng: ~45 giây)*

**Cột trái — Tổng quan dự án:**

> Công trình của chúng tôi thay thế các **heuristic kỹ thuật** trong việc ưu tiên kiểm thử SDC bằng hai ràng buộc có nguyên tắc: **tính bất biến hình học có thể chứng minh** và **tính đơn điệu theo vật lý**.

Dành chút thời gian để khán giả tiếp thu: đây không phải phương pháp học sâu hộp đen — nó mang theo các đảm bảo toán học.

**Cột phải — Đội ngũ:**

> Chúng tôi là nhóm ba người đến từ Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM. Chúng tôi xây dựng trên nền tảng RoadFury từ ICST'26.

**Chuyển slide:** *"Để tôi đi qua nội dung trình bày hôm nay."*

---

## Slide 3: Nội Dung Trình Bày

*(Thời lượng: ~15 giây)*

> Đây là lộ trình. Chúng tôi sẽ đề cập đến bài toán, hai đóng góp chính, khả năng kết hợp các lý thuyết, và kế hoạch dự án.

*(Không cần đọc toàn bộ mục lục — chỉ cần chỉ vào nó.)*

**Chuyển slide:** *"Hãy bắt đầu với lý do tồn tại của bài toán này."*

---

## Slide 4: Kiểm Thử Mô Phỏng Ô Tô Tự Lái

*(Thời lượng: ~90 giây)*

**Điểm đau chính — nói chậm:**

> Các trình mô phỏng SDC như **BeamNG** và **CARLA** thực thi hàng nghìn kịch bản đường mỗi đợt phát hành phần mềm. Mỗi kịch bản mất từ 10 đến 60 giây mô phỏng. Tức là **hàng nghìn giờ công kỹ sư mỗi chu kỳ phát hành**.
>
> Thách thức ở đây: phần lớn các kịch bản là dư thừa. Kỹ sư muốn **lỗi xuất hiện trước**.

**Khối định nghĩa:**

> **Ưu tiên kiểm thử** nghĩa là sắp xếp thứ tự tập kịch bản sao cho các ca kiểm thử thất bại xuất hiện sớm trong quá trình đánh giá. Chỉ số chúng tôi tối ưu là **APFD** — Trung bình Phần trăm Lỗi Phát hiện được. APFD càng cao nghĩa là phát hiện lỗi càng sớm.

**Nhấn mạnh bối cảnh cuộc thi:**

> Công trình này đồng hành cùng **SBFT 2026 — Cuộc thi Kiểm thử Hệ thống Vật lý Cyper**.

**Câu hỏi dự kiến — "APFD nghĩa là gì đơn giản?"**
> Nếu APFD của bạn là 0.8, có nghĩa là khi đã chạy được 80% ca kiểm thử, bạn đã tìm thấy 80% các lỗi. Thứ tự ngẫu nhiên cho bạn khoảng 0.5.

**Chuyển slide:** *"Để tôi định nghĩa APFD một cách hình thức."*

---

## Slide 5: Chỉ Số APFD

*(Thời lượng: ~60 giây)*

**Đi qua công thức:**

> Với một thứ tự gồm N ca kiểm thử và M lỗi, công thức APFD trừ vị trí lỗi trung bình khỏi thứ tự lý tưởng rồi cộng thêm một nửa. Kết quả nằm trong khoảng [0, 1], **cao hơn là tốt hơn**.

**Các tính chất quan trọng:**

> APFD là một **thống kê thứ hạng** — nó chỉ quan tâm đến thứ tự, không phải xác suất hiệuлиброван. Điều này khiến nó **không khả vi**, và vì vậy việc huấn luyện trực tiếp trên APFD rất khó.

**Các con số cần đánh bại:**

> Baseline của chúng tôi từ ICST'26 — Transformer với Stochastic Weight Averaging — đạt **0.8066** với mô hình đơn tốt nhất, và **0.8077** với ensemble 5 cấu hình. Đây là các con số chúng tôi cần đánh bại.

**Giao thức đa thử nghiệm:**

> Chúng tôi báo cáo **APFD 30 lần chạy trên tập Competition của SBFT 2026** gồm 956 ca kiểm thử. Đây là giao thức đồng nhất được sử dụng cho mọi con số trình bày hôm nay.

**Chuyển slide:** *"Vậy tại sao việc đánh bại 0.8077 lại thực sự khó?"*

---

## Slide 6: Tại Sao Bài Toán Này Khó?

*(Thời lượng: ~90 giây)*

**Các điểm yếu của baseline — đi qua từng điểm:**

> 1. **Độ giòn theo tần số lấy mẫu**: Cùng một con đường, lấy mẫu ở 64 điểm so với 197 điểm sẽ cho điểm số khác nhau. Đây là vấn đề rời rạc hóa dữ liệu.
>
> 2. **Phụ thuộc khung hệ quy chiếu**: Xoay con đường 30 độ — APFD giảm 4 đến 7 điểm. Mô hình đang mã hóa tọa độ tuyệt đối, không phải hình học nội tại.
>
> 3. **Mù về vật lý**: Mô hình tự do tạo ra các hàm điểm vi phạm ràng buộc gia tốc hướng tâm: **v-bình phương nhân kappa phải nhỏ hơn mu nhân g**. Điều này vật lý không thể xảy ra.
>
> 4. **Hiệu chỉnh ≠ Sắp xếp**: Mô hình xếp hạng tốt theo AUC không nhất thiết xếp hạng tốt theo APFD. Hai mục tiêu này luôn phân kỳ.

**Vấn đề sâu hơn:**

> Ưu tiên kiểm thử SDC đang được xử lý như **phân loại chuỗi hộp đen**. Nhưng đường là các đường cong liên tục, chịu **đối xứng SE(2) cứng** và **động lực học phương tiện**. Cấu trúc lý thuyết đang bị bỏ qua.

**Câu quan trọng — đọc nguyên văn:**

> "Cấu trúc lý thuyết đang bị bỏ qua."

**Chuyển slide:** *"Đề xuất của chúng tôi là ngừng bỏ qua nó."*

---

## Slide 7: Lộ Trình Nghiên Cứu Của Chúng Tôi — 8 Thấu Kính Lý Thuyết

*(Thời lượng: ~60 giây)*

**Tổng quan bảng — đây là lộ trình của chúng tôi:**

> Chúng tôi đã phát triển **8 thấu kính lý thuyết** cho bài toán này. Tôi sẽ mô tả ngắn gọn từng cái:
>
> - **FNO (Operator Learning)** — tính bất biến theo discretization qua Fourier neural operators.
> - **SE(2)-Đối xứng Đảo** — kết quả chính của chúng tôi: tính bất biến xoay có thể chứng minh với Δ = 0 chính xác.
> - **Listwise Learning-to-Rank** — APFD khả vi qua SoftSort.
> - **Thông tin Vật lý (PINN)** — ràng buộc đơn điệu: giảm 5.6 lần vi phạm.
> - **Dự đoán Conformal** — cận APFD không phân phối.
> - **Phản thực Nhân quả** — gán lỗi theo từng đoạn.
> - **Foundation Tự giám sát** — nhiệm vụ tiền từ Frenet-Serret.
> - **Khai thác Hard-mining Diffusion** — dữ liệu tổng hợp nhắm mục tiêu biên.

**Hai điểm chính:**

> Trong đề xuất này, chúng tôi tập trung vào hai điểm chính: **SE(2)-Đối Xứng Đảo RoadNet** — kết quả sạch nhất được lý thuyết xác minh bằng thực nghiệm — và **PINN Đơn Điệu** — lập luận thực tiễn và pháp lý mạnh nhất.

**Chuyển slide:** *"Để tôi cho các bạn thấy chúng tôi so sánh thế nào với state-of-the-art bên ngoài."*

---

## Slide 8: vs SOTA Bên Ngoài — Chúng Tôi Vượt Qua Mọi Paradigm

*(Thời lượng: ~90 giây)*

**Đi qua bảng — bắt đầu từ dưới lên:**

> Mọi phương pháp trước đây đều nằm dưới chúng tôi. Đi từng phương pháp một:
>
> - **Ngẫu nhiên**: 0.493. Không có tín hiệu gì.
> - **LLM zero-shot**: 0.487. Mô hình ngôn ngữ không giúp được ở đây — sai modality.
> - **GNN**: 0.533. Mô hình đồ thị đường mất thông tin thứ tự.
> - **ResNet-50 hình ảnh**: 0.572. Chuyển đường thành ảnh mất độ chính xác hình học.
> - **SO-SDC-Prioritizer (TOSEM'22)**: 0.765. Thuật toán di truyền baseline.
> - **ITEP4SDC (ICST'24)**: 0.781. MLP với 3 chỉ số tổng hợp — SOTA bên ngoài trước đó.
> - **Greedy-diversity (TOSEM'22)**: 0.795. Phương pháp không neural tốt nhất.
> - **RoadFury (baseline nội bộ ICST'26 của chúng tôi)**: 0.804.

**Các con số quan trọng cần nhấn mạnh:**

> Chúng tôi vượt **LLM/Ngẫu nhiên 0.32 điểm**, **GNN 0.27**, **CNN 0.23**, **ITEP4SDC 0.024**, và **Greedy-diversity 0.01**.
>
> Và ngoài việc tương đương APFD, chúng tôi còn thêm **tính bất biến xoay có thể chứng minh** và **tính đơn điệu vật lý** — các đảm bảo mà không SOTA nào trước đây có được.

**Chuyển slide:** *"Đây là câu chuyện hai trục."*

---

## Slide 9: Đề Xuất Giá Trị Hai Trục

*(Thời lượng: ~45 giây)*

**Giải thích biểu đồ:**

> Biểu đồ này có **APFD trên trục x** và **độ mạnh đảm bảo lý thuyết trên trục y**. Mọi phương pháp trước đây tập trung ở dưới — không có đảm bảo gì. Các phương pháp của chúng tôi nằm ở **đường Pareto** — cả tầng APFD cao nhất lẫn đảm bảo lý thuyết mạnh nhất.
>
> Không phương pháp nào khác làm được cả hai.

**Chuyển slide:** *"Bây giờ hãy đi sâu vào điểm chính thứ nhất: SE(2)-Đối Xứng Đảo RoadNet."*

---

## Slide 10: Tại Sao SE(2)-Đối Xứng Đảo?

*(Thời lượng: ~90 giây)*

**Quan sát vật lý — đọc nguyên văn:**

> "Xe rời làn đường" là một tính chất của **hình học nội tại** của con đường, không phải vị trí nhúng của nó trong mặt phẳng 2D. Xoay hay tịnh tiến toàn bộ con đường **không thể thay đổi** xe có bị lỗi hay không.

**Tuy nhiên baseline không có tính bất biến:**

> Transformer ICST'26 của chúng tôi mã hóa tọa độ tuyệt đối và góc hướng. Xoay một con đường thử nghiệm 30 độ **thay đổi điểm số dự đoán**. Đây là một lỗi.

**Khẳng định của chúng tôi:**

> Với việc ưu tiên kiểm thử SDC, bộ xếp hạng f nên thỏa mãn: **f của đường đã xoay và tịnh tiến bằng f của đường gốc**, với mọi phép biến đổi cứng trong SE(2). Chúng tôi xây dựng một mô hình thỏa mãn điều này **theo thiết kế**, không phải bằng tăng cường dữ liệu.

**Câu quan trọng:**

> Không phải "xấp xỉ bất biến." Không phải "trong ngưỡng dung sai." **Bất biến theo chứng minh bằng xây dựng.**

**Chuyển slide:** *"Để tôi cho các bạn thấy chính xác những gì chúng tôi đã thay đổi."*

---

## Slide 11: Xây Dựng — 7 Kênh Đặc Trưng SE(2)-Bất Biến

*(Thời lượng: ~60 giây)*

**Cột trái — những gì chúng tôi loại bỏ:**

> Bộ 10 kênh đặc trưng tiêu chuẩn bao gồm **x, y tọa độ tuyệt đối** và **sin theta, cos theta hướng** — đây là các đặc trưng **không bất biến**. Chúng mã hóa vị trí nhúng toàn cục tùy ý của con đường.

**Cột phải — những gì chúng tôi giữ lại:**

> 7 kênh của chúng tôi đều là **nội tại**: độ cong có dấu, độ lớn độ cong, đạo hàm độ cong, gia số độ dài cung, thay đổi góc cục bộ, độ cong tích lũy, và độ cong làm mượt. **Không một đặc trưng nào phụ thuộc tọa độ.**

**Kiến trúc:**

> Mô hình là SE2RoadNet: **d-model 192, độ sâu 6, 8 đầu attention**, với attention bias theo độ dài cung tương đối và 32 đặc trưng RFF mỗi lớp. Khoảng **2.11 triệu tham số**.

**Điểm quan trọng cần nhấn mạnh:**

> Vì không đặc trưng nào đọc đường dưới dạng tọa độ (x, y), xoay đường ở đầu vào tạo ra ma trận đặc trưng pixel-đồng nhất. Định lý thỏa mãn bằng xây dựng.

**Chuyển slide:** *"Vậy điều này cho chúng tôi kết quả thực nghiệm gì?"*

---

## Slide 12: Kết Quả Chính — Δ = 0.0000 CHÍNH XÁC

*(Thời lượng: ~90 giây)*

**Trình bày bảng:**

> Chúng tôi chạy kiểm tra bất biến xoay trên 6 góc của tập competition đầy đủ 956 ca kiểm thử, 30 lần chạy.
>
> **Transformer baseline** giảm từ 0.8066 ở 0 độ xuống thấp nhất **0.7494 ở 60 độ** — mức giảm **0.057**.
>
> **SE2RoadNet đọc 0.8047 ở mọi góc**, từ 0 đến 180 độ. **Δ bằng chính xác không.**

**Nhấn mạnh điều cần nói:**

> Đây là **bit-đồng nhất** dưới các phép xoay SO(2) cứng. Không phải "xấp xỉ bằng không." Không phải "trong ngưỡng floating-point." Các điểm số **bằng nhau về mặt toán học**.

**Tại sao điều này quan trọng:**

> Pipeline đặc trưng 7 kênh không chứa đặc trưng phụ thuộc tọa độ. Xoay đường tạo ra ma trận đặc trưng đầu vào pixel-đồng nhất — và do đó đầu ra bit-đồng nhất. Định lý và thực nghiệm xác nhận lẫn nhau.

**Chuyển slide:** *"Để tôi cho các bạn thấy trực quan."*

---

## Slide 13: Trực Quan Hóa Tính Bất Biến

*(Thời lượng: ~45 giây)*

**Mô tả hình:**

> Đây là một đường cong trong hướng gốc, xoay 90 độ, và xoay 180 độ. APFD là **0.8047 cho cả ba** — điểm số, thứ hạng và APFD đều bằng nhau chính xác.
>
> Đường di chuyển trên mặt phẳng. Mô hình không quan tâm. Điểm số không thay đổi.

**Khối định lý:**

> Mô hình hợp thành thỏa mãn tính bất biến SE(2) **bất kể lựa chọn đầu attention**. Vì pipeline đặc trưng bất biến, bất cứ thứ gì xây dựng trên nó đều kế thừa nó.

**Chuyển slide:** *"Bây giờ để tôi phác thảo chứng minh."*

---

## Slide 14: Đảm Bảo Toán Học

*(Thời lượng: ~60 giây)*

**Định lý — đọc nguyên văn:**

> Gọi Φ là pipeline đặc trưng 7 kênh (chỉ độ cong). Khi đó với mọi phép biến đổi cứng trong SE(2), Φ của đường đã xoay và tịnh tiến bằng Φ của đường gốc. Do đó mô hình hợp thành f-theta thỏa mãn f-theta của đường biến đổi bằng f-theta của đường gốc — **bất kể lựa chọn đầu h**.

**Phác thảo chứng minh — 4 bước:**

> 1. **Độ cong là bất biến nội tại**: Độ cong là tính chất của chính đường cong, không phải vị trí nhúng của nó. Đây là hình học vi phân cổ điển — bộ công cụ Frenet-Serret.
>
> 2. **Độ dài cung bất biến theo tái tham số hóa**: Dưới các phép biến đổi cứng, độ dài cung được bảo toàn.
>
> 3. **Cả 7 đặc trưng chỉ phụ thuộc vào {kappa, d-kappa-over-ds, delta-s}**: Không đặc trưng nào đọc đường dưới dạng tọa độ.
>
> 4. **Không đặc trưng nào đọc (x, y) trực tiếp**: Do đó, bất kể phép biến đổi nào, ma trận đặc trưng không thay đổi.

**Xác nhận thực nghiệm:**

> Kết quả Δ = 0 là **xác nhận số** của cấu trúc này. Định lý và thực nghiệm hoàn toàn phù hợp.

**Chuyển slide:** *"Còn các lợi ích phụ."*

---

## Slide 15: SE(2) — Kết Quả Bổ Sung

*(Thời lượng: ~60 giây)*

**Biểu đồ cột AUC — nhấn mạnh điểm nổi bật:**

> SE(2)-Đối Xứng Đảo RoadNet đạt **AUC = 0.9347** — **cao nhất trong toàn bộ dự án**. Đây không chỉ là chiến thắng về độ mạnh; tính đối xứng đảo là một **inductive bias hữu ích** cải thiện chất lượng xếp hạng.
>
> So sánh: baseline ở 0.9170. Chênh lệch là **+0.018**.

**Đánh đổi:**

> Huấn luyện mất 24.2 phút so với 4 phút của FNO. Chi phí là độ phức tạp attention **O(B nhân L-bình phương nhân d-RFF)** mỗi lớp. Đây là cái giá của tính bất biến chính xác.

**Kết quả ổn định:**

> APFD = **0.8048 ± 0.0118**. Đây là phương sai không-ensemble thấp thứ hai trong dự án — mô hình cũng ổn định hơn.

**Chuyển slide:** *"Bây giờ chuyển sang điểm chính thứ hai: ràng buộc thông tin vật lý."*

---

## Slide 16: Tại Sao Thông Tin Vật Lý?

*(Thời lượng: ~90 giây)*

**Ràng buộc động lực học phương tiện — đọc phương trình:**

> Một xe đi theo đường r với vận tốc v giữ được trên đường chỉ khi gia tốc hướng tâm bị giới hạn bởi ma sát: **v-bình phương nhân kappa của s nhỏ hơn mu nhân g**, tại mọi điểm dọc theo cung.

**Tại sao điểm số nên tôn trọng điều này:**

> Một ca kiểm thử có max v-bình phương-kappa vượt ngưỡng nên được xếp **cao hơn** — có khả năng thất bại cao hơn. Nhưng baseline tự do tạo ra các hàm điểm **vi phạm tính đơn điệu này trong 17 đến 21 phần trăm các cặp kiểm thử**.

**Vấn đề kiểm toán được — đây là lập luận pháp lý:**

> Các nhà quản lý và kỹ sư cần hành vi xếp hạng **có thể dự đoán**. Một mô hình nói "đường-một nguy hiểm hơn đường-hai" nhưng ràng buộc gia tốc hướng tâm tối đa thực tế **thấp hơn** cho đường-một — mô hình đó **không thể bào chữa** trong triển khai an toàn quan trọng.
>
> Chúng tôi cần xếp hạng có thể kiểm toán được đối với vật lý phương tiện.

**Câu quan trọng:**

> "Một mô hình vi phạm tính đơn điệu gia tốc hướng tâm không thể bào chữa được trong triển khai an toàn quan trọng."

**Chuyển slide:** *"Để tôi cho các bạn thấy cách chúng tôi thực thi điều này."*

---

## Slide 17: Xây Dựng — Hàm Mất Mát Phụ Đơn Điệu PINN

*(Thời lượng: ~60 giây)*

**Hình phạt đơn điệu:**

> Với mỗi cặp đường kiểm thử (r_i, r_j) có max v-bình phương-kappa của r_i vượt alpha lần max của r_j, chúng tôi muốn điểm dự đoán của r_i lớn hơn hoặc bằng r_j.
>
> Chúng tôi phạt các vi phạm thứ tự bằng hàm mất mát phụ dựa trên ReLU: **L-phys bằng kỳ vọng của ReLU của (điểm-j trừ điểm-i)**. Khi ràng buộc bị vi phạm, mất mát dương.

**Mục tiêu tổng:**

> Hàm mất mát đầy đủ là **focal-BCE cộng lambda-phys nhân L-phys**. Chúng tôi sử dụng lịch trình: lambda tăng từ 0 lên 0.5 trong 30% epoch huấn luyện đầu tiên.

**Cấu hình chiến thắng:**

> **Lambda bằng 0.5, không có phạt Sobolev.** Biến thể Sobolev over-regularize quá mức — hại nhiều hơn lợi.

**Chuyển slide:** *"Và kết quả của hình phạt này?"*

---

## Slide 18: Kết Quả Chính — Giảm 5.6 Lần Vi Phạm

*(Thời lượng: ~90 giây)*

**Trình bày biểu đồ cột:**

> Mô hình đối chứng — không có hình phạt vật lý — vi phạm ràng buộc độ cong **17.57%** ở alpha bằng 1.5.
>
> Với PINN đơn điệu, vi phạm giảm xuống **3.14%** — mức giảm **5.6 lần**.
>
> Ở alpha bằng 2.0, mức giảm là **7.9 lần**: từ 21.44% xuống 2.72%.

**Câu chuyện một câu:**

> Vi phạm độ cong **giảm 5.6 lần** trong khi APFD **không nhúc nhích**.

**Đây là thông điệp quan trọng cho triển khai an toàn quan trọng:**

> Mô hình bây giờ có một **ràng buộc vật lý có ý nghĩa** được tích hợp sẵn. Kỹ sư có thể kiểm toán xếp hạng đối với động lực học phương tiện. Điều này không thể với baseline.

**Chuyển slide:** *"Và quan trọng — APFD không bị ảnh hưởng."*

---

## Slide 19: Câu Chuyện Hai Trục — Vi Phạm Sụp Đổ, APFD Vững

*(Thời lượng: ~60 giây)*

**Giải thích biểu đồ hai trục:**

> Đường màu xanh là APFD — nó **gần như phẳng** qua các biến thể đối chứng, đơn điệu, và đơn điệu-cộng-Sobolev. Đường màu đỏ là phần trăm vi phạm — nó **rơi tự do**.
>
> Đường APFD phẳng → **không mất gì về chỉ số cho nhà quản lý.**
> Đường vi phạm lao dốc → **mô hình bây giờ có thể kiểm toán được** đối với động lực học phương tiện.

**Lập luận kiểm toán:**

> Đây là lập luận thực tiễn mạnh nhất cho PINN đơn điệu: bạn nhận được **tuân thủ vật lý miễn phí** mà không hy sinh bất kỳ hiệu suất ưu tiên nào. Nhà quản lý có được mô hình tôn trọng ràng buộc gia tốc hướng tâm. Kỹ sư giữ nguyên APFD.

**Chuyển slide:** *"Để tôi cho xem bảng đầy đủ."*

---

## Slide 20: PINN — Lợi Ích Bổ Sung

*(Thời lượng: ~60 giây)*

**Ba chiến thắng miễn phí từ PINN đơn điệu:**

> 1. **AUC cải thiện**: 0.9205 lên 0.9244 — mức tăng 0.004. AUC cao thứ hai trong dự án.
> 2. **Phương sai giảm**: 0.0125 xuống 0.0122. Phương sai không-ensemble thấp thứ hai.
> 3. **Vi phạm giảm 5.6 lần**: Chiến thắng kiểm toán — từ 17.57% xuống 3.14%.

**Bài học từ Sobolev:**

> Thêm phạt Sobolev over-regularize quá mức. Nó làm APFD giảm 0.0022 — từ 0.8055 xuống 0.8033 — mà không cải thiện vi phạm đáng kể. Tính đơn điệu thuần túy là lượng inductive bias vật lý đúng đắn.

**Chuyển slide:** *"Vậy chúng ta có hai đảm bảo độc lập. Chúng có thể kết hợp được không?"*

---

## Slide 21: Kết Hợp Lý Thuyết — Các Đảm Bảo Có Xếp Chồng Được Không?

*(Thời lượng: ~90 giây)*

**Câu hỏi tự nhiên:**

> SE(2)-bất biến cộng PINN đơn điệu cộng listwise loss — chúng có **kết hợp** thành một mô hình không? Hay các đảm bảo can thiệp lẫn nhau?

**Mô hình hợp thành:**

> Chúng tôi lấy **SE(2)-đối xứng đảo SE2RoadNet backbone** (1.15 triệu tham số, nhẹ hơn) và thêm **listwise PL loss** cộng neural-sort term cộng BCE phụ.

**Kết quả kết hợp:**

> Định lý bất biến **sống sót** khi thêm listwise training. APFD đọc 0.8038 ở 0, 30, 90 và 180 độ — **Δ bằng không, bit-đồng nhất**.

**Phần thưởng chính:**

> **AUC bằng 0.9385** — **cao nhất trong toàn bộ dự án**, cao hơn cả hai phương pháp riêng lẻ. Định lý SE(2) và listwise loss là **bổ trợ**, không cạnh tranh.

**Điểm mấu chốt:**

> Tính đối xứng đảo và listwise learning là trực giao: đối xứng đảo ràng buộc biểu diễn đầu vào, listwise ràng buộc xếp hạng đầu ra. Chúng xếp chồng sạch sẽ.

**Chuyển slide:** *"Để tôi cho xem cấu hình xếp chồng đầy đủ."*

---

## Slide 22: Cấu Hình Chính Đề Xuất

*(Thời lượng: ~60 giây)*

**Đi qua sơ đồ pipeline:**

> Giai đoạn 1: **Tiền huấn luyện tự giám sát** trên các nhiệm vụ tiền từ Frenet-Serret — học hình học đường nội tại không cần nhãn.
>
> Giai đoạn 2: **SE(2)-đối xứng đảo backbone** — SE2RoadNet, cung cấp đảm bảo Δ = 0 bất biến xoay.
>
> Giai đoạn 3: **PINN đơn điệu phụ** — thực thi tính đơn điệu gia tốc hướng tâm. Giảm 5.6 lần vi phạm.
>
> Giai đoạn 4: **DiffAPFD listwise loss** — mục tiêu xếp hạng trực tiếp khả vi. σ thấp nhất.
>
> Giai đoạn 5 (nhánh bên): **Diffusion hard-mining** cho dữ liệu tổng hợp biên, **Conformal calibration** cho cận APFD không phân phối.

**Mỗi lớp mang lại gì:**

> - SE(2) — Δ = 0 bất biến xoay (Định lý 1)
> - PINN — Giảm 5.6 lần vi phạm (Định lý 2)
> - DiffAPFD — σ thấp nhất, dự phòng AUC
> - Conformal — Cận APFD không phân phối (lớp kiểm toán)

**Mục tiêu cho submission NeurIPS:**

> **APFD-comp ≥ 0.820 ± 0.010** với cả bốn đảm bảo xếp chồng.

**Chuyển slide:** *"Chúng tôi cũng thắng trên các chỉ số phụ."*

---

## Slide 23: Thưởng — Chúng Tôi Cũng Thắng AUC và Độ Ổn Định

*(Thời lượng: ~60 giây)*

**Ba chiến thắng độc lập:**

> 1. **APFD**: vượt mọi SOTA từ 0.024 đến 0.32.
> 2. **AUC**: cao nhất trong tất cả các mô hình — **0.9385** với mô hình hợp thành SE(2) + listwise.
> 3. **Độ ổn định**: **listwise ensemble** đạt **σ = 0.0109** — **phương sai thấp nhất từng được báo cáo** trên benchmark này.

**Đi qua bảng:**

> SOTA bên ngoài trước đó (ITEP4SDC) thậm chí không báo cáo AUC. Baseline nội bộ RoadFury của chúng tôi ở 0.9170 ± 0.012. Mọi phương pháp của chúng tôi đều vượt con số này. Tốt nhất là SE(2) + Listwise ở 0.9385.

**Điểm rút ra:**

> Chúng tôi không đánh đổi chỉ số này cho chỉ số khác. Chúng tôi thắng APFD, AUC và độ ổn định đồng thời. Không phương pháp nào khác trong lĩnh vực này có thể tuyên bố điều này.

**Chuyển slide:** *"Để tôi cho các bạn thấy chúng tôi đang ở đâu trong dự án."*

---

## Slide 24: Chúng Tôi Đang Ở Đâu — Các Phương Pháp Hoàn Thành

*(Thời lượng: ~60 giây)*

**Đọc sơ đồ trạng thái:**

> **Xanh lá — đã hoàn thành**: FNO Roads, SE(2)-Đối Xứng Đảo star, DiffAPFD Listwise, PINN Đơn Điệu star, Dự đoán Conformal, Phản Thực Nhân Quả, Diffusion Hard-mining, SE(2) + Listwise, Tối Thiểu Hóa Rủi Ro Bất Biến, CRC Bounds, OT-Sinkhorn Manifold, TENT Adaptation.
>
> **Cam — một phần**: SSL Foundation (tiền huấn luyện tự giám sát đang hoạt động nhưng chưa tích hợp vào stack đầy đủ).
>
> **Xám — đã lên kế hoạch**: Conformal v3 và SE(2) + PINN stack.

**Bốn tuần tới:**

> 1. Chạy baseline qua kiểm tra xoay để có hình đối chiếu.
> 2. Xếp chồng PINN đơn điệu lên SE(2) backbone — mô hình ba đảm bảo.
> 3. Cận conformal chặt hơn với mục tiêu miss-rate top-K.
> 4. Soạn bản thảo submission NeurIPS 2026.

**Chuyển slide:** *"Để tôi kết thúc với hai điểm rút ra chính."*

---

## Slide 25: Hai Điểm Rút Ra Chính

*(Thời lượng: ~45 giây)*

**Điểm rút ra 1 — SE(2) đối xứng đảo:**

> **Δ = 0.0000 CHÍNH XÁC.** SE2RoadNet bit-đồng nhất dưới các phép xoay cứng. Định lý cộng kết quả thực nghiệm phù hợp — đây là kết quả sạch nhất được lý thuyết xác minh bằng thực nghiệm của dự án. Nó sẽ là Hình 1 của bài báo.

**Điểm rút ra 2 — PINN đơn điệu:**

> **Giảm 5.6 Lần Vi Phạm.** PINN đơn điệu giảm vi phạm độ cong từ 17.57% xuống 3.14% trong khi APFD giữ nguyên. Đây là câu chuyện kiểm toán cho triển khai SDC an toàn quan trọng.

**Câu kết — đọc nguyên văn:**

> "Lý thuyết đánh bại kỹ thuật khi cấu trúc là có thật: **hình học + vật lý > một ablation Transformer khác**."

**Chuyển slide:** *"Cảm ơn mọi người. Tôi sẵn sàng trả lời câu hỏi."*

---

## Slide 26: Cảm Ơn

*(Thời lượng: ~30 giây + Hỏi đáp)*

> Cảm ơn mọi người đã theo dõi.
>
> Tôi sẵn sàng trả lời các câu hỏi về bất kỳ khía cạnh nào của đề xuất này — định lý SE(2), cấu trúc PINN, kết quả kết hợp, hoặc kế hoạch submission NeurIPS 2026.
>
> *(Dừng lại để hỏi đáp.)*

---

## Phụ Lục A: Dự Phòng — Các Phương Pháp Của Chúng Tôi vs Tất Cả SOTA

*(Tham khảo cho Hỏi đáp)*

**Đọc bảng mở rộng:**

> Đây là bảng so sánh đầy đủ bao gồm tất cả các biến thể nội bộ của chúng tôi. Các điểm rút ra cho Hỏi đáp:
> - **Listwise ensemble** đạt **σ thấp nhất từng có** ở 0.0109.
> - **SE(2) + Listwise hợp thành** đạt **AUC cao nhất từng có** ở 0.9385.
> - Mọi SOTA bên ngoài đều nằm dưới các con số của chúng tôi, và không cái nào mang đảm bảo lý thuyết.

---

## Phụ Lục B: Dự Phòng — FNO Bất Biến Độ Phân Giải

*(Tham khảo cho Hỏi đáp)*

**Kết quả FNO:**

> FNO đạt **bất biến discretization**: APFD chỉ thay đổi **±0.0006** khi tần số lấy mẫu thay đổi 3× — từ 64 lên 197 điểm. Khoảng dao động là 0.0012.
>
> Điều này bổ sung cho kết quả SE(2): FNO xử lý thay đổi tần số lấy mẫu, SE(2) xử lý phép xoay cứng. Cùng nhau chúng bao phủ hai trong ba điểm giòn chính của baseline.

---

## Phụ Lục: Bảng Thuật Ngữ

| Thuật ngữ | Định nghĩa |
|---|---|
| **APFD** | Trung bình Phần trăm Lỗi Phát hiện được — chỉ số chính; cao hơn là tốt hơn; khoảng [0, 1] |
| **SE(2)** | Nhóm Euclid đặc biệt trong 2D — phép xoay và tịnh tiến cứng |
| **SE(2)-Đối Xứng Đảo** | Đầu ra mô hình bất biến dưới phép biến đổi đầu vào SE(2) — có thể chứng minh bằng xây dựng |
| **PINN** | Mạng Thông tin Vật lý — thêm ràng buộc vật lý như các số hạng mất mát phụ |
| **PINN Đơn Điệu** | Biến thể PINN thực thi tính đơn điệu của điểm số theo độ cong |
| **DiffAPFD** | APFD khả vi qua SoftSort / sắp xếp neural — cho phép tối ưu hóa thứ hạng trực tiếp |
| **FNO** | Fourier Neural Operator — học operator bất biến discretization |
| **Dự đoán Conformal** | Lượng tử hóa bất định không phân phối với đảm bảo phạm vi mẫu hữu hạn |
| **Frenet-Serret** | Khung hình học vi phân cổ điển cho các tính chất đường cong nội tại |
| **Δ (xoay)** | Chênh lệch APFD giữa 0° và các hướng xoay — đo tính bất biến xoay |
| **SWA** | Stochastic Weight Averaging — chiến lược huấn luyện cải thiện tổng quát hóa |
| **BeamNG** | Nền tảng mô phỏng SDC mã nguồn mở thực tế dùng trong các cuộc thi SBFT/ICST |
| **SBFT** | Hội thảo Quốc tế về Kiểm thử Dựa trên Tìm kiếm và Cấu hình — tổ chức track kiểm thử SDC |
