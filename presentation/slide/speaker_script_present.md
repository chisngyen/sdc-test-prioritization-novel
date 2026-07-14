# Mở đầu

## Cách nói mở đầu (trước khi bấm slide) — [20s]

Em chào thầy và cả lớp ạ. Nhóm em xin trình bày đề tài **ưu tiên kiểm thử xe tự
lái**, tên phương pháp là **RoadFury tiến tới SE2RoadNet**. Buổi hôm nay em xin
đi khoảng 15 phút, trọng tâm nằm ở phần **phương pháp** và phần **kết quả**.

Nếu thầy cho phép, em xin nói trước **một câu để thầy nắm ý chính**: nhóm em xây
một mô hình chấm điểm con đường để xếp thứ tự chạy kiểm thử; điểm mới của nhóm em
là mô hình đó **bất biến với phép xoay con đường — bằng chính thiết kế kiến trúc,
chứng minh được trên giấy, chứ không phải bằng thủ thuật tăng cường dữ liệu**.
Trong suốt bài, em sẽ quay lại làm rõ ý này cho thầy.

## s00 · Trang tiêu đề — [15s]

Đây là đề tài của nhóm em, gồm ba thành viên. Tên phương pháp có hai vế:
**RoadFury** là mô hình nền tảng nhóm em tự xây trước đó, và **SE2RoadNet** là
đề xuất chính mà em sẽ trình bày kỹ hôm nay. Mũi tên giữa hai tên hàm ý: nhóm em
không làm lại từ đầu, mà **vá đúng hai điểm yếu** của RoadFury để ra SE2RoadNet.

## s01 · Mục lục — [20s]

Bài của em có năm phần. Phần một là bối cảnh và bài toán, để thầy thấy vì sao
việc này khó và đáng làm. Phần hai và ba là **phương pháp** — đây là phần em xin
dành nhiều thời gian nhất để thầy hiểu rõ nhóm em đã làm gì. Phần bốn là **đánh
giá thực nghiệm**. Phần năm em xin nói thẳng những hạn chế còn lại. Cuối bài em
có thêm năm slide dự phòng, phòng khi thầy hỏi sâu.

# Phần 1 — Bài toán

## s02 · Chuyển phần 1 — [5s]

Thưa thầy, em xin bắt đầu bằng bài toán ạ.

## s03 · Bối cảnh: kiểm thử xe tự lái trong mô phỏng — [45s]

Để kiểm thử phần mềm lái tự động, người ta không cho xe chạy ngoài đường thật mà
chạy trong **mô phỏng vật lý**, ví dụ nền BeamNG.tech. Mỗi lần build phần mềm,
hệ thống sinh **hàng nghìn** con đường khác nhau một cách tự động, rồi cho xe ảo
chạy giữ làn trên từng con đường: xe giữ được làn thì **PASS**, lệch ra ngoài thì
**FAIL**.

Vấn đề, thưa thầy, là **mỗi con đường chạy mô phỏng vật lý mất từ vài giây tới cả
phút**; gom hàng nghìn kịch bản lại thì tốn rất nhiều giờ máy. Mà phần lớn kịch
bản lại PASS, chỉ chừng ba tới bốn phần mười là FAIL. Cho nên thay vì chạy tràn
lan theo thứ tự ngẫu nhiên, ý tưởng là **xếp những kịch bản dễ FAIL lên chạy
trước**, để lộ lỗi sớm và tiết kiệm chi phí.

## s04 · Phát biểu bài toán — [45s]

Em xin hình thức hóa để thầy tiện theo dõi. **Đầu vào** của mô hình là một con
đường, biểu diễn bằng một chuỗi các điểm hai chiều — tùy con đường mà có từ 64
tới 197 điểm. Xin thầy lưu ý: đầu vào **chỉ có hình học con đường**, không có bất
kỳ dữ liệu hành vi nào của xe.

Mô hình của nhóm em, em gọi là bộ chấm điểm f, nhận con đường đó và trả về **một
xác suất FAIL trong khoảng 0 tới 1**. **Đầu ra** cuối cùng là một thứ tự: xếp các
kịch bản theo điểm giảm dần. Mục tiêu là làm sao **lỗi bị đẩy lên đầu hàng đợi
càng nhiều càng tốt**, và điều đó được đo bằng chỉ số tên là **APFD**, em sẽ
giải thích ngay sau đây.

## s05 · Vì sao cần ưu tiên — [40s]

Để thầy hình dung APFD một cách trực quan: hãy tưởng tượng em vẽ đường cong "đã
chạy bao nhiêu phần trăm test thì phát hiện được bao nhiêu phần trăm lỗi". Nếu
xếp hạng tốt, lỗi dồn lên đầu, đường cong này **dốc đứng lên ngay từ sớm**; nếu
xếp hạng kém, lỗi nằm cuối, đường cong đi ngang mãi rồi mới lên. **APFD chính là
diện tích dưới đường cong đó** — càng cao nghĩa là lộ lỗi càng sớm.

Điểm em muốn thầy nhớ ở đây là: bài toán này **không phải "chạy cho nhanh hơn"**,
mà là **"chạy đúng thứ tự"**. Cùng một tập lỗi, chỉ cần đổi thứ tự chạy là chi
phí đã khác nhau rất nhiều.

## s06 · Bộ dữ liệu SensoDat — [45s]

Nhóm em dùng bộ dữ liệu công khai **SensoDat**, công bố tại hội nghị MSR 2024.
Đây là các kịch bản xe tự lái sinh tự động trên BeamNG. Bộ này chia làm ba phần:
tập **huấn luyện** gần 28.800 kịch bản, tập **kiểm tra** hơn 7.200 kịch bản cùng
phân phối, và một tập riêng gọi là **Competition** gồm 956 kịch bản.

Điều quan trọng, thưa thầy, là tập Competition **lệch phân phối** so với tập
huấn luyện — thuật ngữ là out-of-distribution. Cụ thể, đường trong tập này ngắn
hơn hẳn, chỉ từ 129 tới 229 mét, trong khi huấn luyện có đường dài tới hơn 450
mét. Nhóm em cố tình đánh giá trên tập lệch này để chứng minh mô hình **tổng quát
hóa được**, chứ không chỉ học thuộc phân phối huấn luyện.

## s07 · Ví dụ kịch bản: hình dạng quyết định nhãn — [35s]

Đây là dữ liệu thật, thưa thầy. Quan sát của nhóm em: đường **PASS** thường cong
mượt, còn đường **FAIL** thường có **chicane** — tức những khúc cua zigzag gắt,
độ cong đổi dấu liên tục làm xe không kịp bám làn.

Em xin nhấn mạnh một điểm, vì nó là nền cho toàn bộ phương pháp: **không có tọa
độ tuyệt đối nào của con đường quyết định nhãn PASS hay FAIL — chỉ có hình dạng
quyết định**. Một con đường zigzag thì dù em đặt nó ở đâu trên bản đồ, xoay theo
hướng nào, nó vẫn là con đường zigzag khó lái. Đây chính là gợi ý cho ràng buộc
bất biến ở phần sau.

## s08 · Công trình liên quan: ba hướng — [40s]

Trước nhóm em, có ba hướng tiếp cận. Hướng thứ nhất là **tìm kiếm theo độ đa
dạng** — dùng giải thuật di truyền chọn ra tập test đa dạng hành vi; nhược điểm
là **đa dạng chưa chắc lộ lỗi**. Hướng thứ hai là **học máy trên đặc trưng thủ
công** — ví dụ mô hình ITS4SDC dùng mạng LSTM trên vài đặc trưng chuỗi; rẻ và dễ
giải thích, nhưng ít đặc trưng và **phụ thuộc khung quy chiếu**. Hướng thứ ba là
**học sâu** — mạng đồ thị, ResNet, hay Transformer như RoadFury của nhóm em; cho
APFD cao nhất nhưng vẫn **không có gì bảo đảm tính bất biến**.

Khoảng trống mà nhóm em thấy, và cũng là động lực của đề tài: **chưa phương pháp
nào bảo đảm được tính bất biến với phép xoay hay với tần số lấy mẫu — tất cả đều
chỉ dựa vào tăng cường dữ liệu, tức là gần đúng.**

## s09 · Công trình liên quan: tổng hợp — [35s]

Bảng này tổng hợp lại. Xin thầy đọc theo hai chiều. Chiều APFD thì các phương
pháp tăng dần từ ngẫu nhiên khoảng 0,49 lên tới các mô hình học sâu khoảng 0,80.
Nhưng chiều thứ hai, cột **"bất biến"**, thì **mọi phương pháp trước đều không
có**.

Thông điệp của nhóm em nằm ở đây, thưa thầy: nhóm em **không cố chạy đua thêm vài
phần nghìn APFD**, mà đóng góp một **trục bảo đảm hoàn toàn mới** mà chưa ai có.
Đó là tinh thần của cả đề tài.

# Phần 2 — RoadFury (mô hình nền tảng)

## s10 · Chuyển phần 2 — [5s]

Phần hai, em xin giới thiệu RoadFury — mô hình nền tảng của nhóm em ạ.

## s11 · RoadFury: sơ đồ tổng thể — [30s]

RoadFury hoạt động theo bốn bước. Thứ nhất, chuẩn hóa con đường về 197 điểm và
**trích ra 10 kênh đặc trưng** tại mỗi điểm. Thứ hai, đưa vào một **Transformer**
để đọc con đường như một chuỗi. Thứ ba, dùng kỹ thuật **SWA** — trung bình trọng
số các mô hình cuối quá trình huấn luyện — để mô hình ổn định hơn. Thứ tư, lúc
suy luận thì chấm điểm và sắp xếp giảm dần. Đây là một mô hình rất mạnh, nhưng nó
có **hai điểm mù** mà em sẽ chỉ ra ngay ở slide sau — và chính hai điểm mù này
sinh ra SE2RoadNet.

## s12 · RoadFury chi tiết và hai điểm mù — [55s]

Em xin đi vào chi tiết để thầy thấy rõ hai điểm mù. Bên trái là 10 kênh đặc
trưng: chiều dài đoạn, biến thiên góc, độ cong, độ giật của độ cong, và nhiều
đại lượng khác. **Nhưng có ba kênh — sin của góc, cos của góc, và góc tuyệt đối
— là phụ thuộc khung quy chiếu.** Nghĩa là gì, thưa thầy? Nghĩa là nếu em xoay
con đường đi một góc, ba kênh này đổi giá trị ngay, dù con đường thật ra vẫn y
hệt.

Bên phải là kiến trúc Transformer, khoảng 829 nghìn tham số, và nó dùng **mã hóa
vị trí tuyệt đối** — đây là điểm mù thứ hai. Kết quả thì rất tốt: APFD khoảng
0,80, AUC 0,917, đứng đầu cuộc thi. **Nhưng điểm cao mà không có bảo đảm**: chỉ
cần xoay con đường là điểm số trôi đi. Với một hệ thống an toàn như xe tự lái,
đó là một lỗ hổng nhóm em không thể bỏ qua.

## s13 · Cầu nối: vá hai điểm mù — [30s]

Vậy nên toàn bộ SE2RoadNet, thưa thầy, chỉ là **vá đúng hai điểm mù đó, có chủ
đích**. Điểm mù thứ nhất — nhạy với phép xoay — nhóm em vá bằng cách **bỏ ba kênh
phụ thuộc khung, chỉ giữ 7 kênh bất biến**. Điểm mù thứ hai — nhạy với cách lấy
mẫu — nhóm em vá bằng cách **thay mã hóa vị trí tuyệt đối bằng một cơ chế chỉ dựa
trên khoảng cách dọc đường**.

Tinh thần chung là chuyển từ **tăng cường dữ liệu, vốn chỉ gần đúng**, sang
**ràng buộc bằng kiến trúc, vốn có bảo đảm**. Đây là ý em mong thầy nắm nhất
trước khi vào phần ba.

# Phần 3 — SE2RoadNet (đề xuất chính)

## s14 · Chuyển phần 3 — [5s]

Và đây là phần trọng tâm ạ: SE2RoadNet.

## s15 · SE2RoadNet: tổng quan — [30s]

Sơ đồ tổng thể gồm: 7 kênh đặc trưng bất biến, đưa qua 6 khối mà nhóm em gọi là
**InvariantBlock**, rồi gộp lại và cho ra điểm FAIL. Em xin nói luôn **kết quả
then chốt** để thầy có đích theo dõi: khi nhóm em xoay con đường bằng nhiều góc
khác nhau rồi chấm lại, độ chênh APFD là **đúng bằng 0** — không phải xấp xỉ 0,
mà bằng 0. Bây giờ em xin giải thích bằng bốn bước làm sao đạt được điều đó.

## s16 · Ý tưởng cốt lõi: bất biến SE(2) — [50s]

Trước hết, xin thầy cho em giải thích **SE(2) là gì**. SE(2) là nhóm các **phép
biến đổi cứng của mặt phẳng** — gồm phép **xoay** và phép **tịnh tiến**. Nói nôm
na, thưa thầy: nếu em cầm bản vẽ con đường lên rồi xoay đi một góc bất kỳ, hoặc
dịch sang chỗ khác, hoặc đặt lại gốc tọa độ — thì đó **vẫn là đúng con đường
đó**, chiếc xe vẫn lái y hệt, PASS vẫn PASS và FAIL vẫn FAIL.

Cho nên điểm số mà mô hình chấm **bắt buộc phải giống hệt nhau** trong cả bốn
trường hợp. Trong slide có bốn hình cùng một con đường, và cả bốn đều cho cùng
một điểm. Em xin nhấn mạnh sự khác biệt cốt lõi: **tăng cường dữ liệu chỉ làm mô
hình quen với phép xoay; còn kiến trúc của nhóm em thì không cho phép điểm số đổi
— về mặt toán học là bất khả.**

> Chốt ý cho thầy: bất biến ở đây là **tính chất của kiến trúc**, đúng theo định
> nghĩa, không phụ thuộc vào việc mô hình được huấn luyện tốt hay xấu.

## s17 · Bước 1: bảy kênh đặc trưng bất biến — [55s]

Bước một là chọn đặc trưng. Nhóm em chỉ giữ lại **7 đại lượng nội tại** của con
đường: chiều dài mỗi đoạn, biến thiên góc, **độ cong**, đạo hàm bậc một và bậc
hai của độ cong theo chiều dài, chiều dài cung chuẩn hóa, và độ lệch chuẩn cục bộ
của độ cong. Điểm mấu chốt: **không còn kênh nào phụ thuộc khung quy chiếu** —
nhóm em đã bỏ hẳn sin, cos và góc tuyệt đối.

Thầy có thể hỏi: bỏ tọa độ và hướng đi như vậy thì có mất thông tin không? Câu
trả lời dựa trên định lý **Frenet–Serret**: một đường cong phẳng được xác định
**duy nhất, sai khác đúng một phép xoay-tịnh tiến, bởi hàm độ cong của nó**. Nói
cách khác, nếu em biết con đường "cong bao nhiêu tại mỗi điểm dọc theo chiều
dài", thì em đã nắm trọn hình dạng của nó rồi. Cho nên bỏ ba kênh kia là **không
mất thông tin gì cả** — chỉ bỏ đúng phần dư thừa mà lại chính là phần gây nhạy
cảm với phép xoay.

> Chốt ý cho thầy: tính bất biến được bảo đảm **ngay từ khâu đặc trưng, trước khi
> mô hình học một tham số nào**.

## s18 · Bước 2: kiến trúc InvariantBlock — [45s]

Bước hai là kiến trúc. Mỗi điểm trên đường được biến thành một vector 192 chiều —
gọi là một token. Nhóm em thêm một token tổng hợp đặc biệt, rồi cho qua **6 khối
InvariantBlock**. Mỗi khối là một lớp **chú ý đa đầu** tám đầu, cộng mạng truyền
thẳng và chuẩn hóa, kèm theo một thành phần đặc biệt em sẽ nói ở bước ba. Cuối
cùng gộp thông tin về token tổng hợp và cho ra một con số là xác suất FAIL.

Về quy mô: tổng cộng **2,11 triệu tham số**, gấp khoảng hai lần rưỡi RoadFury,
nhưng vẫn nhẹ — huấn luyện chỉ **24 phút**. Và xin thầy lưu ý: nhóm em dùng
**một cấu hình duy nhất**, không tinh chỉnh riêng cho từng bộ dữ liệu.

## s19 · Bước 3: chú ý theo hiệu chiều dài cung — [60s]

Bước ba là phần kỹ thuật lõi, em xin nói kỹ ạ. Transformer cần biết thứ tự các
điểm, nên thông thường người ta cộng thêm **mã hóa vị trí** — một tín hiệu theo
**chỉ số** của điểm: điểm thứ nhất, thứ hai, thứ ba. Vấn đề, thưa thầy, là chỉ
số tuyệt đối này **phụ thuộc vào cách mình lấy mẫu con đường**: lấy 64 điểm hay
197 điểm thì cùng một vị trí vật lý lại rơi vào chỉ số khác nhau, và thế là tính
bất biến bị phá vỡ.

Giải pháp của nhóm em: thay vì mã hóa vị trí tuyệt đối, nhóm em thêm một **độ
lệch — gọi là bias — vào ma trận chú ý**, và bias này **chỉ phụ thuộc vào hiệu
chiều dài cung giữa hai điểm**, tức khoảng cách dọc theo con đường giữa chúng.
Hiệu số này là đại lượng nội tại — xoay hay dịch con đường đều không làm nó đổi —
nên tính bất biến được giữ trọn vẹn. Về mặt cài đặt, nhóm em đưa hiệu số đó qua
một hàm với 32 tần số ngẫu nhiên cố định rồi qua một mạng nhỏ.

Cái giá phải trả là chi phí tính toán bậc hai theo số điểm, nên huấn luyện mất
24 phút; hướng cải tiến sắp tới của nhóm em là thay bằng một kỹ thuật rẻ hơn tên
là RoPE.

> Chốt ý cho thầy: mã hóa vị trí tuyệt đối là điểm mù thứ hai của RoadFury; nhóm
> em thay nó bằng thứ **chỉ nhìn vào khoảng cách tương đối dọc đường**, nhờ vậy
> giữ được bất biến.

## s20 · Bước 4: huấn luyện — [45s]

Bước bốn là huấn luyện. Điểm đáng nói là **hàm mất mát Focal**. Vì FAIL chỉ chiếm
ba tới bốn phần mười, nếu dùng hàm mất mát thường thì mô hình dễ "đoán PASS hết"
cho chắc ăn. Hàm Focal thêm một hệ số **giảm trọng số các ca dễ và tăng trọng số
các ca khó**, buộc mô hình tập trung vào những ca thật sự khó phân biệt.

Nhóm em có **quét tham số Focal từ 1 đến 5** và thấy APFD **gần như phẳng — mọi
giá trị đều nằm trong khoảng sai số** — chứng tỏ mô hình bền vững với tham số
này; nhóm em chọn giá trị **1,0**. Phần còn lại là các cấu hình huấn luyện tiêu
chuẩn: bộ tối ưu AdamW, lịch học có khởi động, SWA ở giai đoạn cuối. Tất cả gói
gọn trong 24 phút, một cấu hình cho mọi bộ dữ liệu.

# Phần 4 — Đánh giá thực nghiệm

## s21 · Chuyển phần 4 — [5s]

Phần bốn, em xin trình bày kết quả ạ.

## s22 · Chỉ số APFD và giao thức đánh giá — [45s]

Trước hết là chỉ số. **APFD** nằm trong khoảng 0 tới 1, càng cao càng tốt; ngẫu
nhiên rơi vào khoảng 0,5. Nhóm em đánh giá theo **ba lớp** để chắc chắn, thưa
thầy. Lớp một, **chấm một lượt** trên toàn bộ 956 kịch bản, được APFD 0,8047.
Lớp hai, **chấm 30 lần**, mỗi lần lấy ngẫu nhiên một phần ba tập, để loại yếu tố
may rủi do thứ tự cố định — kết quả 0,8048 với độ lệch chuẩn 0,012. Lớp ba là
phép thử em tâm đắc nhất: **thử phép xoay** — xoay cả tập bằng nhiều góc rồi chấm
lại, đo độ chênh APFD.

## s23 · Kết quả headline: bất biến phép xoay bằng 0 — [70s]

Đây là kết quả em muốn thầy chú ý nhất trong cả bài. Nhóm em xoay tập kiểm thử
bằng sáu góc khác nhau — 0, 30, 60, 90, 180 và âm 45 độ — rồi chấm lại. Với
SE2RoadNet, cả sáu góc cho **cùng một con số 0,8047**, độ chênh **đúng bằng
0,0000**. Để so sánh, một mô hình baseline khi bị xoay thì tụt điểm rõ rệt, độ
chênh tới 0,057.

Em xin làm rõ để thầy thấy đây không phải nói quá: **con số này không phải "nằm
trong sai số nhỏ", mà bằng nhau đến từng bit dấu phẩy động**. Lý do là vì bảy
kênh đặc trưng của nhóm em trả về **đúng cùng một vector** sau khi xoay, và mạng
đã ở chế độ suy luận nên hoàn toàn tất định, cho nên đầu ra giống hệt. Đây là một
trong số rất ít trường hợp trong học máy mà nhóm em có thể nói **lý thuyết được
kiểm chứng bằng thực nghiệm, không cần kèm chữ "xấp xỉ"**.

## s24 · Bảng xếp hạng — [55s]

Đây là bảng xếp hạng so với các baseline. Em xin trình bày **trung thực**, thưa
thầy: về APFD thuần, SE2RoadNet ở mức 0,805, tức **ngang bằng baseline tốt nhất
trong phạm vi sai số** — nhóm em **không khẳng định thắng về APFD**. Nhưng điểm
được là: **AUC tăng rõ rệt từ 0,917 lên 0,934**, **và** SE2RoadNet là **phương
pháp duy nhất có bảo đảm bất biến bằng 0**.

Nói theo ngôn ngữ tối ưu, đây là một **cải thiện Pareto**: không thua ở chiều
nào, tốt hơn ở AUC, và có thêm một bảo đảm lý thuyết mà không mô hình nào khác
có. Trong bối cảnh an toàn của xe tự lái, em cho rằng **một bảo đảm chắc chắn
đáng giá hơn vài phần nghìn APFD**.

## s25 · AUC và APFD: vì sao tách bạch — [40s]

Một điểm em muốn giải thích để thầy khỏi thắc mắc: tại sao nhóm em tách riêng
AUC và APFD. **AUC** đo khả năng phân loại đúng PASS với FAIL trên từng cặp.
**APFD** đo tốc độ lộ lỗi theo thứ tự chạy — đây mới là thứ bài toán thực sự cần.
Trong gần như mọi thí nghiệm của nhóm em, hai chỉ số này **không đi cùng nhau**:
ở đây AUC tăng nhưng APFD gần như phẳng. Cho nên khi báo cáo, nhóm em luôn nói
rõ đang tối ưu chỉ số nào, chứ không gộp thành một câu "tốt hơn" chung chung.

# Phần 5 — Hạn chế và hướng phát triển

## s26 · Chuyển phần 5 — [5s]

Cuối cùng, em xin nói thẳng những hạn chế còn lại ạ.

## s27 · Hạn chế — [50s]

Nhóm em xin trung thực với thầy. Thứ nhất, như vừa nói, AUC và APFD phân kỳ, nên
phải lập luận cẩn thận metric nào quan trọng khi nào. Thứ hai, một số cải tiến
phụ chỉ giúp **giảm độ dao động** chứ chưa nâng được APFD trung bình — nhóm em
coi đó là đóng góp về "độ ổn định", không thổi phồng thành kết quả chính. Thứ
ba, và quan trọng nhất về mặt lý thuyết: **bất biến phép xoay là chính xác tuyệt
đối, nhưng bất biến với tần số lấy mẫu thì mới chỉ gần đúng** — độ chênh khoảng
0,001, rất nhỏ nhưng chưa bằng 0. Ngoài ra, phần chặn an toàn bằng conformal thì
nhóm em vẫn đang hoàn thiện. Em xin báo cáo thẳng thay vì che đi.

## s28 · Hướng phát triển — [40s]

Từ những hạn chế đó, nhóm em có bốn hướng đi tiếp. Một, **tổng quát hóa lên hơn
tám bộ benchmark công khai** bằng đúng một công thức, không tinh chỉnh riêng.
Hai, hoàn thiện phần **chặn an toàn conformal** cho vừa đúng vừa hữu ích. Ba,
làm cho **bất biến lấy mẫu cũng chính xác tuyệt đối** bằng kỹ thuật RoPE thay cho
cách hiện tại. Bốn, dùng **học tự giám sát gắn với vật lý** để học biểu diễn nền
tốt hơn. Mục tiêu xa là biến một mô hình bất biến cho xe tự lái thành **một công
thức chung, bất biến và kiểm toán được, cho mọi benchmark**.

## s29 · Tài liệu tham khảo — [15s]

Đây là các tài liệu tham khảo. Nguồn dữ liệu là SensoDat; nền tảng kỹ thuật gồm
cơ chế chú ý, hàm Focal, kỹ thuật SWA, và lý thuyết bất biến nhóm. RoadFury và
SE2RoadNet là phương pháp của nhóm em; các baseline khác nhóm em tái lập trong
cùng một khung đánh giá.

## s30 · Video demo — [20s]

Đây là video demo trình bày phương pháp của nhóm em, có link Google Drive và
Facebook ạ. Nếu thầy muốn, em có thể mở phát trực tiếp.

## s31 · Cảm ơn — [10s]

Phần trình bày của nhóm em đến đây là hết. Em xin cảm ơn thầy và cả lớp đã lắng
nghe, và em xin sẵn sàng nhận câu hỏi ạ.

# Dự phòng — trả lời khi thầy hỏi

## s32 · Dự phòng B1: APFD tính tay — [khi được hỏi]

Nếu thầy muốn thấy APFD tính cụ thể: giả sử 5 test, 2 cái FAIL. Nếu xếp hai cái
FAIL lên vị trí 1 và 2 thì APFD bằng 0,80; nếu xếp xuống cuối, vị trí 4 và 5, thì
APFD chỉ còn 0,20. Cùng một tập lỗi, chỉ khác thứ tự, mà điểm chênh nhau bốn lần
— đó chính là toàn bộ giá trị của việc ưu tiên kiểm thử.

## s33 · Dự phòng B2: vì sao chấm 30 lần — [khi được hỏi]

Chấm một lượt thì nhạy với thứ tự cố định của tập test, dễ may rủi. Nên nhóm em
chấm 30 lần, mỗi lần lấy ngẫu nhiên một mẫu con, rồi báo cáo trung bình kèm độ
lệch chuẩn. Với nhóm em, **độ lệch chuẩn nhiều khi còn quan trọng hơn trung
bình**, vì nó cho biết xếp hạng có ổn định hay không.

## s34 · Dự phòng B3: bất biến lấy mẫu — [khi được hỏi]

Nếu thầy hỏi về bất biến với tần số lấy mẫu: nhóm em lấy cùng một con đường,
resample ở nhiều mật độ điểm khác nhau rồi chấm lại. Độ chênh khoảng 0,001 — rất
nhỏ nhưng **chưa bằng 0 như phép xoay**. Lý do là cơ chế bias theo hiệu chiều dài
cung mới chỉ gần bất biến tại tham số hóa; đây đúng là lý do nhóm em muốn chuyển
sang RoPE để đạt chính xác tuyệt đối.

## s35 · Dự phòng B4: chi phí ở quy mô lớn — [khi được hỏi]

Về chi phí: mô hình chỉ cần một lượt suy luận cho mỗi con đường, nên chấm cả tập
chỉ mất vài giây trên GPU. **Chi phí thật nằm ở mô phỏng vật lý các test được
chọn, không phải ở bộ chấm điểm.** Xếp hạng tốt dồn lỗi lên đầu, nên chỉ cần chạy
một phần nhỏ đầu hàng đợi đã lộ phần lớn lỗi. Huấn luyện chỉ một lần 24 phút nên
tái lập rất rẻ.

## s36 · Dự phòng B5: chứng minh ngắn tính bất biến — [khi được hỏi]

Nếu thầy muốn chứng minh: dưới phép xoay-tịnh tiến, chiều dài mỗi đoạn không đổi
vì phép xoay bảo toàn khoảng cách và phép tịnh tiến thì triệt tiêu; góc giữa hai
đoạn kề cũng không đổi vì phép xoay bảo toàn góc; độ cong và các đạo hàm theo
chiều dài đều là hàm của các đại lượng bất biến này nên cũng bất biến. Bỏ sin,
cos, góc tuyệt đối là điều kiện cần. Hệ quả: bảy kênh đầu vào giống hệt nhau sau
biến đổi, nên mọi tầng phía sau chỉ nhận đầu vào bất biến, và đầu ra bất biến.

# Chuẩn bị cho câu hỏi khó của thầy

## Nếu thầy hỏi: "SE2RoadNet APFD không cao hơn baseline, sao gọi là đóng góp?"

Thưa thầy, nhóm em **không khẳng định thắng về APFD** — APFD ngang trong sai số.
Đóng góp của nhóm em là **cải thiện Pareto**: giữ APFD đỉnh, nâng AUC rõ rệt, và
thêm một **bảo đảm bất biến bằng 0** mà chưa phương pháp nào có. Trong bài toán
an toàn, một bảo đảm chứng minh được đáng giá hơn vài phần nghìn điểm số.

## Nếu thầy hỏi: "Vì sao dám gọi là bất biến chính xác — nó là mạng nơ-ron mà?"

Vì tính bất biến đến từ **khâu đặc trưng, không phải từ tham số học được**. Bảy
kênh không chứa tọa độ hay hướng tuyệt đối, nên xoay con đường rồi trích lại cho
đúng cùng một vector; mạng ở chế độ suy luận là tất định, nên đầu ra giống hệt.
Ở mức số thực có sai số cỡ một phần mười triệu do ma trận xoay chứa sin/cos,
nhưng ở độ chính xác báo cáo thì APFD giống hệt đến từng bit.

## Nếu thầy hỏi: "Vì sao tập Competition được coi là lệch phân phối?"

Vì ba lý do: bộ sinh đường khác nhau, tỉ lệ FAIL khác nhau, và độ dài đường khác
hẳn — Competition chỉ 129 tới 229 mét trong khi huấn luyện dài tới hơn 450 mét.
Bằng chứng là điểm trên tập kiểm tra cùng phân phối và tập Competition chênh nhau
đáng kể.

## Ghi chú nội bộ cho người trình bày (không đọc)

Slide bước 4 hiện để Focal bằng 1,0 theo bản quét tham số mới nhất, nhưng các
slide headline và bảng xếp hạng vẫn đang hiển 0,8047 / 0,805 / 0,934 lấy từ lần
chạy tham số 1,5. Hai con số này lệch nhau không đáng kể và đều nằm trong sai số,
nhưng nếu thầy soi kỹ, cứ trả lời thẳng: mọi giá trị Focal đều cho APFD tương
đương trong phạm vi độ lệch chuẩn, nên kết luận không đổi.
