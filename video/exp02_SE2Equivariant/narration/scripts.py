"""
Vietnamese narration scripts, one per scene.

Each entry is the narration text that will be passed to Edge-TTS to
generate `scene_<NN>.mp3` next to this file.  The scene's `construct()`
calls `self.add_sound("narration/scene_<NN>.mp3")` at the start, so
the audio plays in sync with the animation timeline.

Style notes:
- Conversational technical Vietnamese, ~2 syllables per second.
- Numbers and acronyms are spelled out phonetically so the TTS reads
  them naturally  (e.g. "SE-hai", "A P F D", "B-rel").
- Punctuation drives pacing  (periods = long pause, commas = short).
"""
from __future__ import annotations

# Voice/format for Edge-TTS.  Female Vietnamese, slightly slowed.
VOICE = "vi-VN-HoaiMyNeural"
RATE  = "-5%"   # slow down 5% for clarity on technical terms
PITCH = "+0Hz"

NARRATION = {
    # ---------------------------------------------------------------- 00 --
    "scene_00": (
        "Trong kiểm thử xe tự lái, mỗi ca thử nghiệm là một con đường mô phỏng. "
        "Các trình giả lập tạo ra hàng vạn con đường, "
        "nhưng chỉ khoảng ba mươi phần trăm dẫn đến va chạm. "
        "Mục tiêu là đưa các ca tai nạn lên đầu danh sách. "
        "Chỉ số đánh giá là A P F D — càng nhiều ca tai nạn được phát hiện sớm, "
        "A P F D càng cao. "
        "Dự án này có một baseline và mười bốn thí nghiệm xoay quanh nó. "
        "Video hôm nay tập trung vào thí nghiệm số hai: "
        "SE-hai Equivariant RoadNet — một mô hình mà về mặt toán học, "
        "không thể bị ảnh hưởng bởi việc xoay con đường. "
        "Chúng ta sẽ đi từ điểm đường thô, qua bảy đặc trưng nội tại, "
        "đến điểm dự đoán cuối cùng."
    ),

    # ---------------------------------------------------------------- 01 --
    "scene_01": (
        "Mỗi ca kiểm thử là một hình dạng đường. "
        "Baseline nhìn hình dạng này và dự đoán xác suất xe sẽ đâm. "
        "Bây giờ hãy xoay con đường sáu mươi độ. "
        "Vật lý không đổi, vụ va chạm cũng không đổi. "
        "Chỉ có góc nhìn camera đổi — vậy mà điểm dự đoán sụp đổ. "
        "Điều chúng ta muốn theo thiết kế là: "
        "hàm dự đoán f của R nhân r cộng t, phải bằng f của r. "
        "Nghĩa là mô hình bất biến với mọi phép xoay và mọi phép tịnh tiến. "
        "Đạt được điều này bằng cách chỉ cho mạng nhìn hình học nội tại."
    ),

    # ---------------------------------------------------------------- 02 --
    "scene_02": (
        "Đầu vào là một chuỗi điểm có thứ tự dọc theo con đường. "
        "Mỗi điểm có toạ độ x và y. "
        "Toàn bộ con đường là một tensor L hàng, hai cột. "
        "Nhưng có một vấn đề: toạ độ x và y phụ thuộc vào "
        "việc gốc toạ độ đặt ở đâu, và hướng bắc nằm về phía nào. "
        "Đó chính là điểm yếu mà chúng ta cần khử."
    ),

    # ---------------------------------------------------------------- 03 --
    "scene_03": (
        "Bảy con số mỗi điểm, mà phép xoay không thể chạm tới. "
        "Kênh một: chiều dài đoạn. Khoảng cách giữa hai điểm liên tiếp. "
        "Kênh hai: độ thay đổi hướng tuyệt đối — giá trị tuyệt đối nên bất biến với phép lật. "
        "Kênh ba: độ cong có dấu. Dương là cua trái, âm là cua phải, "
        "tương đương một trên bán kính vòng tròn tiếp xúc. "
        "Kênh bốn: tốc độ thay đổi độ cong — phân biệt khúc cua mượt với khúc cua gấp. "
        "Kênh năm: gia tốc độ cong — phát hiện đoạn giật cục trong hình dạng đường. "
        "Kênh sáu: tỷ lệ chiều dài cung, cho biết điểm đang ở đâu trên đường. "
        "Kênh bảy: độ lệch chuẩn cục bộ của độ cong, đo độ gồ ghề quanh điểm. "
        "Cả bảy đều chỉ là hàm của khoảng cách và góc — không phụ thuộc hệ trục."
    ),

    # ---------------------------------------------------------------- 04 --
    "scene_04": (
        "Cùng một con đường, hai hướng khác nhau. "
        "Bên trái là bản gốc, bên phải đã xoay sáu mươi độ. "
        "Tại cùng chỉ số điểm, ta trích bảy đặc trưng nội tại từ cả hai bản. "
        "Mọi đặc trưng đều trùng khớp đến độ chính xác của máy. "
        "Ngược lại, baseline nuốt vào sin theta và cos theta — "
        "hai đại lượng này xoay theo con đường. "
        "Baseline phải tự học cách bỏ qua chúng, và hiếm khi học được hoàn toàn."
    ),

    # ---------------------------------------------------------------- 05 --
    "scene_05": (
        "Toàn cảnh SE-hai RoadNet. "
        "Đầu vào là tensor L hàng, bảy cột. "
        "Một lớp tuyến tính nâng số chiều lên một trăm chín mươi hai. "
        "Một token C L S học được được gắn vào đầu chuỗi. "
        "Sáu khối InvariantBlock giữ nguyên hình dạng tensor. "
        "Cuối cùng, lấy hàng C L S, đi qua phần đầu, "
        "cho ra một xác suất duy nhất. "
        "Tiếp theo, chúng ta mở một khối InvariantBlock để xem bên trong."
    ),

    # ---------------------------------------------------------------- 06 --
    "scene_06": (
        "Bên trong một InvariantBlock. "
        "Đây là transformer chuẩn: chuẩn hoá lớp, attention nhiều đầu, "
        "phép cộng dư, chuẩn hoá lớp lần nữa, mạng truyền thẳng, cộng dư lần nữa. "
        "Phần thú vị nằm ở chỗ attention được cộng thêm một bias gọi là B-rel. "
        "B-rel được tính từ hiệu chiều dài cung, qua một hàm sin, rồi qua một M L P. "
        "Vì hiệu chiều dài cung bất biến với phép xoay, "
        "B-rel cũng bất biến, do đó toàn bộ attention bất biến. "
        "Tiếp theo, chúng ta theo dõi một tensor thực sự đi qua mạng — "
        "từng phép nhân một."
    ),

    # ---------------------------------------------------------------- 06b -
    "scene_06b": (
        "Hành trình tính toán end-to-end của một tensor. "
        "Để dễ nhìn, ta dùng L bằng bốn, d bằng bốn. "
        "Bước một: phép chiếu tuyến tính. "
        "Mỗi hàng của X được nhân với W-proj, cho ra H-không. "
        "Bước hai: gắn token C L S vào đầu chuỗi. "
        "Hàng đầu tiên giờ là C L S — mạng sẽ ghi câu trả lời vào hàng này. "
        "Bước ba: bên trong một InvariantBlock. "
        "Đầu tiên, chuẩn hoá lớp theo từng hàng. "
        "Sau đó, ba phép chiếu cho Q, K, V. "
        "Tính ma trận điểm S bằng Q nhân K chuyển vị, chia căn d. "
        "Cộng B-rel vào S. "
        "Áp dụng softmax theo từng hàng, ra ma trận trọng số A. "
        "Nhân A với V cho ra ma trận đầu ra O. "
        "Cộng dư, đi qua mạng truyền thẳng, cộng dư lần nữa. "
        "Sáu khối như vậy nối tiếp; hình dạng tensor không đổi. "
        "Bước bốn: lấy hàng C L S, đẩy qua đầu mạng — "
        "tuyến tính, GELU, tuyến tính, rồi sigmoid. "
        "Kết quả là xác suất xe đâm. "
        "Đó chính là con số mà ta sắp xếp theo A P F D."
    ),

    # ---------------------------------------------------------------- 07 --
    "scene_07": (
        "Mô hình hoạt động ra sao? "
        "Đo A P F D qua sáu góc xoay khác nhau. "
        "Các cột vàng SE-hai trùng nhau chính xác đến bốn chữ số thập phân — "
        "đường thẳng phẳng tuyệt đối. "
        "Cột đỏ baseline tụt từ bốn đến tám điểm. "
        "Delta A P F D bằng không. "
        "Bảng tổng kết: A P F D bằng không phẩy tám không bốn bảy; "
        "AUC bằng không phẩy chín ba bốn bảy; "
        "hơn hai triệu tham số. "
        "Bất biến phép xoay theo thiết kế. "
        "Tám bộ dữ liệu công khai, một công thức duy nhất."
    ),
}
