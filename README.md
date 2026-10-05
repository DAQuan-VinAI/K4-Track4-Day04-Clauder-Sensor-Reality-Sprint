# K4-Track4-Day04-Clauder-Sensor-Reality-Sprint

Chủ đề T1: Camera degradation health score. Bài toán: phát hiện mặt tài xế bằng camera cabin RGB khi nắng chiếu thẳng vào mặt (lóa). Phân công xem [TEAMATES.md](TEAMATES.md).

## 1. Research nền (Thành viên 1)

Mục này chỉ ghi **tài liệu nói gì**. Số liệu nhóm tự đo nằm riêng ở [mục 2](#2-nhóm-đo-được).

### 1.1. Thuật toán Viola-Jones (Haar cascade, 2001)

**Link:**

- Paper gốc: P. Viola, M. Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001. [PDF](https://www.cs.ubc.ca/~lowe/425/violaJones01.pdf) · [DOI 10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
- Model nhóm dùng: [`haarcascade_frontalface_default.xml`](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) trong OpenCV
- Hướng dẫn OpenCV: [Cascade Classifier tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

**Input / output:** input là ảnh xám (paper viết: *"working only with the information present in a single grey scale image"*). Output là danh sách bounding box mặt.

**Paper nói gì:**

1. Mỗi đặc trưng Haar là **hiệu giữa tổng độ sáng của các hình chữ nhật kề nhau** trong cửa sổ 24x24 pixel. Nhờ "integral image", mỗi tổng hình chữ nhật chỉ tốn 4 phép tra bảng.
2. AdaBoost tự chọn đặc trưng. Đặc trưng đầu tiên nó chọn dựa trên việc **vùng mắt thường tối hơn vùng má**, đặc trưng thứ hai dựa trên việc **mắt tối hơn sống mũi**.
3. Bộ phát hiện là một cascade 38 tầng với 6061 đặc trưng, huấn luyện trên 4916 ảnh **mặt nhìn thẳng, đứng thẳng** ("frontal upright faces"). Các tầng đầu loại nhanh phần lớn vùng không phải mặt, nên chạy được 15 khung hình/giây với ảnh 384x288 trên Pentium III 700 MHz.
4. Để giảm ảnh hưởng của ánh sáng, mỗi cửa sổ con được **chuẩn hóa phương sai** (variance normalization) trước khi tính đặc trưng.

**Vì sao lóa làm hỏng Haar (suy luận của nhóm từ paper, chưa phải số đo):** chuẩn hóa phương sai chỉ bù được khi cả cửa sổ sáng lên hoặc tối đi đều nhau. Lóa mạnh làm pixel vùng mắt và vùng má cùng bão hòa ở 255, nên hiệu "mắt tối hơn má" về gần 0 và thông tin đã mất thì không chuẩn hóa nào lấy lại được. Các đặc trưng đầu cascade không kích hoạt, cửa sổ bị loại ngay từ tầng đầu.

### 1.2. DMS trong ô tô: vì sao xe thật dùng camera hồng ngoại (IR)?

**Link:**

- E. Nowara, T. K. Marks, H. Mansour, Y. Nakamura, A. Veeraraghavan, *SparsePPG: Towards Driver Monitoring Using Camera-Based Vital Signs Estimation in Near-Infrared*, CVPR Workshops 2018 (MERL). [Trang paper](https://merl.com/publications/TR2018-067) · [PDF](https://www.merl.com/publications/docs/TR2018-067.pdf)
- Konica Minolta Sensing, *Using Near-Infrared Light for Driver And Occupant Monitoring*, AZoSensors, 02/2021. [Bài viết](https://www.azosensors.com/article.aspx?ArticleID=2163)

**Paper nói gì:**

1. Trong xe, ánh sáng trên mặt tài xế thay đổi rất mạnh: ban ngày khi nắng lọt qua tán cây, ban đêm do đèn đường và đèn xe ngược chiều (Nowara và cộng sự, 2018).
2. Các thay đổi này **giảm đáng kể** khi dùng đèn hồng ngoại gần (NIR) chủ động, băng hẹp ở **940 nm**, kèm **kính lọc thông dải** cùng bước sóng đặt trước camera (Nowara và cộng sự, 2018). Camera gần như chỉ nhận ánh sáng do chính đèn của xe phát ra, nên độ sáng trên mặt ổn định.
3. Quanh 930-950 nm, bức xạ mặt trời bị khí quyển hấp thụ nhiều, nên nắng ít gây nhiễu hơn. Mắt người cũng gần như không thấy bước sóng này, nên đèn không làm tài xế chói và dùng được cả ngày lẫn đêm (Konica Minolta, 2021).
4. Hệ thống cũ dùng 850-890 nm thì tài xế còn thấy đốm đỏ, gây mất tập trung. Đó là lý do chuyển sang 940 nm (Konica Minolta, 2021).

**Câu trả lời ngắn:** xe thật dùng camera IR 940 nm có đèn chủ động và kính lọc vì nó **tự tạo ánh sáng ổn định ở dải bước sóng mà nắng yếu**. Camera RGB như của nhóm thì phụ thuộc hoàn toàn vào ánh sáng môi trường, nên nắng chiếu thẳng là cháy mặt.

**Lưu ý:** nắng vẫn có một phần ở 940 nm, chỉ yếu hơn. IR giảm vấn đề lóa chứ không xóa hẳn.

### 1.3. Limitation của Haar cascade

- **Chỉ bắt mặt nhìn thẳng.** Paper huấn luyện trên "frontal upright faces". OpenCV phải dùng một file riêng cho mặt nghiêng ([`haarcascade_profileface.xml`](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_profileface.xml)). Tài xế quay đầu nhìn gương hoặc cúi xuống thì dễ bị mất mặt.
- **Nhạy với ánh sáng.** Toàn bộ đặc trưng là chênh lệch sáng tối giữa các vùng. Lóa (cháy sáng) hoặc bóng đổ một bên mặt làm sai lệch đúng thứ mà detector dựa vào. Xem thêm nghiên cứu về tiền xử lý ánh sáng cho Viola-Jones: [Afifi và cộng sự, 2017](https://arxiv.org/abs/1709.07720).
- **Dễ nhận nhầm (false positive).** Detector có thể nhận nhầm vùng có hoa văn sáng tối giống mặt là mặt.
- **Không cho biết vì sao không thấy mặt.** Với cách gọi thông thường (`detectMultiScale`), output chỉ là danh sách box, rỗng thì là rỗng. Detector không phân biệt được "không có tài xế" với "camera đang bị lóa". Đây là lý do nhóm cần thêm health score.

## 2. Nhóm đo được

*(Thành viên 4 điền từ `results/summary.csv`. Chỉ ghi số nhóm tự đo, không trộn với mục 1.)*

| Mức lóa | Recall proxy (%) | Pixel bão hòa trên mặt (%) | Mean IoU |
| --- | --- | --- | --- |
| 0 | | | |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
