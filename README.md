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

**Bộ số liệu chính:** `results/` — 16 ảnh gốc sạch trong `images/`, thêm lóa nhân tạo 5 mức. Detector và tham số giữ nguyên ở mọi mức. 14/16 ảnh phát hiện được mặt ở baseline nên **recall chỉ tính trên 14 ảnh này**. Nguồn: `results/summary.csv` (run 2026-10-05 17:18, OpenCV 4.14.0, `haarcascade_frontalface_default`).

| Mức lóa | Recall proxy (%) | Pixel bão hòa trên mặt (%) | Mean IoU |
| --- | --- | --- | --- |
| 0 | 100.00 | 0.06 | 1.00 |
| 1 | 100.00 | 3.89 | 0.96 |
| 2 | 64.29 | 32.31 | 0.58 |
| 3 | 14.29 | 75.24 | 0.13 |
| 4 | 0.00 | 93.99 | 0.01 |

- Recall proxy = tỉ lệ mặt ở ảnh gốc còn tìm thấy ở ảnh lóa với **IoU ≥ 0,5**. Ảnh gốc đóng vai trò "nhãn giả" (không có nhãn thật).
- Pixel bão hòa trên mặt = tỉ lệ pixel có mức xám **≥ 250** trong vùng mặt ở ảnh gốc (tính trên ảnh lóa).

### 2.1. Ba câu hỏi

1. **Recall bắt đầu tụt ở mức lóa 2:** 100% (mức 0-1) → **64,29%** (mức 2) → 14,29% (mức 3) → 0% (mức 4). Mức 1 lóa đã làm mặt sáng lên (độ sáng vùng mặt 124 → 179) và IoU tụt nhẹ (1,00 → 0,96) nhưng detector **vẫn còn thấy mặt**. Đến mức 2 thì bắt đầu mất hàng loạt.
2. **Ngưỡng health score ≈ 30% pixel bão hòa trên mặt.** Khi recall bắt đầu tụt (mức 2), trung bình **32,31%** diện tích vùng mặt đã bão hòa trắng; ở mức 1 chỉ 3,89% nên vẫn an toàn. Vì vậy lấy **~30%** làm ngưỡng cảnh báo "camera không đáng tin", còn dưới ~5% coi như bình thường. *Lưu ý:* đây là ngưỡng trung bình; từng ảnh chịu được khác nhau (xem 2.3), nên thực tế nên để vùng đệm và tính theo từng frame.
3. **Có false positive, nhưng thưa.** Ảnh **07.jpg** ở ảnh gốc *không có mặt*, vậy mà ở mức lóa 1 Haar nhận nhầm ra 1 "mặt" (`false_pos = 1`). Trên 14 ảnh có mặt, chỉ **37.jpg** sinh thêm box lạ ở mức 3-4, đưa số false positive lên **0,07 box/ảnh**. Đúng như dự đoán ở mục 1.3: lóa làm vùng hoa văn sáng tối giống mặt bị nhận nhầm.

### 2.2. Failure case

Lấy theo mục `FAILURE CASE` trong `results/run_log.txt`: các ảnh **06, 17, 21, 22, 26** mất mặt sớm nhất ở **mức lóa 2**. Nhóm chọn **21.jpg** vì thể hiện rõ nhất cú lật trạng thái:

| Mức | Detect | IoU | Pixel bão hòa trên mặt | Độ sáng vùng mặt |
| --- | --- | --- | --- | --- |
| 1 | Còn mặt | 1,00 | 17,35% | 198,31 |
| 2 | **Mất mặt** | 0,00 | 48,36% | 236,70 |

Ý nghĩa: chỉ cần tăng lóa từ mức 1 lên mức 2, độ sáng vùng mặt vượt ~237 và gần nửa vùng mặt cháy trắng, Haar mất dấu mặt hoàn toàn — trong khi detector **không có cách nào báo rằng nó đang bị mù**. Ghép ảnh mức 1 (còn mặt) cạnh mức 2 (mất mặt) cho slide.

### 2.3. Đề xuất cải tiến

- **Health gate (quan trọng nhất):** nếu % pixel bão hòa trong vùng mặt vượt ngưỡng đo được (~30%) thì báo **"camera không tin cậy / đang mù"** thay vì báo **"không có tài xế"**. Log các frame bão hòa để đưa vào data loop.
- **Camera IR 940 nm + đèn chủ động + kính lọc thông dải:** tự tạo nguồn sáng ổn định ở dải mà nắng yếu, giảm hẳn phụ thuộc ánh sáng môi trường (xem mục 1.2). Đây là cách xe thật đang làm.
- **Auto-exposure theo vùng mặt:** giảm phơi sáng cục bộ để vùng mặt không bão hòa, thay vì phơi sáng toàn khung.
- **Detector bền hơn Haar:** YuNet hoặc RetinaFace (deep learning) chịu ánh sáng tốt hơn; nếu vẫn Haar thì thêm `haarcascade_profileface` cho mặt nghiêng.
- **Cảnh báo sớm mức 1:** ngay khi pixel bão hòa vượt ~5% đã có thể hạ độ tin cậy, không đợi tới lúc mất mặt.

**Hạn chế của bộ đo:** n = 14 ảnh, lóa là **nhân tạo** (blob Gaussian + veil), ngưỡng bão hòa 250 là quy ước của nhóm. Kết quả cho thấy một điểm đáng lưu ý: ảnh **06.jpg** mất mặt ở mức 2 dù chỉ **3,16%** pixel bão hòa, còn **37.jpg** vẫn nhận được mặt ở mức 3 khi đã **68,64%** bão hòa — nên "% bão hòa" là tín hiệu health tốt nhưng **không tách bạch tuyệt đối**, cần thêm dữ liệu thật.

### 2.4. Kiểm chứng chéo: bộ ảnh đã có lóa sẵn (`results_software/`)

Chạy trên 24 ảnh trong `software_glare/` (ảnh chụp đã có lóa sẵn, thêm lóa phần mềm), 12/24 baseline thấy mặt:

| Mức lóa | Recall proxy (%) | Pixel bão hòa trên mặt (%) | Mean IoU |
| --- | --- | --- | --- |
| 0 | 100.00 | 4.25 | 1.00 |
| 1 | 83.33 | 25.13 | 0.79 |
| 2 | 0.00 | 73.10 | 0.00 |
| 3 | 0.00 | 95.71 | 0.00 |
| 4 | 0.00 | 99.08 | 0.00 |

Xu hướng giống hệt nhưng **sụp sớm hơn**: recall rơi ngay từ mức 1 và về 0 ở mức 2. Lý do: nền ảnh đã chói sẵn (baseline đã có 4,25% bão hòa, độ sáng vùng mặt 172 so với 124 ở bộ sạch) nên chịu được ít lóa hơn. Kết luận: **lóa thật nguy hiểm hơn lóa nhân tạo**, và ngưỡng health score nên hiệu chỉnh theo điều kiện ánh sáng nền.
