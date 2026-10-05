# Phân công nhóm — Sensor Reality Sprint (Track 4, Ngày 4)

Chủ đề T1: Camera degradation health score. Nhóm 5 người, 120 phút làm việc, 3-5 phút trình bày.

## Thành viên

| TV | Họ tên | Mã học viên | Vai trò |
| --- | --- | --- | --- |
| 1 | Phạm Minh Hiếu | 2A202602919 | Research |
| 2 | Hoàng Đức Dũng | 2A202602798 | Thu dữ liệu |
| 3 | Đoàn Anh Quân | 2A202602803 | Chạy code |
| 4 | Nguyễn Phúc Huy | 2A202602911 | Benchmark và phân tích |
| 5 | Trần Thị Lan | 2A202602621 | Trình bày |

## Bảng chốt

| Nhóm cần chốt | Ghi ngắn |
| --- | --- |
| Nền tảng, tính năng, sensor | Xe ADAS. **Phát hiện mặt tài xế**, là bước đầu của hệ thống cảnh báo buồn ngủ/mất tập trung. Sensor là camera cabin RGB. |
| Failure case | Nắng chiếu thẳng vào mặt tài xế làm mặt bị cháy sáng. |
| Claim ban đầu | Khi lóa tăng từ mức 0 đến 4, **tỉ lệ pixel bão hòa trên mặt tăng** và **recall phát hiện mặt giảm**. Nếu không thấy mặt thì toàn bộ hệ thống cảnh báo buồn ngủ phía sau mất tác dụng. |
| Metric và đơn vị | (1) **Recall proxy (%)**: tỉ lệ mặt ở ảnh gốc vẫn được tìm thấy ở ảnh lóa (IoU ≥ 0,5). Đây là proxy vì không có nhãn thật. (2) Tỉ lệ pixel bão hòa trên mặt (%). (3) Mean IoU. |
| Baseline và điều kiện lỗi | Baseline: ảnh gốc. Lỗi: **chính các ảnh đó** thêm lóa 4 mức. Giữ nguyên detector và tham số. |

## Mốc thời gian

| Thời gian (phút) | Việc |
| --- | --- |
| 15-40 | TV1 research, TV2 chụp ảnh, TV3 cài môi trường và chạy thử bằng ảnh mẫu |
| 40-70 | TV3 chạy với ảnh thật, TV1 viết README |
| 70-100 | TV4 phân tích, TV5 bắt đầu dựng slide |
| 100-120 | Ghép slide, tập pitch, tất cả commit lên repo |

## Thành viên 1 — Phạm Minh Hiếu (2A202602919): Research (phút 15-45)

1. Tìm hiểu 2 thứ và ghi link vào `README.md` trong repo:
   - Thuật toán Viola-Jones (Haar cascade, 2001): input là ảnh xám, output là bounding box mặt. Nó dựa vào **chênh lệch sáng tối** giữa vùng mắt và má, nên lóa làm hỏng nó.
   - Một tài liệu về DMS trong ô tô (từ khóa: *"driver monitoring system infrared camera sunlight"*). Mục đích là trả lời: vì sao xe thật dùng camera hồng ngoại (IR)?
2. Viết 3-4 câu "**Paper nói gì**", tách riêng với phần "nhóm đo được".
3. Ghi limitation của Haar: chỉ bắt mặt nhìn thẳng, nhạy với ánh sáng.

## Thành viên 2 — Hoàng Đức Dũng (2A202602798): Thu dữ liệu (phút 15-40)

1. Chụp **30-50 ảnh mặt** của các thành viên (xin đồng ý trước) bằng điện thoại hoặc webcam.
2. Yêu cầu: mặt nhìn gần thẳng, đủ sáng, nhiều góc nhỏ, nhiều người khác nhau, có người đeo kính.
3. Đặt tất cả vào thư mục `images/` trong repo.
4. *Tùy chọn:* chụp thêm 5 ảnh **lóa thật** (đứng cạnh cửa sổ nắng hoặc rọi đèn pin vào mặt) để vào thư mục `real_glare/`. Bộ này dùng để so sánh lóa nhân tạo với lóa thật. *Thực tế: nhóm chưa có ảnh lóa thật. 24 ảnh trong `software_glare/` là ảnh gốc được thêm lóa bằng hiệu ứng phần mềm (3 mức cho mỗi cảnh), không phải lóa thật.*

## Thành viên 3 — Đoàn Anh Quân (2A202602803): Chạy code (phút 40-70)

1. Cài môi trường:

   ```bash
   pip install "opencv-python<5" numpy pandas matplotlib
   ```

   Nếu không cài được OpenCV 4 thì chạy `pip install scikit-image`, script sẽ tự chuyển sang bộ phát hiện dự phòng.
2. Chạy:

   ```bash
   python face_glare_benchmark.py --images ./images --out ./results
   ```

3. Kiểm tra dòng *"Anh co mat o baseline: X/Y"*. Nếu X quá thấp (dưới 70%), báo thành viên 2 chụp lại ảnh rõ hơn.
4. Chạy thêm với ảnh lóa thật nếu có: `--images ./real_glare --out ./results_real`. *Thực tế: đã chạy với bộ lóa phần mềm: `--images ./software_glare --out ./results_software`.*
5. Chụp màn hình terminal, rồi commit cả thư mục `results/` lên repo.

## Thành viên 4 — Nguyễn Phúc Huy (2A202602911): Benchmark và phân tích (phút 70-100)

1. Mở `results/summary.csv` và điền bảng:

   | Mức lóa | Recall (%) | Pixel bão hòa trên mặt (%) |
   | --- | --- | --- |
   | 0 | | |
   | 1 | | |
   | 2 | | |
   | 3 | | |
   | 4 | | |

2. Trả lời 3 câu:
   - Recall bắt đầu tụt ở mức nào?
   - Khi đó **bao nhiêu % mặt bị cháy sáng**? Con số này chính là **ngưỡng health score**.
   - Có false positive nào không? (Ví dụ detector nhận nhầm vật thể khác là mặt, như ở ảnh mẫu.)
3. Chọn **1 failure case**: lấy tên ảnh trong log mục "FAILURE CASE", rồi cắt ảnh mức trước khi mất và mức bị mất để đặt cạnh nhau.
4. Viết **đề xuất cải tiến**:
   - Thêm **health gate**: nếu pixel bão hòa trên mặt vượt ngưỡng đo được thì báo "camera không tin cậy" thay vì báo "không có tài xế".
   - Dùng camera IR kèm đèn chủ động.
   - Dùng auto-exposure theo vùng mặt.
   - Dùng detector deep learning (YuNet, RetinaFace) bền hơn Haar.

## Thành viên 5 — Trần Thị Lan (2A202602621): Trình bày (phút 95-120)

1. Làm **1 slide** theo mẫu trang 8 của đề:
   - **Problem:** DMS cần thấy mặt tài xế, nắng làm mặt cháy sáng.
   - **Method:** Haar cascade (OpenCV), thêm lóa nhân tạo 4 mức.
   - **Benchmark:** `metrics_vs_glare.png` và bảng của thành viên 4.
   - **Failure case:** ảnh từ `glare_grid.jpg`.
   - **Engineering decision:** health gate, camera IR, log các frame bão hòa để đưa vào data loop.
2. Pitch 3-5 phút. Câu chốt: *"Lỗi nguy hiểm nhất không phải là mất mặt, mà là hệ thống không biết mình đang mù."*
3. Nhắc cả nhóm: **mỗi người nộp bản riêng trên VLearn**, có thêm một đoạn "phần tôi làm".
