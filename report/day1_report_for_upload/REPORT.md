# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python3; Ultralytics `8.4.145`.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35

**Thay đổi so với notebook nguồn:**

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=`468` ; `class_name`=`cab` ; `rank`=`1` ; `score`=`0.510915` ; `taxonomy_name`=`ImageNet-1K`.
2. Record là dự đoán cho **cả bức ảnh**: đây là output phân loại ảnh, chỉ có lớp và điểm số, không có box hay tọa độ để xác định một vật thể riêng. Ảnh có nhiều phương tiện nên lớp đứng đầu không đồng nghĩa với việc model đã định vị một chiếc car cụ thể.
3. Danh sách lớp thuộc taxonomy đi kèm checkpoint, ở đây là `ImageNet-1K`; model chọn từ danh sách đó chứ không tự tạo tên lớp.
4. `class_id` là mã lớp ổn định trong taxonomy, `class_name` giúp người đọc hiểu lớp, còn `taxonomy_name` cho biết mã/tên đó thuộc bộ nhãn nào. Chỉ ghi tên có thể gây nhầm khi các taxonomy dùng tên hoặc cách đánh số khác nhau.
5. Guideline cần xác định đơn vị phân loại (chủ thể chính hay toàn cảnh), cách chọn khi có nhiều vật/chủ đề, quy tắc ưu tiên hoặc xử lý đồng hạng, và khi nào được chọn lớp “khác/không rõ” hay cần escalation.
6. `score` chỉ thể hiện mức độ model nghiêng về dự đoán, không chứng minh nhãn đúng theo quy tắc gán nhãn. Ground truth phải được xác định từ ảnh và guideline, rồi được kiểm tra độc lập.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record. 
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chọn prediction `person`: `class_name`=`person` ; `score`=`0.912625` ; `bbox_xyxy`=`[385.33, 69.24, 498.92, 348.92]` ; `bbox_width`=`113.58` ; `bbox_height`=`279.68`.
2. Với ảnh rộng 640 và cao 427 pixel, gốc tọa độ ở góc trên-trái: box bắt đầu tại `(x=385.33, y=69.24)` và kết thúc tại `(x=498.92, y=348.92)`. Box rộng khoảng `113.58` pixel và cao khoảng `279.68` pixel.
3. Chọn ngưỡng `0.35` và `0.60`: JSON hiện có 11 prediction cho `kitchen` ở ngưỡng `0.35`; trong số đó 6 prediction có `score >= 0.60`. JSON đã lọc ở `0.35`, nên không chứa các box bị loại ở ngưỡng thấp hơn.
4. Reviewer cần kiểm tra thêm các box điểm thấp có phải vật thật không, loại false positive, kiểm tra box trùng/lặp, lớp sai, vật bị bỏ sót và độ khít của box.
5. **Quy tắc box:** vẽ box chữ nhật khít quanh phần nhìn thấy của một đối tượng, không lấy thêm nền; nếu đối tượng chạm/cắt mép ảnh thì box dừng tại mép ảnh.
6. Một số bát/đồ vật trên bàn bị cắt ở mép trái ảnh; JSON cũng có box `bowl` bắt đầu sát mép trái (`x=0.32`). Nếu guideline chưa quy định rõ có gán vật bị cắt mép và box theo phần nhìn thấy hay không, cần escalation tới Lab Coach thay vì tự đặt quy tắc cho ca này.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chọn record người: `instance_id`=`kitchen-001` ; `class_name`=`person` ; `score`=`0.899318` ; `polygon_point_count`=`348` ; 5 điểm đầu của `polygon_xy`=`[(446, 70), (445, 71), (444, 71), (443, 72), (442, 72)]`.
2. Box chỉ mô tả hình chữ nhật bao quanh; polygon theo đường biên nên thể hiện hình dạng và vùng pixel của đối tượng sát hơn, đồng thời loại bớt nền nằm ngoài hình dạng.
3. `instance_id` phân biệt từng đối tượng được gán trong ảnh hiện tại, kể cả khi nhiều đối tượng cùng lớp. Nó không nói đối tượng thuộc lớp nào (class ID) và cũng không liên kết cùng một đối tượng qua các khung hình (tracking ID).
4. **Quy tắc mask:** đi theo ranh giới nhìn thấy của đối tượng, không lấy nền; ở phần bị che chỉ vẽ phần quan sát được và không tự suy diễn đường biên khuất.
5. Biên tóc/vai người ở gần các nồi treo và vật dụng phía sau có chỗ chồng lấp, khó xác định chính xác đường biên bị che. Dùng guideline cho phần ranh giới nhìn thấy; nếu không thể xác định vật nào thuộc mask ở chỗ bị che thì escalation, không tự vẽ phần khuất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho ảnh theo taxonomy (ở đây ImageNet-1K). | Ảnh `traffic` có nhiều loại xe nhưng top-1 là `cab` (score `0.510915`); cần xác định guideline muốn mô tả chủ thể hay toàn cảnh. | Chọn nhãn theo quy tắc và bằng chứng trong ảnh, không sao chép prediction/score làm nhãn đúng. | Kiểm tra nhãn nhất quán với taxonomy, đúng đơn vị phân loại và cách xử lý ảnh nhiều chủ đề. |
| Phát hiện vật thể | Một class và một box `xyxy` cho mỗi đối tượng. | `kitchen` có 11 box ở ngưỡng `0.35`; ảnh/JSON cho thấy nhiều box bát và một box `bowl` sát mép trái (`x=0.32`), cần kiểm tra box trùng và vật bị cắt. | Gán từng đối tượng theo guideline, vẽ box khít phần nhìn thấy; escalation ca cắt/che nếu quy tắc chưa rõ. | Kiểm tra bỏ sót, false positive, box trùng, lớp và độ khít; đặc biệt rà soát các vật sát mép và box điểm thấp. |
| Instance segmentation | Một class, mask/polygon và ID riêng cho mỗi instance. | `kitchen-001` là `person`, polygon có 348 điểm; ảnh có các vật chạm/chồng lên nhau và biên bị che, nên cần xem mask có ăn vào nền hoặc vật khác không. | Vẽ biên phần nhìn thấy theo guideline; tách từng instance cùng lớp và không tự đoán phần khuất. | Kiểm tra biên mask, vùng nền bị gộp, vùng bị thiếu, instance chồng lấn và tính nhất quán của ID. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu trong phạm vi bài lab, không đưa dữ liệu cá nhân/nhạy cảm vào báo cáo hoặc nơi chia sẻ công khai, và tuân thủ hướng dẫn lưu trữ của lớp.
2. Dừng xử lý và báo cho Lab Coach/giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md`
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS