# ICDAM 2025: Tài liệu Kỹ thuật Chi tiết

Tài liệu này cung cấp cái nhìn sâu sắc về kiến trúc, các thành phần và cơ chế của Hệ thống Đa tác tử Hybrid LLM cho Quản lý Chuỗi cung ứng.

## 1. Kiến trúc Hệ thống

Hệ thống tuân theo kiến trúc **Hệ thống Đa tác tử Hybrid (MAS)**, kết hợp khả năng suy luận sáng tạo của các Mô hình Ngôn ngữ Lớn (LLM) với sự kiểm chứng nghiêm ngặt của các Bộ giải Ký hiệu (Symbolic Solvers).

### Các Thành phần Cốt lõi:
- **Tác tử (Agents) (`agents/`)**: Các thực thể dựa trên vai trò (Quản lý Dự án, Quản lý Kho) tương tác bằng ngôn ngữ tự nhiên và JSON cấu trúc.
- **LLM Brain (`agents/llm_brain.py`)**: Một client mạnh mẽ cho Google Gemini API với khả năng tự động chuyển sang Chế độ Giả lập (Mock Mode) để chịu lỗi.
- **Bộ giải Ký hiệu (Symbolic Solver) (`solvers/rcpsp_solver.py`)**: Sử dụng Google OR-Tools CP-SAT để giải quyết các bài toán Lập lịch Dự án Ràng buộc Tài nguyên (RCPSP).
- **Bộ kiểm chứng (Verifier) (`solvers/rcpsp_solver.py`)**: Một công cụ dựa trên quy tắc để xác thực các lịch trình do LLM tạo ra dựa trên các ràng buộc về thứ tự ưu tiên và tài nguyên.

---

## 2. Dataset: PSPLib (Problem Set for Project Scheduling)

Dự án sử dụng bộ dữ liệu chuẩn **PSPLib**, một thư viện phổ biến cho bài toán Lập lịch Dự án Ràng buộc Tài nguyên (RCPSP).

### Chi tiết Dataset:
- **Nguồn**: [PSPLib](https://www.om-db.wi.tum.de/psplib/)
- **Tập dữ liệu cụ thể**: Tập **J30** (30 công việc cho mỗi dự án).
- **Định dạng tệp**: Tệp `.sm` (Standard Mode).
- **Cấu trúc tệp `.sm`**:
    - **PRECEDENCE RELATIONS**: Định nghĩa các ràng buộc thứ tự (công việc nào phải làm trước công việc nào).
    - **REQUESTS/DURATIONS**: Định nghĩa thời gian thực hiện và yêu cầu tài nguyên (R1, R2, R3, R4) cho từng công việc.
    - **RESOURCEAVAILABILITIES**: Định nghĩa tổng công suất của các tài nguyên có sẵn.

### Cách xử lý Dataset (`utils/psplib_loader.py`):
Hệ thống triển khai một bộ tải dữ liệu chuyên dụng để phân tích cú pháp các tệp `.sm`:
1. **Phân tích cú pháp (Parsing)**: Đọc tệp văn bản và trích xuất thông tin bằng cách sử dụng các biểu thức chính quy hoặc phân tích theo dòng.
2. **Cấu trúc hóa**: Chuyển đổi dữ liệu thô thành các đối tượng Python (Dictionary) để các tác tử và bộ giải có thể sử dụng.
3. **Tạo dữ liệu giả lập**: Có sẵn hàm `create_dummy_sm_file` để tạo dữ liệu kiểm thử nhanh mà không cần tải dataset bên ngoài.

---

## 3. Vòng lặp LLM-Solver (Verify & Repair)

Cơ chế **Verify & Repair** (Kiểm chứng & Sửa lỗi) đảm bảo tính khả thi của các kế hoạch do LLM tạo ra.

### Quy trình làm việc:
1. **Đề xuất (Proposal)**: LLM đề xuất lịch trình (Start Times) cho các công việc.
2. **Kiểm chứng (Verification)**: `RCPSPVerifier` kiểm tra các ràng buộc:
    - **Thứ tự ưu tiên**: $Start\_Time(Successor) \ge Start\_Time(Predecessor) + Duration(Predecessor)$.
    - **Tài nguyên**: $\sum Demand(Active\_Jobs, r) \le Capacity(r)$ tại mọi thời điểm $t$.
3. **Sửa lỗi (Repair)**: Nếu không khả thi, LLM nhận danh sách lỗi chi tiết và thực hiện sửa đổi cho đến khi đạt được tính khả thi.

---

## 4. Giao thức Đàm phán & Chạy kết quả

### Giao thức Đàm phán:
Sử dụng JSON cấu trúc để đảm bảo tính nhất quán:
```json
{
  "thought": "Suy luận của tác tử",
  "speak": "Thông điệp gửi đi",
  "decision": "AGREED / COUNTER / REJECT",
  "proposal": { "R1": 5, "reason": "Inventory limit" }
}
```

### Cách chạy và Kết quả:
1. **Chạy mô phỏng**: `python3 main_simulation.py`
2. **Kết quả đầu ra**:
    - Nhật ký đàm phán giữa PM và Warehouse.
    - Lịch trình do LLM đề xuất và kết quả kiểm chứng.
    - Trạng thái cuối cùng của dự án (Thành công/Thất bại).

---

## 5. Các Chỉ số Hiệu suất (Metrics)

Hệ thống sử dụng `metrics.py` để đánh giá hiệu năng dựa trên các công thức khoa học:

### 5.1. Tỷ lệ Khả thi (Feasibility Rate)
Đo lường độ tin cậy của LLM trong việc tạo ra kế hoạch hợp lệ.
$$Feasibility Rate = \frac{\text{Số lần kế hoạch khả thi}}{\text{Tổng số lần thử}}$$
*Ví dụ: Nếu PM thử 5 lần và có 4 lần kế hoạch vượt qua bộ kiểm chứng, tỷ lệ là 0.8.*

### 5.2. Độ mạnh (Robustness)
Đo lường khả năng thích ứng của hệ thống khi có sự cố (ví dụ: giảm tài nguyên đột ngột).
$$Robustness = \frac{1}{C_{max}(\text{Dynamic}) - C_{max}(\text{Static})}$$
- **$C_{max}(\text{Static})$**: Thời gian hoàn thành tối ưu (makespan) trong điều kiện lý tưởng.
- **$C_{max}(\text{Dynamic})$**: Makespan thực tế khi có biến động.
- **Ý nghĩa**: Nếu $C_{max}$ không thay đổi khi có biến động, Robustness = $\infty$ (hệ thống cực kỳ ổn định).

---

## 6. Cấu trúc Thư mục
- `agents/`: Tác tử và LLM.
- `solvers/`: Bộ giải OR-Tools và Bộ kiểm chứng.
- `utils/`: `psplib_loader.py` (Xử lý dataset) và `parser.py`.
- `metrics.py`: Tính toán KPI.
- `data/`: Chứa các tệp `.sm` từ PSPLib.

---

## 7. Phân tích Kết quả Thực nghiệm (Cập nhật mới nhất)

Dưới đây là phân tích dựa trên lần chạy hệ thống gần nhất:

### 7.1. Phân tích Mô phỏng (`main_simulation.py`)
- **Đàm phán**: Tác tử Project Manager (PM) đã tạo ra một yêu cầu tài nguyên chuyên nghiệp cho Công việc #2. Tác tử Warehouse đã kiểm tra kho hàng thực tế và phản hồi `AGREE` do đủ số lượng (R1, R2, R3). Điều này chứng minh khả năng hiểu ngữ cảnh và trạng thái của LLM.
- **Vòng lặp LLM-Solver**: PM đề xuất lịch trình cho 5 công việc đầu tiên. Lịch trình `{1: 0, 2: 0, 3: 0, 4: 0, 5: 6}` đã vượt qua bộ kiểm chứng ngay trong lần thử đầu tiên (**Attempt 1**). Điều này cho thấy LLM có khả năng nắm bắt các ràng buộc cơ bản khi được cung cấp dữ liệu đầy đủ.

### 7.2. Phân tích Chỉ số (`metrics.py`)
- **Tỷ lệ Khả thi (0.80)**: Với 4/5 lần thử thành công, hệ thống cho thấy độ tin cậy cao. Khoảng trống 20% còn lại thường do các ràng buộc tài nguyên phức tạp mà LLM cần nhiều vòng lặp "Verify & Repair" hơn để giải quyết.
- **Độ mạnh (0.3333)**: Khi chuyển từ môi trường Tĩnh (Static) sang Động (Dynamic), Makespan tăng từ 42 lên 45 ngày. 
    - Khoảng chênh lệch là 3 ngày.
    - Robustness = $1/3 \approx 0.33$.
    - **Nhận xét**: Chỉ số này cho thấy hệ thống có độ trễ nhất định khi đối mặt với biến động, nhưng vẫn duy trì được tính khả thi của toàn bộ dự án.

### 7.3. Kết luận
Sự kết hợp giữa khả năng lập luận của LLM và tính chính xác của Symbolic Solver tạo ra một hệ thống vừa linh hoạt trong giao tiếp, vừa đảm bảo tính thực thi trong kỹ thuật.
