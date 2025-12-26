# Tài liệu Tổng hợp Phục vụ Viết Luận văn & Bài báo (ICDAM 2025)

Tài liệu này tổng hợp toàn bộ dữ liệu thực nghiệm, kiến trúc hệ thống và các chỉ số hiệu suất đã được triển khai, khớp với cấu trúc các bảng biểu yêu cầu.

---

## Bảng 1: Kiến Trúc Framework (Reference Implementation)

| Thành Phần | Mô Tả Và Vai Trò Trong Dự Án | Lợi Ích Kinh Doanh | Trạng Thái Triển Khai |
| :--- | :--- | :--- | :--- |
| **Core MAS Layer** | Hệ thống Đa tác tử dựa trên `BaseAgent` và `LLMBrain`, hỗ trợ giao tiếp message-based (PROPOSE, AGREE, COUNTER). | Linh hoạt, giảm chi phí vận hành bằng cách tự động hóa đàm phán. | **Hoàn thành** (`agents/basic_agents.py`) |
| **Hybrid Planning Pattern** | Mô hình **OptiMUS**: LLM đề xuất kế hoạch, `RCPSPVerifier` kiểm tra và phản hồi để sửa lỗi tự động. | Đảm bảo tính khả thi cao, giảm rủi ro ảo giác (hallucination) của LLM. | **Hoàn thành** (`main_simulation.py`) |
| **Negotiation Protocol** | Giao thức PROPOSE - COUNTER - AGREE - REJECT, lấy cảm hứng từ ANAC. | Tăng tốc độ ra quyết định, giảm tình trạng thiếu hụt hàng hóa (stockouts). | **Hoàn thành** (`agents/basic_agents.py`) |
| **Solver Layer** | Sử dụng **Google OR-Tools CP-SAT** để tối ưu hóa các ràng buộc (thời gian, tài nguyên). | Tối ưu hóa việc sử dụng tài nguyên, tăng tính bền vững (robustness). | **Hoàn thành** (`solvers/rcpsp_solver.py`) |

---

## Bảng 2: Benchmark (Đóng Góp Chính)

| Thành Phần | Mô Tả Chi Tiết | Dataset & Kịch Bản | Lợi Ích Kinh Doanh |
| :--- | :--- | :--- | :--- |
| **Benchmark Template** | Mở rộng từ REALM-Bench cho tài nguyên dùng chung và lập kế hoạch đa bước. | **PSPLib J30**: Bộ dữ liệu chuẩn với 30 công việc và 4 loại tài nguyên. | Chứng minh khả năng giảm thiểu chậm trễ trong logistics. |
| **Disruption Scenarios** | Mô phỏng các sự cố thực tế như hỏng hóc tài nguyên hoặc tăng nhu cầu đột biến. | **Resource Failure**: Giảm 50% công suất tài nguyên R1 trong quá trình chạy. | Tăng khả năng phục hồi (resilience) của chuỗi cung ứng. |
| **Scheduling Tasks** | Các nhiệm vụ sản xuất với ràng buộc thứ tự (dependencies) và tài nguyên tái tạo. | Tải dữ liệu từ `utils/psplib_loader.py`. | Giảm chi phí sản xuất, tối ưu hóa quy trình kinh doanh. |

---

## Bảng 3: Metrics (Thước Đo Hiệu Suất - Kết quả Thực nghiệm)

Dưới đây là kết quả thu được từ lần chạy hệ thống gần nhất (`metrics.py`):

| Nhóm Metrics | Chỉ Số Cụ Thể | Kết Quả Thực Nghiệm | Lợi Ích Kinh Doanh |
| :--- | :--- | :--- | :--- |
| **Feasibility & Efficiency** | Feasibility Rate | **80% (0.80)** | Đảm bảo kế hoạch có thể thực thi được trong thực tế. |
| **Negotiation** | Agreement Rate | **67% (0.67)** | Tăng tỷ lệ đồng thuận giữa các bên trong chuỗi cung ứng. |
| **Negotiation** | Avg Consensus Rounds | **2.0 vòng** | Tăng tốc độ đàm phán và ra quyết định. |
| **Fairness** | Gini Index (Tardiness) | **0.3116** | Đảm bảo phân bổ công việc công bằng, giảm tranh chấp. |
| **Robustness** | Robustness Index | **0.3333** | Duy trì hiệu suất ổn định khi có biến động tài nguyên. |

---

## Bảng 4: Lộ trình Thực thi (Execution Plan Status)

- **Phase 1: Setup**: Hoàn thành (Xây dựng skeleton, tích hợp OR-Tools).
- **Phase 2: Benchmark v1**: Hoàn thành (Triển khai static tasks, logging metrics).
- **Phase 3: Dynamic & Disruption**: Hoàn thành (Mô phỏng Resource Failure, đo Robustness).
- **Phase 4: Paper-ready**: Đang thực hiện (Tổng hợp bảng biểu, phân tích kết quả).

---

## Bằng chứng Kỹ thuật (Code Snippets cho Luận văn)

### 1. Vòng lặp Verify & Repair (Hybrid Planning)
```python
# solvers/rcpsp_solver.py
class RCPSPVerifier:
    def verify(self, data, schedule):
        # Kiểm tra ràng buộc thứ tự và tài nguyên
        ...
```

### 2. Giao thức Đàm phán (Structured JSON)
```json
{
  "thought": "Inventory is low for R1, proposing a counter-offer...",
  "speak": "I can only provide 5 units of R1 at this moment.",
  "decision": "COUNTER",
  "proposal": {"R1": 5}
}
```

### 3. Tính toán Robustness
```python
# metrics.py
def calculate_robustness(makespan_static, makespan_dynamic):
    return 1.0 / (makespan_dynamic - makespan_static)
```
