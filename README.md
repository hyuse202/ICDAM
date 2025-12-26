# ICDAM 2025: Hệ thống Đa tác tử Hybrid LLM cho SCM

## Tổng quan Dự án

Dự án nghiên cứu xây dựng hệ thống Đa tác tử (Multi-Agent System) kết hợp giữa **Large Language Models (Gemini)** và **Symbolic Solvers (OR-Tools)** để giải quyết các vấn đề lập lịch và đàm phán trong Chuỗi cung ứng (Supply Chain Management).

Dự án này hướng tới việc gửi bài báo khoa học tại hội nghị **ICDAM 2025**.

## Các Tính năng Chính

- **Kiến trúc Hybrid LLM-Solver**: Kết hợp khả năng suy luận của LLM với sự xác thực của bộ giải ký hiệu.
- **Vòng lặp LLM-Solver (Verify & Repair)**: Cơ chế tự động kiểm chứng và sửa lỗi kế hoạch.
- **Giao thức Đàm phán Nâng cao**: Đàm phán dựa trên JSON cấu trúc với logic đề xuất phản hồi (COUNTER-offer).
- **Đánh giá dựa trên Benchmark**: Tích hợp với PSPLib cho các tập bài toán RCPSP.
- **Khả năng Chịu lỗi Mạnh mẽ**: Tự động chuyển sang Chế độ Giả lập khi API LLM không khả dụng.

## Tài liệu

Để biết thêm thông tin kỹ thuật chi tiết, vui lòng tham khảo:
- [Tài liệu Chi tiết](DOCUMENTATION.md)
- [Hướng dẫn Triển khai](file:///home/hyuse/.gemini/antigravity/brain/ce6091fc-8283-4e1a-9e6e-09c9153c83f9/walkthrough.md)

## Bắt đầu

1. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
2. Thiết lập tệp `.env` với `GOOGLE_API_KEY` của bạn.
3. Chạy mô phỏng:
   ```bash
   python3 main_simulation.py
   ```
