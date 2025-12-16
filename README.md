# ICDAM 2025: Benchmark-centric Hybrid LLM Multi-Agent System for SCM

## Project Overview

Dự án nghiên cứu xây dựng hệ thống Đa tác tử (Multi-Agent System) kết hợp giữa **Large Language Models (Gemini)** và **Symbolic Solvers (OR-Tools)** để giải quyết các vấn đề lập lịch và đàm phán trong Chuỗi cung ứng (Supply Chain Management).

Project này hướng tới việc submit bài báo khoa học tại hội nghị **ICDAM 2025**.

## Key Features (Phase 1 Status)

- **Multi-Agent Skeleton:** Cấu trúc Agent (Warehouse, Project Manager) dựa trên role-playing.
- **Robust Hybrid LLM Brain:** - Module kết nối AI thông minh (`agents/llm_brain.py`).
  - **Fault Tolerance:** Tự động chuyển sang chế độ **Mock Mode** (Giả lập) khi API bị lỗi (429 Quota/Network), đảm bảo hệ thống không bao giờ crash.
- **Environment:** Tích hợp thành công thư viện `or-tools` và `google-generativeai`.
