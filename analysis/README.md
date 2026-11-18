# Tái tạo hình phân tích DimABSA
Chỉ tái phân tích predictions đã ship trên CPU, không training hay chạy lại mô hình.
Cài thư viện: `pip install -r requirements-analysis.txt`.
Từ repo root: `python -m analysis.make_all`.
Sửa tham số trong `analysis/config.yaml`; nhận 15 PNG tại `outputs/analysis/figs/` và `outputs/analysis/numbers.md`.
Gold DimABSA2026 và parser/results tác giả được `common.fetch.fetch_repo` tải theo commit cố định vào `.cache/`.
`reference_outputs/logsigma/task3_test_pairs_seed42.jsonl` là dữ liệu dự đoán đã ship; cần mô hình GPU để tái sinh file này.
| Output (đuôi `.png`) | Hình trong bài |
|---|---|
| fig_scsg_cf1_waterfall | 9 — phân rã hao hụt cF1 |
| fig_logsigma_error_by_gold | 10 — sai số theo gold |
| fig_logsigma_mixed_sentiment | 11 — mixed sentiment |
| fig_scsg_error_types | 12 — loại lỗi extraction |
| fig_span_near_miss | 13 — lệch biên span |
| fig_scsg_by_category | 14 — cF1 theo category |
| fig_category_shift_aspects | 15 — category shift |
| fig_A4_votes_vs_precision | 8 — phiếu và precision |
| fig_B2_sigma_dev | 17 — lịch sử sigma và DEV (phụ lục D) |
| fig_A2_sc_at_k | 18 — SC theo k (phụ lục F) |
| fig_A3_vote_threshold | 19 — ngưỡng phiếu (phụ lục F) |
| fig_sc_decompose | 20 — tách hiệu ứng SC (phụ lục F) |
| fig_style_options_counts | 21 — số nhãn qua split (phụ lục F) |
| fig_models_follow_train | 22 — gán nhãn theo train (phụ lục F) |
| fig_va_llm_vs_regressor | 23 — VA trên tuple khớp (phụ lục F) |
Ảnh và số đã sinh sẵn nằm ở `analysis/figs/` và `analysis/numbers.md` để xem nhanh mà không cần chạy.
