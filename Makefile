.PHONY: review

check-pair-judge:
	uv run streamlit run streamlits/review_pairs.py
