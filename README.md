# Transfer Entropy Network with Macro

Computes directed information flow (transfer entropy) from macro variables (VIX, DXY, yields) to ETF returns. The per‑ETF score is the average transfer entropy across selected macros – a measure of how much macro conditions predict the ETF's future.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Selected macros: VIX, DXY, T10Y2Y, DGS10 (configurable)
- Transfer entropy with lag=1, discretisation bins=5
- Score = average TE(macro -> ETF)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-transfer-entropy-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High transfer entropy → ETF returns are strongly driven by macro – potentially more predictable, especially around macro announcements.
- Low TE → ETF is driven by idiosyncratic factors.

## Requirements

See `requirements.txt`.
