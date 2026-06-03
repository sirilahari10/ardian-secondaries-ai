# Private Equity Secondaries: AI Pricing & Agentic Screening

In the PE Secondaries market, accurately pricing LP stakes (estimating the discount or premium to Net Asset Value) and rapidly screening fund documents are critical bottlenecks. 

This repository demonstrates a foundational Python-based architecture for Ardian's Secondaries & Primaries team:
1. Data Engineering: Simulates ingestion of historical fund performance metrics (Vintage, TVPI, DPI, RVPI, Strategy).
2. Predictive Modeling: Uses a Random Forest Regressor to predict the secondary market pricing (Discount/Premium to NAV) based on fund metrics and macroeconomic indicators.
3. Agentic Workflow: Simulates an LLM-driven agent that analyzes the ML pricing output alongside fund strategy to generate an automated Deal Screening Memo.

