# Carbon-Aware Portfolio Optimization  
Efficient Frontier Analysis of Low-Emission S&P 500 Constituents

## Overview
This project investigates the construction and performance of **carbon-aware portfolios** derived from the lowest-emission companies in the S&P 500. Using mean-variance optimization, we construct efficient frontiers under a carbon-screened investment universe and evaluate their performance **in-sample** and **out-of-sample**. The analysis compares optimized portfolios against both the **broad S&P 500 index** and a **naive equal-weight portfolio**.

The work was conducted as part of academic research at the **University of Twente** (2025).  

## Motivation
Climate change and the energy transition are reshaping capital markets. Investors increasingly want to align portfolios with **low-carbon goals** without sacrificing returns. This project explores whether **carbon-aware portfolios** can remain competitive with conventional benchmarks, and how optimization performs relative to simple diversification under different market regimes.

## Methodology
1. **Data Collection**  
   - S&P 500 constituents (Refinitiv / LSEG)  
   - Carbon emissions (Scope 1 & 2) from LSEG ESG dataset  
   - Daily price data (10-year history)

2. **Carbon Screening**  
   - Rank companies by emissions  
   - Select the **50 lowest emitters**  
   - Require ≥10 years of daily data → final universe: **42 stocks**

3. **Portfolio Optimization**  
   - Compute annualized expected returns (μ) and covariance matrix (Σ)  
   - Solve mean-variance optimization problem (via CVXPY)  
   - Constraints: fully invested, long-only, max 10% per asset  
   - Construct the **efficient frontier**

4. **Target Portfolio Selection**  
   - Identify the portfolio on the frontier with **volatility closest to the S&P 500**  

5. **Evaluation**  
   - **Scenario 1:** Train 2015–2018, Test 2019  
   - **Scenario 2:** Train 2015–2022, Test 2023–2024  
   - Compare **Target Portfolio**, **Equal Weight**, and **S&P 500**

6. **Visualization & Reporting**  
   - Plot in-sample frontier vs. out-of-sample realized performance  
   - Report annualized return/volatility comparisons  

A simplified workflow is shown below:

Data → Carbon Screening → Final Universe (42) → Efficient Frontier → Target Portfolio → OOS Evaluation → Benchmark Comparison → Results


## Results (Highlights)

| Scenario | Portfolio       | Return (IS) | Volatility (IS) | Return (OOS) | Volatility (OOS) |
|----------|-----------------|-------------|-----------------|--------------|------------------|
| 2015–2018 / 2019 | Target       | 18.6% | 13.6% | 35.6% | 13.5% |
|          | Equal Weight   | 12.7% | 15.2% | 38.4% | 14.0% |
|          | S&P 500       | 5.9%  | 13.7% | 26.2% | 12.5% |
| 2015–2022 / 2023–2024 | Target       | 28.1% | 24.6% | 15.9% | 17.1% |
|          | Equal Weight   | 17.5% | 21.4% | 22.6% | 14.4% |
|          | S&P 500       | 9.5%  | 18.8% | 22.2% | 12.9% |

**Key insights:**
- In 2019, carbon-aware portfolios (optimized and equal-weight) **outperformed** the S&P 500.  
- In 2023–2024, the S&P 500 and equal-weight portfolios **outperformed** the optimized low-carbon portfolio.  
- Equal-weight portfolios provided **surprisingly strong and robust performance** across both periods.  
- Optimization tended to overweight past winners, which proved fragile out-of-sample.  

## Requirements
- Python 3.9+  
- Dependencies:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `cvxpy`
  - `seaborn` (optional, for nicer plots)
  - `jupyter` (if using notebooks)
- LSEG/Refintiv license for data

Install requirements:
```bash
pip install -r requirements.txt
```

## License
This project is released under the MIT License.  
See the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or analysis in your research, please cite:

> Mathis Jander (2025). *Carbon-Aware Portfolio Optimization: Efficient Frontier Analysis of Low-Emission S&P 500 Constituents*. University of Twente.


## Acknowledgement

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.