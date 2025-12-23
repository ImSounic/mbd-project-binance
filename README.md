# README  
## Liquidity, Volatility, and Shock Propagation in Cryptocurrency Markets  

**Dataset:** Binance Full Cryptocurrency Trading History  
**Source:** https://www.kaggle.com/datasets/jorijnsmit/binance-full-history  

---

## 1. Project Overview

This project analyzes **cryptocurrency market microstructure** using high-frequency trade-level data from Binance. The focus is on understanding how **liquidity**, **volatility**, and **trading intensity shocks** behave across different cryptocurrency tiers and market regimes.

Bitcoin (BTC) and Ethereum (ETH) are treated as benchmark assets to study spillover effects and cross-market dependencies among smaller altcoins.

---

## 2. Research Questions

### RQ1: Liquidity and Volatility Under Market Stress  
**To what extent do liquidity and volatility vary across different cryptocurrency tiers (Large-Cap vs. Small-Cap) during periods of extreme market stress?**

- Compare liquidity and volatility metrics across market-cap tiers  
- Identify periods of extreme market stress (e.g., volatility spikes, crashes)  
- Examine whether small-cap cryptocurrencies are disproportionately affected  

---

### RQ2: Shock Propagation from Bitcoin and Ethereum  
**How do shocks in the trading intensity (number of trades) of Bitcoin and Ethereum propagate to the liquidity and volatility of smaller altcoins, and what is the typical time lag for this influence?**

- Identify shocks in BTC and ETH trading intensity  
- Measure altcoin responses in liquidity and volatility  
- Estimate lead–lag relationships and spillover dynamics  

---

### RQ3: Hyper-Sensitivity to Bitcoin Price Movements  
**Which specific cryptocurrency pairs exhibit “Hyper-Sensitivity” to Bitcoin’s price movements, and does this sensitivity increase during bearish market regimes compared to bullish ones (or vice versa)?**

- Estimate rolling sensitivity (beta) of altcoins relative to BTC  
- Classify market regimes into bullish and bearish phases  
- Analyze regime-dependent amplification of BTC influence  

---

## 3. Dataset Description

### 3.1 Data Source

The dataset consists of **full historical trade-level data** from the Binance exchange. Each file corresponds to a single trading pair (e.g., `BTCUSDT`, `ETHUSDT`, `ADAUSDT`) and contains all executed trades over time.

---

### 3.2 Key Variables

| Column Name        | Description |
|--------------------|-------------|
| `id`               | Unique trade identifier |
| `price`            | Executed trade price |
| `qty`              | Quantity of base asset traded |
| `quote_qty`        | Quantity in quote currency |
| `time`             | Trade timestamp (milliseconds) |
| `is_buyer_maker`   | Indicator of trade direction |
| `is_best_match`    | Trade matching flag |

---

## 4. Asset Classification

Cryptocurrencies are grouped by **market capitalization**:

- **Large-Cap:** Bitcoin (BTC), Ethereum (ETH)  
- **Mid-Cap:** Cryptocurrencies ranked approximately 10–30 by market cap  
- **Small-Cap:** Lower-liquidity altcoins  

Market capitalization information may be merged from external sources (e.g., CoinMarketCap).

---

## 5. Derived Metrics

### Liquidity Measures
- Trade count  
- Trading volume  
- Amihud illiquidity measure  
- Buyer-initiated vs. seller-initiated trade ratios  

### Volatility Measures
- Log returns  
- Realized volatility  
- Absolute return volatility  

### Shock Identification
- Extreme changes in trade count  
- High-percentile trading intensity events  
- Structural breaks in BTC and ETH activity  

---

## 6. Methodology Overview

- **Time aggregation:** 1-minute, 5-minute, or hourly intervals  
- **Market stress identification:** volatility thresholds and drawdowns  
- **Spillover analysis:** Vector Autoregression (VAR) and impulse responses  
- **Sensitivity analysis:** rolling regressions and regime-based comparisons  

---

## 7. Expected Contributions

- Evidence of asymmetric liquidity degradation during market stress  
- Quantification of time-lagged spillovers from BTC and ETH to altcoins  
- Identification of cryptocurrencies with extreme sensitivity to BTC price movements  

---

