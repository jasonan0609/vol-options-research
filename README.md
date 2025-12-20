# Volatility Research & Options Analytics Dashboard

## Overview
This repository tracks my personal journey of learning and implementing analytical skills in volatility and options trading, culminating in a modular research platform and unified dashboard for monitoring volatility dynamics, option-implied information, cross-asset relationships, and event-driven risk.

---

## Core Capabilities

### Volatility Analytics
- Historical volatility (multiple rolling windows)
- EWMA volatility for shock sensitivity
- Realized volatility diagnostics
- Event-based volatility comparisons (CPI, FOMC)

### Implied Volatility & Options
- Black–Scholes pricing, Greeks, and IV solver
- Implied volatility surface & term structure
- IV–RV spread analysis and carry intuition
- Volatility risk premium proxies

### Forecasting & Simulation
- Volatility forecasting models:
  - HAR
  - GARCH-family benchmarks
- Forecast evaluation (RMSE, bias, horizon analysis)
- Monte Carlo simulation using forecasted volatility
- Scenario and stress testing

### Event Impact Framework
- Scheduled macro events (CPI, FOMC, earnings)
- Event windows with abnormal returns
- Volatility changes and drawdowns
- Cross-asset comparison of reaction speed and magnitude

### Correlation & Regime Analysis
- Rolling cross-asset correlations
- Heatmaps and regime shifts
- Stress-period correlation behavior

### Options Strategy Backtesting
- Greeks-aware PnL attribution
- Transaction costs and re-hedging logic
- Simple volatility strategies (straddles, calendars)
- Event stress testing

### ML-Based Volatility Signals
- Feature engineering (lags, term structure, events)
- Predictive models for volatility
- Out-of-sample evaluation and diagnostics

---

## Research Dashboard
An integrated dashboard combining:
- Volatility estimators and forecasts  
- IV/RV spreads and term structure  
- Correlation regimes  
- Event impact analysis  
- Strategy performance and risk  

Designed for daily research workflows.

---

## Project Structure
data/ -> raw & processed market data  
core/ -> returns, volatility, correlations  
options/ -> Black-Scholes, IV surface, Greeks  
forecasting/ -> HAR, GARCH, ML models  
events/ -> event calendars & impact analytics  
backtesting/ -> options strategy engine  
simulation/ -> Monte Carlo & stress scenarios  
dashboard/ -> unified research interface  
utils/ -> plotting, configs, helpers  