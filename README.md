# 🍽️ Campus Canteen Demand & Food Waste Optimization

A Statistical Machine Learning forecasting and decision-support system built to reduce daily food waste and prevent meal shortages in university campus canteens (**UN-SDG 12: Responsible Consumption and Production**).

## Overview
Campus meal demand fluctuates sharply due to weekday class schedules, weekend student travel, examination weeks, rainfall surges, special menu days, and month-end budget reliance. This project models 730 days (2 academic years) of campus dining dynamics, benchmarks four regression models, calibrates adaptive **90% and 95% statistical prediction intervals**, and deploys an interactive **Streamlit** dashboard for daily kitchen planning.

### Key Results (146 Unseen Test Days)
- **Best Model:** Random Forest Regressor ($R^2 = 0.9570$, $\text{MAE} = 26.65\text{ meals}$, $\text{RMSE} = 34.97\text{ meals}$)
- **Uncertainty Calibration:** 93.84% empirical coverage at the 90% prediction interval using 5-fold Out-of-Fold conformal calibration
- **Waste Impact:** Cuts simulated food waste by up to **70.5%** and meal shortages by **88.7%–99.8%** compared to a fixed daily cooking quota

---

## Run locally by following the below steps:
- clone the repo
- pip install -r requirements.txt          //to install the required pacakges
- streamlit run app.py                   // To run the streamlit app

Streamlit App:
<img width="1449" height="832" alt="Screenshot 2026-11-12 at 7 05 37 PM" src="https://github.com/user-attachments/assets/c220a3cc-dc6a-4d91-8612-a3e440d21aa3" />
  
