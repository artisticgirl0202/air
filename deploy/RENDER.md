# 🍃 Air Quality Monitoring
본 프로젝트는 공공데이터(대기질 및 기상)를 바탕으로 대기 오염도를 모니터링하고 예측하는 웹 대시보드 및 REST API 서비스입니다.
Timescale/Postgres 데이터베이스에 적재된 시계열 데이터를 조회하며, 머신러닝/딥러닝 앙상블 모델(XGBoost, LSTM)을 통해 향후 대기질 예측 지표를 시각화하여 제공합니다.

## 🛠 기술 스택
Frontend: Streamlit
Backend: Python (FastAPI)
Database: PostgreSQL, TimescaleDB
Machine Learning: PyTorch (LSTM), XGBoost, Hugging Face Hub
Deployment: Render (Backend API - Docker), Streamlit Community Cloud (Frontend)


## ⚖️ License & Copyright

- 사전 협의 및 서면 동의 없는 무단 복제, 배포, 상업적 이용, 리버스 엔지니어링을 엄격히 금지합니다.
- **Copyright ⓒ 2026 Yula Jeong. All rights reserved.**
