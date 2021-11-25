# StockPrice_Prediction

### 목표
- 주식 데이터와 전력소비 데이터를 구조가 간단한 LSTM, CNN, Seq2Seq 부터 Transformer를 최적화한 informer까지 학습하여 예측 후 모델 별 성능 평가
- Linformer (2020)모델을 Time Series에서 사용할 수 있는 형태로 변환, 파라미터를 조정하며 성능평가
- Sequence Length, Label Length, Prediction Length set을 조절하여 성능 평가

### Datasets
1. AMD
2. NVIDIA
- 수집 기간 : 2019년 5월 1일 ~ 2021년 5월24일 (hourly)
- 일별 데이터 개수 : 7개 - UTC 시간 기준 13:30 ~ 19:30 / 14:30 ~ 20:30 (summer time)
- Target : close(한 시간이 끝나는 시점의 가격)
- 데이터 총 개수 : 3,635개 (Train : 2,949개 / Test : 686개)
<img width="490" alt="image" src="https://user-images.githubusercontent.com/62350977/143505958-421d3736-5b84-4eb0-ac85-5861e166e6c1.png">
3. American Electric Power

4. Dominion Virginia Power
- 출처 : Kagge
- 수집 기간 : 2005년 5월 1일 ~ 2019년 8월 2일 (hourly)
- Target : MW(Megawatts)
- 데이터 총 개수 : 116,185개 (Train : 93,484개 / Test : 22,701개)

### Experiment setting
<img width="800" alt="image" src="https://user-images.githubusercontent.com/62350977/143506153-004d4a51-150d-49e8-a2f0-2c93fea044f5.png">

### Run
```python
python main.py --model informer --gpu_id 0 --output_attention
```

### Metrics
- MSE
- RMSE
- MAE

### Experimental results
<img width="1175" alt="image" src="https://user-images.githubusercontent.com/62350977/143506219-f419ea20-5579-421b-a5fc-b60eebe18c18.png">

### References
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)

[Linformer: Self-attention with linear complexity](https://arxiv.org/abs/2006.04768)

https://github.com/zhouhaoyi/Informer2020

https://github.com/lucidrains/linformer
