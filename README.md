# Prediction-of-the-number-of-people-getting-on-and-off-the-bus-at-work-hours-in-Jeju-Island

### 교내 경진 대회 참여
 * Dacon에서 예전에 진행했던 제주도 퇴근시간 버스승차인원 예측을 주제로 교내 AI 경진대회가 진행되었다.



### Dacon 제주도 퇴근시간 버스승차인원 예측
* [Dacon 제주도 퇴근시간 버스승차인원 예측 링크](https://dacon.io/competitions/official/229255/overview/description)  



### 데이터 설명
* 해당 데이터에는 버스카드를 통해 결제를 한 경우에 대한 정류소 승, 하차 데이터로 모든 승차정보의 경우는 기록이 되어있지만, 스에서 하차를 할 때, 버스카드를 찍지 않는 경우, 해당 기록이 비어 있는 상태입니다. 따라서, 승차 인원수와 하차 인원수가 동일하지 않고 다소 차이가 있음을 미리 알려드립니다.
* 교내 경진대회의 경우에는 기존 경진대회에 있던 여러 데이터가 누락 된 데이터가 주어졌다.


### 제공 파일
1. train.csv – 2019년 9월 제주도의 각 날짜, 출근시간(6시~12시)의 버스 정류장별 승하차 인원, 퇴근시간(18시~20시)의 버스 정류장별 승차 인원이 기록되어 있습니다.
2. test.csv – 2019년 10월의 각 날짜, 출근시간(오전 6시~12시)의 버스 정류장별 승하차 인원이 기록되어 있습니다.
3. bus.csv - 버스카드별로 승하차 정보가 기록이 되어있습니다. 해당 데이터는 탑승 시간대가오전 6시부터 12시 사이인 경우만 있습니다.

### 깃허브 파일 설명
1. AI competition2 : lgbm 모델 사용, 많은 양의 데이터를 반복 학습시켜 public상에서 점수를 높게 받을 수 있게 설정
2. AI Competition Ver2 : lgbm 모델 사용, 학습시킬 컬럼을 늘리는데 주력 fold 교차검증 방식을 이용하여 과적합 방지
3. AI competition : k-fold 모델 구성 (학습 결과 판별을 위해 생성)
4. GitHub Competition : 공공데이터포털 데이터 중 공항 일일예상승객 openAPI를 이용해 새로운 데이터를 생성
5. Position : 공공데이터포털 데이터 중 제주특별자치도_버스정류소현황_20170831 파일을 이용하여 train, test정류소 이름에 해당하는 위도, 경도 생성 제주특별자치도_버스정류소현황_20170831만 사용할 경우 Null인 정류소가 생기기 때문에 제주데이터허브(https://www.jejudatahub.net/) API를 이용하여 Null인 정류소의 위도, 경도 생성


### 목표

* 여러 파생 변수들을 생성하는 것을 목표로 삼았다.
* 기존 경진대회에 있던 데이터를 복구하는 것을 목표로 삼았다.



### 진행 과정

* 이미 진행되었던 경진대회라 코드가 공유되어 있어 해당 코드를 분석했다.
* 기존 경진대회에 있던 데이터를 복구할 수 있는 방향으로 코드를 개선했다.
* Public에서 좋은 결과를 내더라도 Private에서 동일하게 좋은 결과를 낼 것이라는 보장이 없어, 2개의 파일 제출 중 하나는 과적합을 생각하지 않은 것과 다른 하나는 과적합을 고려한 것을 제출했다.
* 지난 경진대회에서는 Cross Validation을 적용하지 못한 점이 아쉬워 어떻게든 적용할 수 있도록 노력했다.

1. [기존 경진대회 코드 공유](https://dacon.io/competitions/official/229255/codeshare)
    

2. 데이터 복구
    2-1. [공공데이터포털](https://www.data.go.kr/data/15010850/fileData.do)의 데이터 중 제주특별자치도_버스정류소현황_20170831 파일을 이용하여 train, test정류소 이름에 해당하는 위도, 경도 생성. 해당 데이터만 사용할 경우 NULL인 정류소가 생기기 때문에 추가적인 데이터가 필요했다.
    2-2. [제주데이터허브](https://www.jejudatahub.net/data/view/data/612)의 API를 이용해 NULL에 해당하는 데이터의 정보를 획득했다. 
    2-3. [공공데이터포털](https://www.data.go.kr/data/15004024/openapi.do)의 데이터 중 공항 일일예상승객 openAPI를 이용해 새로운 데이터를 생성해 여러 파생 변수를 생성했다.
    (위도, 경도 데이터 복구 : Position.ipynb, 공항 일일예상승객 데이터 : GitHub Competition.ipynb)

3. 참고 사이트
    3-1. [학습 모델 선정 및 Cross Validation](https://hyemin-kim.github.io/2020/08/04/S-Python-sklearn4/#1-randomizedsearchcv)





### 결과

* 위도, 경도 관련 데이터를 어느정도 복구할 수 있었지만 같은 이름의 버스 정류장은 구분이 불가능 했다.(같은 이름의 버스 정류장인데 노선 방향이 다른 경우)

* 주요 관광지가 있는 경우 해당 관광지에 사람이 어느정도 방문했는지와 해당 유적지와 버스 정류장의 거리를 계산해서 모델을 학습시켰지만 정확도는 향상되지 않았다. (퇴근시간 버스 승차 인원 예측이므로 관련이 없어 보임)

* 공항 일일예상승객 데이터로 만든 파생 변수들은 정확도 향상에 유의미한 영향을 끼쳤다.



### 보완사항

* k-fold 교차 검증 코드를 작성하는데 많은 어려움이 있었다. (아래는 구현한 k-fold 코드로 RandomForestRegressor 이외에 LGBMRegressor, GradientBoostingRegressor, XGBRegressor, VotingRegressor, StackingRegressor 모델 구현 및 검증했다. AI competition.ipynb 코드 참고)
``` python
# Random Forest K-fold
from sklearn.ensemble import RandomForestRegressor

n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=1, shuffle=True)

i = 1
total_error = 0

X = np.array(train_data[input_var_0])
Y = np.array(y_train.values.ravel())

rfr_fold = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=1)

for train_index, test_index in kfold.split(X):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    rfr_fold_fit = rfr_fold.fit(x_train_fold, y_train_fold)
    rfr_fold_pred = rfr_fold_fit.predict(x_test_fold)
    error = np.sqrt(mean_squared_error(y_test_fold, rfr_fold_pred))
    print('Fold = {}, prediction score = {:.2f}'.format(i, error))
    total_error += error
    i+=1

print('---'*10)
print('Average Error: %s' % (total_error / n_splits))

# n_estimator : 200 => 30분 Average Error : 2.165 input_var_0

plot_feature_importances(rfr_fold_fit, train_data[input_var_0])
```
* 다양한 모델을 생성한 만큼 하이퍼 파라미터를 조정하는데 많은 시간이 걸렸다. 최종적으로는 LGBMRegressor를 단독으로 사용하고 반복횟수를 충분히 많이 주는 것의 정확도가 제일 높았다.

* LGBMRegressor를 생성하고 50000번 반복시켰을 때 커널이 죽는 현상이 발생했다. (20000번 반복 수행시킴)


* 반복 수행시킬 경우 rmse가 계속적으로 감소했다. epoch를 반복시킬 경우 어느정도 감소하는 것은 맞다고 생각하지만 커널 연결이 끊어질 때 까지 반복시킨 경우에도 감소했다. 이 부분에 대해서는 추가적으로 공부해야 할 필요성을 느꼈다.
...
[2100]	training's rmse: 1.31259
[2200]	training's rmse: 1.2926
[2300]	training's rmse: 1.27387
...
[19700]	training's rmse: 0.520722
[19800]	training's rmse: 0.51907
[19900]	training's rmse: 0.517365
[20000]	training's rmse: 0.515717

* 데이터가 커 한복 반복 수행할 때 마다 상당한 시간이 걸려 다양한 경우에 대해서 확인하지 못했다.

* 노트북 파일 명을 대충 지으니 나중에 어떤 파일인지 헷갈리는 사태가 발생했다. 각 파일 별로 기능을 한 번에 파악할 수 있도록 파일 명을 정해야 될 필요성을 느꼈다.



</br> </br> </br> </br>
***
## English












