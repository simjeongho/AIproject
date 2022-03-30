
from random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
               31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
               35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
               10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
               500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
               700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
print(fish_data)

# np의 column_stack 함수를 이용하여 생성
np.column_stack(([1, 2, 3], [4, 5, 6]))


# np의 column_Stack 함수는 각각의 데이터를 매칭시켜준다.
fish_data = np.column_stack((fish_length, fish_weight))
print("np_column stack 사용하여 데이터 매칭")
print(fish_data)

print(fish_data[:5])

print(np.ones(5))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

print(fish_target)

# Converting a Python List to a Numpy Array
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# Data shuffling
# The random.seed() function sets the seed needed to generate random numbers
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

# Create train set
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
# 35까지는 training set 36부터는 test set
# Create test set
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
# 원래는 0과 1의 인덱스를 생성하여 1인 인덱스는 training data 0인 인덱스는 test data로 나눠야한다.
# 하지만 scikit-learn 의 train_test_Split함수를 사용하면 한 번에 나눌 수 있음
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
# 처음 2개는 입력 데이터 그 다음은 타깃 데이터가 나온다.

# 처음에 카테고리 별로 있는 데이터들을 정제하면 fish data라는 것이 나온다.
# 그 후 0과 1의 인덱스로 생성한 fish target 데이터들을 만들어준다.

# 그 후 data와 target 데이터들을 numpy 배열로 만들어준다.
# numpy.array()로 만들어진 배열은 파이썬의 리스트나 튜플로 만들어진 자료구조와 다른 구조를 갖는다.
# numpy.ndarray는 numpy.array()함수로 만들 수 있는 자료 형태로 shape, ndim, dtype, itemSize, size등 행렬의 형태를 갖는다.
print("train_test_split 함수로 만들어진 결과의 ndarry shape")
print(train_input.shape, test_input.shape)
# target 데이터들의 데이터 형태
print(train_target.shape, test_target.shape)
print(test_target)
# 하지만 train_test_split 함수에 아무 것도 주지 않으면 target의 비율 대로 섞이지 않는다.
# 따라서 stratify라는 매개변수를 주어 target의 비율대로 섞어줄 것을 명시한다.
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
print(train_target.shape, test_target.shape)
print("stratify 매개 변수 준 후")
print(test_target)  # 비율 대로 잘 섞였다.
print("print input")
print(train_input)
print("test input")
print(test_input)
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)  # kneighsbor 모델 knn 모델을 훈련시키는 함수는 fit
kn.score(test_input, test_target)  # test를 하는 것 score가 1이라면 모든 데이터를 맞췄다는 것
# 결과 예측
print("knn training 후 예상치와 답지")
kn.predict(test_input)  # test를 어떻게 하나염


# training data는 문제와 같은 것 target data는 답지와 같은 것

# 그래프를 그릴 거에여
plt.scatter(train_input[:, 0], train_input[:, 1])  # x좌표, y좌표
# marker 매개변수는 모양을 지정합니다. 특정한 값만 다른 모양으로 그래프를 그리겠다.
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# kneighbors함수는 거리와 index들을 반환한다.
distances, indexes = kn.kneighbors([[25, 150]])
# 다시 그래프를 그린 후 데이터 하나를 특별하게 표시해준 후
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
# marker='D'로 지정하면 산점도를 마름모로 그립니다.아까 받은 인덱스를 x좌표, y좌표로 준다.
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 거리를 측정해봤는데 이상함
print("train input [index]의 데이터")
print(train_input[indexes])
print("train input[index]정답은??")
print(train_target[indexes])
# 아까 kneighbors 메소드가 반환한 distances indexes들과의 거리가 저장되어있다.
print("distances")
print(distances)
# 거리 비율이 이상함 각 데이터들의 범위가 다르기 떄문이다.
# x 출을 y축과 동일하게 1000으로 설정해준다. xlim함수를 이용해서
# 그러면 x축은 엄청 작기 떄문에 데이터들을 찾는데 고려되지 못한다. 두 특성의 scale이 다르기 때문이다.
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 그래서 하는 것이 data preprocessing (전처리)

# 가장 흔하게 할 수 있는 전처리 방법은 표준점수 각 값이 평균에서 표준 편차의 몇 배만큼 떨어져 있는지를 나타낸다.
# 평균을 빼고 표준 편차를 나눠준다.
mean = np.mean(train_input, axis=0)  # 평균
std = np.std(train_input, axis=0)  # 표준편차
print(mean, std)

train_scaled = (train_input - mean) / std  # 정규화를 해줬다.
new = ([25, 150] - mean) / std  # 새로운 데이터 또한 정규화 과정을 거친 후 넣어준다.
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 아까의 그래프와 비슷하나 x축과 y축의 범위가 달라짐
# 다시 훈련
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)

print(kn.predict([new]))
# 제대로 예측

# kneighbors 메소드는 distances와 indexes배열을 리턴한다.
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 다시 그리면 정상적으로 나온다.
