
import numpy as np
from sklearn.model_selection import train_test_split
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
print(train_target.shape, test_target.shape)
print("stratify 매개 변수 준 후")
print(test_target)  # 비율 대로 잘 섞였다.
