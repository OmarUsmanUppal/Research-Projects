
import numpy as np
import pandas as pd

'''=========READ FILE================'''

car_data = pd.read_csv("Datasets-UCI//6_car.csv", delimiter=';')
print("The column headings are : ", list(car_data.columns.values))
print("="*60)
num_of_attr = car_data.shape[1]-1
num_of_samples = car_data.shape[0]
class_types = car_data["label"].unique()
class_size = len(class_types)
print("Total attributes in the dataset are := ", num_of_attr)
print("="*60)
print("Total number of samples in the dataset are := ", num_of_samples)
print("="*60)
print("Total classes in the dataset are := ", len(class_types))
print("="*60)
print("Class Labels are := ", class_types)
print("="*60)

'''=========PANDAS to NUMPY================'''

X_attr = np.array(car_data.iloc[:, 0:num_of_attr])
y_label = np.array(car_data['label'])

'''=========TRAINING and TESTING SAMPLES SPLIT================'''

train_size = int(num_of_samples * 0.8)
test_size = num_of_samples - train_size

X_train = X_attr[:train_size, :]
X_test = X_attr[train_size:, :]

y_train = y_label[:train_size]
y_test = y_label[train_size:]

'''=========CALCULATE PRIOR PROBABILITY===p(0), p(1), p(2), p(3)================'''

lbl_counts = np.array([0, 0, 0, 0])
lbl_counts[0] = len([x for x in y_train if x == 0])
lbl_counts[1] = len([x for x in y_train if x == 1])
lbl_counts[2] = len([x for x in y_train if x == 2])
lbl_counts[3] = len([x for x in y_train if x == 3])

class_prior_p = lbl_counts / train_size

for i in range(class_size):
    print("Prior probability p(class x = {}) is = {}".format(i, class_prior_p[i]))

'''=========CALCULATE LIKELIHOOD FOR EACH FEATURE|CLASS === p(c|x)=============='''


def calc_likelihood(att_list, n_or_p):
    return_list = []
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for sample_i in range(len(att_list)):
        if att_list[sample_i] == n_or_p:
            if y_train[sample_i] == 0:
                count_0 += 1
            if y_train[sample_i] == 1:
                count_1 += 1
            if y_train[sample_i] == 2:
                count_2 += 1
            if y_train[sample_i] == 3:
                count_3 += 1

    return_list.append(count_0/lbl_counts[0])
    return_list.append(count_1/lbl_counts[1])
    return_list.append(count_2/lbl_counts[2])
    return_list.append(count_3/lbl_counts[3])
    return return_list

p_attr_p_ive = np.zeros((num_of_attr, class_size))
p_attr_n_ive = np.zeros((num_of_attr, class_size))

print("="*60)
for attr_i in range(num_of_attr):
    p_attr_p_ive[attr_i] = calc_likelihood(X_train[:, attr_i], 1)
    p_attr_n_ive[attr_i] = calc_likelihood(X_train[:, attr_i], -1)
    print("p(attr{}= 1|x=0) = {}  p(attr{}= 1|x=1) = {}  p(attr{}= 1|x=2) = {}  p(attr{}= 1|x=3) = {}".format(attr_i, p_attr_p_ive[attr_i][0], attr_i, p_attr_p_ive[attr_i][1], attr_i, p_attr_p_ive[attr_i][2], attr_i, p_attr_p_ive[attr_i][3]))
    print("p(attr{}=-1|x=0) = {}  p(attr{}=-1|x=1) = {}  p(attr{}=-1|x=2) = {}  p(attr{}=-1|x=3) = {}\n".format(attr_i, p_attr_n_ive[attr_i][0], attr_i, p_attr_n_ive[attr_i][1], attr_i, p_attr_n_ive[attr_i][2], attr_i, p_attr_n_ive[attr_i][3]))
print("="*60)

'''=========NAIVE BAYES CLASSIFIER IS TRAINED----SINCE EVIDENCE IS COMMON FOR ALL CLASSES SO IT IS IGNORED=========='''

'''=========USE OF TEST SET NOW TO CHECK THE ACCURACY OF THE CLASSIFIER=============='''
# p_attr_p_ive[21][4] :: P(Yes Attr|class)
# p_attr_n_ive[21][4] :: P(No Attr|class)
# class_prior_p[0-3] :: P(Class)
# X_test : Find the probability


def calc_post_pr(attr, cls_lbl):
    prob = class_prior_p[cls_lbl]
    for attr_j in range(num_of_attr):
        if attr[attr_j] == 1:
            prob = prob * p_attr_p_ive[attr_j][cls_lbl]
        else:
            prob = prob * p_attr_n_ive[attr_j][cls_lbl]

    return prob

# Below lines of code will predict the class of the Test Data

pred_class_test = []
for j in range(test_size):
    pred_class = np.zeros((1, 4))
    for i in range(len(lbl_counts)):
        pred_class[0][i] = calc_post_pr(X_test[j], i)

    pred_class_test.append((pred_class.argmax()))
    # print(pred_class.argmax())


# Below lines of code will predict the class of the Train Data
pred_class_train = []
for j in range(train_size):
    pred_class = np.zeros((1, 4))
    for i in range(len(lbl_counts)):
        pred_class[0][i] = calc_post_pr(X_train[j], i)

    pred_class_train.append((pred_class.argmax()))
    # print(pred_class.argmax())


# Below lines of code will find the accuracy of the prediction done
def accuracy(predicted_class, y_class):
    correct = 0
    for class_i in range(len(predicted_class)):
        if predicted_class[class_i] == y_class[class_i]:
            correct += 1
            # print("predicted label {} matches the orignial lable {}".format(pred_class_train[i], y_train[i]))
    return round(correct*100/len(predicted_class), 2)

print(accuracy(pred_class_train, y_train), "% Accuracy for train data")
print("="*60)
print(accuracy(pred_class_test, y_test), "% Accuracy for test data")
