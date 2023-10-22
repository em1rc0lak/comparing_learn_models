from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd



# necessary data arrangements
training_file = "training_set.csv"
training_set_pd = pd.read_csv(training_file)    

test_file = "test_set.csv"
test_set_pd = pd.read_csv(test_file)


training_gender = list(training_set_pd["Sex"])
training_measurements = training_set_pd.drop(columns=["Sex"]).values.tolist()
test_measurements = test_set_pd.drop(columns=["Sex"]).values.tolist()

correct_prediciton = list(test_set_pd["Sex"])

#function to test different classifiers
def get_prediction_rate(obj, train_set1 , train_set2 , test_set, correct_set):
    obj = obj.fit(train_set1,train_set2)
    obj_prediction = obj.predict(test_set)
    
    correct_predict = 0
    wrong_predict = 0
    i = 0
    while i < len(obj_prediction):
        if obj_prediction[i] == correct_set[i]:
            correct_predict += 1
        else :
            wrong_predict += 1
        i+=1

    prediction_rate = correct_predict/(correct_predict+wrong_predict)*100
    return prediction_rate


tree = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=40)
svc = SVC(gamma="auto")

tree_prediction_rate = get_prediction_rate(tree, training_measurements , training_gender , test_measurements , correct_prediciton)
knn_prediction_rate = get_prediction_rate(knn , training_measurements , training_gender , test_measurements , correct_prediciton)
svc_prediction_rate = get_prediction_rate(svc , training_measurements , training_gender , test_measurements , correct_prediciton)

print(tree_prediction_rate)
print(knn_prediction_rate)
print(svc_prediction_rate)

