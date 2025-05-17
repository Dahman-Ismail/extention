# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt.matplotlib inline
# # import seaborn as sns
# # from sklearn import metrics 
# # import warnings
# # warnings.filterwarnings('ignore')

# # #Loading data into dataframe

# # data = pd.read_csv("phishing.csv")
# # data.head()

# # #Shape of dataframe

# # data.shape

# # #Listing the features of the dataset

# # data.columns

# # data.info()

# # data.nunique()

# # data = data.drop(['Index'],axis = 1)

# # data.describe().T

# # plt.figure(figsize=(15,15))
# # sns.heatmap(data.corr(), annot=True)
# # plt.show()


# # #pairplot for particular features

# # df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]
# # sns.pairplot(data = df,hue="class",corner=True);

# # data['class'].value_counts().plot(kind='pie',autopct='%1.2f%%')
# # plt.title("Phishing Count")
# # plt.show()


# # # Splitting the dataset into dependant and independant fetature
# # y = data['class']
# # X = data.drop('class',axis=1)
# # X.shape, y.shape


# # # Creating holders to store the model performance results
# # ML_Model = []
# # accuracy = []
# # f1_score = []
# # recall = []
# # precision = []

# # #function to call for storing the results
# # def storeResults(model, a,b,c,d):
# #   ML_Model.append(model)
# #   accuracy.append(round(a, 3))
# #   f1_score.append(round(b, 3))
# #   recall.append(round(c, 3))
# #   precision.append(round(d, 3))


# # # Splitting the dataset into train and test sets: 80-20 split

# # from sklearn.model_selection import train_test_split

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# # X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # # Gradient Boosting Classifier Model
# # from sklearn.ensemble import GradientBoostingClassifier

# # # instantiate the model
# # gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# # # fit the model 
# # gbc.fit(X_train,y_train)


# # #predicting the target value from the model for the samples
# # y_train_gbc = gbc.predict(X_train)
# # y_test_gbc = gbc.predict(X_test)


# # #computing the accuracy, f1_score, Recall, precision of the model performance

# # acc_train_gbc = metrics.accuracy_score(y_train,y_train_gbc)
# # acc_test_gbc = metrics.accuracy_score(y_test,y_test_gbc)
# # print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
# # print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
# # print()

# # f1_score_train_gbc = metrics.f1_score(y_train,y_train_gbc,average='macro')
# # f1_score_test_gbc = metrics.f1_score(y_test,y_test_gbc,average='macro')
# # print("Gradient Boosting Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
# # print("Gradient Boosting Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
# # print()

# # recall_score_train_gbc = metrics.recall_score(y_train,y_train_gbc,average='macro')
# # recall_score_test_gbc =  metrics.recall_score(y_test,y_test_gbc,average='macro')
# # print("Gradient Boosting Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
# # print("Gradient Boosting Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
# # print()

# # precision_score_train_gbc = metrics.precision_score(y_train,y_train_gbc,average='macro')
# # precision_score_test_gbc = metrics.precision_score(y_test,y_test_gbc,average='macro')
# # print("Gradient Boosting Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
# # print("Gradient Boosting Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))


# # #computing the classification report of the model

# # print(metrics.classification_report(y_test, y_test_gbc))


# # training_accuracy = []
# # test_accuracy = []
# # # try learning_rate from 0.1 to 0.9
# # depth = range(1,10)
# # for n in depth:
# #     forest_test =  GradientBoostingClassifier(learning_rate = n*0.1)

# #     forest_test.fit(X_train, y_train)
# #     # record training set accuracy
# #     training_accuracy.append(forest_test.score(X_train, y_train))
# #     # record generalization accuracy
# #     test_accuracy.append(forest_test.score(X_test, y_test))
    

# # #plotting the training & testing accuracy for n_estimators from 1 to 50
# # plt.figure(figsize=None)
# # plt.plot(depth, training_accuracy, label="training accuracy")
# # plt.plot(depth, test_accuracy, label="test accuracy")
# # plt.ylabel("Accuracy")  
# # plt.xlabel("learning_rate")
# # plt.legend();


# # training_accuracy = []
# # test_accuracy = []
# # # try learning_rate from 0.1 to 0.9
# # depth = range(1,10,1)
# # for n in depth:
# #     forest_test =  GradientBoostingClassifier(max_depth=n,learning_rate = 0.7)

# #     forest_test.fit(X_train, y_train)
# #     # record training set accuracy
# #     training_accuracy.append(forest_test.score(X_train, y_train))
# #     # record generalization accuracy
# #     test_accuracy.append(forest_test.score(X_test, y_test))
    

# # #plotting the training & testing accuracy for n_estimators from 1 to 50
# # plt.figure(figsize=None)
# # plt.plot(depth, training_accuracy, label="training accuracy")
# # plt.plot(depth, test_accuracy, label="test accuracy")
# # plt.ylabel("Accuracy")  
# # plt.xlabel("max_depth")
# # plt.legend();

# # #storing the results. The below mentioned order of parameter passing is important.

# # storeResults('Gradient Boosting Classifier',acc_test_gbc,f1_score_test_gbc,
# #              recall_score_train_gbc,precision_score_train_gbc)


# # # Decision Tree Classifier model 
# # from sklearn.tree import DecisionTreeClassifier

# # # instantiate the model 
# # tree = DecisionTreeClassifier(max_depth=30)

# # # fit the model 
# # tree.fit(X_train, y_train)


# # #predicting the target value from the model for the samples

# # y_train_tree = tree.predict(X_train)
# # y_test_tree = tree.predict(X_test)

# # #computing the accuracy, f1_score, Recall, precision of the model performance

# # acc_train_tree = metrics.accuracy_score(y_train,y_train_tree)
# # acc_test_tree = metrics.accuracy_score(y_test,y_test_tree)
# # print("Decision Tree : Accuracy on training Data: {:.3f}".format(acc_train_tree))
# # print("Decision Tree : Accuracy on test Data: {:.3f}".format(acc_test_tree))
# # print()

# # f1_score_train_tree = metrics.f1_score(y_train,y_train_tree)
# # f1_score_test_tree = metrics.f1_score(y_test,y_test_tree)
# # print("Decision Tree : f1_score on training Data: {:.3f}".format(f1_score_train_tree))
# # print("Decision Tree : f1_score on test Data: {:.3f}".format(f1_score_test_tree))
# # print()

# # recall_score_train_tree = metrics.recall_score(y_train,y_train_tree)
# # recall_score_test_tree = metrics.recall_score(y_test,y_test_tree)
# # print("Decision Tree : Recall on training Data: {:.3f}".format(recall_score_train_tree))
# # print("Decision Tree : Recall on test Data: {:.3f}".format(recall_score_test_tree))
# # print()

# # precision_score_train_tree = metrics.precision_score(y_train,y_train_tree)
# # precision_score_test_tree = metrics.precision_score(y_test,y_test_tree)
# # print("Decision Tree : precision on training Data: {:.3f}".format(precision_score_train_tree))
# # print("Decision Tree : precision on test Data: {:.3f}".format(precision_score_test_tree))

# # #computing the classification report of the model

# # print(metrics.classification_report(y_test, y_test_tree))

# # training_accuracy = []
# # test_accuracy = []
# # # try max_depth from 1 to 30
# # depth = range(1,30)
# # for n in depth:
# #     tree_test = DecisionTreeClassifier(max_depth=n)

# #     tree_test.fit(X_train, y_train)
# #     # record training set accuracy
# #     training_accuracy.append(tree_test.score(X_train, y_train))
# #     # record generalization accuracy
# #     test_accuracy.append(tree_test.score(X_test, y_test))
    

# # #plotting the training & testing accuracy for max_depth from 1 to 30
# # plt.plot(depth, training_accuracy, label="training accuracy")
# # plt.plot(depth, test_accuracy, label="test accuracy")
# # plt.ylabel("Accuracy")  
# # plt.xlabel("max_depth")
# # plt.legend();

# # #storing the results. The below mentioned order of parameter passing is important.

# # storeResults('Decision Tree',acc_test_tree,f1_score_test_tree,
# #              recall_score_train_tree,precision_score_train_tree)


# # # Random Forest Classifier Model
# # from sklearn.ensemble import RandomForestClassifier

# # # instantiate the model
# # forest = RandomForestClassifier(n_estimators=10)

# # # fit the model 
# # forest.fit(X_train,y_train)


# # #predicting the target value from the model for the samples
# # y_train_forest = forest.predict(X_train)
# # y_test_forest = forest.predict(X_test)

# # #computing the accuracy, f1_score, Recall, precision of the model performance

# # acc_train_forest = metrics.accuracy_score(y_train,y_train_forest)
# # acc_test_forest = metrics.accuracy_score(y_test,y_test_forest)
# # print("Random Forest : Accuracy on training Data: {:.3f}".format(acc_train_forest))
# # print("Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
# # print()

# # f1_score_train_forest = metrics.f1_score(y_train,y_train_forest)
# # f1_score_test_forest = metrics.f1_score(y_test,y_test_forest)
# # print("Random Forest : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
# # print("Random Forest : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
# # print()

# # recall_score_train_forest = metrics.recall_score(y_train,y_train_forest)
# # recall_score_test_forest = metrics.recall_score(y_test,y_test_forest)
# # print("Random Forest : Recall on training Data: {:.3f}".format(recall_score_train_forest))
# # print("Random Forest : Recall on test Data: {:.3f}".format(recall_score_test_forest))
# # print()

# # precision_score_train_forest = metrics.precision_score(y_train,y_train_forest)
# # precision_score_test_forest = metrics.precision_score(y_test,y_test_tree)
# # print("Random Forest : precision on training Data: {:.3f}".format(precision_score_train_forest))
# # print("Random Forest : precision on test Data: {:.3f}".format(precision_score_test_forest))

# # #computing the classification report of the model

# # print(metrics.classification_report(y_test, y_test_forest))

# # training_accuracy = []
# # test_accuracy = []
# # # try max_depth from 1 to 20
# # depth = range(1,20)
# # for n in depth:
# #     forest_test =  RandomForestClassifier(n_estimators=n)

# #     forest_test.fit(X_train, y_train)
# #     # record training set accuracy
# #     training_accuracy.append(forest_test.score(X_train, y_train))
# #     # record generalization accuracy
# #     test_accuracy.append(forest_test.score(X_test, y_test))
    

# # #plotting the training & testing accuracy for n_estimators from 1 to 20
# # plt.figure(figsize=None)
# # plt.plot(depth, training_accuracy, label="training accuracy")
# # plt.plot(depth, test_accuracy, label="test accuracy")
# # plt.ylabel("Accuracy")  
# # plt.xlabel("n_estimators")
# # plt.legend();

# # #storing the results. The below mentioned order of parameter passing is important.

# # storeResults('Random Forest',acc_test_forest,f1_score_test_forest,
# #              recall_score_train_forest,precision_score_train_forest)

# # #creating dataframe
# # result = pd.DataFrame({ 'ML Model' : ML_Model,
# #                         'Accuracy' : accuracy,
# #                         'f1_score' : f1_score,
# #                         'Recall'   : recall,
# #                         'Precision': precision,
# #                       })

# # # dispalying total result
# # result

# # #Sorting the datafram on accuracy
# # sorted_result=result.sort_values(by=['Accuracy', 'f1_score'],ascending=False).reset_index(drop=True)

# # # dispalying total result
# # sorted_result
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Create a DataFrame from the sorted results
# # data = {
# #     'ML Model': [
# #         'Gradient Boosting Classifier',
# #         'Random Forest',
# #         'Decision Tree',
# # '
# #     ],
# #     'Accuracy': [0.974, 0.967, 0.964, 0.963, 0.962, 0.956, 0.934, 0.605],
# #     'f1_score': [0.974, 0.971, 0.968, 0.963, 0.966, 0.961, 0.941, 0.454],
# #     'Recall': [0.988, 0.993, 0.980, 0.984, 0.991, 0.991, 0.943, 0.292],
# #     'Precision': [0.989, 0.990, 0.965, 0.984, 0.993, 0.989, 0.927, 0.997]
# # }

# # df = pd.DataFrame(data)

# # # Set 'ML Model' as index
# # df.set_index('ML Model', inplace=True)

# # # Plot the scores for each model
# # fig, ax = plt.subplots(figsize=(10, 10))
# # df.plot(kind='bar', ax=ax)
# # ax.set_xticklabels(df.index, rotation=45, ha='right')
# # ax.set_ylim([0, 1])  # Assuming the scores range from 0 to 1
# # ax.set_xlabel('Model')
# # ax.set_ylabel('Score')
# # ax.set_title('Model Scores')
# # plt.legend(loc='lower right')
# # plt.show()

# # import pickle
# # # dump information to that file
# # pickle.dump(gbc, open('newmodel.pkl', 'wb'))

# # gbc = pickle.load(open("newmodel.pkl","rb"))

# # import extractUrl
# # url="http://8csdg3iejj.lilagoraj.pl/"
# # #can provide any URL. this URL was taken from PhishTank
# # obj = FeatureExtraction(url)
# # x = np.array(obj.getFeaturesList()).reshape(1,30)
# # print("Feature Array: ",x)
# # y_pro_phishing = gbc.predict_proba(x)[0,0]
# # y_pro_non_phishing = gbc.predict_proba(x)[0,1]
# # print(y_pro_phishing, y_pro_non_phishing,"\n")
# # y_pred =gbc.predict(x)[0]
# # print("Prediction = ",y_pred)
# # if y_pred==1:
# #   print("It is a safe website")
# # else:
# #   print("Caution! Suspicious website detected")



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics 
# import warnings
# warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split

# # Load data
# data = pd.read_csv("phishing.csv")
# data = data.drop(['Index'], axis=1)

# # Splitting features and labels
# y = data['class']
# X = data.drop('class', axis=1)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Holders for results
# ML_Model = []
# accuracy = []
# f1_score = []
# recall = []
# precision = []

# def storeResults(model, a, b, c, d):
#     ML_Model.append(model)
#     accuracy.append(round(a, 3))
#     f1_score.append(round(b, 3))
#     recall.append(round(c, 3))
#     precision.append(round(d, 3))

# # ------------------- Gradient Boosting -------------------
# from sklearn.ensemble import GradientBoostingClassifier
# gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
# gbc.fit(X_train, y_train)

# y_test_gbc = gbc.predict(X_test)
# storeResults('Gradient Boosting', 
#              metrics.accuracy_score(y_test, y_test_gbc),
#              metrics.f1_score(y_test, y_test_gbc),
#              metrics.recall_score(y_train, gbc.predict(X_train)),
#              metrics.precision_score(y_train, gbc.predict(X_train)))

# # ------------------- Decision Tree -------------------
# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(max_depth=30)
# tree.fit(X_train, y_train)

# y_test_tree = tree.predict(X_test)
# storeResults('Decision Tree', 
#              metrics.accuracy_score(y_test, y_test_tree),
#              metrics.f1_score(y_test, y_test_tree),
#              metrics.recall_score(y_train, tree.predict(X_train)),
#              metrics.precision_score(y_train, tree.predict(X_train)))

# # ------------------- Random Forest -------------------
# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(n_estimators=10)
# forest.fit(X_train, y_train)

# y_test_forest = forest.predict(X_test)
# storeResults('Random Forest', 
#              metrics.accuracy_score(y_test, y_test_forest),
#              metrics.f1_score(y_test, y_test_forest),
#              metrics.recall_score(y_train, forest.predict(X_train)),
#              metrics.precision_score(y_train, forest.predict(X_train)))

# # ------------------- Result Summary -------------------
# result = pd.DataFrame({
#     'ML Model': ML_Model,
#     'Accuracy': accuracy,
#     'f1_score': f1_score,
#     'Recall': recall,
#     'Precision': precision
# })

# sorted_result = result.sort_values(by=['Accuracy', 'f1_score'], ascending=False).reset_index(drop=True)
# print(sorted_result)

# # ------------------- Plotting -------------------
# sorted_result.set_index('ML Model', inplace=True)
# fig, ax = plt.subplots(figsize=(10, 6))
# sorted_result.plot(kind='bar', ax=ax)
# ax.set_xticklabels(sorted_result.index, rotation=45, ha='right')
# ax.set_ylim([0, 1])
# ax.set_xlabel('Model')
# ax.set_ylabel('Score')
# ax.set_title('Model Scores')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load and prepare data
data = pd.read_csv("phishing.csv")
data = data.drop(['Index'], axis=1)

# Features and target
X = data.drop('class', axis=1)
y = data['class']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train Gradient Boosting
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
gbc.fit(X_train, y_train)
pickle.dump(gbc, open("model_gbc.pkl", "wb"))

# ------------------------
# Train Random Forest
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
pickle.dump(forest, open("model_forest.pkl", "wb"))

# ------------------------
# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=30)
tree.fit(X_train, y_train)
pickle.dump(tree, open("model_tree.pkl", "wb"))

# ------------------------
# Evaluate
models = {
    "Gradient Boosting": gbc,
    "Random Forest": forest,
    "Decision Tree": tree,
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print()
