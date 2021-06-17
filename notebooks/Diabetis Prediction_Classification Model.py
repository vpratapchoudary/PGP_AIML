import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

def solution():
    train=pd.read_csv('res/diabetes_train.csv')
    test=pd.read_csv('res/diabetes_test.csv')
    #print(train.head())
    #print(test.head())
    '''Write your code here....
    .......
    .......
    '''
    Xtrain = train.iloc[:,:-1]
    ytrain = train.iloc[:,-1]
    Xtest = test.iloc[:,:-1]
    ytest = test.iloc[:,-1]
    #print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)

    params = {'learning_rate':[0.3,0.6,0.9,1], 'n_estimators':[50,100]}
    ada_gsv = GridSearchCV(estimator=AdaBoostClassifier(random_state=42), param_grid=params, cv=5, scoring='accuracy')
    ada_gsv.fit(Xtrain,ytrain)
    print(ada_gsv.best_params_)
    res1_lr = ada_gsv.best_params_['learning_rate']
    res2_nest = ada_gsv.best_params_['n_estimators']

    ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.3, random_state=42)
    ada_clf.fit(Xtrain, ytrain)
    ada_predict = ada_clf.predict(Xtest)
    accuracy_ada = round(accuracy_score(ytest, ada_predict)*100,2)
    cmat_ada = confusion_matrix(ytest, ada_predict)
    print('AdaBoost: Accuracy:{}, cmat:{}'.format(accuracy_ada,cmat_ada))
    tn_ada, tp_ada, fn_ada, fp_ada = cmat_ada[0,0], cmat_ada[1,1], cmat_ada[1,0], cmat_ada[0,1]
    sensitivity_ada = round(tp_ada/float(tp_ada+fn_ada)*100,2)
    specificity_ada = round(tn_ada/float(tn_ada+fp_ada)*100,2)
    print('AdaBoost Sensitivity:{}, Specificity:{}'.format(sensitivity_ada,specificity_ada))

    gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.3, random_state=42)
    gb_clf.fit(Xtrain, ytrain)
    gb_predict = gb_clf.predict(Xtest)
    accuracy_gb = round(accuracy_score(ytest, gb_predict)*100,2)
    cmat_gb = confusion_matrix(ytest, gb_predict)
    print('GradientBoost: Accuracy:{}, cmat:{}'.format(accuracy_gb,cmat_gb))
    tn_gb, tp_gb, fn_gb, fp_gb = cmat_gb[0,0], cmat_gb[1,1], cmat_gb[1,0], cmat_gb[0,1]
    sensitivity_gb = round(tp_gb/float(tp_gb+fn_gb)*100,2)
    specificity_gb = round(tn_gb/float(tn_gb+fp_gb)*100,2)
    print('GradientBoost Sensitivity:{}, Specificity:{}'.format(sensitivity_gb,specificity_gb))

    if accuracy_ada>accuracy_gb:
        res3_acc=accuracy_ada
    else:
        res3_acc=accuracy_gb

    if sensitivity_ada>sensitivity_gb:
        res4_sen=sensitivity_ada
    else:
        res4_sen=sensitivity_gb

    if specificity_ada>specificity_gb:
        res5_spe=specificity_ada
    else:
        res5_spe=specificity_gb

    # Creating a list of the answer
    result=[res1_lr, res2_nest, res3_acc, res4_sen, res5_spe]
    print(result)


    # Finally create a dataframe of the final output  and write the output to output.csv

    result=pd.DataFrame(result)

    # writing output to output.csv
    result.to_csv('output/output.csv', header=False, index=False)