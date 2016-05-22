import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb

def showHist(a):
    tmap = {}
    for i in a:
        if i in tmap:
            tmap[i] += 1
        else:
            tmap[i] = 1
    v = tmap.values()
    v.sort()
    plt.plot(range(len(v)),v)
    plt.show()
    dict= sorted(tmap.iteritems(), key=lambda d:d[1], reverse = True)
    print dict[0:10]

data = []
tfile = '../exp2_raw_data/train11w.data'
train = pd.read_csv(tfile,sep = '\t')

#preprocess
cateMap = {}
tmp = np.array(train[train['category_id']>0][['creative_id','category_id']])
for i,j in tmp:
    cateMap[i] = j

train['category_id'] = train['creative_id'].map(cateMap)
train = train.dropna(axis = 0)

#init train
x = np.array(train.drop(['qq','description','imp_time','pic_url','web_url', 'product_id','advertiser_id','series_id','creative_id','product_type','click_num', 'pos_id'], axis = 1))
y = np.array(train['click_num'])

xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)

# some model

if __name__ == '__main__':
    # clf = MultinomialNB(alpha = 0.1)
    # clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth= 20, min_samples_split = 100 , class_weight = 'balanced')
    clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    # clf = AdaBoostClassifier(n_estimators=350, learning_rate=0.03)
    #clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

    clf.fit(xTrain, yTrain)
    print clf.score(xTrain, yTrain)
    print clf.score(xTest, yTest)