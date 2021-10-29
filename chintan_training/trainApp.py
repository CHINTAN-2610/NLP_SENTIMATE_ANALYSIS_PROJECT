import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from chintan_utils.utils import preprocess_training_data


class TrainApi:

    def __init__(self, stopWordsFilePath):
        self.stop_words = stopWordsFilePath

    def training_model(self, jsonFilePath, modelPath):
        self.jsonFilePath = jsonFilePath
        data_df = preprocess_training_data(self.jsonFilePath, self.stop_words)

        TfidfVect = TfidfVectorizer()  #initialize TFDTF
        TfidfVect.fit(data_df['text']) #fit the data to convert
        self.modelPath = modelPath

        # saving vector for prediciton
        with open(self.modelPath + '/vectorizer.pickle', 'wb') as f:
            pickle.dump(TfidfVect, f)

        #  train set and prediction set
        x = data_df['text']
        y = data_df['target']

        # splitting training and test data
        # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state= 42)

        x_vector = TfidfVect.transform(x)
        # x_test_vector = TfidfVect.transform(x_test)

        # training data using SVM ##
        # model = svm.SVC(C=1.0, kernel='gaussian', degree=3, gamma='auto')
        model = naive_bayes.MultinomialNB()
        model.fit(x_vector, y)

        # y_pred = model.predict(x_test_vector)
        # score1 = metrics.accuracy_score(y_test, y_pred)

        # save the model to disk
        with open(self.modelPath + '/modelForPrediction.sav', 'wb') as f:
            pickle.dump(model, f)

        return ("Success")
