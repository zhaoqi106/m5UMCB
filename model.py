import numpy as np
import random
import os
import pandas as pd

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout,MaxPooling1D,Attention,Flatten,Input
from tensorflow.keras.layers import Conv1D,LSTM,Bidirectional,concatenate
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

splitcharBy=3
overlap_interval=1
window = 2

def cal_base(y_true, y_pred):
    y_pred_positive = np.round(np.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)

    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP )
    return SP


m5U_list_train=list(open(r'm5U_CV.txt','r'))
len_seq = 41
num_in=len(m5U_list_train)
label=[]
feature =[]
random.shuffle(m5U_list_train)
for i in range(num_in):
    seq = str(m5U_list_train[i][0:41])
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    feature.append(TempArray)
    if m5U_list_train[i][-2]=='1':
        label.append(1)
    else:
        label.append(0)



m5U_list_test=list(open(r'm5U_IND.txt','r'))

len_seq = 41
num_in=len(m5U_list_test)
Y_test=[]
X_test=[]

random.shuffle(m5U_list_test)
for i in range(num_in):
    seq = str(m5U_list_test[i][0:41])
    seq = seq.replace('-', 'X')
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    X_test.append(TempArray)
    if m5U_list_test[i][-2]=='1':
        Y_test.append(1)
    else:
        Y_test.append(0)


def load_embedding_vectors(filename):
    embedding_vectors = dict()
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_vectors[word] = vector

    f.close()
    return embedding_vectors


wordembedding = load_embedding_vectors("GloVe.txt")
feature = np.array(feature)
label = np.array(label)

def generate_arrays_from_feature(feature, label, batch_size):
    while 1:
        train_data = []
        train_label = []
        cnt = 0
        for i in range(len(feature)):
            temp_list = []
            for j in feature[i]:
                word = j
                if word in wordembedding.keys():
                    temp_list.append(wordembedding[word])
                else:
                    word = "<unk>"
                    temp_list.append(wordembedding[word])
            train_data.append(temp_list)
            train_label.append(label[i])
            while 1:
                train_data = []
                train_label = []
                cnt = 0
                for i in range(len(feature)):
                    temp_list = []
                    for j in feature[i]:
                        word = j
                        if word in wordembedding.keys():
                            temp_list.append(wordembedding[word])
                        else:
                            word = "<unk>"
                            temp_list.append(wordembedding[word])
                    train_data.append(temp_list)
                    train_label.append(label[i])
                    cnt += 1
                    if cnt == batch_size:
                        cnt = 0
                        yield (np.array(train_data), np.array(train_label))
                        train_data = []
                        train_label = []




# 5 fold CV
testAcc1 = 0
testTime1 = 0
seed = 100
kfold = 5
kf = KFold(n_splits = kfold, shuffle=True,random_state = seed)
label_predict = np.zeros(label.shape)
foldi = 0
label_pre = []
cvscores = []
cvroc_auc_score = []
cvmatthews_corrcoef = []
cvaccuracy_score = []
cvprecision_score = []
cvrecall_score = []
cvspe = []
cvsen = []
cvaccuracy_score1 = []
cvprecision_score1 = []
cvmatthews_corrcoef1 = []
cvspe1 = []
cvsen1 = []
roc_auc_scores_max = 0
thres = 0.9

batch_size=32
epochs=100

def ANN(optimizer='adam', neurons=64,kernel_size=5, batch_size=64, epochs=60, activation='relu', patience=50,drop=0.2,
        loss='categorical_crossentropy'):

    inp1 = Input(shape=(39, 300), dtype='float32')

    # model cnn +BiLSTM
    cnn1 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(inp1)
    cnn3 = MaxPooling1D(pool_size=2)(cnn1)
    drop1 = Dropout(0.2)(cnn3)

    cnn11 = Conv1D(64, 5, padding='same', strides=1,activation='relu')(inp1)
    cnn13 = MaxPooling1D(pool_size=2)(cnn11)
    drop11 = Dropout(0.2)(cnn13)

    cnn21 = Conv1D(64, 7, padding='same', strides=1,activation='relu')(inp1)
    cnn23 = MaxPooling1D(pool_size=2)(cnn21)
    drop21 = Dropout(0.2)(cnn23)
    cnn = concatenate([drop1,  drop11,drop21], axis=-1)

    Bi1 = Bidirectional(LSTM(64,dropout=0.2, recurrent_dropout=0.2,return_sequences = True))(cnn)#

    flat = Flatten()(Bi1)
    x1 = Dense(256, activation='relu')(flat)
    x2 = Dropout(0.5)(x1)
    x3 = Dense(128, activation='relu')(x2)
    x4 = Dropout(0.5)(x3)
    x5 = Dense(64, activation='relu')(x4)
    x6 = Dropout(0.5)(x5)
    final_output = Dense(1, activation='sigmoid')(x6)
    model = Model(inputs =inp1,outputs = final_output)      #

    model.summary()

    return  model


model = ANN()


model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=3e-4),
              metrics=['accuracy'])
# 防止过拟合
earlystopping=EarlyStopping(monitor='val_loss',patience=3,verbose=0, mode='min')
callbacks = [earlystopping]

model.fit_generator(generate_arrays_from_feature(feature,label, batch_size),
                    epochs=epochs,
                    steps_per_epoch=len(feature) // batch_size,
                    verbose=2,
                    validation_data=generate_arrays_from_feature(X_test,Y_test, batch_size),
                    validation_steps=len(X_test) // batch_size)


model.save('m5UMCB.h5')

model4 = load_model("m5UMCB.h5")
def Glove(m5U_list_test,num_in):
    splitcharBy = 3
    overlap_interval = 1
    wordembedding = load_embedding_vectors("GloVe.txt")
    test_data = []
    for i in range(num_in):
        seq = str(m5U_list_test[i][0:41])
        TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
        test_data.append(TempArray)
    test_data1 = []
    for i in range(len(test_data)):
        temp_list = []
        for j in test_data[i]:
            word = j
            if word in wordembedding.keys():
                temp_list.append(wordembedding[word])
            else:
                word = "<unk>"
                temp_list.append(wordembedding[word])
        test_data1.append(temp_list)
    feature_test = np.array(test_data1)
    return feature_test

m5U_list_test=list(open(r'm5U_IND.txt','r'))
len_seq = 41
num_in=len(m5U_list_test)

l_test=[]
random.shuffle(m5U_list_test)
for i in range(num_in):
    if m5U_list_test[i][-2]=='1':
        l_test.append(1)
    else:
        l_test.append(0)
l_test = np.array(l_test)
f_test_Glove=Glove(m5U_list_test,num_in)
label_predict_Glove = model4.predict(f_test_Glove)


label_pre = label_predict_Glove
label_predict = [0 if item<=0.5 else 1 for item in label_pre]
print("AUROC: %f " %roc_auc_score(l_test, label_pre))
print("AP: %f " %average_precision_score(l_test,label_pre))
print("MCC: %f " %matthews_corrcoef(l_test,label_predict))
print( "ACC:  %f "  %accuracy_score(l_test,label_predict))
print("Precision: %f " %precision_score(l_test,label_predict))
print( "Recall:  %f "  %recall_score(l_test,label_predict))
Spe=specificity(l_test,label_predict)
print("specificity: ",round(Spe*100,2))

fpr, tpr, thre = roc_curve(l_test, label_pre)
fpr = pd.DataFrame(fpr)
fpr.to_csv("fpr.csv")

tpr = pd.DataFrame(tpr)
tpr.to_csv("tpr.csv")


precision, recall, thresholds = precision_recall_curve(l_test, label_pre)
aupr_score = auc(recall, precision)
plt.plot(recall, precision, lw='2', color='gray', closed=False, label='PR Curve(AUPR=%0.3f)' % aupr_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower left")
plt.show()