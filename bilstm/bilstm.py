from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import progressbar as pb
import pickle


def get_model(maxlen):
    try:
        embedding_matrix = pickle.load(open('embedding_matrix.pkl','rb'))
        maxin = embedding_matrix.shape[1]
    except FileNotFoundError:
        print('\tLoading worddict')
        worddict = {}
        with open('worddict') as infile:
            for line in infile:
                sline = line.rstrip().split('\t')
                worddict[sline[0]] = int(sline[1])
        print('\tCreating embedding matrix')
        maxin = max(worddict.values()) + 1
        embedding_matrix = np.zeros((1, maxin, 100))
        print('\t\tLoading pre-trained embeddings')
        with open('glove.840B.300d.txt') as infile:
            for i, line in enumerate(infile):
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:101], dtype='float32')
                embedding_matrix[0,i] = coefs
        print('\t\tGenerating novel initializations')
        limit = np.sqrt(3/maxin)
        pretrained_len = i
        for i in range(maxin):
            if i < pretrained_len:
                continue
            else:
                embedding_matrix[0,i] = np.random.uniform(-limit, limit, 100)
        pickle.dump(embedding_matrix, open('embedding_matrix.pkl', 'wb'))
    print('\tCompiling model')
    inputs = Input((maxlen, ))
    emb = Embedding(maxin, 100, weights=embedding_matrix, input_shape=(maxlen,), mask_zero=True)(inputs)
    print(emb)
    bilstm = Bidirectional(LSTM(25, return_sequences=True, input_shape=(maxlen,100), dropout=0.3))(emb)
    print(bilstm)
    output1 = Dense(4, activation='softmax', input_shape=(maxlen,50), name="output1")(bilstm)
    print(output1)
    output2 = Dense(7, activation='softmax', input_shape=(maxlen,50), name="output2")(bilstm)
    print(output2)
    output3 = Dense(3, activation='softmax', input_shape=(maxlen,50), name="output3")(bilstm)
    print(output3)
    model = Model(inputs=inputs, outputs=[output1,
                                          output2,
                                          output3])
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1),
                  sample_weight_mode='temporal',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def parse_data_file(infile, data, labels, datatype, maxlen, maxtoknum):
    curdata = []
    curlabels1 = []
    curlabels2 = []
    curlabels3 = []
    print('\tReading file...')
    lines = infile.readlines()
    print('\tParsing file...')
    with pb.ProgressBar(max_val=len(lines)) as p:
        for i, line in enumerate(lines):
            rline = line.rstrip()
            if not rline:
                if len(curdata) > maxlen:
                    maxlen = len(curdata)
                if curdata and (maxtoknum == -1 or len(data[datatype]) < maxtoknum):
                    data[datatype].append(curdata)
                    labels[datatype][0].append(curlabels1)
                    labels[datatype][1].append(curlabels2)
                    labels[datatype][2].append(curlabels3)
                curdata = []
                curlabels1 = []
                curlabels2 = []
                curlabels3 = []
            else:
                sline = rline.split('\t')
                curdata.append(int(sline[0]) + 1)
                curlabels1.append([0,0,0,0])
                curlabels2.append([0,0,0,0,0,0,0])
                curlabels3.append([0,0,0])
                if sline[1] == '0':
                    curlabels1[-1][0] = 1
                else:
                    curlabels1[-1][int(sline[1])] = 100000
                if sline[2] == '0':
                    curlabels2[-1][0] = 1
                else:
                    curlabels2[-1][int(sline[2])] = 100000
                if sline[3] == '0':
                    curlabels3[-1][0] = 1
                else:
                    curlabels3[-1][int(sline[3])] = 100000
    return data, labels, maxlen


def get_data():
    data = {'training': [],
            'dev': [],
            'test': []}
    labels = {'training': [[], [], []],
              'dev': [[], [], []],
              'test': [[], [], []]}
    print('Creating training data')
    with open('training.conll') as infile:
        data, labels, maxlen = parse_data_file(infile, data, labels, 'training', 0, 100000)
    print('Creating dev data')
    with open('dev.conll') as infile:
        data, labels, maxlen = parse_data_file(infile, data, labels, 'dev', maxlen, 1000)
    print('Creating test data')
    with open('test.conll') as infile:
        data, labels, maxlen = parse_data_file(infile, data, labels, 'test', maxlen, 1000)
    data['training'] = pad_sequences(data['training'], maxlen=maxlen)
    labels['training'][0] = pad_sequences(labels['training'][0], maxlen=maxlen)
    labels['training'][1] = pad_sequences(labels['training'][1], maxlen=maxlen)
    labels['training'][2] = pad_sequences(labels['training'][2], maxlen=maxlen)
    data['dev'] = pad_sequences(data['dev'], maxlen=maxlen)
    labels['dev'][0] = pad_sequences(labels['dev'][0], maxlen=maxlen)
    labels['dev'][1] = pad_sequences(labels['dev'][1], maxlen=maxlen)
    labels['dev'][2] = pad_sequences(labels['dev'][2], maxlen=maxlen)
    data['test'] = pad_sequences(data['test'], maxlen=maxlen)
    labels['test'][0] = pad_sequences(labels['test'][0], maxlen=maxlen)
    labels['test'][1] = pad_sequences(labels['test'][1], maxlen=maxlen)
    labels['test'][2] = pad_sequences(labels['test'][2], maxlen=maxlen)
    return data, labels, maxlen


def main():
    print('Generating data')
    data, labels, maxlen = get_data()
    print('Defining model')
    model = get_model(maxlen)
    print(labels['training'][0].shape)
    print(labels['training'][0][0].shape)
    print(labels['dev'][0][0].shape)
    print('Fitting model')
    model.fit(data['training'],
              [(labels['training'][0]/100000).astype(int),
               (labels['training'][1]/100000).astype(int),
               (labels['training'][2]/100000).astype(int)
              ],
              sample_weight={'output1': np.argmax(labels['training'][0], axis=2),
                             'output2': np.argmax(labels['training'][1], axis=2),
                             'output3': np.argmax(labels['training'][2], axis=2)},
              epochs=100, batch_size=32,
              validation_data=(data['dev'],
                               [(labels['dev'][0]/100000).astype(int),
                                (labels['dev'][1]/100000).astype(int),
                                (labels['dev'][2]/100000).astype(int)
                               ]),
              callbacks=[EarlyStopping(monitor='val_loss',
                                       min_delta=0.00001,
                                       mode='min',
                                       patience=2)],
              verbose=1)
    test_preds = model.predict(data['test'],
                               verbose=1,
                               batch_size=512)
    true1 = np.argmax(labels['test'][0], axis=2).flatten()
    true2 = np.argmax(labels['test'][1], axis=2).flatten()
    true3 = np.argmax(labels['test'][2], axis=2).flatten()
    pred1 = np.argmax(pad_sequences(test_preds[0], maxlen=maxlen), axis=2).flatten()
    pred2 = np.argmax(pad_sequences(test_preds[1], maxlen=maxlen), axis=2).flatten()
    pred3 = np.argmax(pad_sequences(test_preds[2], maxlen=maxlen), axis=2).flatten()
    print('Confusion Matrices')
    print(confusion_matrix(true1, pred1))
    print(confusion_matrix(true2, pred2))
    print(confusion_matrix(true3, pred3))
    print('Classification Report')
    print(classification_report(true1, pred1))
    print(classification_report(true2, pred2))
    print(classification_report(true3, pred3))


if __name__ == '__main__':
    main()
