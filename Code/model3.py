import tensorflow as tf
from tensorflow.python.client import device_lib

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

ds_train = tf.data.experimental.make_csv_dataset("D80M_1.tsv.gz",
                                                 field_delim = "\t",
                                                 compression_type = "GZIP",
                                                 batch_size = 1000,
                                                 label_name = "Click",
                                                 shuffle = False,
                                                 num_epochs = 1

                                                 )

columns = list(ds_train.element_spec[0].keys())
columns.insert(0, 'Click')

ds_test = tf.data.experimental.make_csv_dataset("D80M_2.tsv.gz",
                                                field_delim = "\t",
                                                compression_type = "GZIP",
                                                batch_size = 1000,
                                                column_names = columns,
                                                label_name = "Click",
                                                shuffle = False,
                                                num_epochs = 1
                                                )

next(iter(ds_train))


# map functions
def spliter(x, y):
    sp1 = tf.strings.to_number(
        tf.strings.split(x["AdTitle_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))
    sp2 = tf.strings.to_number(
        tf.strings.split(x["AdDescription_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 30))
    x['Query_tokens'] = tf.convert_to_tensor(x['Query_tokens'], tf.string)
    sp3 = tf.strings.to_number(
        tf.strings.split(x["Query_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))
    sp4 = tf.strings.to_number(
        tf.strings.split(x['AdKeyword_tokens'], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))
    sp5 = tf.convert_to_tensor(x['UserID'], tf.int32)
    x['AdTitle_tokens'] = sp1
    x["AdDescription_tokens"] = sp2
    x["Query_tokens"] = sp3
    x['AdKeyword_tokens'] = sp4
    x['UserID'] = sp5
    return x, y


def spliter1(x, y):
    sp1 = tf.strings.to_number(
        tf.strings.split(x["AdTitle_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))

    x['AdTitle_tokens'] = sp1
    return sp1, y


def spliter2(x, y):
    sp2 = tf.strings.to_number(
        tf.strings.split(x["AdDescription_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))
    x['AdDescription_tokens'] = sp2
    return sp2, y


# functional pitu pitu
data_train = ds_train.map(spliter).map(lambda x, y: ({'title': x['AdTitle_tokens'],
                                                      'description': x["AdDescription_tokens"],
                                                      'query': x['Query_tokens'],
                                                      'Depth': x['Depth'],
                                                      'Position': x['Position'],
                                                      'numeric': tf.concat(
                                                          [tf.reshape(x['Depth'], (-1, 1)), tf.reshape(x['Position'],
                                                                                                       (-1, 1)),
                                                           tf.reshape(x['Age'], (-1, 1))], axis = 1
                                                      ),
                                                      'categorical': x['Gender'],
                                                      'keyword': x['AdKeyword_tokens'],
                                                      'UserID': x['UserID']},
                                                     y))

data_test = ds_test.map(spliter).map(lambda x, y: ({'title': x['AdTitle_tokens'],
                                                    'description': x["AdDescription_tokens"],
                                                    'query': x['Query_tokens'],
                                                    'Depth': x['Depth'],
                                                    'Position': x['Position'],
                                                    'numeric': tf.concat(
                                                        [tf.reshape(x['Depth'], (-1, 1)), tf.reshape(x['Position'],
                                                                                                     (-1, 1)),
                                                         tf.reshape(x['Age'], (-1, 1))], axis = 1
                                                    ),
                                                    'Gender': x['Gender'],
                                                    'keyword': x['AdKeyword_tokens'],
                                                    'UserID': x['UserID']},
                                                   y))

variables = []
# depth = tf.feature_column.numeric_column('Depth')
# variables.append(depth)
position = tf.feature_column.numeric_column('Position')
variables.append(position)
# age = tf.feature_column.numeric_column('Age')
# variables.append(age)
preprocessing_layer = tf.keras.layers.DenseFeatures(variables)

feature_columns = []
user = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_hash_bucket('UserID', 1000, dtype = tf.int32))
# user = tf.feature_column.categorical_column_with_hash_bucket('UserID', 1000)
feature_columns.append(user)

numeric_input = tf.keras.Input(shape = (3,), name = "numeric")
title_input = tf.keras.Input(shape = (None,), name = "title")
description_input = tf.keras.Input(shape = (None,), name = "description")
query_input = tf.keras.Input(shape = (None,), name = "query")
keyword_input = tf.keras.Input(shape = (None,), name = "keyword")

hidden2 = Dense(258)(numeric_input)
hidden2 = tf.keras.layers.BatchNormalization()(hidden2)
hidden2 = tf.keras.layers.Dense(128, activation = 'relu')(hidden2)
hidden2 = tf.keras.layers.Dropout(0.1)(hidden2)
hidden2 = tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.000001))(
    hidden2)

title_features0 = tf.keras.layers.Embedding(1000, 128)(title_input)
description_features = tf.keras.layers.Embedding(1500, 128)(description_input)
description_features3 = tf.keras.layers.Embedding(2000, 128)(description_input)
query_features0 = tf.keras.layers.Embedding(1000, 128)(query_input)
keyword_features = tf.keras.layers.Embedding(1000, 128)(keyword_input)

description_features3 = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(
    description_features3)
description_features3 = tf.keras.layers.GlobalAveragePooling1D()(description_features3)
description_features3 = tf.keras.layers.Dense(128, activation = 'relu',
                                              kernel_regularizer = tf.keras.regularizers.l2(0.0001))(
    description_features3)
description_features3 = tf.keras.layers.Dropout(0.2)(description_features3)
description_features3 = tf.keras.layers.Dense(128, activation = 'relu',
                                              kernel_regularizer = tf.keras.regularizers.l2(0.0001))(
    description_features3)

title_features = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(title_features0)
title_features = tf.keras.layers.GlobalMaxPooling1D()(title_features)
title_features = tf.keras.layers.Dense(256, activation = 'relu',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.0001))(title_features)
title_features = tf.keras.layers.Dropout(0.2)(title_features)
title_features = tf.keras.layers.Dense(128, activation = 'relu')(title_features)

keyword_features = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(keyword_features)
keyword_features = tf.keras.layers.MaxPooling1D(3)(keyword_features)
keyword_features = tf.compat.v1.keras.layers.CuDNNLSTM(128,
                                             return_sequences = False)(keyword_features)
keyword_features = tf.keras.layers.Dense(256, activation = 'relu',
                                         kernel_regularizer = tf.keras.regularizers.l2(0.0001))(keyword_features)
keyword_features = tf.keras.layers.Dropout(0.2)(keyword_features)
keyword_features = tf.keras.layers.Dense(128, activation = 'relu')(keyword_features)

description_features = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(
    description_features)
description_features = tf.keras.layers.MaxPooling1D(3)(description_features)
description_features = tf.compat.v1.keras.layers.CuDNNLSTM(64,
                                                 return_sequences = False)(description_features)
description_features = tf.keras.layers.Dropout(0.2)(description_features)
description_features = tf.keras.layers.Dense(128, activation = 'relu')(description_features)

description_merge = tf.keras.layers.Average()([description_features, description_features3])
description_merge = tf.keras.layers.Dropout(0.2)(description_merge)
description_merge = tf.keras.layers.Dense(128, activation = 'relu',
                                          kernel_regularizer = tf.keras.regularizers.l2(0.0001))(description_merge)

query_features = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(query_features0)
query_features = tf.keras.layers.GlobalAveragePooling1D()(query_features)
query_features = tf.keras.layers.Dense(128, activation = 'relu',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.0001))(query_features)
query_features = tf.keras.layers.Dropout(0.2)(query_features)
query_features = tf.keras.layers.Dense(128, activation = 'relu')(query_features)

dotted = tf.keras.layers.Dot(axes = 1)([query_features0, title_features0])
dotted = tf.keras.layers.Conv1D(128, 5, activation = 'relu', input_shape = (None, 128))(dotted)
dotted = tf.keras.layers.GlobalAveragePooling1D()(dotted)
dotted = tf.keras.layers.Dense(128, activation = 'relu',
                               kernel_regularizer = tf.keras.regularizers.l2(0.0001))(dotted)
dotted = tf.keras.layers.Dropout(0.2)(dotted)
dotted = tf.keras.layers.Dense(128, activation = 'relu')(dotted)

x = tf.keras.layers.concatenate(
    [title_features, description_merge, query_features, hidden2, keyword_features, description_features3, description_features, dotted])

dense0 = tf.keras.layers.Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.0001))(x)
norm = tf.keras.layers.BatchNormalization()(dense0)
dense1 = tf.keras.layers.Dense(128, activation = 'relu')(norm)
dense1 = tf.keras.layers.Dropout(0.2)(dense1)
dense1 = tf.keras.layers.Dense(64, activation = 'relu')(dense1)
dense2 = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense1)

model = keras.Model(
    inputs = [title_input, description_input, query_input, numeric_input, keyword_input],
    outputs = [dense2]
)
# v for v in feature_layer_inputs.values()
model.compile(
    loss = tf.keras.losses.binary_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['AUC']
)

model.fit(
    data_train,
    epochs = 2,
    validation_data = data_test
)