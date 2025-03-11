import tensorflow as tf
from tensorflow.python.client import device_lib

from tensorflow import keras
from tensorflow.keras.layers import Dense

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


ds_train = tf.data.experimental.make_csv_dataset("D100k_1.tsv.gz",
                                                 field_delim = "\t",
                                                 compression_type = "GZIP",
                                                 batch_size = 1000,
                                                 label_name = "Click",
                                                 shuffle = False,
                                                 num_epochs = 1
                                                 )

columns = list(ds_train.element_spec[0].keys())
columns.insert(0, 'Click')

ds_test = tf.data.experimental.make_csv_dataset("D100k_2.tsv.gz",
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
    sp3 = tf.strings.to_number(
        tf.strings.split(x["Query_tokens"], sep = "|", maxsplit = -1, name = None), tf.int32).to_tensor(
        shape = (None, 10))
    x['AdTitle_tokens'] = sp1
    x["AdDescription_tokens"] = sp2
    x["Query_tokens"] = sp3
    return x, y


# functional pitu pitu
data_train = ds_train.map(spliter).map(lambda x, y: ({'title': x['AdTitle_tokens'],
                                                      'description': x["AdDescription_tokens"],
                                                      'query': x['Query_tokens'],
                                                      'numeric': tf.concat(
                                                          [tf.reshape(x['Depth'], (-1, 1)), tf.reshape(x['Position'],
                                                                                                       (-1, 1)),
                                                           tf.reshape(x['Age'], (-1, 1))], axis = 1
                                                      ),
                                                      'categorical': x['Gender']},
                                                     y))

data_test = ds_test.map(spliter).map(lambda x, y: ({'title': x['AdTitle_tokens'],
                                                    'description': x["AdDescription_tokens"],
                                                    'query': x['Query_tokens'],
                                                    'numeric': tf.concat(
                                                        [tf.reshape(x['Depth'], (-1, 1)), tf.reshape(x['Position'],
                                                                                                     (-1, 1)),
                                                         tf.reshape(x['Age'], (-1, 1))], axis = 1
                                                    ),
                                                    'categorical': x['Gender']},
                                                   y))

variables = []
depth = tf.feature_column.numeric_column('Depth')
variables.append(depth)
position = tf.feature_column.categorical_column_with_vocabulary_list('Position', [1, 2, 3])
one_hot = tf.feature_column.indicator_column(position)
variables.append(one_hot)
age = tf.feature_column.numeric_column('Age')
variables.append(age)
preprocessing_layer = tf.keras.layers.DenseFeatures(variables)

numeric_input = tf.keras.Input(shape = (3,), name = "numeric")
title_input = tf.keras.Input(shape = (None,), name = "title")
description_input = tf.keras.Input(shape = (None,), name = "description")
query_input = tf.keras.Input(shape = (None,), name = "query")
categorical_input = tf.keras.Input(shape = (1,), name = "categorical")

one_hot = tf.keras.layers.CategoryEncoding(
    num_tokens = 3, output_mode = "one_hot")(categorical_input)
hidden = Dense(4)(one_hot)
hidden2 = Dense(128)(numeric_input)
title_features = tf.keras.layers.Embedding(1000, 128)(title_input)
description_features = tf.keras.layers.Embedding(1000, 128)(description_input)
query_features = tf.keras.layers.Embedding(1000, 128)(query_input)

title_features = tf.keras.layers.SimpleRNN(128, dropout = 0.2, recurrent_dropout = 0.2,
                                           return_sequences = False)(title_features)
title_features = tf.keras.layers.Dense(256, activation = 'relu',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.0001))(title_features)
description_features1 = tf.keras.layers.SimpleRNN(64, dropout = 0.2, recurrent_dropout = 0.2,
                                                  return_sequences = False)(description_features)
description_features1 = tf.keras.layers.Dense(128, activation = 'relu',
                                              kernel_regularizer = tf.keras.regularizers.l2(0.0001))(
    description_features1)

# description_features = tf.keras.layers.Conv1D(128, 5, activation = 'relu')(description_features)
# description_features = tf.keras.layers.MaxPooling1D(3)(description_features)
description_features = tf.keras.layers.SimpleRNN(64, dropout = 0.2, recurrent_dropout = 0.2,
                                                 return_sequences = False)(description_features)

query_features = tf.keras.layers.SimpleRNN(64, dropout = 0.2, recurrent_dropout = 0.2,
                                           return_sequences = False)(query_features)
query_features = tf.keras.layers.Dense(128, activation = 'relu',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.0001))(query_features)

x = tf.keras.layers.concatenate(
    [title_features, description_features, query_features, hidden2, hidden, description_features1
     ])

dense0 = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.0001))(x)
norm = tf.keras.layers.BatchNormalization()(dense0)
dense1 = tf.keras.layers.Dense(64, activation = 'relu')(norm)
dense2 = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense1)

model = keras.Model(
    inputs = [title_input, description_input, query_input, numeric_input, categorical_input],
    outputs = [dense2]
)

model.compile(
    loss = tf.keras.losses.binary_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['AUC']
)

model.fit(
    data_train,
    epochs = 3,
    validation_data = data_test
)
