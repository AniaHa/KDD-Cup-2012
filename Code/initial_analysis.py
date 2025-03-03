import tensorflow as tf
from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

ds1 = tf.data.experimental.make_csv_dataset("/home/azureuser/advml/D10M.tsv.gz",
                                            field_delim = "\t",
                                            compression_type = "GZIP",
                                            batch_size = 100,
                                            label_name = "Click",
                                            shuffle = False,
                                            num_epochs = 1)  # False important for test data

ds2 = tf.data.experimental.make_csv_dataset("/home/azureuser/advml/D80M.tsv.gz",
                                            field_delim = "\t",
                                            compression_type = "GZIP",
                                            batch_size = 100,
                                            label_name = "Click",
                                            shuffle = False,
                                            num_epochs = 1)  # False important for test data

ds3 = tf.data.experimental.make_csv_dataset("/home/azureuser/advml/D100k.tsv.gz",
                                            field_delim = "\t",
                                            compression_type = "GZIP",
                                            batch_size = 1000,
                                            label_name = "Click",
                                            shuffle = False,
                                            num_epochs = 1
                                            )  # False important for test data

iterator = ds2.as_numpy_iterator()
columns = list(ds2.element_spec[0].keys())

variables = []
# Ad position as real number
depth = tf.feature_column.numeric_column('Depth')
variables.append(depth)
print(variables)
position = tf.feature_column.categorical_column_with_vocabulary_list('Position', [1, 2, 3])
one_hot = tf.feature_column.indicator_column(position)
variables.append(one_hot)
gender = tf.feature_column.categorical_column_with_vocabulary_list('Gender', [0, 1, 2])
one_hot_g = tf.feature_column.indicator_column(gender)
variables.append(one_hot_g)
age = tf.feature_column.numeric_column('Age')
variables.append(age)


preprocessing_layer = tf.keras.layers.DenseFeatures(variables)
print(preprocessing_layer(next(iter(ds2))[0]))

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.000001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.000001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['AUC'])

model.fit(ds2, epochs = 5)
