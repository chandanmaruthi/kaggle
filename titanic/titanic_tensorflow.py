from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))
import pandas as pd



class_names = [1,0]

def get_data_set(data_url="", shuffle=False) :

    dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(data_url),
                                               origin=data_url)
    dataset__cleansed_fp ="cleansed.csv"
    df = pd.read_csv(dataset_fp)
    df.Name = pd.factorize(df.Name)[0]
    df.Cabin = pd.factorize(df.Cabin)[0]
    df.Sex = pd.factorize(df.Sex)[0]
    df.Embarked = pd.factorize(df.Embarked)[0]
    df.Ticket = pd.factorize(df.Ticket)[0]
    df.Fare = pd.factorize(df.Fare)[0]
    df.Age = pd.factorize(df.Age)[0]
    df.to_csv(dataset__cleansed_fp, index=False)

    print("Local copy of the dataset file: {}".format(dataset_fp))

    # column order in CSV file
    column_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

    feature_names = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    label_name = 'Survived'

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))


    batch_size = 32
    train_dataset = tf.contrib.data.make_csv_dataset(
        dataset__cleansed_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)


    features, labels = next(iter(train_dataset))
    #print(features[:5])
    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))
    print(features[:5])
    return train_dataset, features, labels


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def train_model(train_dataset, features,labels):
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10000, activation=tf.nn.relu, input_shape=(11,)),  # input shape required
      tf.keras.layers.Dense(10000, activation=tf.nn.relu),
      tf.keras.layers.Dense(10000, activation=tf.nn.relu),
      tf.keras.layers.Dense(10000, activation=tf.nn.softmax),
      tf.keras.layers.Dense(2)
    ])

    predictions = model(tf.cast(features,tf.float32))
    predictions[:5]

    tf.nn.softmax(predictions[:5])



    print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    print("    Labels: {}".format(labels))



    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    global_step = tf.train.get_or_create_global_step()

    loss_value, grads = grad(model, tf.cast(features,tf.float32), labels)

    print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.variables), global_step)

    print("Step: {},         Loss: {}".format(global_step.numpy(),
                                              loss(model, tf.cast(features,tf.float32), labels).numpy()))

    ## Note: Rerunning this cell uses the same model variables

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 1000
    with tf.device("/gpu:0"):

        for epoch in range(num_epochs):
            epoch_loss_avg = tfe.metrics.Mean()
            epoch_accuracy = tfe.metrics.Accuracy()

            # Training loop - using batches of 32
            for x, y in train_dataset:
                # Optimize the model
                x = tf.cast(x, tf.float32)
                loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step)

                # Track progress
                epoch_loss_avg(loss_value)  # add current batch loss
                # compare predicted label to actual label
                epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

            # end epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 2 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results);
    return model

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


#l = loss(model, tf.cast(features,tf.float32), labels)
#print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)



def validate_model(model, validate_dataset, features, labels):


    #validate_dataset = validate_dataset.map(pack_features_vector)
    test_accuracy = tfe.metrics.Accuracy()

    for (x,y) in validate_dataset:
      x= tf.cast(x,tf.float32)
      logits = model(x)
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
      test_accuracy(prediction, y)

    print(prediction)
    print(y)
    print("Validate set accuracy: {:.3%}".format(test_accuracy.result()))


    tf.stack([y,prediction],axis=1)


def predict():
    test_url = "https://s3.amazonaws.com/chandan-kaggle-datasets/titanic/test.csv"
    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),origin=test_url)
    test_dataset_cleansed = "test_new.csv"

    df_test = pd.read_csv(test_fp)
    print(df_test.head())
    df_test.Name = pd.factorize(df_test.Name)[0]
    df_test.Cabin = pd.factorize(df_test.Cabin)[0]
    df_test.Sex = pd.factorize(df_test.Sex)[0]
    df_test.Embarked = pd.factorize(df_test.Embarked)[0]
    df_test.Ticket = pd.factorize(df_test.Ticket)[0]
    df_test.Fare = pd.factorize(df_test.Fare)[0]
    df_test.Age = pd.factorize(df_test.Age)[0]
    df_test.to_csv(test_dataset_cleansed, index=False)

    test_dataset = tf.convert_to_tensor(df_test)



    test_accuracy = tfe.metrics.Accuracy()

    logits = model(test_dataset)


# Pre Process Training Data
train_dataset, features, labels = get_data_set("https://s3.amazonaws.com/chandan-kaggle-datasets/titanic/train.csv", shuffle=False)
# Train Model
model = train_model(train_dataset, features, labels)
# Prepare Dev Data
validate_dataset, features, labels = get_data_set("https://s3.amazonaws.com/chandan-kaggle-datasets/titanic/train.csv",shuffle=True)
# Validate with Dev Data
validate_model(model, validate_dataset,features,labels)
# Validate Model with Dev Data
#   Do Something
# Prepare Test Data
#   Do something
# Predict on Test Data
#   Do something









