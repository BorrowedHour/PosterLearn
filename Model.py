import glob
import io
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

totalLines = 14

def parse_labels_and_features(dataset):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[0]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:, 1:(182*268)]
    # Scale the data to [0, 1] by dividing out the max value, 255.
    features = features / 255

    return labels, features
def create_training_input_fn(features, labels, batch_size):
  """A custom input_fn for sending mnist data to the estimator for training.

  Args:
    features: The training features.
    labels: The training labels.
    batch_size: Batch size to use during training.

  Returns:
    A function that returns batches of training features and labels during
    training.
  """
  def _input_fn():
    raw_features = tf.constant(features.values)
    raw_targets = tf.constant(labels.values)
    dataset_size = len(features)

    return tf.train.shuffle_batch(
        [raw_features, raw_targets],
        batch_size=batch_size,
        enqueue_many=True,
        capacity=2 * dataset_size,  # Must be greater than min_after_dequeue.
        min_after_dequeue=dataset_size)  # Important to ensure uniform randomness.

  return _input_fn

def create_predict_input_fn(features, labels):
  """A custom input_fn for sending mnist data to the estimator for predictions.

  Args:
    features: The features to base predictions on.
    labels: The labels of the prediction examples.

  Returns:
    A function that returns features and labels for predictions.
  """
  def _input_fn():
    raw_features = tf.constant(features.values)
    raw_targets = tf.constant(labels.values)
    return tf.train.limit_epochs(raw_features, 1), raw_targets

  return _input_fn

def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create a linear classifier object.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        training_examples)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=28,
        hidden_units=hidden_units,
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
        gradient_clip_norm=5.0,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "LogLoss error (on validation data):"
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.fit(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = list(classifier.predict_proba(input_fn=predict_training_input_fn))
        validation_predictions = list(classifier.predict_proba(input_fn=predict_validation_input_fn))
        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_predictions,labels=range(0,28))
        validation_log_loss = metrics.log_loss(validation_targets, validation_predictions,labels=range(0,28))
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, validation_log_loss)
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print "Model training finished."
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = list(classifier.predict(validation_examples))
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print "Final accuracy (on validation data): %0.2f" % accuracy

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier

def model():
    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    chunksize = 1

    poster_dataframe = pd.read_csv(io.open("output.csv", "r"), sep=",", header=None)
    poster_dataframe = poster_dataframe.reindex(np.random.permutation(poster_dataframe.index))

    trainLines = int(totalLines * .8)
    validationLines = totalLines - trainLines
    training_targets, training_examples = parse_labels_and_features(poster_dataframe.head(trainLines))
    validation_targets, validation_examples = parse_labels_and_features(poster_dataframe.tail(validationLines))

    classifier = train_nn_classification_model(
        learning_rate=0.05,
        steps=1000,
        batch_size=100,
        hidden_units=[100, 100],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

model()