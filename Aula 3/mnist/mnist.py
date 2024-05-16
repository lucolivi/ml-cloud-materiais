import tensorflow as tf
import numpy as np

import argparse
import os, json
import mlflow

def _is_chief_task(task_type, task_id):
    return task_type is None or task_type == "chief" or (task_type == "worker" and task_id == 0)

def _get_temp_dir(dirpath, task_id):
    base_dirpath = "workertemp_" + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief_task(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--per-worker-batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=str, default="outputs")

    args = parser.parse_args()

    mlflow.tensorflow.autolog()

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    global_batch_size = args.per_worker_batch_size * num_workers

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(60000)
        .repeat()
        .batch(global_batch_size)
    )

    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int64)
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(global_batch_size)
    )

    with strategy.scope():

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=(28,28,1)),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10)
            ]
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=["accuracy"]
        )

        model.fit(
            train_dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, validation_data=test_dataset
        )

        task_type = tf_config["task"]["type"]
        task_id = tf_config["task"]["index"]

        write_model_path = write_filepath(args.model_dir, task_type, task_id)

        model.save(write_model_path)
        
if __name__ == "__main__":
    main()




