import argparse
import datasets
import pandas as pd
import tensorflow as tf
import numpy as np 
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import layers, models, callbacks

# Enable mixed precision for reduced memory usage
set_global_policy('mixed_float16')

model_name = 'bert-base-uncased'
pre_trained_model = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pre_trained_model.trainable = False

# Pre define hyperparameters
MAX_LENGTH = 32
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

tf.config.list_physical_devices('GPU')

def tokenize(examples):
    """Tokenize the input text."""
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

# Add embeddings (pre_trained_model)
def add_ptm_embeddings(examples):
    chunk_size = 100
    embeddings = []
    for i in range(0, len(examples['text']), chunk_size):
        batch = examples['text'][i:i+chunk_size]
        tokens = tokenizer(batch, return_tensors="tf", max_length=MAX_LENGTH, padding="max_length", truncation=True)
        outputs = pre_trained_model(**tokens)
        embeddings.append(outputs.last_hidden_state.numpy())
    return {"embeddings": np.concatenate(embeddings)}

# Create a model
def create_model(output_dim):
    model = models.Sequential([
        layers.Input(shape=(MAX_LENGTH, 768)),                          # Input
        layers.Bidirectional(layers.GRU(90, return_sequences=True)),    # Bidirectional GRU layer
        layers.GlobalMaxPooling1D(),                                    # Global max pooling 
        layers.Dense(90, activation='relu'),                            # Dense Layer with relu activation
        layers.Dropout(0.33),                                           # Dropout .33
        layers.Dense(50, activation='relu'),                            # Dense Layer with relu activation
        layers.Dense(30, activation='tanh'),                            # Dense Layer with tanh activation
        layers.Dropout(0.33),                                           # Dropout .33
        layers.Dense(output_dim, activation='sigmoid')                  # Fully connected layer
    ])
    return model

# Create a HF dataset from csv
def load_hf_dataset(train_path, dev_path):
    # Load the CSVs into Huggingface datasets
    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Convert label columns into a list of floats."""
        return {"labels": [float(example[l]) for l in labels]}
    
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(add_ptm_embeddings, batched=True)
    
    # Convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["embeddings"],
        label_cols="labels",
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["embeddings"],
        label_cols="labels",
        batch_size=BATCH_SIZE
    )
  
    return labels, train_dataset, dev_dataset

# Compile the model
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )
    return model

# Fit the model
def fit_model(model, train_dataset, dev_dataset, model_path):
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=dev_dataset,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True),
            callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True)
        ]
    )
    return model

# Training process starts
def train(model_path="model.keras", train_path="train.csv", dev_path="dev.csv"):

    # Get data from huggingface dataset
    labels, train_dataset, dev_dataset = load_hf_dataset(train_path, dev_path)

    # define a model with a single fully connected layer
    model = create_model(len(labels))

    # Compile the model
    model = compile_model(model)

    # Train the model
    model = fit_model(model, train_dataset, dev_dataset, model_path)

# Prediction process starts
# input_path="dev.csv"      : FOR DEVELOPMENT PHASE
# input_path="test-in.csv"  : FOR TEST PHASE
def predict(model_path="model.keras", input_path="test-in.csv"):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the data for prediction
    df = pd.read_csv(input_path)

    # Preprocess input features 
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(add_ptm_embeddings, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["embeddings"],
        batch_size=BATCH_SIZE
    )

    # Generate predictions
    predictions = np.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # Assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # Write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name='submission.csv'))

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
