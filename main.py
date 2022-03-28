import csv
import pickle

from tqdm import tqdm

import wandb
import torch
import random
import datasets
import transformers

from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer

from torch.optim import AdamW
from transformers import BartConfig, BartTokenizer, BartForSequenceClassification

default_config = {
    "max_length": 50,
    "batch_size": 24,
    "epochs": 4,
    "lr": 1e-6
}

project = wandb.init(project="multisimplify", entity="jemoka", config=default_config)
# project = wandb.init(project="multisimplify", entity="jemoka", config=default_config, mode="disabled")
config = project.config

MAX_LENGTH = config.max_length
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LR = config.lr

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Plan of attack ##
# - Generate a dataset of three levels to predict sari
# - Train a BART with a linear head on it with this stuff
# - Use it as a loss function?? :shrug:

with open("./data/asset.valid.sari.bin", "rb") as df:
    dataset = pickle.load(df)

# Shuffle the dataset
random.shuffle(dataset)

# Let's now batchify the dataset
def batchify(data, batch_size):
    # Create batches and iterate
    batches = []

    # Iterate over batch
    for i in range(0, len(data)-batch_size):
        batches.append(data[i:i+batch_size])

    return batches

dataset_batches = batchify(dataset, BATCH_SIZE)

# Awesome, now, we instatiate a BART and start training!
config = BartConfig.from_pretrained("facebook/bart-base")
config.num_labels = 1

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForSequenceClassification.from_pretrained("facebook/bart-base", config=config).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

wandb.watch(model)
model.train()

# We go through batches
for e in range(EPOCHS):
    print(f"Training epoch {e}!")

    # shuffle
    random.shuffle(dataset_batches)

    # And iterate through batches
    bar = tqdm(dataset_batches)

    for i, batch in enumerate(bar):
        # unpack each batch
        inputs, labels = zip(*batch)

        # concat inputs 
        input_concat = [i[0] + "<mask>" + i[1] for i in inputs]

        # tokenize
        input_tensor = tokenizer(input_concat, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        label_tensor = torch.tensor(labels).unsqueeze(1).to(DEVICE)/100 # normalize

        # get outputs
        outputs = model(**input_tensor, labels=label_tensor)

        # backwards the loss
        outputs.loss.backward()

        # step the optimizer
        optimizer.step()

        # Update the bar and log
        bar.set_description_str(f"batch: {i} | loss: {outputs.loss.cpu().item()}")

        if i % 10 == 0:
            wandb.log({"loss": outputs.loss})

