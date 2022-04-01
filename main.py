import csv
import pickle
from collections import defaultdict

from tqdm import tqdm

import os
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
    "lr": 6e-8,
    "base_model": "./models/easy-puddle-9"
}

# project = wandb.init(project="multisimplify", entity="jemoka", config=default_config)
project = wandb.init(project="multisimplify", entity="jemoka", config=default_config, mode="disabled")
config = project.config

BASE_MODEL = config.base_model
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

dataset = sorted(dataset, key=lambda x: x[-1])

# Parse it into bins by tens
bins = defaultdict(list)
for i in dataset:
    bins[int(i[-1]//10)].append(i)
# Freeze the bins
bins = dict(bins)
# Create final dataset by sampling
# 1500 each bin, which seems pretty normal
dataset = []
for i in bins.keys():
    dataset = dataset + random.sample(bins[i], min(1500, len(bins[i])))

# <template Code to plot dist>
# dist = [i[-1] for i in dataset]
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots()
# axs.hist(dist, bins=100)
# fig.savefig("./test.png")
# </template Code to plot dist>

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
config = BartConfig.from_pretrained(BASE_MODEL)
config.num_labels = 1

tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)
model = BartForSequenceClassification.from_pretrained(BASE_MODEL, config=config).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

def attempt(model, tokenizer):
    while True:
        # get input
        src = input("src: ")
        tgt = input("tgt: ")
        # concat
        input_concat = src+"<mask>"+tgt

        # tokenize and predict
        input_tensor = tokenizer(input_concat, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        outputs = model(**input_tensor)

        # print
        print(f"The model claims: {round(outputs.logits.cpu().item(), 4)}")

TRAIN = False

if not TRAIN:
    model.eval()
    attempt(model, tokenizer)

else: 
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

    # Save model
    os.mkdir(f"./models/{project.name}")
    model.save_pretrained(f"./models/{project.name}")
    tokenizer.save_pretrained(f"./models/{project.name}")

    # Break
    attempt(model, tokenizer)

