# Let's get some data
with open("./asset.valid.bin", "rb") as df:
    asset = pickle.load(df)
# Get the original sentences
original = asset["original"]
# Get the reference simplifications
references = asset["simplified"]

# And, let's get ourselves a simplification metric
sari = datasets.load_metric("sari")

# We will convert the simplifications w.r.t. the sari by filtering subsets out
dataset = []
for osent, rsent in tqdm(zip(original, references), total=len(references)):
    for target in rsent:
        dataref = list(filter(lambda x:x!=target, rsent))
        score = sari.compute(sources=[osent], predictions=[target], references=[dataref])["sari"]
        dataset.append(((osent, target), score))

