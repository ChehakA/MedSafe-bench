from datasets import load_dataset

print("Downloading med_qa dataset from HuggingFace...")
print("This may take a minute (dataset is ~132MB)\n")

dataset = load_dataset(
    "bigbio/med_qa",
    name="med_qa_en_source",
    trust_remote_code=True
)

print("Download complete!")
print("\nDataset splits available:", list(dataset.keys()))
print("Train size   :", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Test size    :", len(dataset["test"]))

dataset.save_to_disk("./med_qa_data")
print("\nDataset saved locally to ./med_qa_data folder")