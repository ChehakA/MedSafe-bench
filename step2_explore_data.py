from datasets import load_from_disk

print("Loading dataset from disk...\n")
dataset = load_from_disk("./med_qa_data")

print("=== DATASET STRUCTURE ===")
print(dataset)
print()

print("=== COLUMNS ===")
print(dataset["train"].column_names)
print()

# Check what options actually looks like
print("=== RAW SAMPLE (to see options format) ===")
print(dataset["test"][0])
print()

print("=== SAMPLE QUESTIONS (Test Set) ===\n")
for i in range(3):
    sample = dataset["test"][i]
    print(f"--- Question {i+1} ---")
    print("Question :", sample["question"])
    print("Options  :", sample["options"])   # print raw first
    print("Answer   :", sample["answer_idx"], "->", sample["answer"])
    print()