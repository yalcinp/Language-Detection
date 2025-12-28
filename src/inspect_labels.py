from datasets import load_dataset

ds = load_dataset("MartinThoma/wili_2018")
train = ds["train"]
feat = train.features["label"]

print("Text columns:", train.column_names)
print("Label feature type:", type(feat))
if hasattr(feat, "names"):
    print("Num labels:", len(feat.names))
    print("First 30 label names:", feat.names[:30])

# Print a few samples
for i in range(5):
    ex = train[i]
    lab = ex["label"]
    name = feat.int2str(lab) if hasattr(feat, "int2str") else str(lab)
    print(i, "label_id=", lab, "label_name=", name, "text_snip=", ex.get("sentence", ex.get("text",""))[:60])

