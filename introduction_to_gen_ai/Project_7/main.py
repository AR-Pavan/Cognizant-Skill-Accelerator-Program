from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric

# Loading pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: Positive/Negative
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Loading IMDb dataset
dataset = load_dataset("imdb")
print(dataset)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]



training_args = TrainingArguments(
    output_dir="./bert-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=0e-0,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()



metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer.evaluate()
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    return "Positive" if prediction == 1 else "Negative"


print(predict_sentiment("This movie was fantastic!"))
print(predict_sentiment("I did not enjoy this film at all."))
