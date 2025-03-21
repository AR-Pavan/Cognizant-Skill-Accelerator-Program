from datasets import load_dataset
from transformers import DistilBertTokenizer

# Load the IMDB dataset
dataset = load_dataset('imdb')

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)


trainer.train()


model.save_pretrained('./fine_tuned_distilbert')
tokenizer.save_pretrained('./fine_tuned_distilbert')


eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
