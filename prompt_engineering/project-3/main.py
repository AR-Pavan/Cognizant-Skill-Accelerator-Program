import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


data = {
    'train': [
        {"text": "I can't log in to my account.", "label": 0},
        {"text": "Why was I charged extra on my bill?", "label": 1},
        {"text": "What are your business hours?", "label": 2},
        {"text": "The app keeps crashing on my phone.", "label": 0},
        {"text": "How do I cancel my subscription?", "label": 1}
    ],
    'test': [
        {"text": "Do you offer any discounts for students?", "label": 2},
        {"text": "I need help resetting my password.", "label": 0},
        {"text": "There's a mistake in my latest invoice.", "label": 1},
        {"text": "What services do you provide?", "label": 2},
        {"text": "The website is not loading properly.", "label": 0}
    ]
}


dataset = DatasetDict({
    'train': Dataset.from_list(data['train']),
    'test': Dataset.from_list(data['test'])
})

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)


trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
