from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments

dataset_name = 'rotten_tomatoes'
model_name='google-bert/bert-base-cased'

# Loading model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load data
data = load_dataset(dataset_name)
train, test = data['train'], data['test']

train_data_text = train.map(function=lambda data: tokenizer(data['text'], truncation=True), batched=True)
test_data_text = test.map(function=lambda data: tokenizer(data['text'], truncation=True), batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval):
	logits, labels = eval
	metric = evaluate.load("f1")
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(references=labels, predictions=predictions)

args = TrainingArguments(
	"model",
	learning_rate=2e-5,
	num_train_epochs=2,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	save_strategy="epoch",
	report_to="none",
	weight_decay=0.01
)

trainer = Trainer(
	args=args,
	processing_class=tokenizer,
	model=model,
	data_collator=data_collator,
	compute_metrics=compute_metrics,
	eval_dataset=test_data_text,
	train_dataset=train_data_text
)

trainer.train()

print(trainer.evaluate())
