from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

dataset_name = 'rotten_tomatoes'
data = load_dataset(dataset_name)

model_name='google-bert/bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train, test = data['train'], data['test']

train_data_text = train.map(function=lambda data: tokenizer(data['text'], truncation=True), batched=True)
test_data_text = test.map(function=lambda data: tokenizer(data['text'], truncation=True), batched=True)

train_data_text = train_data_text.remove_columns('label')
test_data_text = test_data_text.remove_columns('label')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

args = TrainingArguments(
	"masked-llm",
	learning_rate=2e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=10,
	weight_decay=0.01,
	save_strategy="epoch",
	report_to="none"
)

trainer = Trainer(
	model=model,
	args=args,
	train_dataset=train_data_text,
	eval_dataset=test_data_text,
	tokenizer=tokenizer,
	data_collator=data_collator
)

## Save pretrained tokenizer
tokenizer.save_pretrained("mlm")

trainer.train()

model.save_pretrained("mlm")


