from setfit import Trainer, TrainingArguments, sample_dataset, SetFitModel
from datasets import load_dataset 

model_name='sentence-transformers/all-mpnet-base-v2'
dataset_name='kannanrbk/multidomain-sentences-dataset'

data = load_dataset(dataset_name)

train, test = data['train'], data['test']

model = SetFitModel.from_pretrained(model_name)

#generate sample data 

sample_data = sample_dataset(num_samples=16, dataset=train)


args = TrainingArguments(num_iterations=20, num_epochs=3)

trainer = Trainer(args=args, model=model, eval_dataset=test, train_dataset=train, metric='f1', metric_kwargs={"average": "macro"})

trainer.train()

print(trainer.evaluate())
