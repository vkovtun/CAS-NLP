from datasets import DatasetDict, Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import argparse
import numpy as np
import os

class FineTuner:
    LABEL_LIST = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-MISC", "I-MISC"]
    label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    id2label = {i: label for label, i in label2id.items()}

    def __init__(self):
        super().__init__()

        self.tokenizer = None

    def read_conll(self, path):
        tokens, labels = [], []
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        data.append({"tokens": tokens, "ner_tags": [self.__class__.label2id.get(l, 0) for l in labels]})
                        tokens, labels = [], []
                else:
                    splits = line.split()
                    if len(splits) >= 2:
                        tokens.append(splits[0])
                        labels.append(splits[-1])
        return data

    def load_dataset_from_txt(self, dataset_dir):
        return DatasetDict({
            "train": Dataset.from_list(self.read_conll(os.path.join(dataset_dir, "train.txt"))),
            "validation": Dataset.from_list(self.read_conll(os.path.join(dataset_dir, "dev.txt")))
        })

    def tokenize_and_align(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []

        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
            labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    labels.append(examples["ner_tags"][i][word_idx])
                else:
                    label = examples["ner_tags"][i][word_idx]
                    if self.__class__.LABEL_LIST[label].startswith("I-"):
                        labels.append(label)
                    else:
                        labels.append(-100)
                previous_word_idx = word_idx
            all_labels.append(labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def compute_metrics(self, p):
        metric = evaluate.load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds = [[self.__class__.LABEL_LIST[p] for (p, l) in zip(pred, label) if l != -100]
                      for pred, label in zip(predictions, labels)]
        true_labels = [[self.__class__.LABEL_LIST[l] for (p, l) in zip(pred, label) if l != -100]
                       for pred, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def main(self):
        parser = argparse.ArgumentParser(description="Fine tune a xlmr token classification model")
        parser.add_argument("--model", default="ivlcic/xlmr-ner-slavic", help="Model name or local path")
        parser.add_argument("--language", required=True, help="Language")
        args = parser.parse_args()


        dataset_dir=f"datasets/wikianc/{args.language}"
        output_dir=f"models/wikianc/{args.language}"
        model = args.model

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        dataset = self.load_dataset_from_txt(dataset_dir)
        tokenized_dataset = dataset.map(self.tokenize_and_align, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(self.__class__.LABEL_LIST),
                                                                id2label=self.__class__.id2label, label2id=self.__class__.label2id)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(output_dir)

if __name__ == "__main__":
    FineTuner().main()
