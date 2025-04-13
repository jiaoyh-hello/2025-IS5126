# Import necessary libraries
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments,TrainerCallback
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer
# callback to log evaluation results at each epoch
class EpochLoggerCallback(TrainerCallback):
    def __init__(self):
        self.history = []
    # store training metrics
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        entry = {
            "Epoch": round(state.epoch, 2) if state.epoch is not None else None,
            "Training Loss": metrics.get("loss", None),
            "Validation Loss": metrics.get("eval_loss", None),
            "Accuracy": metrics.get("eval_accuracy", None),
            "F1 Macro": metrics.get("eval_f1_macro", None),
            "F1 Weighted": metrics.get("eval_f1_weighted", None),
        }
        self.history.append(entry)

    def print_summary(self):
        df = pd.DataFrame(self.history)
        print("\n Training Summary:")
        print(df.round(6))
        return df

# Main model class for BERT
class BertExtractorModel:
    def __init__(self, model_name="bert-base-uncased",config=None):
        # Load pre-trained tokenizer and QA model
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        # Load training configuration
        self.config = config or {}
        self.learning_rate = self.config.get("learning_rate", 3e-5)
        self.batch_size = self.config.get("batch_size", 16)
        self.num_train_epochs = self.config.get("num_train_epochs", 3)
        self.max_length = self.config.get("max_length", 128)
        # Initialize the epoch logger callback
        self.epoch_logger = EpochLoggerCallback()

    # convert text and answer to token
    def prepare_dataset(self, dataset):
        def process(example):
            # Tokenize input with context and question
            encoding = self.tokenizer(
                example["question"], example["context"],
                truncation="only_second",
                max_length=192,
                padding="max_length",
                return_offsets_mapping=True
            )
            offsets = encoding.pop("offset_mapping")
            # Locate start and end character indices
            answer = example["answers"]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])
            # Define token range
            token_start_index = encoding["input_ids"].index(self.tokenizer.sep_token_id) + 1
            token_end_index = len(encoding["input_ids"]) - 1
            # Map character-level answer span to token-level span
            start_token = end_token = 0
            for idx in range(token_start_index, token_end_index):
                if offsets[idx] is None:
                    continue
                if offsets[idx][0] <= start_char < offsets[idx][1]:
                    start_token = idx
                if offsets[idx][0] < end_char <= offsets[idx][1]:
                    end_token = idx
                    break

            encoding["start_positions"] = start_token
            encoding["end_positions"] = end_token
            return encoding

        # Apply processing to entire dataset
        return dataset.map(process, remove_columns=dataset.column_names)



    def train(self, train_dataset, val_dataset=None):
        train_dataset = self.prepare_dataset(train_dataset)
        if val_dataset is not None:
            val_dataset = self.prepare_dataset(val_dataset)
        else:
            print("No validation set detected，can not conduct evaluation！")
        learning_rate = self.learning_rate
        per_device_train_batch_size = self.batch_size
        num_train_epochs = self.num_train_epochs

        # Define training arguments
        args = TrainingArguments(
            output_dir="./bert_model",
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=200,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=200,
            save_strategy="steps",
            save_steps= 200,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            weight_decay = 0.05 # adding a penalty to the loss function to prevent overfitting
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[self.epoch_logger],
        )
        # Start training
        trainer.train()
        if hasattr(self, 'epoch_logger'):
            self.epoch_logger.print_summary()

    # Predict answer span given a single context and sentiment
    def predict(self, text, sentiment):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = self.tokenizer(sentiment, text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get start and end token index
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        # Invalid prediction case
        if start >= end or end > inputs["input_ids"].shape[1]:
            return ""
        # Decode answer span from token IDs
        tokens = inputs["input_ids"][0][start:end]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # compute accuracy and F1 scores
    def compute_metrics(self,pred):
        start_preds = pred.predictions[0].argmax(axis=-1)
        end_preds = pred.predictions[1].argmax(axis=-1)
        start_labels = pred.label_ids[0]
        end_labels = pred.label_ids[1]
        # Only evaluate non-padding tokens
        mask = (start_labels != -100) & (end_labels != -100)

        acc = accuracy_score(start_labels[mask], start_preds[mask])
        f1_macro = f1_score(start_labels[mask], start_preds[mask], average="macro")
        f1_weighted = f1_score(start_labels[mask], start_preds[mask], average="weighted")
        return {
            "eval_accuracy": acc,
            "eval_f1_macro": f1_macro,
            "eval_f1_weighted": f1_weighted
        }

    # Save the model
    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # Load the model
    def load_model(self, path):
        self.model = BertForQuestionAnswering.from_pretrained(path)
        self.tokenizer = BertTokenizerFast.from_pretrained(path)


