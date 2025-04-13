# Import Auxiliary modules
import MachineLearningProject.Data.tweetDataset as tweetDataset
import MachineLearningProject.MLUtilities.Evaluations.Evaluationfunction as Evaluationfunction
import MachineLearningProject.Data.dataprocessing as dataprocessing
import MachineLearningProject.models.BERT as BERT
# Import necessary libraries
from collections import Counter
from datasets import Dataset
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import json
import torch
import ast
import matplotlib.pyplot as plt

# Set up the Control Module of USE_AUGMENTATION
USE_AUGMENTATION = True
# Print current hardware device being used (GPU or CPU)
print(" current device：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else " No GPU，using CPU")
# Set up the training parameters
training_config = {
    "learning_rate": 3e-5,
    "batch_size": 16,
    "num_train_epochs": 3,
    "max_length": 192
}
# load the model
bert_model = BERT.BertExtractorModel(config=training_config)
# Define data path and load dataset
proj_dir = r"E:\NUS\IS5126\Module BERT\MachineLearningProject\Data\raw"
path = os.path.join(proj_dir, "train.csv")
train_data = tweetDataset.load_data(path)
df = pd.read_csv(path)

# Apply Data processing
processor = dataprocessing.TextCleaner()
df['text'] = df['text'].apply(lambda x: processor.preprocess(x))
df["selected_text"] = df["selected_text"].apply(lambda x: processor.preprocess(x))

## Helper function to locate the answer span within the context
def find_answer_span(text, answer):
    start_idx = text.lower().find(answer.lower())
    if start_idx == -1:
        return None
    return {"text": answer, "answer_start": start_idx}

# Build QA-format dataset
qa_data = []
for _, row in df.iterrows():
    context = row["text"]
    question = row["sentiment"]
    answer_text = row["selected_text"]
    answer_span = find_answer_span(context, answer_text)
    if answer_span:
        qa_data.append({
            "context": context,
            "question": question,
            "answers": answer_span
        })

# Augmentation
if USE_AUGMENTATION:
    augmented_data = []
    for row in qa_data:
        context = row["context"]
        question = row["question"]
        answer_text = row["answers"]["text"]
        # apply synonym replace
        aug_context = processor.synonym_replace(context, n=2)
        aug_answer_span = find_answer_span(aug_context, answer_text)
        if aug_answer_span:
            augmented_data.append({
                "context": aug_context,
                "question": question,
                "answers": aug_answer_span
            })

    qa_data.extend(augmented_data)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# split dataset into training set，validation set and test set with 0.75,0.15,0.15
df_qa = pd.DataFrame(qa_data)
train_df, rest_df = train_test_split(df_qa, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)


# 4.transform into Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))


# Model training
if __name__ == "__main__":
    DO_TRAINING = False
    DO_EVALUATION = True
    DO_TEST_EVALUATION = True
    DO_PLOTTING = True

    if DO_TRAINING:
        bert_model.train(train_dataset, val_dataset)
        bert_model.save_model("./bert_model")
    val_results = test_results = None

    bert_model.load_model("./bert_model")

    def compute_weighted_average(results, counts, metric_name):
        total = sum(counts.values())
        weighted_sum = sum(results[cat][metric_name] * counts[cat] for cat in results)
        return weighted_sum / total

    if DO_EVALUATION:
        df_val = val_df.copy()
        if isinstance(df_val["answers"].iloc[0], str):
            df_val["answers"] = df_val["answers"].apply(lambda x: ast.literal_eval(x))
        # Re-preprocess for the text column
        df_val["text"] = df_val["context"].apply(lambda x: processor.preprocess(x))
        # Extract the actual selected answer text from the dictionary
        df_val["selected_text"] = df_val["answers"].apply(lambda x: x["text"])
        # Store predictions for each sample
        predicted_texts = []
        for _, row in df_val.iterrows():
            if not row["text"].strip():
                predicted_texts.append("")
                continue
            # Run the model to get the predicted span of text
            result = bert_model.predict(row["text"], row["question"])
            predicted_texts.append(result)
        # Extract the ground truth answer spans and sentiment labels
        ground_truths = df_val["selected_text"].tolist()
        categories = df_val["question"].tolist()
        # Evaluation with ground truths, predictions, and their sentiment category
        evaluator = Evaluationfunction.TextEvaluatorByCategory(ground_truths, predicted_texts, categories)
        val_results = evaluator.evaluate()

        print("\n=== evaluation result ===")
        for cat, metrics in val_results.items():
            print(f"\ncategory: {cat}")
            for metric, score in metrics.items():
                print(f"{metric}: {score:.4f}")
        # Count the number of samples belong to each category
        counts = Counter(categories)
        overall_metrics = {}
        for metric in ["Accuracy", "Jaccard", "Precision", "Recall", "F1 Score"]:
            overall_metrics[metric] = compute_weighted_average(val_results, counts, metric)

        print("\n=== Weighted Average (Overall Metrics - validation) ===")
        for metric, score in overall_metrics.items():
            print(f"{metric}: {score:.4f}")

        results = val_results

    # Evaluation on test data
    if DO_TEST_EVALUATION:
        df_test = test_df.copy()
        if isinstance(df_test["answers"].iloc[0], str):
            df_test["answers"] = df_test["answers"].apply(lambda x: ast.literal_eval(x))
        # Re-preprocess for the text column
        df_test["text"] = df_test["context"].apply(lambda x: processor.preprocess(x))
        # Extract the actual selected answer text from the dictionary
        df_test["selected_text"] = df_test["answers"].apply(lambda x: x["text"])
        # Store predictions for each sample
        predicted_texts = []
        for _, row in df_test.iterrows():
            if not row["text"].strip():
                predicted_texts.append("")
                continue
            result = bert_model.predict(row["text"], row["question"])
            predicted_texts.append(result)
        # Extract the ground truth answer spans and sentiment labels
        ground_truths = df_test["selected_text"].tolist()
        categories = df_test["question"].tolist()
        # Evaluation with ground truths, predictions, and their sentiment category
        evaluator = Evaluationfunction.TextEvaluatorByCategory(ground_truths, predicted_texts, categories)
        test_results = evaluator.evaluate()

        print("\n=== Test Evaluation Result ===")
        for cat, metrics in test_results.items():
            print(f"\ncategory: {cat}")
            for metric, score in metrics.items():
                print(f"{metric}: {score:.4f}")
        # Count the number of samples belong to each category
        counts = Counter(categories)
        overall_metrics = {}
        for metric in ["Accuracy", "Jaccard", "Precision", "Recall", "F1 Score"]:
            overall_metrics[metric] = compute_weighted_average(test_results, counts, metric)

        print("\n=== Weighted Average (Overall Metrics - test) ===")
        for metric, score in overall_metrics.items():
            print(f"{metric}: {score:.4f}")

        results = test_results

    if DO_PLOTTING:
        # Draw the metrics plot
        def plot_results(results, title_suffix):
            categories = list(results.keys())
            metrics_to_plot = ["Accuracy", "Jaccard", "Precision", "Recall", "F1 Score"]
            for metric in metrics_to_plot:
                values = [results[cat][metric] for cat in categories]
                plt.figure()
                plt.bar(categories, values)
                plt.title(f"{metric} by Sentiment Category ({title_suffix})")
                plt.ylabel(metric)
                plt.xlabel("Sentiment")
                plt.ylim(0, 1)
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.show()

        if val_results:
            plot_results(val_results, "Validation Set")
        if test_results:
            plot_results(test_results, "Test Set")

        # Draw the loss plot
        log_file = r"E:\NUS\IS5126\Module BERT\MachineLearningProject\src\bert_model\checkpoint-3570\trainer_state.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f).get("log_history", [])
            steps, losses, val_steps, val_losses = [], [], [], []
            for log in logs:
                if "loss" in log and "step" in log:
                    steps.append(log["step"])
                    losses.append(log["loss"])
                if "eval_loss" in log and "step" in log:
                    val_steps.append(log["step"])
                    val_losses.append(log["eval_loss"])
            plt.plot(steps, losses, label="Train Loss")
            plt.plot(val_steps, val_losses, label="val Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training and validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("not find the file")

# Sample test
def predict_single(text: str, sentiment: str):
        text_cleaned = processor.preprocess(text)
        if not text_cleaned.strip():
            print(" context is empty")
            return ""
        bert_model.load_model("./bert_model")
        result = bert_model.predict(text,sentiment)
        print("\n tweet text:", text)
        print(" sentiment:", sentiment)
        print(" extraction:", result)
        return result
predict_single("I am so excited to see you!!!", "positive")
predict_single("I can't believe how bad this is...", "negative")

