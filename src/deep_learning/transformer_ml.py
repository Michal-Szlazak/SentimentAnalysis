import os

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Optimization for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def run_transformer_analysis(
    name, data_bundle, model_checkpoint="distilbert-base-uncased"
):
    print(f"\n🚀 Starting training: {model_checkpoint} on {name}...")

    # 1. Device check for M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Convert to HF Datasets
    train_ds = Dataset.from_dict(
        {"text": data_bundle["X_train"], "label": data_bundle["y_train"]}
    )
    test_ds = Dataset.from_dict(
        {"text": data_bundle["X_test"], "label": data_bundle["y_test"]}
    )

    # 3. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=False, max_length=256
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn, batched=True)

    # 4. Model Setup
    num_labels = len(set(data_bundle["y_train"]))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    ).to(device)

    # 5. Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "Accuracy": acc_metric.compute(predictions=predictions, references=labels)[
                "accuracy"
            ],
            "F1": f1_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )["f1"],
        }

    # 6. Minimal Training Args for maximum compatibility
    training_args = TrainingArguments(
        output_dir=f"./results/{name}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=10,
        fp16=False,  # Crucial for MPS (M4)
        report_to="none",
        dataloader_pin_memory=False,
    )

    # 7. Trainer - Changed 'tokenizer' to 'processing_class'
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        # In latest transformers, use processing_class instead of tokenizer
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return {
        "Dataset": name,
        "Model": model_checkpoint,
        "Accuracy": eval_results["eval_Accuracy"],
        "F1": eval_results["eval_F1"],
    }
