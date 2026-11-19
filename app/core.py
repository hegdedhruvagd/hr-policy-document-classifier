"""Complete HR Policy Document Classifier pipeline."""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"
MODEL_DIR = Path("artifacts/hr_policy_classifier")

CATEGORIES = [
    "LEAVE_ATTENDANCE",
    "COMPENSATION_BENEFITS",
    "RECRUITMENT_HIRING",
    "WORKPLACE_CONDUCT",
    "EXPENSE_REIMBURSEMENT",
]

LABEL_TOKENS = [f"<|{category}|>" for category in CATEGORIES]

HR_EXAMPLES = [
    {"text": "Employees receive 20 days of paid annual leave each year.", "category": "LEAVE_ATTENDANCE"},
    {"text": "Sick leave requires a medical certificate for absences longer than two days.", "category": "LEAVE_ATTENDANCE"},
    {"text": "Employees must record their attendance using the office system.", "category": "LEAVE_ATTENDANCE"},
    {"text": "Unapproved absence may result in disciplinary action.", "category": "LEAVE_ATTENDANCE"},
    {"text": "Unused annual leave can be carried forward up to five days.", "category": "LEAVE_ATTENDANCE"},
    {"text": "Monthly salaries are paid on the last working day.", "category": "COMPENSATION_BENEFITS"},
    {"text": "Employees are eligible for medical insurance coverage.", "category": "COMPENSATION_BENEFITS"},
    {"text": "Annual performance bonuses depend on individual performance.", "category": "COMPENSATION_BENEFITS"},
    {"text": "Salary revisions are reviewed during the annual appraisal.", "category": "COMPENSATION_BENEFITS"},
    {"text": "The company contributes to the employee provident fund.", "category": "COMPENSATION_BENEFITS"},
    {"text": "All candidates must submit a resume when applying.", "category": "RECRUITMENT_HIRING"},
    {"text": "Shortlisted applicants will be invited for an interview.", "category": "RECRUITMENT_HIRING"},
    {"text": "Background verification is required before employment.", "category": "RECRUITMENT_HIRING"},
    {"text": "New employees must complete the onboarding process.", "category": "RECRUITMENT_HIRING"},
    {"text": "The recruitment team evaluates candidates against job requirements.", "category": "RECRUITMENT_HIRING"},
    {"text": "Employees must treat colleagues with respect.", "category": "WORKPLACE_CONDUCT"},
    {"text": "Harassment and discrimination are strictly prohibited.", "category": "WORKPLACE_CONDUCT"},
    {"text": "Confidential company information must not be shared externally.", "category": "WORKPLACE_CONDUCT"},
    {"text": "Employees must disclose conflicts of interest.", "category": "WORKPLACE_CONDUCT"},
    {"text": "Personal use of company equipment must follow IT policies.", "category": "WORKPLACE_CONDUCT"},
    {"text": "Employees must submit travel expense claims with receipts.", "category": "EXPENSE_REIMBURSEMENT"},
    {"text": "Taxi expenses are reimbursed only for approved business travel.", "category": "EXPENSE_REIMBURSEMENT"},
    {"text": "Expense claims must be submitted within 30 days.", "category": "EXPENSE_REIMBURSEMENT"},
    {"text": "Hotel bookings must follow the company's travel budget.", "category": "EXPENSE_REIMBURSEMENT"},
    {"text": "Managers must approve reimbursement requests before payment.", "category": "EXPENSE_REIMBURSEMENT"},
]


def milestone_1():
    return {"categories": CATEGORIES.copy(), "examples": HR_EXAMPLES.copy()}


def milestone_2():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


class HRDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.rows = []
        for item in examples:
            prompt = f"Classify this HR policy paragraph:\n{item['text']}\nCategory:"
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True,
                max_length=max_length - 2
            )["input_ids"]
            label_id = tokenizer.convert_tokens_to_ids(
                f"<|{item['category']}|>"
            )
            input_ids = prompt_ids + [label_id, tokenizer.eos_token_id]
            labels = [-100] * len(prompt_ids) + [label_id, tokenizer.eos_token_id]
            self.rows.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class HRDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(
                item["input_ids"] + [self.tokenizer.pad_token_id] * pad
            )
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value) for key, value in batch.items()}


def milestone_3(epochs=8):
    tokenizer, model = milestone_2()
    dataset = HRDataset(HR_EXAMPLES, tokenizer)

    args = TrainingArguments(
        output_dir="artifacts/training",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=HRDataCollator(tokenizer),
    )
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return str(MODEL_DIR)


def load_classifier():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Trained model not found. Run milestone_3() first."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def classify_text(text, tokenizer=None, model=None):
    if tokenizer is None or model is None:
        tokenizer, model = load_classifier()

    prompt = f"Classify this HR policy paragraph:\n{text}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    label_ids = [tokenizer.convert_tokens_to_ids(t) for t in LABEL_TOKENS]
    scores = torch.softmax(logits[label_ids], dim=0)
    best = int(torch.argmax(scores).item())

    return {
        "category": CATEGORIES[best],
        "confidence": float(scores[best].item()),
    }


def milestone_4(text, tokenizer=None, model=None):
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    ]
    return [
        {"text": paragraph, **classify_text(paragraph, tokenizer, model)}
        for paragraph in paragraphs
    ]


def milestone_5():
    import streamlit as st

    st.set_page_config(page_title="HR Policy Classifier", page_icon="📄")
    st.title("HR Policy Document Classifier")
    st.write("Upload a plain-text HR policy document to classify its sections.")

    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        content = uploaded.getvalue().decode("utf-8", errors="replace")
        if st.button("Classify document", type="primary"):
            try:
                with st.spinner("Classifying sections..."):
                    results = milestone_4(content)
                st.success(f"Classified {len(results)} sections")
                for index, result in enumerate(results, 1):
                    st.subheader(f"Section {index}: {result['category']}")
                    st.caption(f"Confidence: {result['confidence']:.1%}")
                    st.write(result["text"])
                output = "\n\n".join(
                    f"[{r['category']} | {r['confidence']:.1%}]\n{r['text']}"
                    for r in results
                )
                st.download_button(
                    "Download classified document",
                    output,
                    file_name="classified_hr_policy.txt",
                    mime="text/plain",
                )
            except FileNotFoundError as error:
                st.error(str(error))
                st.info("Train the model first by running the training command.")
