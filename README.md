# HR Policy Document Classifier

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/duuuve63az/badge.svg)](https://ask.thili.ai/b/duuuve63az)

This project develops an HR Policy Document Classifier, a tool designed for HR administrators. It automatically categorizes internal policy documents, such as "leave policy" or "expense claim," facilitating quick retrieval and efficient departmental routing. The core is a fine-tuned GPT-2 model that assigns specific tags to document sections.

## What you are building

**Goal.** You will have built a Streamlit application that automatically classifies sections of HR policy documents into predefined categories using a fine-tuned GPT-2 model.

**Done when.** The project is complete when a fine-tuned GPT-2 model can accurately categorize HR document paragraphs by generating special label tokens, and a Streamlit app allows users to upload text files and view the classified sections.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect example HR document paragraphs and assign 3-5 categories.**  
      Hint: Identify distinct categories relevant to HR policies and manually label a dataset of paragraphs.
- [ ] **2. Extend the GPT-2 tokenizer and model for these HR categories.**  
      Hint: Add special tokens for your categories to the tokenizer and resize the model's embedding layer.
- [ ] **3. Train the model using the prepared dataset and Hugging Face Trainer.**  
      Hint: Prepare your dataset in a format suitable for Hugging Face `Trainer` and configure training arguments.
- [ ] **4. Implement a function to classify new document text.**  
      Hint: Use the fine-tuned model to predict category tokens for input paragraphs.
- [ ] **5. Create a Streamlit app to upload a text file and display classified sections.**  
      Hint: Build a user interface that takes text input, calls your classification function, and presents the results clearly.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display the confidence score for each classified section in the Streamlit app.
- Allow users to manually override or edit a classification within the Streamlit app interface.
- Implement a feature to download the classified document with the assigned tags embedded.

