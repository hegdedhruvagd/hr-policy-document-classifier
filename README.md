# HR Policy Document Classifier

A Streamlit application that classifies HR policy paragraphs using fine-tuned DistilGPT-2.

## Categories
LEAVE_ATTENDANCE, COMPENSATION_BENEFITS, RECRUITMENT_HIRING, WORKPLACE_CONDUCT, EXPENSE_REIMBURSEMENT

## Dataset
25 manually written and labelled examples (5 per category). No external training dataset was used.

## Install
python -m pip install -r requirements.txt

## Train
python -c "from app.core import milestone_3; milestone_3()"

## Run
streamlit run app/main.py

## Test
python -m pytest -q

## Results and limitations
A sample run processed five sections. All five were predicted as WORKPLACE_CONDUCT at approximately 22% confidence. This confirms the pipeline runs, not that predictions are reliable. No independent evaluation or validated accuracy, precision, recall or F1 score is available. This is a small-data prototype, not suitable for production HR decisions. Model weights are excluded from Git due to their size and must be generated locally by training.
