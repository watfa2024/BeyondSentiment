from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
FIGURE_DIR = ROOT / 'figures'
PR_FILE = DATA_DIR / 'pr_data.csv'
COMMENTS_FILE = DATA_DIR / 'comments_with_sentiment.csv'
MILESTONE_FILE = DATA_DIR / 'milestone_dataset.csv'
ISSUES_FILE = DATA_DIR / 'issues.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
PHI_HIGH_QUANTILE = 0.75
