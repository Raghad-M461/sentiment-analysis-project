from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1) The dataset: are short customer feedback sentences, each labelled
#    as Positive or Negative.

texts = [
    'A fantastic quality overall.',
    'A friendly update overall.',
    'A great website overall.',
    'A helpful delivery overall.',
    'A perfect service overall.',
    'A reliable service overall.',
    'A wonderful onboarding overall.',
    'An impressive product overall.',
    'Honestly the dashboard is great.',
    'Honestly the service is impressive.',
    'Honestly the setup process is perfect.',
    'I found the delivery really impressive.',
    'I found the experience really outstanding.',
    'I found the platform really great.',
    'I found the service really wonderful.',
    'I found the tool really impressive.',
    'I found the update reall
