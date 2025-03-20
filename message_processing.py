import nltk
#nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist

from nltk.sentiment.vader import SentimentIntensityAnalyzer

training_data = [
    ("I love this movie", "positive"),
    ("Such an amazing experience","positive"),
    ("Best day of my life","positive"),
    ("This movie is terrible", "negative"),
    ("I hate this product", "negative"),
    ("Worst service ever", "negative"),
    ("Such a bad experience", "negative"),

]

sid = SentimentIntensityAnalyzer()
def analyze_sentiment(sentence):
    sentiment_score = sid.polarity_scores(sentence)
    if sentiment_score['compound'] > 0:
        return "positive"
    elif sentiment_score['compound'] < 0:
        return "negative"
    else:
        return "neutral"

new_sentences = ["I experience this movie", "Best day of my life"]
for sentence in new_sentences:
    print(f"Sentence: {sentence}, Sentiment: {analyze_sentiment(sentence)}")


###################Posterior probabilities
def categorize_sentence(sentence, classifier):
    features = extract_features(sentence)
    prob_dist = classifier.prob_classify(features)
    label = prob_dist.max()
    return f"{label} ({prob_dist.prob(label):.2f})"

def extract_features(sentence):
    words = set(word.lower() for word in word_tokenize(sentence) if word.lower() not in stopwords.words('english'))
    return {word: True for word in words}

def train_classifier(labeled_sentences):
    featuresets = [(extract_features(sentence), label) for (sentence, label) in labeled_sentences]
    return NaiveBayesClassifier.train(featuresets)

classifier = train_classifier(training_data)


new_sentences = ["I experience this movie", "Best day of my life"]

for sentence in new_sentences:
    print(f"Sentence: {sentence}, Sentiment (Naive Bayes): {categorize_sentence(sentence, classifier)}")
