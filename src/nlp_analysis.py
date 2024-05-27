import spacy

def analyze_sentiment(text: str) -> float:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentiment = doc.sentiment
    return sentiment