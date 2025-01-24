import spacy
from spacy.tokens import DocBin

# Load your spaCy binary data
# doc_bin = DocBin().from_disk("/home/viktor/Work/Projects/UniBe/CAS-NLP/4/en/validation/validation1.spacy")
doc_bin = DocBin().from_disk("/home/viktor/Work/Projects/UniBe/CAS-NLP/4/en/train/train1.spacy")

# Load a blank NLP model (adjust the language as needed, e.g., "en")
nlp = spacy.blank("en")

# Convert the DocBin to a JSON-compatible format
docs = list(doc_bin.get_docs(nlp.vocab))
json_data = []

for doc in docs:
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    json_data.append({
        "text": doc.text,
        "entities": entities
    })

# Print JSON-formatted data
import json
print(json.dumps(json_data, indent=2))