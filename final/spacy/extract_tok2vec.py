import sys
import spacy

if len(sys.argv) != 3:
    print("Usage: extract_tok2vec.py <path_to_model> <output_tok2vec.bin>")
    sys.exit(1)

model_path = sys.argv[1]
output_path = sys.argv[2]

nlp = spacy.load(model_path)
tok2vec = nlp.get_pipe("transformer")

tok2vec.to_disk(output_path)
print(f"Saved tok2vec weights to {output_path}")
