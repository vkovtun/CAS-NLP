import random

# Define your vocabulary
vocabulary = {
    'Noun': ['dog', 'cat', 'man', 'woman', 'apple', 'park', 'car'],
    'Verb': ['eats', 'drives', 'sees', 'likes', 'loves', 'hates', 'finds'],
    'Adjective': ['big', 'small', 'red', 'green', 'fast', 'slow'],
    'Determiner': ['the', 'a'],
    'Preposition': ['in', 'on', 'under', 'with', 'without']
}

grammar = {
    'S': [['NP', 'VP']],
    'NP': [['Determiner', 'Noun'], ['Determiner', 'Adjective', 'Noun']],
    'VP': [['Verb', 'NP'], ['Verb', 'PP']],
    'PP': [['Preposition', 'NP']],
    'Determiner': vocabulary['Determiner'],
    'Noun': vocabulary['Noun'],
    'Verb': vocabulary['Verb'],
    'Adjective': vocabulary['Adjective'],
    'Preposition': vocabulary['Preposition']
}

def generate(symbol):
    if symbol in vocabulary:
        return random.choice(vocabulary[symbol])
    else:
        expansion = random.choice(grammar[symbol])
        return ' '.join(generate(sym) for sym in expansion)

# Generate sentences
for _ in range(10):
    sentence = generate('S')
    print(sentence.capitalize() + '.')
