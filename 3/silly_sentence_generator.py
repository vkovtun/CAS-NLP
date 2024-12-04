import random

# Define your vocabulary
vocabulary = {
    'Noun': ["apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat", "box", "boy", "bread", "brother", "cake", "car",
             "cat", "chair", "chicken", "children", "Christmas", "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye", "farm",
             "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden", "girl", "goat", "grass", "ground", "hand", "head",
             "hill", "home", "horse", "house", "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name", "nest", "night",
             "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin", "Santa Claus", "school", "seed", "sheep", "shoe", "sister",
             "snow", "song", "squirrel", "stick", "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water", "way", "wind",
             "window", "woman", "women", "wood"],
    'Verb': ["am", "are", "ate", "be", "came", "can", "come", "did", "do", "does", "don't", "draw", "drink", "eat", "fall", "fly", "found", "gave",
             "get", "give", "go", "goes", "grow", "had", "has", "have", "help", "hurt", "is", "jump", "keep", "know", "laugh", "let", "like", "live",
             "look", "made", "make", "may", "must", "pick", "play", "put", "ran", "read", "ride", "run", "saw", "say", "see", "show", "sing", "sit",
             "sleep", "start", "take", "tell", "thank", "think", "try", "use", "walk", "want", "was", "went", "were", "will", "wish", "work",
             "write"],
    'Adjective': ["big", "black", "blue", "brown", "clean", "cold", "fast", "first", "five", "four", "funny", "good", "green", "hot", "kind", "light",
                  "little", "long", "many", "much", "new", "old", "one", "pretty", "red", "round", "seven", "six", "small", "ten", "three", "two",
                  "white", "yellow"],
    'Determiner': ["a", "an", "any", "her", "his", "its", "my", "one", "our", "some", "that", "the", "their", "these", "this", "those", "your"],
    'Preposition': ["about", "around", "at", "before", "by", "for", "from", "in", "into", "of", "off", "on", "out", "over", "to", "under", "upon",
                    "with"],
    'Pronoun': ["I", "he", "him", "me", "she", "they", "us", "we", "who", "you"],
    'Conjunction': ["and", "but", "or", "so", "because"],
    'Adverb': ["again", "always", "away", "here", "just", "never", "no", "not", "now", "soon", "there", "together", "too", "very", "well", "where",
               "why", "yes", "yet"]
}

grammar = {
    'S': [['NP', 'VP'], ['Pronoun', 'VP'], ['NP', 'VP', 'Conjunction', 'S'], ['VP']],
    'NP': [['Determiner', 'Noun'], ['Determiner', 'Adjective', 'Noun'], ['Pronoun'], ['Adjective', 'Noun'], ['NP', 'PP'], ['NP', 'Conjunction', 'NP']],
    'VP': [['Verb', 'NP'], ['Verb', 'PP'], ['Verb', 'Adverb'], ['Verb'], ['Verb', 'Adverb', 'PP'], ['Verb', 'NP', 'Conjunction', 'VP']],
    'PP': [['Preposition', 'NP'], ['Preposition', 'Determiner', 'Noun']],
    'Determiner': vocabulary['Determiner'],
    'Noun': vocabulary['Noun'],
    'Verb': vocabulary['Verb'],
    'Adjective': vocabulary['Adjective'],
    'Preposition': vocabulary['Preposition'],
    'Pronoun': vocabulary['Pronoun'],
    'Conjunction': vocabulary['Conjunction'],
    'Adverb': vocabulary['Adverb'],
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
