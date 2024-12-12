from pprint import pprint

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import statistics


# Function to generate text with constrained vocabulary
def generate_constrained_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Function to map nltk POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


# Function to tokenize and lemmatize
def process_sentence(sentence):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Lowercase and remove punctuation
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Get part-of-speech tags
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return lemmatized_tokens


# Function that computes how many words are from the vocabulary
def compute_accuracy(sentence, vocabulary):
    lemmatized_tokens = process_sentence(sentence)
    values = list(map(lambda token: 1 if token in vocabulary else 0, lemmatized_tokens))
    for index, value in enumerate(values):
        if value == 0:
            print(f"Missing token: {lemmatized_tokens[index]}")
    return statistics.mean(values)


# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-xl'  # You can also use 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure pad_token_id is set to eos_token_id if not already
tokenizer.pad_token_id = tokenizer.eos_token_id

# # Define your limited vocabulary
# limited_vocab = [
#     'dog', 'cat', 'man', 'woman', 'child', 'say', 'go', 'come', 'see',
#     'is', 'on', 'in', 'with', 'and', 'but', 'the', 'a', 'an', 'to',
#     'of', 'for', 'it', 'he', 'she', 'they', 'have', 'that',
#     'as', 'at', 'from', 'by', 'her', 'his', 'house', 'day', 'night', 'walk',
#     'look', 'find', 'think', 'know', 'feel', 'time', 'world', 'story'
# ]

pre_primer = ["a", "and", "away", "big", "blue", "can", "come", "down", "find", "for", "funny", "go", "help", "here", "i", "in", "is", "it", "jump", "little", "look", "make", "me", "my", "not", "one", "play", "red", "run", "said", "see", "the", "three", "to", "two", "up", "we", "where", "yellow", "you"]
primer = ["all", "am", "are", "at", "ate", "be", "black", "brown", "but", "came", "did", "do", "eat", "four", "get", "good", "have", "he", "into", "like", "must", "new", "no", "now", "on", "our", "out", "please", "pretty", "ran", "ride", "saw", "say", "she", "so", "soon", "that", "there", "they", "this", "too", "under", "want", "was", "well", "went", "what", "white", "who", "will", "with", "yes"]
grade_1 = ["after", "again", "an", "any", "as", "ask", "by", "could", "every", "fly", "from", "give", "going", "had", "has", "her", "him", "his", "how", "just", "know", "let", "live", "may", "of", "old", "once", "open", "over", "put", "round", "some", "stop", "take", "thank", "them", "then", "think", "walk", "were", "when"]
grade_2 = ["always", "around", "because", "been", "before", "best", "both", "buy", "call", "cold", "does", "don't", "fast", "first", "five", "found", "gave", "goes", "green", "its", "made", "many", "off", "or", "pull", "read", "right", "sing", "sit", "sleep", "tell", "their", "these", "those", "upon", "us", "use", "very", "wash", "which", "why", "wish", "work", "would", "write", "your"]
grade_3 = ["about", "better", "bring", "carry", "clean", "cut", "done", "draw", "drink", "eight", "fall", "far", "full", "got", "grow", "hold", "hot", "hurt", "if", "keep", "kind", "laugh", "light", "long", "much", "myself", "never", "only", "own", "pick", "seven", "shall", "show", "six", "small", "start", "ten", "today", "together", "try", "warm"]
nouns = ["apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat", "box", "boy", "bread", "brother", "cake", "car", "cat", "chair", "chicken", "children", "Christmas", "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye", "farm", "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden", "girl", "goat", "grass", "ground", "hand", "head", "hill", "home", "horse", "house", "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name", "nest", "night", "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin", "Santa Claus", "school", "seed", "sheep", "shoe", "sister", "snow", "song", "squirrel", "stick", "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water", "way", "wind", "window", "woman", "women", "wood"]

limited_vocab = pre_primer + primer + grade_1 + grade_2 + grade_3 + nouns

# with open('en_top_3000.txt', 'r') as file:
#     limited_vocab = [line.strip() for line in file]

punctuation = ["ago", ".", ",", "?", "!", ";", ":", "-", "(", ")", "'", '"']
limited_vocab += punctuation


# Get the IDs of the words and punctuation in your vocabulary
limited_vocab_ids = []
for word in limited_vocab:
    # Handle both with and without prefix space
    tokens = tokenizer.encode(word, add_special_tokens=False)
    tokens_with_space = tokenizer.encode(' ' + word, add_special_tokens=False)
    limited_vocab_ids.extend(tokens)
    limited_vocab_ids.extend(tokens_with_space)

# Create a set of all token IDs and determine the forbidden tokens
all_token_ids = set(tokenizer.get_vocab().values())
allowed_token_ids = set(limited_vocab_ids)
forbidden_token_ids = all_token_ids - allowed_token_ids

# Convert forbidden token IDs into the format required by 'bad_words_ids'
bad_words_ids = [[token_id] for token_id in forbidden_token_ids]

# Generate a story
prompt = "Long time ago"
# story = generate_constrained_text(prompt, max_length=50)
# print(story)

# Evaluation

# Download required datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

for i in range(5):
    story = generate_constrained_text(prompt, max_length=50)
    print(f"{i}: {story}")
    print(f"Vocabulary accuracy: {compute_accuracy(story, limited_vocab)}")
