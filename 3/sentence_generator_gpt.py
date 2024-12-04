from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-medium'  # You can also use 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

pre_primer = ["a", "and", "away", "big", "blue", "can", "come", "down", "find", "for", "funny", "go", "help", "here", "I", "in", "is", "it", "jump", "little", "look", "make", "me", "my", "not", "one", "play", "red", "run", "said", "see", "the", "three", "to", "two", "up", "we", "where", "yellow", "you"]
primer = ["all", "am", "are", "at", "ate", "be", "black", "brown", "but", "came", "did", "do", "eat", "four", "get", "good", "have", "he", "into", "like", "must", "new", "no", "now", "on", "our", "out", "please", "pretty", "ran", "ride", "saw", "say", "she", "so", "soon", "that", "there", "they", "this", "too", "under", "want", "was", "well", "went", "what", "white", "who", "will", "with", "yes"]
grade_1 = ["after", "again", "an", "any", "as", "ask", "by", "could", "every", "fly", "from", "give", "going", "had", "has", "her", "him", "his", "how", "just", "know", "let", "live", "may", "of", "old", "once", "open", "over", "put", "round", "some", "stop", "take", "thank", "them", "then", "think", "walk", "were", "when"]
grade_2 = ["always", "around", "because", "been", "before", "best", "both", "buy", "call", "cold", "does", "don't", "fast", "first", "five", "found", "gave", "goes", "green", "its", "made", "many", "off", "or", "pull", "read", "right", "sing", "sit", "sleep", "tell", "their", "these", "those", "upon", "us", "use", "very", "wash", "which", "why", "wish", "work", "would", "write", "your"]
grade_3 = ["about", "better", "bring", "carry", "clean", "cut", "done", "draw", "drink", "eight", "fall", "far", "full", "got", "grow", "hold", "hot", "hurt", "if", "keep", "kind", "laugh", "light", "long", "much", "myself", "never", "only", "own", "pick", "seven", "shall", "show", "six", "small", "start", "ten", "today", "together", "try", "warm"]
nouns = ["apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat", "box", "boy", "bread", "brother", "cake", "car", "cat", "chair", "chicken", "children", "Christmas", "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye", "farm", "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden", "girl", "goat", "grass", "ground", "hand", "head", "hill", "home", "horse", "house", "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name", "nest", "night", "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin", "Santa Claus", "school", "seed", "sheep", "shoe", "sister", "snow", "song", "squirrel", "stick", "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water", "way", "wind", "window", "woman", "women", "wood"]
punctuation=[".", ",", "?", "!", ";", ":", "-"]


# Define your limited vocabulary
# limited_vocab = [
#     'dog', 'cat', 'man', 'woman', 'child', 'said', 'went', 'came', 'saw',
#     'was', 'were', 'on', 'in', 'with', 'and', 'but', 'the', 'a', 'an', 'to',
#     'of', 'for', 'it', 'he', 'she', 'they', 'had', 'has', 'have', 'that',
#     'as', 'at', 'from', 'by', 'her', 'his', 'house', 'day', 'night', 'walked',
#     'looked', 'found', 'thought', 'knew', 'felt', 'time', 'world', 'story'
# ]

limited_vocab = pre_primer + primer + grade_1 + grade_2 + grade_3 + nouns + punctuation

# Get the IDs of the words in your vocabulary
limited_vocab_ids = []
for word in limited_vocab:
    tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
    limited_vocab_ids.extend(tokens)

limited_vocab_ids = list(set(limited_vocab_ids))  # Remove duplicates

# Create a set of all token IDs and determine the forbidden tokens
all_token_ids = set(tokenizer.get_vocab().values())
forbidden_token_ids = all_token_ids - set(limited_vocab_ids)

# Convert forbidden token IDs into the format required by 'bad_words_ids'
bad_words_ids = [[token_id] for token_id in forbidden_token_ids]

# Function to generate text with constrained vocabulary
def generate_constrained_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Generate a story
prompt = "Once upon a time"
story = generate_constrained_text(prompt, max_length=200)
print(story)
