from transformers import AutoModelForCausalLM, GPT2TokenizerFast, LogitsProcessor, LogitsProcessorList
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import torch

pre_primer = ["a", "and", "away", "big", "blue", "can", "come", "down", "find", "for", "funny", "go", "help", "here", "I", "in", "is", "it", "jump", "little", "look", "make", "me", "my", "not", "one", "play", "red", "run", "said", "see", "the", "three", "to", "two", "up", "we", "where", "yellow", "you"]
primer = ["all", "am", "are", "at", "ate", "be", "black", "brown", "but", "came", "did", "do", "eat", "four", "get", "good", "have", "he", "into", "like", "must", "new", "no", "now", "on", "our", "out", "please", "pretty", "ran", "ride", "saw", "say", "she", "so", "soon", "that", "there", "they", "this", "too", "under", "want", "was", "well", "went", "what", "white", "who", "will", "with", "yes"]
grade_1 = ["after", "again", "an", "any", "as", "ask", "by", "could", "every", "fly", "from", "give", "going", "had", "has", "her", "him", "his", "how", "just", "know", "let", "live", "may", "of", "old", "once", "open", "over", "put", "round", "some", "stop", "take", "thank", "them", "then", "think", "walk", "were", "when"]
grade_2 = ["always", "around", "because", "been", "before", "best", "both", "buy", "call", "cold", "does", "don't", "fast", "first", "five", "found", "gave", "goes", "green", "its", "made", "many", "off", "or", "pull", "read", "right", "sing", "sit", "sleep", "tell", "their", "these", "those", "upon", "us", "use", "very", "wash", "which", "why", "wish", "work", "would", "write", "your"]
grade_3 = ["about", "better", "bring", "carry", "clean", "cut", "done", "draw", "drink", "eight", "fall", "far", "full", "got", "grow", "hold", "hot", "hurt", "if", "keep", "kind", "laugh", "light", "long", "much", "myself", "never", "only", "own", "pick", "seven", "shall", "show", "six", "small", "start", "ten", "today", "together", "try", "warm"]
nouns = ["apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat", "box", "boy", "bread", "brother", "cake", "car", "cat", "chair", "chicken", "children", "Christmas", "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye", "farm", "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden", "girl", "goat", "grass", "ground", "hand", "head", "hill", "home", "horse", "house", "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name", "nest", "night", "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin", "Santa Claus", "school", "seed", "sheep", "shoe", "sister", "snow", "song", "squirrel", "stick", "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water", "way", "wind", "window", "woman", "women", "wood"]

dolch_word_list = pre_primer + primer + grade_1 + grade_2 + grade_3 + nouns

# Vocabulary
vocab_list = ['[PAD]', '[UNK]', '[EOS]', '.', ',', '!', '?', ';', ':'] + dolch_word_list
vocab = {word: idx for idx, word in enumerate(vocab_list)}

# Initialize the tokenizer with a word-level model
word_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))

# Set up pre-tokenizer and decoder for word-level processing
word_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
word_tokenizer.decoder = decoders.WordPiece()

# Convert tokens to token IDs
vocabulary_token_ids = set(vocab.values())

class VocabularyFilterLogitsProcessor(LogitsProcessor):
    def __init__(self, vocabulary_token_ids):
        self.vocabulary_token_ids = vocabulary_token_ids

    def __call__(self, input_ids, scores):
        # Create a mask over the vocabulary tokens
        vocab_mask = torch.full(scores.shape, -float('inf'), dtype=scores.dtype, device=scores.device)
        vocab_mask[..., list(self.vocabulary_token_ids)] = 0.0
        # Set scores for tokens not in the vocabulary to negative infinity
        scores += vocab_mask
        return scores

# Initialize the logits processor with your vocabulary
vocabulary_processor = VocabularyFilterLogitsProcessor(vocabulary_token_ids)
logits_processor = LogitsProcessorList([vocabulary_processor])

# Set the initial prompt
prompt = "Here is a sentence:"
input_ids = word_tokenizer.encode(prompt).ids

# Load a word-level compatible model
# model = AutoModelForCausalLM.from_pretrained("leroyrr/roberta-ulmfit", is_decoder=True)
model = AutoModelForCausalLM.from_pretrained("leroyrr/bert-base-head")

# Generate text
input_ids = torch.tensor([input_ids], dtype=torch.long)  # Convert input_ids to a tensor
output_ids = model.generate(
    input_ids,
    max_length=50,
    logits_processor=logits_processor,
    do_sample=True,
    temperature=0.9,
    top_k=10,
    eos_token_id=vocab['[EOS]'],
    pad_token_id=vocab['[PAD]'],
)

# Decode the generated tokens
generated_text = word_tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(generated_text)
