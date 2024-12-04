from transformers import AutoModelForCausalLM, GPT2TokenizerFast, LogitsProcessor, LogitsProcessorList, TrainingArguments, Trainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

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
model = AutoModelForCausalLM.from_pretrained("leroyrr/bert-base-head", is_decoder=True)
# model = AutoModelForCausalLM.from_pretrained("vocab-transformers/distilbert-word2vec_256k-MLM_best")
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.to('cuda')  # Move the model to GPU

# Load Simple English Wikipedia dataset
dataset = load_dataset("rahular/simple-wikipedia", split="train")
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='no',
    save_strategy='no',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    load_best_model_at_end=True,
    # report_to='none'  # Disable built-in progress bars
)

# Define the tokenizer function for dataset processing
def tokenize_function(examples):
    max_length = 512
    return {'input_ids': [word_tokenizer.encode(text).ids[:max_length] for text in examples['text']]}

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids'])

# Define a custom loss function to add a penalty for out-of-vocabulary tokens
def custom_loss_function(logits, labels, vocab_token_ids, penalty_weight=5.0):
    # Apply penalty for out-of-vocabulary tokens
    vocab_mask = torch.full(logits.shape, -float('inf'), dtype=logits.dtype, device=logits.device)
    vocab_mask[..., list(vocab_token_ids)] = 0.0  # No penalty for in-vocab tokens
    logits += vocab_mask * penalty_weight

    # Calculate cross-entropy loss
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

# Define a custom Trainer class to incorporate the custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['input_ids']
        loss = custom_loss_function(logits, labels, vocabulary_token_ids, penalty_weight=5.0)
        return (loss, outputs) if return_outputs else loss

# Define a custom data collator for padding
class CustomDataCollator:
    def __call__(self, features):
        # Extract input_ids from each example and pad them to the same length
        input_ids = [torch.tensor(feature['input_ids'], dtype=torch.long) for feature in features]
        max_length = 512
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=vocab['[PAD]'])
        padded_input_ids = padded_input_ids[:, :max_length]  # Truncate if needed
        return {'input_ids': padded_input_ids}

data_collator = CustomDataCollator()

# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator  # Add data collator to handle padding
)

# Train the model
trainer.train()

# Generate text
input_ids = torch.tensor([input_ids], dtype=torch.long).to('cuda')  # Convert input_ids to a tensor and move to GPU
output_ids = model.generate(
    input_ids,
    max_length=50,
    logits_processor=logits_processor,
    do_sample=True,
    temperature=0.9,
    top_k=1000,
    eos_token_id=vocab['[EOS]'],
    pad_token_id=vocab['[PAD]'],
)

# Decode the generated tokens
generated_text = word_tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(generated_text)
