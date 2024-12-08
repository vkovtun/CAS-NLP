from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login



# Load the pre-trained GPT-2 model and tokenizer
# model_name = 'gpt2-xl'  # You can also use 'gpt2-medium', 'gpt2-large', etc.
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# model_name = "meta-llama/Llama-3.3-70B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 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
login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openGPT-X/Teuken-7B-instruct-research-v0.4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True,
)

# model_name = "Qwen/QwQ-32B-Preview"
# model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your limited vocabulary
# limited_vocab = [
#     'dog', 'cat', 'man', 'woman', 'child', 'said', 'went', 'came', 'saw',
#     'was', 'were', 'on', 'in', 'with', 'and', 'but', 'the', 'a', 'an', 'to',
#     'of', 'for', 'it', 'he', 'she', 'they', 'had', 'has', 'have', 'that',
#     'as', 'at', 'from', 'by', 'her', 'his', 'house', 'day', 'night', 'walked',
#     'looked', 'found', 'thought', 'knew', 'felt', 'time', 'world', 'story'
# ]

pre_primer = ["a", "and", "away", "big", "blue", "can", "come", "down", "find", "for", "funny", "go", "help", "here", "I", "in", "is", "it", "jump", "little", "look", "make", "me", "my", "not", "one", "play", "red", "run", "said", "see", "the", "three", "to", "two", "up", "we", "where", "yellow", "you"]
primer = ["all", "am", "are", "at", "ate", "be", "black", "brown", "but", "came", "did", "do", "eat", "four", "get", "good", "have", "he", "into", "like", "must", "new", "no", "now", "on", "our", "out", "please", "pretty", "ran", "ride", "saw", "say", "she", "so", "soon", "that", "there", "they", "this", "too", "under", "want", "was", "well", "went", "what", "white", "who", "will", "with", "yes"]
grade_1 = ["after", "again", "an", "any", "as", "ask", "by", "could", "every", "fly", "from", "give", "going", "had", "has", "her", "him", "his", "how", "just", "know", "let", "live", "may", "of", "old", "once", "open", "over", "put", "round", "some", "stop", "take", "thank", "them", "then", "think", "walk", "were", "when"]
grade_2 = ["always", "around", "because", "been", "before", "best", "both", "buy", "call", "cold", "does", "don't", "fast", "first", "five", "found", "gave", "goes", "green", "its", "made", "many", "off", "or", "pull", "read", "right", "sing", "sit", "sleep", "tell", "their", "these", "those", "upon", "us", "use", "very", "wash", "which", "why", "wish", "work", "would", "write", "your"]
grade_3 = ["about", "better", "bring", "carry", "clean", "cut", "done", "draw", "drink", "eight", "fall", "far", "full", "got", "grow", "hold", "hot", "hurt", "if", "keep", "kind", "laugh", "light", "long", "much", "myself", "never", "only", "own", "pick", "seven", "shall", "show", "six", "small", "start", "ten", "today", "together", "try", "warm"]
nouns = ["apple", "baby", "back", "ball", "bear", "bed", "bell", "bird", "birthday", "boat", "box", "boy", "bread", "brother", "cake", "car", "cat", "chair", "chicken", "children", "Christmas", "coat", "corn", "cow", "day", "dog", "doll", "door", "duck", "egg", "eye", "farm", "farmer", "father", "feet", "fire", "fish", "floor", "flower", "game", "garden", "girl", "goat", "grass", "ground", "hand", "head", "hill", "home", "horse", "house", "kitty", "leg", "letter", "man", "men", "milk", "money", "morning", "mother", "name", "nest", "night", "paper", "party", "picture", "pig", "rabbit", "rain", "ring", "robin", "Santa Claus", "school", "seed", "sheep", "shoe", "sister", "snow", "song", "squirrel", "stick", "street", "sun", "table", "thing", "time", "top", "toy", "tree", "watch", "water", "way", "wind", "window", "woman", "women", "wood"]

# pre_primer = ["", "і", "далеко", "великий", "синій", "можна", "приходь", "вниз", "знайти", "для", "смішний", "йти", "допомогти", "тут", "я", "в", "є",
#               "це", "стрибати", "маленький", "дивитися", "робити", "мене", "мій", "не", "один", "грати", "червоний", "бігти", "сказав", "бачити", "",
#               "три", "до", "два", "вгору", "ми", "де", "жовтий", "ти"]
# primer = ["все", "", "є", "на", "з'їв", "бути", "чорний", "коричневий", "але", "прийшов", "зробив", "робити", "їсти", "чотири", "отримати", "добрий",
#           "мати", "він", "в", "подобатися", "повинен", "новий", "ні", "зараз", "на", "наш", "зовні", "будь ласка", "гарний", "біг", "їздити",
#           "побачив", "казати", "вона", "так", "скоро", "той", "там", "вони", "це", "також", "під", "хотіти", "був", "добре", "пішов", "що", "білий",
#           "хто", "буде", "з", "так"]
# grade_1 = ["після", "знову", "", "будь-який", "як", "питати", "від", "міг", "кожний", "літати", "з", "дати", "йду", "мав", "має", "її", "його",
#            "його", "як", "просто", "знати", "дозволити", "жити", "може", "з", "старий", "одного разу", "відкрити", "над", "покласти", "круглий",
#            "деякі", "зупинити", "брати", "дякую", "їх", "тоді", "думати", "ходити", "були", "коли"]
# grade_2 = ["завжди", "навколо", "тому що", "був", "до", "найкращий", "обидва", "купити", "дзвонити", "холодний", "робить", "не", "швидкий",
#            "перший", "п'ять", "знайшов", "дав", "йде", "його", "зробив", "багато", "від", "або", "тягнути", "читати", "правий", "співати", "сидіти",
#            "спати", "сказати", "їх", "ці", "ті", "на", "нам", "використовувати", "дуже", "мити", "який", "чому", "бажати", "працювати", "би",
#            "писати", "твій"]
# grade_3 = ["про", "кращий", "принести", "нести", "чистий", "різати", "зроблено", "малювати", "пити", "вісім", "падати", "далеко", "повний", "отримав",
#            "рости", "тримати", "гарячий", "боліти", "якщо", "тримати", "добрий", "сміятися", "світло", "довгий", "багато", "сам", "ніколи", "тільки",
#            "власний", "вибирати", "сім", "буду", "показати", "шість", "маленький", "почати", "десять", "сьогодні", "разом", "пробувати", "теплий"]
# nouns = ["яблуко", "дитина", "спина", "м'яч", "ведмідь", "ліжко", "дзвін", "птах", "день народження", "човен", "коробка", "хлопчик", "хліб", "брат",
#          "торт", "машина", "кіт", "стілець", "курка", "діти", "Різдво", "пальто", "кукурудза", "корова", "день", "собака", "лялька", "двері", "качка",
#          "яйце", "око", "ферма", "фермер", "батько", "ноги", "вогонь", "риба", "підлога", "квітка", "гра", "сад", "дівчинка", "коза", "трава",
#          "земля", "рука", "голова", "пагорб", "дім", "кінь", "будинок", "кошеня", "нога", "лист", "чоловік", "чоловіки", "молоко", "гроші", "ранок",
#          "мати", "ім'я", "гніздо", "ніч", "папір", "вечірка", "картина", "свиня", "кролик", "дощ", "кільце", "малинівка", "Святий Миколай", "школа",
#          "насіння", "вівця", "взуття", "сестра", "сніг", "пісня", "білка", "палиця", "вулиця", "сонце", "стіл", "річ", "час", "верх", "іграшка",
#          "дерево", "годинник", "вода", "шлях", "вітер", "вікно", "жінка", "жінки", "деревина"]

limited_vocab = pre_primer + primer + grade_1 + grade_2 + grade_3 + nouns

# with open('en_top_3000.txt', 'r') as file:
#     limited_vocab = [line.strip() for line in file]

punctuation=[".", ",", "?", "!", ";", ":", "-", "(", ")", "'", '"']
limited_vocab += punctuation


# Get the IDs of the words and punctuation in your vocabulary
limited_vocab_ids = []
for word in limited_vocab:
    # Handle both with and without prefix space
    tokens = tokenizer.encode(word, add_special_tokens=False)
    tokens_with_space = tokenizer.encode(' ' + word, add_special_tokens=False)
    limited_vocab_ids.extend(tokens)
    limited_vocab_ids.extend(tokens_with_space)

limited_vocab_ids = list(set(limited_vocab_ids))  # Remove duplicates

# Create a set of all token IDs and determine the forbidden tokens
all_token_ids = set(tokenizer.get_vocab().values())
allowed_token_ids = set(limited_vocab_ids)
forbidden_token_ids = all_token_ids - allowed_token_ids

# Convert forbidden token IDs into the format required by 'bad_words_ids'
bad_words_ids = [[token_id] for token_id in forbidden_token_ids]

# Function to generate text with constrained vocabulary
def generate_constrained_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
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

# Generate a story
prompt = "Long time ago,"
# prompt = "Колись давно"
story = generate_constrained_text(prompt, max_length=50)
print(story)

# Next steps
# - Try another model (LLAMA)
# - Think about performance metrics - words out of vocabulary, syntactic accuracy and semantic accuracy.