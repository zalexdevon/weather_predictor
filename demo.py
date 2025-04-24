# Load model lên
model = AutoModelForSeq2SeqLM.from_pretrained(f"Source_code/train_model_clack/model_47")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"Source_code/train_model_clack/tokenizer")

# Load val_dataset
val_ds = utils.load_python_object("Dataset/val_dataset_10_items/val_dataset_1.pkl")

eng_sentences, true_vie_sents = zip(*val_ds)

# Tiến hành dịch các câu trong sentences
## Đặt thiết bị: CUDA nếu có GPU, CPU nếu không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## Tokenize câu đầu vào
inputs = tokenizer(eng_sentences, return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

batch_size = 32
translated_text = []

for i in tqdm(range(0, len(eng_sentences), batch_size)):
    batch_sentences = eng_sentences[i : i + batch_size]

    inputs = tokenizer(
        batch_sentences, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=MAX_LENGTH, num_beams=2, early_stopping=False
        )

    decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated_text.extend(decoded_batch)


smooth = SmoothingFunction()
bleu_scores = []
for true, translated in zip(true_vie_sents, translated_text):
    bleu = sentence_bleu([true], translated, smoothing_function=smooth.method1)
    bleu_scores.append(bleu)

np.mean(bleu_scores)
