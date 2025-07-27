import evaluate

def evaluate_model(model, tokenizer, val_texts, val_glosses, device):
    model.eval()
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    references, predictions = [], []

    for text, gloss in zip(val_texts, val_glosses):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        pred_gloss = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_gloss.split())
        references.append([gloss.split()])

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_output = rouge_metric.compute(
        predictions=[" ".join(p) for p in predictions],
        references=[" ".join(r[0]) for r in references],
        use_stemmer=True
    )

    return bleu_score, rouge_output
