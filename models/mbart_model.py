from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model_and_tokenizer(model_name="facebook/mbart-large-50", lang="en_XX"):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = lang
    tokenizer.tgt_lang = lang
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[lang]
    return model, tokenizer
