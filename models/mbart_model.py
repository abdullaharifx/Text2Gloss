from transformers import MBartForConditionalGeneration, MBartTokenizer

def load_model_and_tokenizer_cc25(model_name="facebook/mbart-large-cc25", lang="en_XX"):
    tokenizer = MBartTokenizer.from_pretrained(model_name, src_lang=lang, tgt_lang=lang)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[lang]
    return model, tokenizer
