# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from g2p.g2p.japanese import japanese_to_ipa
from g2p.g2p.mandarin import chinese_to_ipa
from g2p.g2p.english import english_to_ipa
from g2p.g2p.french import french_to_ipa
from g2p.g2p.korean import korean_to_ipa
from g2p.g2p.german import german_to_ipa


def cjekfd_cleaners(text, sentence, language, text_tokenizers):
    tokenizer = text_tokenizers.get(language)
    if tokenizer is None:
        print(f"Warning: No tokenizer available for language '{language}', skipping.")
        return ""

    if language == "zh":
        return chinese_to_ipa(text, sentence, tokenizer)
    elif language == "ja":
        return japanese_to_ipa(text, tokenizer)
    elif language == "en":
        return english_to_ipa(text, tokenizer)
    elif language == "fr":
        return french_to_ipa(text, tokenizer)
    elif language == "ko":
        return korean_to_ipa(text, tokenizer)
    elif language == "de":
        return german_to_ipa(text, tokenizer)
    else:
        raise Exception("Unknown language: %s" % language)
        return None
