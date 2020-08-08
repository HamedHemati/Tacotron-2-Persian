from collections import Counter
from tac2persian.utils.g2p.phonemizer_api.phonemize import phonemize
from tac2persian.utils.g2p.char_list import char_list, _punctuations, _pad


class Grapheme2Phoneme():
    def __init__(self):
        self.char_list = char_list
        self.punctutations = _punctuations
        # Char to id and id to char conversion
        self.char_to_id = {s: i for i, s in enumerate(self.char_list)}
        self.id_to_char = {i: s for i, s in enumerate(self.char_list)}
        
        # Set first and second languages in 'bilingual' mode
        self.set_bilingual_languages()

    def set_bilingual_languages(self, first_lang="fa", second_lang="en-us"):
        """Sets languages in bilingual mode."""
        self._bilingual_first_lang = first_lang
        self._bilingual_second_lang = second_lang

    def text_to_phone(self, text, language="fa"):
        """Converts text to phoneme."""
        # Count number stars (for bilingual mode)
        char_counts = Counter(text)
        even_stars = char_counts["*"] > 0 and char_counts["*"] % 2 == 0

        # If language is set to 'bilingual', split sentence with stars and convert each part separately
        if language == "bilingual" and even_stars:
            ph = ""
            for i, p in enumerate(text.split("*")):
                if i % 2 == 0:
                    ph_fa = phonemize(p, 
                                      strip=False, 
                                      with_stress=True, 
                                      preserve_punctuation=True, 
                                      punctuation_marks=self.punctutations,
                                      njobs=1, 
                                      backend='espeak', 
                                      language=self._bilingual_first_lang, 
                                      language_switch="remove-flags")
                    ph += ph_fa
                else:
                    ph_en = phonemize(p, 
                                      strip=False, 
                                      with_stress=True, 
                                      preserve_punctuation=True, 
                                      punctuation_marks=self.punctutations,
                                      njobs=1, 
                                      backend='espeak', 
                                      language=self._bilingual_second_lang, 
                                      language_switch="remove-flags")
                    ph += ph_en
        else:
            # If the language is 'bilingual' but no stars or not odd number of stars
            if language == "bilingual":
                lang = self._bilingual_first_lang
            # Otherwise
            else:
                lang = language
            ph = phonemize(text, 
                           strip=False, 
                           with_stress=True, 
                           preserve_punctuation=True, 
                           punctuation_marks=self.punctutations,
                           njobs=1, 
                           backend='espeak', 
                           language=lang, 
                           language_switch="remove-flags")
        return ph

    def _should_keep_char(self, 
                          p):
        """Checks if char is valid and is not pad char."""
        return p in self.char_list and p not in [_pad]

    def phone_to_sequence(self, phons):
        sequence = [self.char_to_id[s] for s in list(phons) if self._should_keep_char(s)]
        return sequence
        
    def text_to_sequence(self, text, language="de"):
        """Converts text to sequence of indices."""
        sequence = []
        # Get the phoneme for the text
        phons = self.text_to_phone(text, language=language)
        # Convert each phone to its corresponding index
        sequence = [self.char_to_id[s] for s in list(phons) if self._should_keep_char(s)]
        if sequence == []:
            print("!! After phoneme conversion the result is None. -- {} ".format(text))
        return sequence
