
# Phonemes definition
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(_vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))

_pad = '_'
_punctuations = '.!;:,?' # list of punctuations that espeak engine will keep after phonemization
_space = ' '

char_list = [_pad] + list(_phonemes) + list(_space) + list(_punctuations) 


if __name__ == "__main__":
    print(f"Char list:\n{char_list}")