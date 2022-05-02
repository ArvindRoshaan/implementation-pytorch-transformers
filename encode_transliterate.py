## Transliterate source language file data to target language provided in the arguments
# first argument > file extension
# second argument > input language
# third argument > output language

import glob, sys
import aksharamukha.transliterate as transliterate

file_ext = sys.argv[1]
input_lang = sys.argv[2]
output_lang = sys.argv[3]

for filename in ['train', 'test', 'valid']:
    sentences = []
    with open(f"{filename}.{file_ext}", 'r') as f_read_ptr:
        sentences = f_read_ptr.readlines()
    with open(f"{filename}.{file_ext}_temp", 'w') as f_write_ptr:
        for sentence in sentences:
            processed = transliterate.process(input_lang, output_lang, sentence)
            f_write_ptr.write(processed)
    filenm = filename + "." + file_ext
    print(filenm)
