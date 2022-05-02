import aksharamukha.transliterate as transliterate
import os, sys, shutil

directory = sys.argv[1]
src_filename = sys.argv[2]
tgt_filename = sys.argv[3]
src_lang = sys.argv[4]
tgt_lang = sys.argv[5]
intrm_lang = sys.argv[6]
model_desc = sys.argv[7]

parent_dir = os.getcwd()
#path1 = os.path.join(parent_dir, "data-bin", directory)
#path2 = os.path.join(parent_dir, "checkpoints")
#path3 = os.path.join(parent_dir, "output_custom_fldr", directory)
path3 = os.path.join(parent_dir, directory)
#path4 = os.path.join(parent_dir, "log_custom_fldr", directory)
#path5 = os.path.join(parent_dir, "examples/translation", directory)
#path6 = os.path.join(parent_dir, "examples/translation", directory, "tmp")
#path7 = os.path.join(path2, directory)
"""
if os.path.isdir(path1):
    print(path1 + " already exists.")
else: 
    os.mkdir(path1)
    print(path1 + "successfully created.")
if os.path.isdir(path7):
    print(path7 + " already exists.")
else:
    os.mkdir(path7)
    print(path7 + "successfully created.")
if os.path.isdir(path3):
    print(path3 + " already exists.")
else:
    os.mkdir(path3)
    print(path3 + "successfully created.")
if os.path.isdir(path4):
    print(path4 + " already exists.")
else:
    os.mkdir(path4)
    print(path4 + "successfully created.")
if os.path.isdir(path5):
    print(path5 + " already exists.")
else:
    os.mkdir(path5)
    print(path5 + "successfully created.")
if os.path.isdir(path6):
    print(path6 + " already exists.")
else:
    os.mkdir(path6)
    print(path6 + "successfully created.")

src_raw_file = path5+"/../data-prep/"+src_filename
tgt_raw_file = path5+"/../data-prep/"+tgt_filename

if os.path.exists(src_raw_file):
    shutil.copy2(src_raw_file, path6)
    print("src file copied successfully.")
else:
    print("src file doesnot exist.")
    sys.exit()
if os.path.exists(tgt_raw_file):
    shutil.copy2(tgt_raw_file, path6)
    print("tgt file copied successfully.")
else:
    print("tgt file doesnot exists.")
    sys.exit()

os.chdir(path6)
print(os.getcwd())
"""
src_ext = os.path.splitext(src_filename)[1].split('.')[1]
tgt_ext = os.path.splitext(tgt_filename)[1].split('.')[1]
print(src_ext)
print(tgt_ext)
"""
for ext in [src_ext, tgt_ext]:
    sentences1 = []
    sentences2 = []
    count1 = 1
    count2 = 1
    if ext == src_ext:
        with open(f"{src_filename}", 'r') as f_read_ptr:
            sentences1 = f_read_ptr.readlines()
        trans_src = src_lang
        trans_tgt = intrm_lang
    if ext == tgt_ext:
        with open(f"{tgt_filename}", 'r') as f_read_ptr:
            sentences1 = f_read_ptr.readlines()
        trans_src = tgt_lang
        trans_tgt = intrm_lang

    with open(f"valid.{ext}", 'w') as f_write_ptr1:
        for sentence in sentences1:
            if count1%23 == 0:
                processed_sentence = transliterate.process(trans_src, trans_tgt, sentence)
                f_write_ptr1.write(processed_sentence)
            else:
                sentences2.append(sentence)
            count1 += 1

    with open(f"test.{ext}", 'w') as f_write_ptr1, open(f"train.{ext}", 'w') as f_write_ptr2, open(f"train.{src_ext}-{tgt_ext}", 'a') as f_write_ptr3:
        for sentence in sentences2:
            if count2%23 == 0:
                processed_sentence = transliterate.process(trans_src, trans_tgt, sentence)
                f_write_ptr1.write(processed_sentence)
            else:
                processed_sentence = transliterate.process(trans_src, trans_tgt, sentence)
                f_write_ptr2.write(processed_sentence)
                f_write_ptr3.write(processed_sentence)
            count2 += 1

comb_train = path6 + "/train." + src_ext + "-" + tgt_ext
bpe_tokens = "10000"
code_file = path5 + "/code"
learn_bpe_path = path5 + "/../subword-nmt/subword_nmt/learn_bpe.py"
apply_bpe_path = path5 + "/../subword-nmt/subword_nmt/apply_bpe.py"
learn_bpe_command = "python " + learn_bpe_path + " -s " + bpe_tokens + " < " + comb_train + " > " + code_file

print("running learn_bpe.pl ")
os.system(learn_bpe_command)

for ext in [src_ext, tgt_ext]:
    for f in ["train", "valid", "test"]:
        fullname = f +"."+ ext
        fullpath = path6 + "/" + fullname
        finalpath = path5 + "/" + fullname
        print("apply_bpe.py to {}".format(fullname))
        apply_bpe_command = "python " + apply_bpe_path + " -c " + code_file + " < " + fullpath + " > " + finalpath
        os.system(apply_bpe_command)

os.chdir(parent_dir)

print("executing preprocess command.")
preprocess_cmd = "fairseq-preprocess --source-lang " + src_ext + " --target-lang " + tgt_ext + " --trainpref " + path5 + "/train --validpref " + path5 + "/valid --testpref " + path5 + "/test --destdir " + path1 + " --workers 20"
os.system(preprocess_cmd)


print("Starting the training of the model.")
train_cmd = "CUDA_VISIBLE_DEVICES=0 fairseq-train " + path1 + " --arch transformer_iwslt_de_en --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --keep-last-epochs 1 --max-epoch 350 --fp16 > " + path4 + "/train_log.txt"
os.system(train_cmd)
"""

output_test_file = path3 + "/bkp_custom/" + model_desc + "/output_test_" + src_ext + "-" + tgt_ext + ".txt"
output_train_file = path3 + "/bkp_custom/" + model_desc + "/output_train_" + src_ext + "-" + tgt_ext + ".txt"
output_valid_file = path3 + "/bkp_custom/" + model_desc + "/output_valid_" + src_ext + "-" + tgt_ext + ".txt"
"""
print("running evaluation command for test.")
evaluate_cmd = "fairseq-generate " + path1 + " --path " + path2 + "/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > " + output_test_file
os.system(evaluate_cmd)

os.chdir(path5)

for ext in [src_ext, tgt_ext]:
    os.rename("test."+ext, "org_test."+ext)

for f in ["train", "valid"]:
    for ext in [src_ext, tgt_ext]:
        shutil.copy2(f + "." + ext, "orig_" + f + "." + ext)
        os.rename(f + "." + ext , "test." + ext)
    if f == "train":
        evaluate_cmd = "fairseq-generate " + path1 + " --path " + path2 + "/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > " + output_train_file
    elif f == "valid":
        evaluate_cmd = "fairseq-generate " + path1 + " --path " + path2 + "/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > " + output_valid_file
    print("running evaluation command for {}.".format(f))
    os.system(evaluate_cmd)
    for ext in [src_ext, tgt_ext]:
        os.rename("orig_" + f + "." + ext, f + "." + ext)

for ext in [src_ext, tgt_ext]:
    os.rename("org_test." + ext, "test." + ext)
"""
os.chdir(path3)
print("Moving to directory :", os.getcwd())

for filename in [output_test_file, output_train_file, output_valid_file]: 
    sentences = []
    with open(f"{filename}", 'r') as f_read_ptr:
        sentences = f_read_ptr.readlines()
    print("Sentences stored for {}".format(filename))

    os.rename(filename, filename+"_intrm")
    print("Renaming to intermediate file for {}".format(filename))

    with open(f"{filename}", 'w') as f_write_ptr:
        for sentence in sentences:
            if sentence.startswith("S-") and src_lang != 'en':
                sent_split = sentence.split('\t', 1)
                f_write_ptr.write(sent_split[0])
                f_write_ptr.write("\t")
                processed_sent = transliterate.process(intrm_lang, src_lang, sent_split[1])
                f_write_ptr.write(processed_sent)
            elif sentence.startswith("T-") and tgt_lang != 'en':
                sent_split = sentence.split('\t', 1)
                f_write_ptr.write(sent_split[0])
                f_write_ptr.write("\t")
                processed_sent = transliterate.process(intrm_lang, tgt_lang, sent_split[1])
                f_write_ptr.write(processed_sent)
            elif (sentence.startswith("H-") or sentence.startswith("D-")) and tgt_lang != 'en':
                sent_split = sentence.split('\t', 2)
                f_write_ptr.write(sent_split[0])
                f_write_ptr.write("\t")
                f_write_ptr.write(sent_split[1])
                f_write_ptr.write("\t")
                processed_sent = transliterate.process(intrm_lang, tgt_lang, sent_split[2])
                f_write_ptr.write(processed_sent)
            elif sentence.startswith("P-"):
                f_write_ptr.write(sentence)
            else:
            	f_write_ptr.write(sentence)
    print("Successfully Completed for {}".format(filename))

#print("running model stats bash script.")
#os.chdir(parent_dir)
#os.system("bash get_model_stats.sh -n " + directory)

print("All the steps completed.")
