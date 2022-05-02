import re
import sys, os

directory = sys.argv[1]
input_lang = sys.argv[2]
output_lang = sys.argv[3]

parent_dir = os.getcwd()
path1 = os.path.join(parent_dir, directory, "tmp")
os.chdir(path1)

for i in [input_lang, output_lang]:
	if i == 'en':
		for initial in ['train', 'test', 'valid']:
			sentences = []
			if os.path.exists(initial+"."+i):
				with open(f"{initial}.{i}", 'r') as f_read_ptr:
					sentences = f_read_ptr.readlines()
				os.rename(initial+"."+i, initial+"."+i+"_temp")
				with open(f"{initial}.{i}", 'w') as f_write_ptr:
					for sentence in sentences:
						temp = sentence.lower()
						temp = re.sub(r"[^a-z0-9\$&£:;\./\-\\\'\n ]*", '', temp)
						temp = re.sub(r"[ ]*\'((?=[a-z ]))", '', temp)

						temp = re.sub(r"((?<=[0-9 ])+)\/((?=[ 0-9])+)", ' by ', temp)
						temp = re.sub(r"((?<=[0-9 ]){2,})\-((?=[ 0-9]){2,})", ' to ', temp)

						temp = temp.replace(':', '.').replace(';', ',').replace('\\', ' ').replace('/', ' ').replace(',', ' , ').replace('-', ' ').replace('&', ' and ')
						temp = re.sub(r"((?<=[a-z0-9 ])*)\$((?=[0-9])+)", ' dollar ', temp)
						temp = re.sub(r"((?<=[a-z0-9 ])*)\$((?![0-9])+)", '', temp)
						temp = re.sub(r"((?<=[a-z0-9 ])*)\£((?=[0-9])+)", ' euro ', temp)
						temp = re.sub(r"((?<=[a-z0-9 ])*)\£((?![0-9])+)", '', temp)
						f_write_ptr.write(temp)
	elif i == 'hi':
		for initial in ['train', 'test', 'valid']:
			sentences = []
			if os.path.exists(initial+"."+i):
				with open(f"{initial}.{i}", 'r') as f_read_ptr:
					sentences = f_read_ptr.readlines()
				os.rename(initial+"."+i, initial+"."+i+"_temp")
				with open(f"{initial}.{i}", 'w') as f_write_ptr:
					for sentence in sentences:
#						temp = sentence.lower()
						temp = sentence
#						temp = re.sub(r"((?<=[0-9 ])+)\/((?=[ 0-9])+)", ' by ', temp)
						temp = re.sub(r"((?<=[0-9 ]){2,})\-((?=[ 0-9]){2,})", ' से  ', temp)
						temp = temp.replace(':', '.').replace(';', ',').replace('\\', ' ').replace('/', ' ').replace(',', ' , ').replace('|', ' | ').replace('-', ' ').replace('\'', '')
#						temp = re.sub(r"((?<=[a-z0-9 ])*)\$((?=[0-9])+)", ' dollar ', temp)
#						temp = re.sub(r"((?<=[a-z0-9 ])*)\$((?![0-9])+)", '', temp)
#						temp = re.sub(r"((?<=[a-z0-9 ])*)\£((?=[0-9])+)", ' euro ', temp)
#						temp = re.sub(r"((?<=[a-z0-9 ])*)\£((?![0-9])+)", '', temp)
#						temp.replace('<', '').replace('>', '').replace('*', '').replace('^', '')..replace('`', '').replace('{', '').replace('}', '').replace('~', '').replace('§' ,'').replace().replace('±', '').replace()
#						temp = re.sub(r"((?<![a-z0-9 ]){2,})\&", '', temp)
#						temp = temp.replace('&', ' and ')
						f_write_ptr.write(temp)
