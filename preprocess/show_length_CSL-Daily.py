from tqdm import tqdm 
with open('./preprocess/CSL-Daily/video_map.txt','r', encoding='utf-8') as f:
    inputs_list = f.readlines()
total_length = 0
for file_idx, file_info in tqdm(enumerate(inputs_list[1:]), total=len(inputs_list)-1):  # Exclude first line
    index, name, length, gloss, char, word, postag = file_info.strip().split("|")
    total_length += int(length)
print(f'average length : {total_length/len(inputs_list)-1:}')