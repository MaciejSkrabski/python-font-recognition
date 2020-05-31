#%%
def subset_of_csv(path, output_size):
    from csv import reader
    from random import sample
    arr = []
    with open(path) as csvfile:
        reader = reader(csvfile, delimiter=' ')
        for row in reader:
            arr.append(row[0])
    new_word_list = sample(arr, k=output_size)
    return new_word_list
subset = subset_of_csv("words.csv", 600)
#%%
def append_dict_as_row(file_name, dict_of_elem, field_names):
    from csv import DictWriter
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
# %%
def words_to_imgs(wordlist=['raz', 'dwa', 'trzy'], font='', label='', csv_name=''):
    from random import randint
    from PIL import Image, ImageDraw, ImageFont

	
    field_names = ['Word','Label']
    

    for word in wordlist:
        row_dict = {'Word': word, 'Label': label}
        append_dict_as_row(csv_name, row_dict, field_names)
        img = Image.new('RGB', (300, 300), color = (255, 255, 255))
            
        fnt = ImageFont.truetype(font, randint(20,70))
        d = ImageDraw.Draw(img)
        d.text((randint(5, 200), randint(5,200)), word, font=fnt, fill=(0, 0, 0))

        img = img.rotate(randint(-20,20), resample = Image.BICUBIC, fillcolor="white", expand=1).resize([300, 300])
        img.save('dataset/'+word+'.jpg')
# %%
if __name__ == "__main__":
    wordlist = subset_of_csv("words.csv", 600)
    words_to_imgs(wordlist[:5], "fonty/Lato-Regular.ttf", "Lato", "testowe.csv")
    