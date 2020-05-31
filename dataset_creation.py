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
def shuffle_csv(input_file, output_file):
    from csv import writer, reader
    from random import shuffle

    results = []
    
    # read csv to array
    with open(input_file) as csvfile:
        reader = reader(csvfile)
        for row in reader: # each row is a list
            results.append(row)

    print("length on read: ", len(results))

    # shuffle
    to_shuffle = results #skip row info
    shuffle(to_shuffle)

    results = to_shuffle
    print("length after shuffle: ", len(results))

    #assign ids
    for idx, row in enumerate(results):
        row.insert(0, idx)
    
    # add "id" column
    print(results[0])
    results.insert(0, ["id", "word", "label"]) 
    print(results[1])
    print("length with id column: ", len(results))

    #write 
    with open('dataset/'+output_file, 'w', newline='') as csvfile:
        cwriter = writer(csvfile)
        #for w in results:
         #   cwriter.writerow(w)
        cwriter.writerows(results)
# %%
if __name__ == "__main__":
    sub = subset_of_csv("words.csv", 600)
    lato=sub[:200]
    sans=sub[200:400]
    serif=sub[400:]
    print(lato[-5:], sans[:5], sans[-5:], serif[:5], serif[-5:], sub[-5:])
    print(len(lato)==len(serif)==len(sans)==200)

    words_to_imgs(lato, "fonty/Lato-Regular.ttf", "Lato-Regular", "testowe.csv")
    words_to_imgs(sans, "fonty/LiberationSans-Regular.ttf", "LiberationSans-Regular", "testowe.csv")
    words_to_imgs(serif, "fonty/LiberationSerif-Regular.ttf", "LiberationSerif-Regular", "testowe.csv")
    
    shuffle_csv('testowe.csv', 'dataset.csv')


# %%
