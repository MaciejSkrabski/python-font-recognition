#%%
import csv
import random
from PIL import Image, ImageDraw, ImageFont
#%%
arr = []
with open("words.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        arr.append(row[0])

l=len(arr)
ints=[]
i=0
# while i < 600:
#     r=random.randint(0, l-i)
#    print(i)
#     if (r not in ints):
#         ints.append(r)
#         i+=1

# new_word_list=[]
# for i in ints:
#     new_word_list.append(arr[i])
new_word_list = random.sample(arr, k=600)


# %%
def words_to_imgs(wordlist=['raz', 'dwa', 'trzy'], font='', output_directory=''):

    for word in wordlist:
        img = Image.new('RGB', (300, 300), color = (255, 255, 255))
            
        fnt = ImageFont.truetype(font, random.randint(20,70))
        d = ImageDraw.Draw(img)
        d.text((random.randint(5, 200), random.randint(5,200)), word, font=fnt, fill=(0, 0, 0))

        img = img.rotate(random.randint(-20,20), resample = Image.BICUBIC, fillcolor="white", expand=1).resize([300, 300])
        img.save(output_directory+'/'+word+".jpg")



# %%
words_to_imgs(new_word_list[:200], 'fonty/Lato-Regular.ttf', 'Lato')
words_to_imgs(new_word_list[200:400], 'fonty/LiberationSans-Regular.ttf', 'Sans')
words_to_imgs(new_word_list[400:600], 'fonty/LiberationSerif-Regular.ttf', 'Serif')
# %%
