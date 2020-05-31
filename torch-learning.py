#%%
from dataset_creation import subset_of_csv, words_to_imgs

if __name__ == "__main__":
    wordlist = subset_of_csv("words.csv", 600)
    print(wordlist)
    words_to_imgs(wordlist[:5], "fonty/Lato-Regular.ttf", "Lato", "testowe.csv")

# %%
