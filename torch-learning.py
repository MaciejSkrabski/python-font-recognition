#%%

#%%
if __name__ == "__main__":
    from dataset_creation import subset_of_csv, words_to_imgs
    sub = subset_of_csv("words.csv", 600)
    words_to_imgs(sub[:200], "fonty/Lato-Regular.ttf", "Lato-Regular", "testowe.csv")
    words_to_imgs(sub[200:400], "fonty/LiberationSans-Regular.ttf", "LiberationSans-Regular", "testowe.csv")
    words_to_imgs(sub[400:], "fonty/LiberationSerif-Regular.ttf", "LiberationSerif-Regular", "testowe.csv")
# %%
