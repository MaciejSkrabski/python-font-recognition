#%%

#%%
if __name__ == "__main__":
    from csv import reader

    l=[]
    with open("dataset/dataset.csv") as csvfile:
        r = reader(csvfile)
        for row in r:
            l.append(row)
    print(l[:5], l[-5:], len(l))

# %%
