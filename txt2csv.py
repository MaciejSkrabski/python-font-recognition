#%% 
def txt2csv(argv):
    import csv
    try:
        with open(argv, 'r') as f:
            read_txt = f.read()
            read_txt = read_txt.split()

    except IOError as e:
        print('reader, ', e)
        
    except:
        print("reader, other error")

    else:
        try:
            with open('words.csv', 'w') as output:
                writer = csv.writer(output)
                for row in read_txt:
                    writer.writerow([row])
        except IOError as f:
            print('writer, ', f)
        except:
            print('writer other error')
        else:
            print("write successful")
        

# %%
if __name__ == '__main__':
    txt2csv("haba")
