import os

def delete_dir(path):
    
    while True:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            break
        try:
            sub_path = path
            k = 0
            while True:
                listt = os.listdir(sub_path)
                lenght = len(listt)
                if lenght == 0:
                    os.rmdir(listt)
                    break
                i = 0
                while i < lenght: 
                    if i >= len(listt):
                        break
                    ppath = os.path.join(sub_path, listt[i])
                    if os.path.isfile(ppath):
                        os.remove(ppath)   
                        listt.remove(listt[i])
                    else:
                        if len(os.listdir(ppath)) == 0:   
                            os.rmdir(ppath)
                    i += 1
                if len(listt) != 0:
                    sub_path = f'{sub_path}/{listt[k]}'
                else:
                    sub_path = os.path.split(sub_path)[0]
        except:
            continue
