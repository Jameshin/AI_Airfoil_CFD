 
path_dir = './'
file_list = os.listdir(path_dir)
n_remove = 2
numd = len(file_list)
#print(numd, file_list[0])
for i in range(numd):
    with open(file_list[i],'r', encoding='cp949') as f:
        #filename = os.path.splitext(file_list[i])[0]
        with open("../UIUC_DB_KISTI/"+file_list[i],'w') as f1:
            next(f)
            next(f)
            for line in f:
                f1.write(line)
