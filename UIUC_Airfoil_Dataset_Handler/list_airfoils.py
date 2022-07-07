# -*- coding: utf-8 -*-
import os
 
path_dir = './UIUC_DB_KISTI/'
file_list = os.listdir(path_dir)
file_list.sort()

n_remove = 2
numd = len(file_list)
#print(numd, file_list[0])
f = open("list_airfoils.dat", "w")
for i in range(0,numd):
    filename = os.path.splitext(file_list[i])[0]
    f.write('AirfoilName(' + str(i+1).rjust(4,'0') + ')' + str(filename) + '\n')
    #os.makedirs("./make_p2d/"+filename) #, exist_ok=True)
f.close()
