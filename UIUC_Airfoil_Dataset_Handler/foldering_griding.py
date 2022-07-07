# -*- coding: utf-8 -*-
import os
 
path_dir = './UIUC_DB_KISTI/'
file_list = os.listdir(path_dir)
file_list.sort()

n_remove = 2
numd = len(file_list)
#print(numd, file_list[0])
for i in range(169,209):
    filename = os.path.splitext(file_list[i])[0]
    os.system('./kgrid -inp inp2.inp -grd ./UIUC_DB_KISTI/'+file_list[i])
    os.system('ln -s ../preflow ./result/preflow')
    os.system('cp ' + path_dir + file_list[i] + ' preflow.rc ./result')
    os.chdir('result')
    os.system('./preflow ../grid_cpu1.in')
    os.chdir('../')
    os.rename('./result','Airfoil_'+str(i+1).rjust(4,'0'))
    #os.makedirs("./make_p2d/"+filename) #, exist_ok=True)
