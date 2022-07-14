# -*- coding: utf-8 -*-
import os
 
path_dir = './'
#file_list = os.listdir(path_dir)
#file_list.sort()
numd = 1550
#print(numd, file_list[0])
#f = open("list_airfoils.dat", "w")
for i in range(138,numd+1):
    #os.system("mv Airfoil_"+str(i)+" Airfoil_"+str(i).rjust(4,'0'))
    os.system("mv Airfoil_"+str(i).rjust(4,'0') +" Airfoil_"+str(i-1).rjust(4,'0'))
    #os.makedirs("./make_p2d/"+filename) #, exist_ok=True)
#f.close()
