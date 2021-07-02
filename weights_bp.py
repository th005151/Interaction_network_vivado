from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import re
import glob

def replace(file_path, pattern, subst):
    #Create temp file                                                                                                             
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern,subst,line))
    #Copy the file permissions from the old file to the new file                                                                  
    copymode(file_path, abs_path)
    #Remove original file                                                                                                         
    remove(file_path)
    #Move new file                                                                                                                
    move(abs_path, file_path)
bit = 16
for f in ['parameters.h']+glob.glob('weights/*.h'):
    replace(f,r'ap_fixed<[0-9]{1,2},6>','ap_fixed<%i,6>'%bit)
