import sys
import numpy as np 

def numbersites(inpfile):
    count = 0
    with open(inpfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            if len(line_split) >= 5:
                _type = line_split[0]
                x, y, z = map(float, line_split[1:4])
                maxl = int(line_split[4])
                print(_type, x, y, z, maxl)
                for i in range(maxl + 1):
                    skip_lines = f.readline()
                count += 1
    return count

# integration bounds 
r1 = 6.78 # inner radius 
r2 = 12.45 # outer radius
small = 1.0e-4 # SVD threshold 
maxl = 4 # maximum multipole order 

inpfile = sys.argv[1] if len(sys.argv) > 1 else "temp_format.dma"
multsites = numbersites(inpfile)
print(f"multsites={multsites}")

multipoles = np.zeros((multsites, maxl+1, maxl+1, 2))
xyzmult    = np.zeros((multsites, 3))

midbond = np.zeros((multsites, multsites), dtype=int)
lmax = np.zeros(multsites, dtype=int)
rvdw = np.zeros(multsites, dtype=np.float64)
atomtype = np.full(multsites, '', dtype='<U2')

