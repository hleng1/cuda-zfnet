from PIL import Image

im = Image.open('fish.jpg','r') #input image
pix = im.load()
pix_val = list(im.getdata())

f=open('input.txt',"x")#output file name
for i in pix_val:
    for j in i:
        f=open('input.txt',"a")#output file name
        f.write(str(float(j)))
        f.write('\n')
        f.close()
