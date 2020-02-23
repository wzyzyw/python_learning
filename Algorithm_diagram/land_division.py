"""
算法图解第四章分而治之，划分土地python实现
"""
import math
from PIL import Image
from PIL import ImageDraw
square_width=[]


def division(length,width):
    remainder=length%width
    if remainder==0:
        multiple = int(math.floor(length / width))
        for i in range(multiple):
            square_width.append(width)
    else:
        multiple=int(math.floor(length/width))
        for i in range(multiple):
            square_width.append(width)
        remain=length-multiple*width
        length=width
        width=remain
        division(length,width)


if __name__=='__main__':
    length=168
    width=64
    division(length,width)
    print(u'划分的正方形边长：',square_width)
    im=Image.new('RGB',(length,width),'white')
    draw=ImageDraw.Draw(im)
    draw.line([(0,0),(length-1,0),(0,width-1),(length-1,width-1)],fill='black')

    #绘制划分后的土地
    x1=0
    x2=square_width[0]
    y1=0
    y2=square_width[0]
    draw.rectangle((x1,y1,x2,y2),fill='blue',outline='black')
    x1=square_width[0]
    x2=square_width[0]*2
    draw.rectangle((x1, y1, x2,y2), fill='blue',outline='black')
    x1=x2
    x2=square_width[2]+2*square_width[1]
    y2=square_width[2]
    draw.rectangle((x1, y1, x2, y2), fill='orange',outline='black')
    y1=y2
    y2=y1+square_width[3]
    x2=x1+square_width[3]
    draw.rectangle((x1, y1, x2, y2), fill='green',outline='black')
    x1=x2
    x2=x1+square_width[4]
    y2=y1+square_width[4]
    draw.rectangle((x1, y1, x2, y2), fill='red',outline='black')
    y1+=square_width[4]
    x2=x1+square_width[5]
    y2=y1+square_width[5]
    draw.rectangle((x1, y1, x2, y2), fill='purple',outline='black')
    x1=x2
    x2=x1+square_width[6]
    draw.rectangle((x1, y1, x2, y2), fill='purple',outline='black')
    im.save('../Algorithm_diagram/land_division.png')
    im.show()


