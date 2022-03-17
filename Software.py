import tkinter
import tkinter.filedialog
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms as transforms
import os
import tkinter.font as tkFont
from tkinter import *
from PIL import Image,ImageTk
import numpy as np

def get_img(filename, width, height):
    im = Image.open(filename).resize((width, height))
    im = ImageTk.PhotoImage(im)
    return im

#设置图片保存路径
outfile = './'

#创建一个界面窗口
win = tkinter.Tk()
#tT=tkinter.Text()
#ft2 = tkFont.Font(size=30, weight=tkFont.BOLD)

win.title("This is an image-to-image translation process")#, font=ft2)
#tT.insert(END,'This is an image-to-image translation process')


#win.title(string='')
win.geometry("1280x1080")
#win.geometry('400x300')
"""
frame = LabelFrame(
    win,
    text='This is an image-to-image translation process',
    bg='#f0f0f0',
    font=(30)
)
frame.pack(expand=True, fill=BOTH)
"""
#背景
canvas = tkinter.Canvas(win, width=1280, height=1080)
#imgpath = 'AI.jpg'
#img = Image.open(imgpath)
photo = get_img('AI.jpg', 1580, 1380)#ImageTk.PhotoImage(img)
 
canvas.create_image(639, 500, image=photo)
canvas.pack()
entry=tkinter.Entry(win,insertbackground='blue', highlightthickness =2)
entry.pack()
 


"""
canvas_root = tkinter.Canvas(win, width=1280, height=1080)
im_root = get_img('AI.jpg', 1580, 1380)
canvas_root.create_image(639, 500, image=im_root)
canvas_root.pack()
"""
# canvas.create_window(100, 50, width=100, height=20,
#                                        window=entry)
#     # label 中设置图片
# im_root1 = get_img('./background/one.jpg', 100, 40)
# img_label = Label(win, text='欢迎使用J波检测', image=im_root1)
# img_label.place(x=3, y=3, width=100, height=40)
ft = tkFont.Font(family='Fixdsys', size=20, weight=tkFont.BOLD)
ft1 = tkFont.Font(size=25, slant=tkFont.ITALIC,weight=tkFont.BOLD)
ft2 = tkFont.Font(size=30, weight=tkFont.BOLD)
#设置全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count =0
img2 = tkinter.Label(win)
pic_name  = ''
MODEL =''
"""
#设置选项
var = tkinter.StringVar()
# 为变量设置值
var.set(('Unet'))
# 创建Listbox,将var的值赋给Listbox,可以显示8行记录
lb = tkinter.Listbox(win, listvariable=var, height=2,font=ft1)
# 创建一个list并将值循环添加到Listbox控件中
lb.insert('end', 'Segnet')
lb.pack()
# 显示已经被选择的list记录
def choose_model():
	global MODEL  
	print(lb) 
	value = lb.get(lb.curselection())   # 获取当前选中的文本
	#tkinter.messagebox.showinfo(title='my message', message=value)
	print(value) 
	MODEL =value 
"""
var = tkinter.StringVar()
var.set('A')
l = tkinter.Label(win, text='Choose one model to be tested:',font=ft1)
#l.pack()

# def show_selected():
#     value = var.get()  # 获取当前选中项
    #tkinter.messagebox.showinfo(title='my message', message=value)
def choose_model():
	global MODEL  
	#print(lb) 
	value = var.get()   # 获取当前选中的文本
	#tkinter.messagebox.showinfo(title='my message', message=value)
	print(value) 
	MODEL =value 

r1 = tkinter.Radiobutton(win,text='A.Unet  ',
                         variable=var, value='Unet',font=ft1,
                         command=choose_model, compound=tkinter.CENTER)
r1.pack()
r2 = tkinter.Radiobutton(win, text='B.Segnet',
                         variable=var, value='Segnet ',font=ft1,
                         command=choose_model, compound=tkinter.CENTER)
r2.pack()
canvas.create_window(550, 100, #width=10, height=1,
                                      window=l) 
button1_window = canvas.create_window(550, 150, window=r1)
button1_window = canvas.create_window(550, 200, window=r2)
#canvas.create_window(1200, 1050, width=10, height=1,
                                      # window=r1)
						
"""
canvas_root = tkinter.Canvas(win, width=1280, height=1080)
im_root = get_img('AI.jpg', 1580, 1380)
canvas_root.create_image(639, 500, image=im_root)
canvas_root.pack()
"""
#实现在本地电脑选择图片
def choose_file():
	global pic_name
	select_file = tkinter.filedialog.askopenfilename(title='选择图片')
	pic_name = select_file
	var.set(select_file)
	load = Image.open(select_file)
	load = transforms.Resize((400,500))(load)
	#声明全局变量
	global original
	original = load
	render = ImageTk.PhotoImage(load)
	
	img  = tkinter.Label(win,image=render)
	img.image = render
	img.place(x=100,y=350)
	"""
	if 'test_label' in os.listdir('datasets/cityscapes/'):
		os.system('rm datasets/cityscapes/test_label/*')
		os.system('cp {} datasets/cityscapes/test_label'.format(select_file))
	else:
		os.mkdir('datasets/cityscapes/test_label')
		os.system('cp {} datasets/cityscapes/test_label'.format(select_file))
	"""

def carryout_shell():
	global  pic_name
	os.system('python test.py --test_image_path {} --Model {}'.format(pic_name, MODEL))
	select_file = './Image_translated/{}'.format(pic_name.split('/')[-1].split('.')[0]+'.png')
	image = cv2.imread(select_file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #得到灰度图
	ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  #设置阈值，转化成二值图像

	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)  #返回轮廓和属性
# #cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

	#bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
	minbound= [cv2.minAreaRect(cnt) for cnt in contours]
	for box in minbound:
		points=np.int0(cv2.boxPoints(box))
		cv2.drawContours(image,[points],0,(255,255,255),2)

	# for bbox in bounding_boxes:
	# 	[x, y, w, h] = bbox
	# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imwrite(select_file, image)

	var.set(select_file)
	load = Image.open(select_file)
	load = transforms.Resize((400,500))(load)
	#声明全局变量
	global original
	original = load
	render = ImageTk.PhotoImage(load)
	
	img  = tkinter.Label(win,image=render)
	img.image = render
	img.place(x=650,y=350)
    

#保存函数
def save():
	global count
	count += 1
	save_img.save(os.path.join(outfile,'test'+str(count)+'.jpg'))

#显示路径
#e = tkinter.StringVar()
#e_entry = tkinter.Entry(win, width=68, textvariable=var)
#e_entry.pack()

#设置选择图片的按钮

#b1 = tkinter.Button(window, text='show selection', width=15, height=2, command=show_selection)
#b1.pack()
#button0 = tkinter.Button(win, text ="Select_Model", command = choose_model,font=ft1)
#button0.place(x=500,y=100)
#button0.pack()
button1 = tkinter.Button(win, text ="Select_RGB", command = choose_file,font=ft1)
button1.place(x=100,y=300) 
"""
button1 = tkinter.Button(win, text ="Select_label", command = choose_file_label,font=ft1)
button1.place(x=100,y=450)
"""
#设置随机比例缩放的按钮
button2 = tkinter.Button(win,text="Translated_Image",command=carryout_shell,font=ft1)
button2.place(x=650,y=300)
win.mainloop()
