import tkinter
import tkinter.messagebox
window = tkinter.Tk()
window.title('my window')
window.geometry('200x200')

var = tkinter.StringVar()
# 为变量设置值
var.set((2020, 2021, 2022))
# 创建Listbox,将var的值赋给Listbox,可以显示8行记录
lb = tkinter.Listbox(window, listvariable=var, height=8)
# 创建一个list并将值循环添加到Listbox控件中
list_items = ['1-watermelon', '2-strawberry', '3-grape']
for item in list_items:
    # 从最后一个位置开始加入值
    lb.insert('end', item)
# 在第一个位置加入'first'字符
lb.insert(1, 'first')
# 在第二个位置加入'second'字符
lb.insert(2, 'second')
# 删除第二个位置的字符
lb.delete(2)
lb.pack()
print(lb)

# 显示已经被选择的list记录
def show_selection():
    value = lb.get(lb.curselection())   # 获取当前选中的文本
    tkinter.messagebox.showinfo(title='my message', message=value)


b1 = tkinter.Button(window, text='show selection', width=15, height=2, command=show_selection)
b1.pack()

window.mainloop()


