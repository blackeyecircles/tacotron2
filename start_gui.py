#!/usr/bin/python
# -*- coding: UTF-8 -*-
from threading import Thread

import tkinter as tk
import tkinter.font as tkFont
from tts.client_test import run
# 指定字体名称、大小、样式

window = tk.Tk()
window.title('语音合成')

window.geometry('700x180+600+350')

ft = tkFont.Font(family='宋体', size=10, weight=tkFont.NORMAL)
ft = tkFont.Font(family='华文隶书', size=10, weight=tkFont.NORMAL)  #设置字体
# Lab = tk.Label(window, text='请输入文本', height=2, fg="white", bg='#0000cd', \
#                font=ft, compound='left')
# Lab.pack()  #设置label显示

label = tk.Label(window, text="请输入文本", height=3, font=ft, fg='black')
label.pack()

name_input = tk.Text(window, width=90, height=5, font=ft)		# width宽 height高
name_input.pack()
name_input.insert(tk.INSERT, "中瑞福宁的人工智能系统。")


def synthetic():
    # 可以用get()方法获取Text的文本内容, 其中第一个参数是起始位置，'1.1'就是从第一行第一列后，到第一行第五列后
    # print(name_input.get('0.0', tk.END))
    # run(0, name_input.get('0.0', tk.END), set_flag)
    th = Thread(target=run, args=(0, name_input.get('0.0', tk.END), set_flag))
    th.start()


global_lable = None
def set_flag(text):
    global global_lable
    if not global_lable:
        global_lable = tk.Label(window, text=text, height=3, font=ft, fg='black')
        global_lable.pack()
    else:
        global_lable['text'] = text


tk.Button(window, text='合成语音', command=synthetic, font=ft, width=8, height=2).pack()


window.mainloop()
