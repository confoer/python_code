import sqlite3
from openpyxl import Workbook

class text_sys:
    def __init__(self):  
        conn = sqlite3.connect('student_text.db')
        yb = conn.cursor() 
        yb.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students';")
        if yb.fetchone() is None:
            sql = """
            create table students(
            id int primary key,
            name varchar,
            age int,
            sex varchar);
            """
            yb.execute(sql)
            conn.commit()
        yb.close()
        conn.close()
    
    def add_student(self):
        conn = sqlite3.connect('student_text.db') 
        stu = input("id,姓名，年龄，性别\n")
        values = stu.split()
        id = values[0]
        name = values[1]
        age = values[2]
        sex = values[3]
        sql = """insert into students(id,name, age, sex) values (?,?, ?, ?);"""
        yb = conn.cursor()
        yb.execute(sql, [id,name, age, sex])
        conn.commit()
        yb.close()
        conn.close()
        
    
    def remove_student(id):
        conn = sqlite3.connect('student_text.db') 
        sql =""" DELETE FROM students WHERE id = ?;"""
        yb = conn.cursor()
        yb.execute(sql,(id,))
        conn.commit()
        yb.close()
        conn.close()
        

    def remove_student_all():
        conn = sqlite3.connect('student_text.db') 
        sql ="""DELETE FROM students;"""
        yb = conn.cursor()
        yb.execute(sql)
        conn.commit()
        yb.close()
        conn.close()
    
    def add_list_column(name,type):
        conn = sqlite3.connect('student_text.db')
        sql = "ALTER TABLE students ADD {} {};".format(name,type)
        yb =conn.cursor()
        yb.execute(sql)
        conn.commit()
        yb.close()
        conn.close()
    
    def delete_list_column(name):
        conn = sqlite3.connect('student_text.db')
        sql = "ALTER TABLE students DROP COLUMN {};".format(name)
        yb = conn.cursor()
        yb.execute(sql)
        conn.commit()
        yb.close()
        conn.close()

    def export_form():
        conn = sqlite3.connect('student_text.db')
        cursor = conn.execute("SELECT * FROM student")
        workbook = Workbook()
        sheet = workbook.active
        for row_index, row_data in enumerate(cursor):
            for column_index, column_data in enumerate(row_data):
                sheet.cell(row=row_index+1, column=column_index+1, value=column_data)
        save_filename = input('请输入保存路径:')
        workbook.save(filename=save_filename+'student.xlsx')
        conn.commit()
        conn.close()

if __name__ =="__main__":
    y=1
    ts = text_sys()
    while(y):
        print("选择模式")
        k = input("1.添加学生信息\n2.清除学生信息\n3.清除所有信息\n4.修改表格\n5.导出表格\n")
        x = int(k)
        if x==1:
            ts.add_student()
        elif x==2:
            No = input("请输入学生编号:")
            ts.remove_student(No)
        elif x==3:
            ts.remove_student_all()
        elif x==4:
            print("1.添加列\n2.删除列\n")
            a = input("请选择:")
            if a == 1:
                b = input("请输入列名:")
                ts.add_list_column(b)
            elif a==2:
                b = input("请输入列名:")
                ts.delete_list_column(b)
            else:
                print("输入错误")
        elif x==5:
            ts.export_form()
        else :
            print("输入错误")
            break
        z = input("是否继续?y/n")
        if(z =='y' or z == 'Y'):
            continue
        else:
            break