import sqlite3

class text_sys:
    def __init__(self):   
        conn = sqlite3.connect('student_text.db') 
        sql = """
        create table students(
        id int primary key,
        name varchar,
        age int,
        sex varchar);
        """
        yb = conn.cursor()
        yb.execute(sql)
        conn.commit()
        yb.close()
        conn.close()
    
    def add_student():
        conn = sqlite3.connect('student_text.db') 
        stu = input("id,姓名，年龄，性别")
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
        
        
if __name__ =="__main__":
    y=1
    ts = text_sys()
    while(y):
        print("选择模式")
        x = input("1.添加学生信息\n2.清除学生信息\n3.清除所有信息\n4.修改表格\n5.导出表格")
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
        z = input("是否继续?y/n")
        if(z =='y' or z == 'Y'):
            y=1
        else:
            y=0