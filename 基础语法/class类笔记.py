import pyautogui
class people:
    def __init__(self,name,age):
        self.name = name 
        self.age= age
    def sit(self):
        print(f"{self.name} is sit down")
    def jump(self):
        print(f"{self.name} is jumping")
    def sleeping(self):
        print(f"{self.name} is sleeping")
a=pyautogui.prompt(text="请输入名字",title="名字",default="按Cancel取消")
b=pyautogui.prompt(text="请输入年龄",title="年龄",default="按Cancel取消")
c=people(a,b)
if a == None:
    print("未检测到名字")
elif b== None:
    print("未检测到年龄")
else:
    x=input("请输入名字")
    y=input("请输入年龄")
one_people = people(x,y)
print(f"{one_people.name} is a {one_people.age} student.")
print(f"{c.name} is a {c.age} people")
# class Car:
#     def __init__(self,make,model,year):
#         self.make=make
#         self.model=model
#         self.year=year
#         self.odometer_reading=30000
#     def update_odometer(self,odometer):
#         if odometer <=self.odometer_reading:
#             if odometer ==0:
#                 print("your car is new car.")
#             else:    
#                 self.odometer_reading = odometer
#         else:
#             print("you do not drive this car")
#     def increment_odometer(self,miles):
#         self.odometer_reading+=miles        
#     def read_odometer(self):
#         print(f"this car is {self.odometer_reading} miles in it.")
#     def get_descripive_name(self):
#         long_name=f"{self.year} {self.make} {self.model}"
#         return long_name.title()
#     def odometer(self):
#         print(f"my new car has {self.odometer_reading} miles on it")
# # my_new_car=Car('audi','a4',2019)
# # x=input("请输入里程数")
# # x=int(x)
# # my_new_car.update_odometer(x)
# # print(my_new_car.get_descripive_name())
# class other_car:
#     def __init__(self,make,model,year):
#         super().__init__(make,model,year)