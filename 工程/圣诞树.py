import turtle
import random

def setup_turtle():
    turtle.speed("fastest")
    turtle.title("圣诞树")
    turtle.tracer(0)
    turtle.screensize(bg='pink')
    turtle.setup(900, 800)
    turtle.pensize(5)

def draw_tree():
    turtle.left(90)
    turtle.forward(3 * n)
    turtle.color("orange", "yellow")
    turtle.begin_fill()
    turtle.left(126)
    for _ in range(5):
        turtle.forward(n / 5)
        turtle.right(144)
        turtle.forward(n / 5)
        turtle.left(72)
    turtle.end_fill()
    turtle.right(126)

def draw_lights():
    if random.randint(0, 30) == 0:
        turtle.color('tomato')
        turtle.circle(6)
    elif random.randint(0, 30) == 1:
        turtle.color('orange')
        turtle.circle(3)
    else:
        turtle.color('dark green')

def tree(d, s):
    if d <= 0:
        return
    turtle.forward(s)
    tree(d - 1, s * .8)
    turtle.right(120)
    tree(d - 3, s * .5)
    draw_lights()
    turtle.right(120)
    tree(d - 3, s * .5)
    turtle.right(120)
    turtle.backward(s)

def draw_stars():
    turtle.color("dark green")
    turtle.backward(n * 4.8)
    tree(15, n)

def draw_presents():
    for _ in range(200):
        a = 200 - 400 * random.random()
        b = 10 - 20 * random.random()
        turtle.up()
        turtle.forward(b)
        turtle.left(90)
        turtle.forward(a)
        turtle.down()
        color = 'tomato' if random.randint(0, 1) == 0 else 'wheat'
        turtle.color(color)
        turtle.circle(2)
        turtle.up()
        turtle.backward(a)
        turtle.right(90)
        turtle.backward(b)

def draw_message():
    turtle.color("dark red", "red")
    turtle.write("Merry Christmas", align="center", font=("Comic Sans MS", 40, "bold"))
    turtle.goto(-10, -280)
    turtle.write("㊗️杨欣雨圣诞快乐，2024年12月25日", align="center", font=("Comic Sans MS", 20, "bold"))
    turtle.goto(-10, -310)
    turtle.write("🧊🍦❄❄️❄️🎄🎄🎄🎅🎅🎅️", align="center", font=("Comic Sans MS", 20, "bold"))

def draw_snow():
    turtle.hideturtle()
    turtle.pensize(2)
    for _ in range(200):
        turtle.pencolor("white")
        turtle.penup()
        turtle.setx(random.randint(-350, 350))
        turtle.sety(random.randint(-100, 350))
        turtle.pendown()
        dens = 6
        snowsize = random.randint(1, 10)
        for _ in range(dens):
            turtle.forward(int(snowsize))
            turtle.backward(int(snowsize))
            turtle.right(int(360 / dens))

n = 100.0
setup_turtle()
draw_tree()
draw_stars()
draw_presents()
draw_message()
draw_snow()
turtle.done()