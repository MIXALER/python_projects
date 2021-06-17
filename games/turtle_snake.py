import turtle
import time
import random
import math
import sys


# 动态创建类实例
def create_instance(module_name, class_name, *args, **kwargs):
    module_meta = __import__(module_name, globals(), locals(), [class_name])
    class_meta = getattr(module_meta, class_name)
    obj = class_meta(*args, **kwargs)
    return obj


def start_game():
    global game_state
    game_state = 'start'


# 设置贪吃蛇的运动方向
def go_up():
    if head.direction != 'down':
        head.direction = 'up'


def go_down():
    if head.direction != 'up':
        head.direction = 'down'


def go_right():
    if head.direction != 'left':
        head.direction = 'right'


def go_left():
    if head.direction != 'right':
        head.direction = 'left'


# 移动贪吃蛇，一次移动20个像素
def move():
    if head.direction == 'up':
        head.sety(head.ycor() + 20)
    if head.direction == 'down':
        head.sety(head.ycor() - 20)
    if head.direction == 'right':
        head.setx(head.xcor() + 20)
    if head.direction == 'left':
        head.setx(head.xcor() - 20)


def monster_move():
    if head.direction != 'Paused':
        diff_x = head.xcor() - monster.xcor()
        diff_y = head.ycor() - monster.ycor()
        len = math.sqrt(diff_x ** 2 + diff_y ** 2)
        proportion = random.randint(-5, 5) / 100
        monster.setx(monster.xcor() + (15 + proportion) * (diff_x / len))
        monster.sety(monster.ycor() + (15 + proportion) * (diff_y / len))


def game_over():
    global delay
    head.direction = 'Paused'
    time.sleep(1)
    bodies.clear()
    delay = 0.2
    game_over_slogan = turtle.Turtle()
    game_over_slogan.speed(0)
    game_over_slogan.hideturtle()
    game_over_slogan.penup()
    game_over_slogan.shape('square')
    game_over_slogan.color('red')
    game_over_slogan.goto(head.xcor() - 35, head.ycor())
    game_over_slogan.write("Game Over!!",
                           align='left', font=('Arial', 12, 'normal'))
    time.sleep(10)
    sys.exit('游戏结束')


def game_win():
    global delay
    head.direction = 'Paused'
    time.sleep(1)
    bodies.clear()
    delay = 0.2
    winner_slogan = turtle.Turtle()
    winner_slogan.speed(0)
    winner_slogan.hideturtle()
    winner_slogan.penup()
    winner_slogan.shape('square')
    winner_slogan.color('red')
    winner_slogan.goto(head.xcor() - 30, head.ycor())
    winner_slogan.write("Winner !!",
                        align='left', font=('Arial', 12, 'normal'))

    time.sleep(10)
    sys.exit('游戏结束')



# 记录头和身体交叉的次数
def contact_inc():
    global contact
    contact += 1


def status_write():
    status.clear()
    global time_spent
    motion = head.direction
    info = "Contact:{} Time:{}   Motion:{}".format(contact, time_spent, motion)
    status.write(info, align='center', font=('Arial', 20, 'normal'))


# 计时器刷新率设置为0.2
delay = 0.2
contact = 0
base_time = 0
time_spent = 0
motion = 'Paused'
food_num = 10

# 设定食物初始位置随机值
random_x = random.sample(range(-230, 230), 10)
random_y = random.sample(range(-270, 190), 10)

random_location = []
for xy in zip(random_x, random_y):
    random_location.append(xy)

# 设定游戏窗口
window = turtle.Screen()
window.title("Snake by Kinkey Lam")
window.setup(width=660, height=740)  # 窗口大小 宽： 500 + 80*2 高： 500 + 80*2 + 80 状态栏高80
window.tracer(0)

# 边框线绘制
board = turtle.Turtle()
board.color("black")
board.penup()
board.goto(250, -290)
board.pendown()
board.sety(board.ycor() + 580)
board.setx(board.xcor() - 500)
board.sety(board.ycor() - 580)
board.setx(board.xcor() + 500)
board.sety(board.ycor() + 500)
board.setx(board.xcor() - 500)
board.sety(board.ycor() + 1)
board.setx(board.xcor() + 500)
board.hideturtle()

# 贪吃蛇的头
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("red")
head.penup()
head.goto(0, 0)
head.direction = 'Paused'


# 怪物，追赶贪吃蛇的头
monster = turtle.Turtle()
monster.speed(0)
monster.shape("square")
monster.color("purple")
monster.penup()
monster.goto(random.randint(-230, 230), random.randint(-290, 80))
monster.showturtle()

foods = dict()

for i in range(food_num):
    obj = create_instance("turtle", "Turtle")
    foods[i + 1] = obj

for food_i, xy in zip(foods.values(), random_location):
    food_i.penup()  # 隐藏移动轨迹
    food_i.color('black')
    food_i.goto(xy[0], xy[1])
    food_i.hideturtle()

# 状态栏
status = turtle.Turtle()
status.hideturtle()
status.penup()
status.color('black')
status.goto(0, 230)
info = "Contact:{}  Time:{}  Motion:{}".format(contact, time_spent, motion)
status.write(info, align='center', font=('Arial', 24, 'normal'))

# 游戏介绍第一句
intro1 = turtle.Turtle()
intro1.hideturtle()
intro1.penup()
intro1.color('black')
intro1.goto(-200, 190)
intro1.write("Welcome to Kinley's version of snake ....", align='left', font=('Arial', 12, 'normal'))

# 游戏介绍第二句
intro2 = turtle.Turtle()
intro2.hideturtle()
intro2.penup()
intro2.color('black')
intro2.goto(-200, 160)
intro2.write("You are going to use the 4 arrow keys to move the snake ",
             align='left', font=('Arial', 12, 'normal'))
# 游戏介绍第3句
intro3 = turtle.Turtle()
intro3.hideturtle()
intro3.penup()
intro3.color('black')
intro3.goto(-200, 145)
intro3.write("around the screen, trying to consume all the food items ",
             align='left', font=('Arial', 12, 'normal'))
# 游戏介绍第4句
intro4 = turtle.Turtle()
intro4.speed(0)
intro4.hideturtle()
intro4.penup()
intro4.shape('square')
intro4.color('black')
intro4.goto(-200, 130)
intro4.write("before the monster catches you ....",
             align='left', font=('Arial', 12, 'normal'))
# 游戏介绍第5句
intro5 = turtle.Turtle()
intro5.hideturtle()
intro5.penup()
intro5.color('black')
intro5.goto(-200, 100)
intro5.write("Click anywhere on the screen to start the game, have fun!!",
             align='left', font=('Arial', 12, 'normal'))

# 贪吃蛇的身体列表
bodies = []

# 游戏状态 开始界面 游戏开始
game_state = 'introduction'  # 开始界面, 介绍游戏

window.listen()

# todo 不知道在哪里查按键和字符的映射关系。。。。。
window.onkeypress(start_game, 'p')
window.onkeypress(go_up, 'w')
window.onkeypress(go_down, 's')
window.onkeypress(go_left, 'a')
window.onkeypress(go_right, 'd')

time_flag = 1

while True:

    window.update()  # 刷新窗口

    if game_state == 'introduction':
        pass
    elif game_state == 'start':
        intro1.clear()
        intro2.clear()
        intro3.clear()
        intro4.clear()
        intro5.clear()
        if time_flag == 1:
            base_time = time.time()
            time_flag = 0
        time_spent = int((time.time() - base_time) // 1)
        status_write()
        for i, food_i in foods.items():
            food_i.write(str(i), font=('Arial', 12, 'normal'))


    if len(foods) == 0:
        game_win()

    # 判断是否和怪物相撞
    if head.distance(monster) < 20:
        game_over()

    # 检查头部和身体是否有交叉
    for body in bodies:
        if head.distance(body) < 20:
            contact_inc()
            break
    # 用一个字典保存所有食物的坐标，然后判定是否有坐标重复
    for food_item in list(foods.items()):
        if head.distance(food_item[1]) < 20:
            food_item[1].clear()
            del foods[food_item[0]]

            for i in range(food_item[0]):
                new_body = turtle.Turtle()
                new_body.shape('square')
                new_body.color('black')
                new_body.penup()
                if len(bodies) >0:
                    new_body.goto(bodies[len(bodies)-1].xcor(), bodies[len(bodies)-1].ycor())
                if len(bodies) == 0:
                    new_body.goto(head.xcor(), head.ycor())
                new_body.hideturtle()
                bodies.append(new_body)

            delay -= 0.001  # 降低速度，当蛇吃下食物后

    # 第一个身体
    if (len(bodies) > 0) and head.direction != 'Paused':
        bodies[0].goto(head.xcor(), head.ycor())
        bodies[0].showturtle()

    # 让身体跟着头部
    for i in range(len(bodies) - 1, 0, -1):
        if head.direction != 'Paused':
            bodies[i].goto(bodies[i - 1].xcor(), bodies[i - 1].ycor())
            bodies[i].showturtle()



    if game_state != 'introduction':

        move()  # 头部移动
        monster_move()  # 怪物跟踪

    if head.xcor() < -220:
        head.direction = 'Paused'
    if head.xcor() > 220:
        head.direction = 'Paused'
    if head.ycor() < -260:
        head.direction = 'Paused'
    if head.ycor() > 190:
        head.direction = 'Paused'

    time.sleep(delay)

window.mainloop()  # 保持游戏窗口一直打开