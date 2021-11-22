import random

import smtplib

import numpy


class Flower:
    def __init__(self, Name: str, numberOfPetals: int, price: float):
        self.Name = Name
        self.numberOfPetals = numberOfPetals
        self.price = price

    def print_information(self):
        if type(self.Name) == str:
            if type(self.numberOfPetals) == int:
                if type(self.price) == float:
                    print(self.Name, self.numberOfPetals, self.price)
        else:
            print("invalid input")


flower = Flower("juice", 10, float(100))
flower.print_information()
# def eight_queens():
#     def helper(queens, dif, sum):
#         p = len(queens)
#         if p == 8:
#             result.append(queens)
#             return
#         for q in range(8):
#             if q not in queens and p - q not in dif and p + q not in sum:
#                 helper(queens + [q], dif + [p - q], sum + [p + q])
#
#     result = []
#     helper([], [], [])
#
#     # dfs depth first search
#     ret_img = []
#     for i in result[random.randint(0, 91)]:
#         ret_img.append("| " * i + "|" + "Q" + "| " * (8 - i))
#     # ret_img = ["| " * i + "|" + "Q" + "| " * (8 - i) for i in result[random.randint(0, 91)]]
#     # 列表推导式
#
#     for i in ret_img:
#         print(i)


# eight_queens()
