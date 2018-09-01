import os

origin_dir = "./UTKFace"
des_dir = "./data"
imgFiles = [file for file in os.listdir(origin_dir)]

def encodeAge(n):
    if n<=5:
        return 0
    elif n<=10:
        return 1
    elif n<=15:
        return 2
    elif n<=20:
        return 3
    elif n<=30:
        return 4
    elif n<=40:
        return 5
    elif n<=50:
        return 6
    elif n<=60:
        return 7
    elif n<=70:
        return 8
    else:
        return 9


def makeDir():
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)

    for i in range(20):
        new_folder = os.path.join(des_dir,format(i,"<02"))
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

def moveFiles():
    for file in imgFiles:
        lst = file.split("_")
        # age is first given  and gender is given second  
        age = int(lst[0])
        gender = int(lst[1])

        # converts to seperate folder and then it is 00,01,02 format 
        folder = format(encodeAge(age)*2 + gender,"<02")
        origin_file = os.path.join(origin_dir,file)
        des_file = os.path.join(des_dir,folder,file)
        os.rename(origin_file,des_file)
