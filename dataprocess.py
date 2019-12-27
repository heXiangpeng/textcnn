# encoding=utf-8
import csv
import random


with open("data.csv","r",encoding="utf-8") as csvfile:
    read = csv.reader(csvfile)
#     num = 0
    row = []
#     head = ['lable','text']

    for i in read:
        row.append(i)

    random.shuffle(row)
    length = int(len(row)*0.2)

    test = row[:length]
    train = row[length+1:len(row)]
    print(len(train))

    with open('test.csv', 'w')as f:
        f_csv = csv.writer(f)
        # f_csv.writerow(head)
        f_csv.writerows(test)

    with open('train.csv', 'w')as f:
        f_csv = csv.writer(f)
        # f_csv.writerow(head)
        f_csv.writerows(train)

    # for i in read:
    #     print(i)
    #
    #     if i[0] == '1':
    #         item = [1,i[1]]
    #         row.append(item)
    #     else:
    #         item = [0,i[1]]
    #         row.append(item)
    #
    # with open('data.csv', 'w')as f:
    #     f_csv = csv.writer(f)
    #     # f_csv.writerow(head)
    #     f_csv.writerows(row)