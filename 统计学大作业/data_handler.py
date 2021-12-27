import pandas as pd


def describe(a):
    MAX = a.max()
    MIN = a.min()
    MEAN = a.mean()
    STD = a.std()
    return MAX, MIN, MEAN, STD


def handler(a, group_num):
    max_score, min_score, mean_score, std = describe(a)
    stride = (max_score - min_score) / group_num
    array = []
    for i in range(1, group_num + 1):
        array.append(min_score + i * stride)
    return array


def relabel_score(a, n):
    label = []
    arr = handler(a, n)
    for s in a:
        if s >= arr[4]:
            s = 'A'
            label.append(s)
            continue
        elif s >= arr[3]:
            s = 'A'
            label.append(s)
            continue
        elif s >= arr[2]:
            s = 'B'
            label.append(s)
            continue
        elif s >= arr[1]:
            s = 'C'
            label.append(s)
            continue
        elif s >= arr[0]:
            s = 'D'
            label.append(s)
            continue
        else:
            s = 'E'
            label.append(s)
            continue
    return label


def relabel_score_xgb(a, n):
    label = []
    arr = handler(a, n)
    for s in a:
        if s >= arr[4]:
            s = 5
            label.append(s)
            continue
        elif s >= arr[3]:
            s = 4
            label.append(s)
            continue
        elif s >= arr[2]:
            s = 3
            label.append(s)
            continue
        elif s >= arr[1]:
            s = 2
            label.append(s)
            continue
        elif s >= arr[0]:
            s = 1
            label.append(s)
            continue
        else:
            s = 0
            label.append(s)
            continue
    return label


def build_sch_list(num):
    sch_list = pd.read_excel('school_list.xls')['学校名称']
    lis = sch_list[0:num]
    s_list = []
    for s in lis:
        s_list.append(s)
    s_list.append('scut')
    s_list.append('华工')
    s_list.append('华南理工')
    s_list.append('清华')
    s_list.append('pku')
    s_list.append('港大')
    s_list.append('香港中文大学')
    return s_list


def relabel_school(school, sch_list):
    sch_label = []
    for s in school:
        if s in sch_list:
            s = 1
        else:
            s = 0
        sch_label.append(s)
    return sch_label


def group_by_sex(df):
    man = df[df['3、您的性别是:'] == 1]
    man.to_excel('man.xlsx')
    woman = df[df['3、您的性别是:'] == 2]
    woman.to_excel('woman.xlsx')


def group_by_school(df, sch_list):
    class1 = df[df['(1)1、您就读的大学是:___'].isin(sch_list)]
    class1.to_excel('重点大学.xlsx')
    class2 = df[~df['(1)1、您就读的大学是:___'].isin(sch_list)]
    class2.to_excel('普通大学.xlsx')


def group_by_major(df):
    class1 = df[df['2、您的专业类型'] == 1]
    class1.to_excel('文史哲.xlsx')
    class2 = df[df['2、您的专业类型'] == 2]
    class2.to_excel('理工农医.xlsx')
    class3 = df[df['2、您的专业类型'] == 3]
    class3.to_excel('经管法.xlsx')
    class4 = df[df['2、您的专业类型'] == 4]
    class4.to_excel('教育.xlsx')
    class5 = df[df['2、您的专业类型'] == 5]
    class5.to_excel('艺术.xlsx')
