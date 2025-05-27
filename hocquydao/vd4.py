#thuật toán A*

from collections import defaultdict
from queue import PriorityQueue

data = defaultdict(list)
data['A']=['B',2,'C',1,'D',3,6]
data['B']=['E',5,'F',4,3]
data['C']=['G',6,'H',3,4]
data['D']=['I',2,'J',4,5]
data['E']=[3]
data['F']=['K',2,'L',1,'M',4,1]
data['G']=[6]
data['H']=['N',2,'O',4,2]
data['I']=[5]
data['J']=[4]
data['K']=[2]
data['L']=[0]
data['M']=[4]
data['N']=[0]
data['O']=[4]

class Node:
    def __init__(self, name, par = None,g = 0, h = 0):
        self.name = name
        self.par = par
        self.g = g
        self.h = h
    def display(self):
        print(self.name,self.g,self.h)
    def __lt__(self, other):
        if other == None:
            return False
        return self.g + self.h < other.g + other.h
    def __eq__(self, other):
        if other == None:
            return False
        return self.name == other.name
def equal(O,G):
    if O.name == G.name:
        return True
    return False
def checkInPriority(tmp,c):
    if tmp == None:
        return False
    return (tmp in c.queue)
def getPath(O):
    print(O.name)
    if O.par != None:
        getPath(O.par)
    else:
        return

def AStar(S=Node('A'),G=Node('N')):
    Open = PriorityQueue()
    Closed = PriorityQueue()
    S.h = data[S.name][-1]
    Open.put(S)
    while True:
        if Open.empty():
            print('tim kiem that bai')
            return
        O = Open.get()
        Closed.put(O)
        print('duyet: ', O.name, O.g, O.h)
        if equal(O,G) == True:
            print('tim kiem thanh cong')
            getPath(O)
            print('distance: ', (O.g+O.h))
            return
        #tìm con
        i = 0
        while i < len(data[O.name])-1:
            name = data[O.name][i]
            g = O.g + data[O.name][i+1]
            h = data[name][-1]
            tmp = Node(name = name,g = g, h = h)
            tmp.par = O

            ok1 = checkInPriority(tmp,Open)
            ok2 = checkInPriority(tmp,Closed)
            if not ok1 and not ok2:
                Open.put(tmp)
            i += 2

AStar(S=Node('A'),G=Node('N'))


