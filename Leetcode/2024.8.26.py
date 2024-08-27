from typing import List
import collections
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates

#dfs
class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        for employee in employees:
            mp = {employee.id:employee}
        def dfs(idx:int)->int:
            employee = mp[idx]
            total = employee.importance + sum(dfs(subIdx) for subIdx in employee.subordinates)
            return total
    
        return dfs(id)
#bfs
class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        for employee in employees:
            mp = {employee.id:employee}
        total = 0
        que = collections.deque([id])
        while que:
            curIdx = que.popleft()
            employee = mp[curIdx]
            total+=employee.importance
            for subIdx in employee.subordinates:
                que.append(subIdx)
    
        return total        