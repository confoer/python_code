from typing import List
from collections import defaultdict
from bisect import bisect_left
class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        k=(n*(n+1)//2+1)//2
        def check(upper:int)->int:
            cnt = l = 0
            freq = defaultdict(int)
            for r , in_ in enumerate(nums):
                freq[in_] +=1
                while len(freq)>upper:
                    out = nums[l]
                    freq[out]-=1
                    if freq[out]==0:
                        del freq[out]
                    l+=1
                cnt += r-l+1
                if cnt >=k:
                    return True
            return False
        return bisect_left(range(len(set(nums))),True,1,key=check)