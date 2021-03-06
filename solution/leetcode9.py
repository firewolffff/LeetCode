#Design and implement a data structure for Least Recently Used (LRU) cache.
#使用index + pop 比 remove 的速度快
class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity;
        self.dic = {};
        self.recentUnused = [];
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        value = self.dic.get(key,-1);
        if(value != -1):
            index = self.recentUnused.index(key);
            self.recentUnused.pop(key);
            self.recentUnused.insert(0,key);
            
        return value;
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        
        if(key in self.dic):
            self.dic[key] = value;
            index = self.recentUnused.index(key);
            self.recentUnused.pop(key);
            self.recentUnused.insert(0,key);
        else:
            if(len(self.dic) < self.cap):
                self.dic[key] = value;
                self.recentUnused.insert(0,key);
            
            else:
                #print(self.dic.keys());
                #print(self.recentUnused[-1]);
                del self.dic[self.recentUnused.pop(-1)];
                self.dic[key] = value;
                self.recentUnused.insert(0,key);
				

#插入排序
def insertSort(arr):
    base = 0;
    for i in range(1,len(arr)):
        j = i;
        base = arr[i]
        while(j > 0 and base < arr[j-1]):
            arr[j] = arr[j-1];
            j = j - 1;
        arr[j] = base;


#链表的插入排序
def insertionSortList(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if(not head or not head.next):
        return head;

    dumy = ListNode(head.val);

    p = head;
    q = dumy;#dumy 的头 总是值最大的节点
    while(p):
        if(p.val >= q.val):
            q.next = ListNode(p.val);
            p = p.next;
            q = q.next;
        else:
            pre = dumy;
            cur = dumy.next;
            while(cur and cur.val < p.val):
                pre = cur;
                cur = cur.next;

            node = ListNode(p.val);
            pre.next = node;
            node.next = cur;
            p = p.next;

    return dumy.next;
	

#列表 归并排序 
def sortList(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if(head is None or head.next is None):
        return head;
    return mergeSort(head);
    
    
def merge(l1,l2):
    dumy = ListNode(0);
    p = dumy;
    while(l1 and l2):
        if(l1.val < l2.val):
            p.next = ListNode(l1.val);
            l1 = l1.next;
            p = p.next;
        else:
            p.next = ListNode(l2.val);
            l2 = l2.next;
            p = p.next;
    if(l1):
        p.next = l1;
    if(l2):
        p.next = l2;
    return dumy.next;

def mergeSort(head):
    if(head is None or head.next is None):
        return head;
    slow = head;
    fast = head;
    while(fast.next and fast.next.next):
        slow = slow.next;
        fast = fast.next.next;
    l2 = slow.next;
    slow.next = None;
    left = mergeSort(head);
    right = mergeSort(l2);
    return merge(left,right);
	

#列表排序 速度比使用归并排序快
def sortList(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if(not head):
        return None;
    nodes = [];
    while(head):
        nodes.append(head.val);
        head = head.next;

    nodes = sorted(nodes);
    dumy = ListNode(nodes[0]);
    p = dumy;
    for i in nodes[1:]:
        p.next = ListNode(i);
        p = p.next;
    return dumy;
	
	

#求共线最多的点数
#为什么本地调试结果与LeetCode测试结果不一致？
def maxPoints(points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    if(len(points)==0):
        return 0;
    if(len(points)==1):
        return 1;

    maxpoint = 0;

    for i in range(len(points)):
        first = points[i];
        dic = {};
        samepoints = 1;
        count = 0;
        for j in range(i+1,len(points)):
            second = points[j];
            if(first[0]==second[0] and first[1]==second[1]):
                samepoints += 1;
                continue;
            elif(second[0] == first[0]):
                line = ('x',first[0]);
            else:
                k = (second[1] - first[1])/(second[0]-first[0]);
                b = first[1] - k*first[0];
                line = (k,b);

            if(line in dic):
                dic[line] += 1;
            else:
                dic[line] = 1;
            count = max(count,dic[line]);
        #print(count,i);
        maxpoint = max(maxpoint,count+samepoints);
        print(maxpoint,count,i);
    return maxpoint;
	
	
#计算逆波兰表达式
def evalRPN(tokens):
    operand = [];
    for token in tokens:
        if(token in ['+','-','*','/']):
            behind = operand.pop();
            forword = operand.pop();
            if(token == '+'):
                operand.append(behind + forword);
            elif(token == '-'):
                operand.append(forword - behind);
            elif(token == '*'):
                operand.append(forword * behind);
            else:
                if(forword * behind < 0 and abs(forword)<abs(behind)):
                    operand.append(0);
                else:
                    operand.append(forword // behind);
        else:
            operand.append(int(token));
    print(operand);
    if(operand):
        return operand.pop();
    else:
        return 0;
		
		
"""
Given an integer array nums, find the contiguous subarray within an array 
(containing at least one number) which has the largest product.
"""
def maxProduct(nums):
    maxproduct = nums[0];
    minproduct = nums[0];
    res = nums[0];
    for i in range(1,len(nums)):
        ta = max(maxproduct*nums[i],nums[i],minproduct*nums[i]);
        tb = min(minproduct*nums[i],nums[i],maxproduct*nums[i]);
        maxproduct = ta;
        minproduct = tb;

        res = max(res,max(maxproduct,minproduct));
    return res;
	
	
"""
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).
Find the minimum element.
"""
def findMin(nums):
    minval = nums[0];

    for n in nums:
        if(n>=minval):
            continue;
        else:
            minval = n;
            break;
    return minval;
	

#循环小数转化为有限小数表示方式
def fractionToDecimal( numerator, denominator):
        z = numerator // denominator ;
        mod = numerator % denominator ;
        
        if(mod == 0):
            return str(z);
        else:
            mod = mod if numerator*denominator < 0 else denominator-mod;
            fra = [];
            rec_fra = [];
            mod = mod * 10;
            fra.append(str(mod // denominator));
            mod = mod % denominator;
            rec_fra.append(mod);
            index = 0;
            while(mod != 0):
                mod = mod * 10;
                mz = mod // denominator;
                mod = mod % denominator;
                if(mod in rec_fra):
                    fra.append(str(mz));
                    index = rec_fra.index(mod);
                    break;
                else:
                    fra.append(str(mz));
                    rec_fra.append(mod);
            
            k = len(rec_fra) - index;
            print(fra);
            print(rec_fra);
            if(mod == 0):
                return str(z) + '.' + ''.join(fra);
            else:
                n = len(fra);
                if(k == n):
                    return str(z) + '.' +'(' + ''.join(fra) + ')';
                else:
                    
                    nonrec_part = fra[:n-k];
                    rec_part = fra[n-k:];
                    return str(z) + '.' + ''.join(nonrec_part) + '(' + ''.join(rec_part) + ')';
					
					
#Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
def twoSum(numbers, target):
        n = len(numbers);
        k = None;
        for i in range(n):
            if(numbers[i] > target):
                k = i;
                break;
            else:
                continue;
        k = n-1 if k is None else k;
        
        left = 0;
        right = k;
        while(right > left):
            if(numbers[left] + numbers[right] == target):
                break;
            elif(numbers[left] + numbers[right] > target):
                right -= 1;
            else:
                left += 1;
                
        return [left+1,right+1];
		

def convertToTitle( n):
    dic = {1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',
          11:'K',12:'L',13:'M',14:'N',15:'O',16:'P',17:'Q',18:'R',19:'S',
          20:'T',21:'U',22:'V',23:'W',24:'X',25:'Y',26:'Z'};

    if(n<27):
        return dic[n];

    res = [];
    while(n > 0):
        tmp = n % 26;
        tmp = 26 if tmp == 0 else tmp;
        res.append(tmp);
        n -= tmp;
        n = n // 26;

    print(res)
    s = ""
    for r in res[::-1]:
        s += dic[r];
    return s;
	
def titleToNumber(s: str) -> int:
    A = ord('A');
    n = len(s);
    res = 0;
    for i in range(n):
        e = ord(s[i]) - A + 1;
        res = res + e * pow(26,n-i-1);
    return res;
	
#Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
def majorityElement(nums):
    count = 0
    candidate = None

    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)

    return candidate
	
#n的阶乘末尾0 的个数
def trailingZeroes( n):
    if(n==0):
        return 0;

    e = 5;
    res = 0;
    while(n > 0):
        res = res + n//e;
        n = n // 5;

    return res;


def calculateMinimumHP(dungeon: List[List[int]]) -> int:
    m,n = len(dungeon),len(dungeon[0])
    dp = [[0] * n] *m
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if i == m-1 and j == n-1:
                dp[i][j] = max(0,-dungeon[i][j])
            elif i == m-1:
                dp[i][j] = max(0,dp[i][j+1] - dungeon[i][j])
            elif j == n-1:
                dp[i][j] = max(0,dp[i+1][j] - dungeon[i][j])
            else:
                dp[i][j] = max(0,min(dp[i][j+1],dp[i+1][j])- dungeon[i][j])
    return dp[0][0] + 1
	

#数字字符组合最大
from functools import cmp_to_key
def largestNumber(nums):
    str_nums = [str(n) for n in nums];

    def strSort(x,y):
        if(x+y > y+x):
            return 1;
        elif(x+y == y+x):
            return 0;
        else:
            return -1;

    str_nums = sorted(str_nums,key=cmp_to_key(strSort),reverse=True);
    return ''.join(str_nums);
	
class Solution:
    def __It__(self,x,y):
        return x+y > y+x;
    
    def largestNumber(self, nums):
        str_nums = [str(n) for n in nums];

        
        
        str_nums = sorted(str_nums,key=Solution);
        return ''.join(str_nums);
		

def iceNum(P,W,V):
    minw = min(W);
    res = minw;
    W = [w-minw for w in W];
    while(P>0):
        for i in range(len(W)):
            if(W[i]==0):
                P = P - V[i];
            else:
                W[i] = W[i] - 1;
        if(P > 0):
            res += 1;
    return res;