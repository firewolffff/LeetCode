#!/usr/bin/env python
# coding: utf-8


#先成环，后解环
def rotateRight(head: ListNode, k: int) -> ListNode:
    if(head==None):
        return None;
    new_head = head;
    n = 1;
    while(head.next is not None):
        n = n + 1;
        head = head.next;
    head.next = new_head;

    k = k % n;
    for i in range(n-k):
        head = head.next;
    new_head = head.next;
    head.next = None;
    return new_head;

#回溯法速度太慢 统计从(0,0)位置到(m-1,n-1)位置的所有行走路线数目 只能向下和向右移动
def uniquePaths( m, n):
    start = [0,0];
    end = (m-1,n-1);
    count = [0];
    def backtrace(position):
        if(position[0]<m and position[1]<n):
            if(position[0] == end[0] and position[1] == end[1]):
                count[0] = count[0] + 1;
                return ;
            else:
                #down
                position[0] = position[0] + 1;
                backtrace(position);
                position[0] = position[0] - 1;
                #right
                position[1] = position[1] + 1;
                backtrace(position);
                position[1] = position[1] - 1;

        else:
            return ;

    backtrace(start);
    return count[0];

#动态规划速度快
def uniquePaths(m, n):
    # bottom-top dp solution
    # dp[i][j] indicates the total paths at position i,j
    dp = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m-1, -1, -1):
        for j in range(n-1,-1,-1):
            if i == m-1 or j == n-1:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i+1][j]+dp[i][j+1]
    return dp[0][0]


#obstacleGrid == 1 表示此处不通
def uniquePathsWithObstacles(obstacleGrid):
    m = len(obstacleGrid);
    n = len(obstacleGrid[0]);
    if(obstacleGrid[0][0] == 1):
        return 0;
    obstacleGrid[0][0] = 1;

    for i in range(1,n):
        obstacleGrid[0][i] = obstacleGrid[0][i-1] if obstacleGrid[0][i]==0 else 0;

    for j in range(1,m):
        obstacleGrid[j][0] = obstacleGrid[j-1][0] if obstacleGrid[j][0]==0 else 0;


    for i in range(1,m):
        for j in range(1,n):
            if(obstacleGrid[i][j]==0):
                obstacleGrid[i][j] = obstacleGrid[i-1][j]+obstacleGrid[i][j-1];
            else:
                obstacleGrid[i][j] = 0;
    print(obstacleGrid);
    return obstacleGrid[m-1][n-1];


#Given a m x n grid filled with non-negative numbers, 
#find a path from top left to bottom right which minimizes the sum of all numbers along its path.
#You can only move either down or right at any point in time.
"""
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
"""
def minPathSum(grid):
    m = len(grid);
    n = len(grid[0]);
    dp = [[0 for i in range(n)] for j in range(m)];
    dp[0][0] = grid[0][0];
    for i in range(1,m):
        dp[i][0] = dp[i-1][0] + grid[i][0];
    for j in range(1,n):
        dp[0][j] = dp[0][j-1] + grid[0][j];

    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j];

    return dp[m-1][n-1];



#包括科学记数法
def isNumber(s):
    if len(s) == 0:
        return False
    dot = 0
    digit = 0
    s = s.strip() + " "
    i = 0

    if s[i] in ['+', '-']:
        i += 1

    while s[i].isdigit() or s[i] == '.':
        if i < len(s) and s[i] == '.':
            dot += 1
            if dot > 1:
                return False
        if i < len(s) and s[i].isdigit():
            digit += 1
        i += 1

    if digit == 0:
        return False

    if s[i] == 'e':
        if s[i+1] in ['+', '-']:
            i += 1
        sright = s[i + 1:]
        if not sright.strip().isdigit():
            return False
        i += len(sright)
    return len(s) - 1 == i


#计算近似平方根
def mySqrt(x):
    low = 0;
    hight = x;
    if(x <= 1):
        return x;
    while(low < hight):
        mid = (low + hight) // 2;
        if(mid * mid <= x < (mid+1) * (mid+1)):
            return mid;
        elif(mid * mid > x):
            hight = mid;
        else:
            low = mid;


#打印
def fullJustify(words, maxWidth):
    i = 0;
    n = len(words);
    temp = [];
    curlen = 0;
    res = [];
    while(i < n):
        if(curlen + len(words[i]) + len(temp) <= maxWidth):
            temp.append(words[i]);
            curlen = curlen + len(words[i]);
            i = i + 1;
        else:
            print(temp);
            print(curlen);
            spacelen = maxWidth - curlen;
            if(len(temp) == 1):
                res.append(temp[0] + " " * spacelen);
            else:
                if(spacelen % (len(temp) -1) ==0):
                    wordspace = spacelen // (len(temp) - 1);
                    s = temp[0];
                    for w in temp[1:]:
                        s = s + " " * wordspace + w;
                    res.append(s);
                else:
                    tn = len(temp);
                    wordspace = spacelen // (tn -1);
                    remain = spacelen % (tn -1);
                    s = temp[0];
                    k = 1;
                    while(remain > 0 and k < n):
                        s = s + " "*(wordspace + 1) + temp[k];
                        k = k + 1;
                        remain = remain - 1;
                    while(k < tn):
                        s = s + " "*wordspace + temp[k];
                        k = k + 1;
                    res.append(s);
            curlen = 0;
            temp = [];
    print(temp);
    #最后一行
    if(temp):
        spacelen = maxWidth - curlen;
        if(len(temp) == 1):
            res.append(temp[0] + " " * spacelen);
        else:
            s = temp[0];
            for w in temp[1:]:
                s = s + ' ' + w;
            s = s + ' '*(maxWidth - len(s));
            res.append(s);
    return res;


"""
跳楼梯，每次跳一个或二个阶梯，求有几种方式到达第n 个台阶
f(n) = f(n-1) + f(n-2)
"""
def climbStairs(n):
    if(n==1):
        return 1;
    n1 = 1;
    n2 = 2;
    for i in range(3,n+1):
        t = n1 + n2;
        n1 = n2;
        n2 = t;
    return n2;


#将文件路径最简化 linux系统下的文件路径格式为标准
#结果一致 这个通过 下面的不通过
def simplifyPath(path):
    patharr = path.split('/')
    for i in  range(len(patharr)-1,-1,-1):
        if len(patharr[i]) == 0:
            del  patharr[i]
        elif patharr[i] == '/':
            del patharr[i]
        elif patharr[i] == '.':
            del patharr[i]
    i = 0
    while i < len(patharr):
        if patharr[i] in '..':
            del patharr[i]
            i -= 1
            if i >= 0:
                del patharr[i]
                i -=1
        i += 1
    return '/'+'/'.join(patharr)
    
#LeetCode 未通过
def simplifyPath(path):
    n = len(path);
    if(n == 1):
        return '/';
    cp = path[0] if path[0]=='/' else '/';
    for i in range(1,n):
        if(cp[-1] + path[i] == '//'):
            continue;
        elif(cp[-1] + path[i] == '..'):
            index = cp.rfind('/');
            if(index == 0):
                cp = '/';
            else:
                cp = cp[:index];
                index = cp.rfind('/');
                cp = cp[:index] if index > 0 else '/';
        elif(cp[-1] == '.'):
            cp = cp[:-1];
        else:
            cp = cp + path[i];
    if(len(cp) > 1 and (cp[-1] == '/' or cp[-1] == '.')):
        return cp[:-1];
    return cp;
    

#convert word1 to word2 and return the minimum count of operations 
#operations : replace a char,insert a char, remove a char
#https://leetcode.com/problems/edit-distance/discuss/273352/java-7m-dp-solution-with-detailed-explanation
def minDistance(word1, word2):
	dist = [[0] * (len(word2)+1) for _ in range(len(word1) + 1)];
	for i in range(len(word1)): 
		dist[i + 1][0] = i + 1;
	for i in range(len(word2)): 
		dist[0][i + 1] = i + 1;
	for i in range(len(word1)): 
		for j in range(len(word2)): 
			if( word1[i] == word2[j] ):
				dist[i + 1][j + 1] = dist[i][j];
			else:
				dist[i + 1][j + 1] = min(min(dist[i][j + 1], dist[i + 1][j]) + 1, dist[i][j] + 1);
	return dist[-1][-1];


#填充0 将0所在的行和列填充为0
def setZeroes(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix);
    m = len(matrix[0]);
    col = {};
    for i in range(n):
        if(0 in matrix[i]):
            for j in range(m):
                if(matrix[i][j] == 0):
                    col[j] = j;
                matrix[i][j] = 0;

    for j in col.keys():
        for i in range(n):
            matrix[i][j] = 0;


"""
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
"""
def searchMatrix(matrix, target):
    n = len(matrix);
    if(n==0):
        return False;
    m = len(matrix[0]);
    if(m==0):
        return False;
    first_col = [row[0] for row in matrix];
    k = n-1;
    for i in range(n):
        if(first_col[i] == target):
            return True;
        if(i<n-1 and first_col[i] < target and first_col[i+1] > target):
            k = i;
            break;
    if(target in matrix[k]):
        return True;
    else:
        return False;


#将由0，1，2 组成的数组排序
def sortColors(nums):
    p0 = -1;
    p1 = -1;
    p2 = -1;
    for i in range(len(nums)):
        temp = nums[i];
        p2 += 1;
        nums[p2] = 2;
        if(temp <= 1):
            p1 += 1;
            nums[p1] = 1;
        if(temp == 0):
            p0 += 1;
            nums[p0] = 0;
    
import collections
#来自LeetCode 解决方案 找出s 中包含t 的最小字串
def minWindow(s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """

    if not t or not s:
        return ""

    # Dictionary which keeps a count of all the unique characters in t.
    dict_t = collections.Counter(t)

    # Number of unique characters in t, which need to be present in the desired window.
    required = len(dict_t)

    # left and right pointer
    l, r = 0, 0

    # formed is used to keep track of how many unique characters in t are present in the current window in its desired frequency.
    # e.g. if t is "AABC" then the window must have two A's, one B and one C. Thus formed would be = 3 when all these conditions are met.
    formed = 0

    # Dictionary which keeps a count of all the unique characters in the current window.
    window_counts = {}

    # ans tuple of the form (window length, left, right)
    ans = float("inf"), None, None

    while r < len(s):

        # Add one character from the right to the window
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1

        # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1

        # Try and contract the window till the point where it ceases to be 'desirable'.
        while l <= r and formed == required:
            character = s[l]

            # Save the smallest window until now.
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            # The character at the position pointed by the `left` pointer is no longer a part of the window.
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1

            # Move the left pointer ahead, this would help to look for a new window.
            l += 1    

        # Keep expanding the window once we are done contracting.
        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


import collections

#从n个数中取k个数的组合
#C(n+1,m) =  C(n,m) + C(n,m-1);
def combine(n, k):
    if k==1:
        return [[i] for i in range(1,n+1)]
    elif k==n:
        return [[i for i in range(1,n+1)]]
    else:
        rs=[]
        rs+=combine(n-1,k)
        part=combine(n-1,k-1)
        for ls in part:
            ls.append(n)
        rs+=part
        return rs

#组合算法
def combine(nums,k):
    if(k == 0 or k > len(nums)):
        return [];
    if(k == len(nums)):
        return [nums];
    
    n = len(nums);
    mask = [False for i in range(n)];
    for i in range(k):
        mask[i] = True;
    
    #寻找True False 组合
    res = [];
    position = 0;
    last = n - 1;
    preTrueNum = k;
    while(position < n-k):
        temp = [];
        for i in range(n):
            if(mask[i]):
                temp.append(nums[i]);
        res.append(temp);
        for i in range(n-1):
            if(mask[i] and not mask[i+1]):
                mask[i] = False;
                mask[i+1] = True;
                break;
                
        if(mask[last]):
            for i in range(last):
                if(i < preTrueNum-1):
                    mask[i] = True;
                else:
                    mask[i] = False;
                
            last = last - 1;
            preTrueNum = preTrueNum - 1;
        print(mask);
        for i in range(n):
            if(mask[i]):
                position = i;
                break;
    #取出最后一种组合
    temp = [];
    for i in range(n):
        if(mask[i]):
            temp.append(nums[i]);
    res.append(temp);
    return res;

#求nums 的所有子集
def subsets(nums):
    res = [];
    for i in range(0,len(nums)+1):
        res += combin(nums,i);
    return res;
    
def combin(nums,k):
    if(k==0):
        return [[]];
    elif(k==1):
        return [[i] for i in nums];
    elif(k==len(nums)):
        return [nums];

    res = [];
    res += combin(nums[:-1],k);
    c2 = combin(nums[:-1],k-1);
    for ls in c2:
        ls.append(nums[-1]);
    res += c2;
    return res;
    


def subset(nums):
    res = [[]]
    for n in nums:
        for i in range(len(res)):
            res.append(res[i] + [n])
    return res


#判断 board 中的字符是否存在能组合成 word 的情况
def exist(board, word):
    n = len(board);
    m = len(board[0]);

    for i in range(n):
        for j in range(m):
            if(dfs(board,i,j,word)):
                return True;
    return False;
                
    
def dfs(board,i,j,word):
    if(len(word) == 0):
        return True;
    if(i<0 or i >= len(board) or j<0 or j >= len(board[0]) or board[i][j] != word[0]):
        return False;
    t = board[i][j];
    board[i][j] = '#';
    #向四周搜索下一个字符
    res = dfs(board,i+1,j,word[1:]) or dfs(board,i-1,j,word[1:]) or dfs(board,i,j+1,word[1:]) or dfs(board,i,j-1,word[1:]);
    board[i][j] = t;
    return res;


# In[12]:


board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]];
word = "ABCCED";
print(exist(board,word));