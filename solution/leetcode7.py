#!/usr/bin/env python
# coding: utf-8

#杨辉三角 Pascal's Triangle numRows 从1开始
def generate(numRows):
    if(numRows <=0):
        return [];
    if(numRows == 1):
        return [[1]];
    if(numRows == 2):
        return [[1],[1,1]];

    res = [[1],[1,1]];
    for i in range(3,numRows+1):
        temp = [0 for _ in range(i)];
        temp[0] = 1;
        temp[-1] = 1;
        r = res[-1];
        for j in range(1,i-1):
            temp[j] = r[j-1] + r[j];
        res.append(temp);

    return res;



#获得杨辉三角中第rowIndex 行的系数值，rowIndex 从0开始
def getRow(rowIndex):
        
    if(rowIndex < 0):
        return [];
    if(rowIndex == 0):
        return [1];
    if(rowIndex == 1):
        return [1,1];

    pre = [1,1];
    for i in range(3,rowIndex+2):
        temp = [1 for _ in range(i)];
        for j in range(1,i-1):
            temp[j] = pre[j-1] + pre[j];
        pre = temp;
    return pre;



#三角路径最小和
def minimumTotal(triangle):
    if(len(triangle) <= 0):
        return 0;
    for i in range (1,len(triangle)):
        for j in range (1,len(triangle[i])-1):
            triangle[i][j] = min(triangle[i-1][j]+triangle[i][j] , triangle[i][j]+triangle[i-1][j-1]);
        triangle[i][0] = triangle[i-1][0]+triangle[i][0];
        triangle[i][-1] = triangle[i-1][-1]+triangle[i][-1];

    return min(triangle[-1]);



#股票买卖收益，一次买入卖出 求最大收益
def maxProfit(prices):
    n = len(prices);
    if(n <=1):
        return 0;
    profit = 0;
    buy_p = prices[0];
    for i in range(n):
        if(prices[i] < buy_p):
            buy_p = prices[i];
        elif(prices[i] - buy_p > profit):
            profit = prices[i] - buy_p;

    return profit;


#可多次买入卖出 完成一次买卖交易后才能进行第二次，既不能连续两次都是买或者卖
#最大总收益等于相邻波峰波谷差值和
def maxProfit(prices):
    n = len(prices);
    v = prices[0];
    p = prices[0];
    i = 0;
    sum_profit = 0;
    while(i < n-1):
        while(i < n - 1 and prices[i] >= prices[i+1]):
            i = i + 1;
        v = prices[i];

        while(i < n - 1 and prices[i] <= prices[i+1]):
            i = i + 1;
        p = prices[i];

        sum_profit = sum_profit + p - v;
    return sum_profit;



def maxProfit(prices):
    prices.append(0);
    n = len(prices);
    sum_profit = 0;
    buy = 0;
    for i in range(1,n):
        if(prices[i] > prices[i-1]):
            continue;
        else:
            profit = prices[i-1] - prices[buy];
            if(profit>0):
                sum_profit = sum_profit + profit;
        buy = i;
    return sum_profit;



#网友答案
#最多允许两次交易
#release2 第二次卖出当前股票后总钱数
#release1 第一次卖出当前股票后总钱数
#hold2 第二次买入当前股票后剩余的总钱数
#hold1 第一次买入当前股票后剩余的总钱数
def maxProfit(prices):
    hold1 = -10000;
    hold2 = -10000;
    release1 = 0;
    release2 = 0;
    for i in prices:                             # Assume we only have 0 money at first
        release2 = max(release2, hold2+i);     # The maximum if we've just sold 2nd stock so far.
        hold2 = max(hold2,    release1-i);  # The maximum if we've just buy  2nd stock so far.
        release1 = max(release1, hold1+i);     # The maximum if we've just sold 1nd stock so far.
        hold1 = max(hold1,    -i);          # The maximum if we've just buy  1st stock so far. 
    return release2; #Since release1 is initiated as 0, so release2 will always higher than release1.


#求二叉树路径上数字的和的最大值
import sys
max_s = -sys.maxsize
def maxPathSum(root):
    if(not root):
        return 0;
    
    #返回最大和 根，根+左，根+右
    def helper(root):
        if(not root):
            return 0;
        ls = helper(root.left);
        rs = helper(root.right);
        
        #二叉树和的组合形式 根，根+左，根+右，左，右
        max_s = max(max_s,max(root.val + ls + rs,root.val));
        max_s = max(max_s,max(root.val + ls,root.val + rs));
        return max(ls + root.val, max(rs + root.val, root.val));

    helper(root);
    return max_s;  


#判断字符串是否为回文字符串，只判断字符串中的数字和字母
def isPalindrome(s):
    n = len(s);
    if(n == 0):
        return True;

    left = 0;
    right = n-1;
    while(left <= right):

        if(s[left].lower() == s[right].lower()):
            left = left + 1;
            right = right - 1;
            continue;
        else:
            t = s[left] + s[right];
            if(t.isalnum()):
                return False;
            else:
                if(not s[left].isalnum()):
                    left = left + 1;
                if(not s[right].isalnum()):
                    right = right - 1;
    return True;



"""
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:
Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
"""
#LeetCode 解决方案
from collections import defaultdict
def ladderLength(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """

    if endWord not in wordList or not endWord or not beginWord or not wordList:
        return 0

    # Since all words are of same length.
    L = len(beginWord)

    # Dictionary to hold combination of words that can be formed,
    # from any given word. By changing one letter at a time.
    all_combo_dict = defaultdict(list)
    for word in wordList:
        for i in range(L):
            # Key is the generic word
            # Value is a list of words which have the same intermediate generic word.
            all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)


    # Queue for BFS
    queue = [(beginWord, 1)]
    # Visited to make sure we don't repeat processing same word.
    visited = {beginWord: True}
    while queue:
        current_word, level = queue.pop(0)      
        for i in range(L):
            # Intermediate words for current word
            intermediate_word = current_word[:i] + "*" + current_word[i+1:]

            # Next states are all the words which share the same intermediate state.
            for word in all_combo_dict[intermediate_word]:
                # If at any point if we find what we are looking for
                # i.e. the end word - we can return with the answer.
                if word == endWord:
                    return level + 1
                # Otherwise, add it to the BFS Queue. Also mark it visited
                if word not in visited:
                    visited[word] = True
                    queue.append((word, level + 1))
            all_combo_dict[intermediate_word] = []
    return 0


#找出所有最短的从 beginWord转化到endWord 的路径
#方案超时
from collections import defaultdict
def findLadders(beginWord, endWord, wordList):
    if(endWord not in wordList or len(beginWord)==0 or len(endWord)==0):
        return [];

    l = len(beginWord);
    combo_dic = defaultdict(list);
    for word in wordList:
        for i in range(l):
            combo_dic[word[:i]+"*"+word[i+1:]].append(word);
    

    q = [[beginWord]];
    candiate_res = [];
    while(q):
        curlist = q.pop(0);
        cur = curlist[-1];
        if(cur == endWord):
            candiate_res.append(curlist);
            continue;
        for j in range(l):
            intermediate = cur[:j] + "*" + cur[j+1:];
            print(combo_dic[intermediate]);
            for word in combo_dic[intermediate]:
                newlist = curlist.copy();
                if(word not in curlist):
                    newlist.append(word);
                    q.append(newlist);
    
    if(len(candiate_res) == 0):
        return [];
    lenth = [len(r) for r in candiate_res];
    min_lenth = min(lenth);
    res = [];
    for r in candiate_res:
        if(len(r) == min_lenth):
            res.append(r);
    return res;

print(findLadders("hit","cog",["hot","dot","dog","lot","log","cog"]))

#求最长连续数字的长度
def longestConsecutive(nums):
    n = len(nums);
    if(n == 0):
        return 0;
    if(n == 1):
        return 1;

    nums = sorted(nums);

    diff = [nums[i] - nums[i-1] for i in range(1,n)];

    max_count = 0;
    count = 1;
    for d in diff:
        if(d == 1):
            count = count + 1;
        elif(d == 0):
            continue;
        else:
            if(count > max_count):
                max_count = count;
            count = 1;
    return max(max_count,count);

#LeetCode 解决方案，采用hash 解决该问题
def longestConsecutive(nums):
    longest_streak = 0
    num_set = set(nums)

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak



#求二叉树路径组成的数字的和
sumN = 0;
def sumNumbers(root):
    if(not root):
        return 0;
    num = root.val;

    if(root.left is None and root.right is None):
        sumN += num;

    if(root.left):
        root.left.val = num * 10 + root.left.val;

    if(root.right):
        root.right.val = num * 10 + root.right.val;

    sumNumbers(root.left);
    sumNumbers(root.right);

    return sumN;

#网友答案
def sumNumbers(root):
    if not root:
        return 0
    pathsum = []
    sumroot(root, '', pathsum)
    return sum(pathsum)
        
def sumroot(root, current, pathsum):
    if root.left:
        sumroot(root.left, current + str(root.val), pathsum)
    if root.right:
        sumroot(root.right, current + str(root.val), pathsum)
    if not root.left and not root.right:
        current += str(root.val)
        pathsum.append(int(current))


"""
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
A region is captured by flipping all 'O's into 'X's in that surrounded region.
"""
#将被围住的点同化
def solve(board):
    """
    Do not return anything, modify board in-place instead.
    """
    if(not board):
        return ;
    n = len(board);
    m = len(board[0]);

    #从边缘开始，若存在一条路径连入内部，则此路通，不能被围，
    #否则除该点外内部所有点都被围
    for j in range(m):
        if(board[0][j]=='O'):
            helper(board,0,j);

        if(board[n-1][j] == 'O'):
            helper(board,n-1,j);

    for i in range(n):
        if(board[i][0] == 'O'):
            helper(board,i,0);
        if(board[i][m-1] == 'O'):
            helper(board,i,m-1);
    
    print(board);
    for i in range(n):
        for j in range(m):
            if(board[i][j] == 'O'):
                board[i][j] = 'X';
            elif(board[i][j] == 'C'):
                board[i][j] = 'O';


def helper(board,i,j):
    n = len(board);
    m = len(board[0]);
    #O 在边缘上围不住
    if(i>=0 and i < n and j >= 0 and j < m and board[i][j]=='O'):
        board[i][j] = 'C';
        helper(board,i+1,j);
        helper(board,i-1,j);
        helper(board,i,j-1);
        helper(board,i,j+1);



