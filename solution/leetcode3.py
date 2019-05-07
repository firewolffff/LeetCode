#!/usr/bin/env python
# coding: utf-8


#判断是否为数独解
def isValidSudoku( board) -> bool:
        dic_subbox = [{},{},{}];
        for i in range(9):
            dic_i = {};
            if(i%3==0):
                dic_subbox = [{},{},{}];
            for j in range(9):
                if(board[i][j] != '.'):
                    if(board[i][j] in dic_i):
                        print('row')
                        return False;
                    else:
                        dic_i[board[i][j]] = board[i][j];
                
                    if(board[i][j] in dic_subbox[j//3]):
                        print('box')
                        return False;
                    else:
                        dic_subbox[j//3][board[i][j]] = board[i][j];
        
        for i in range(9):
            dic_i = {};
            for j in range(9):
                if(board[j][i] != '.'):
                    if(board[j][i] in dic_i):
                        print('col')
                        return False;
                    else:
                        dic_i[board[j][i]] = board[j][i];
        return True;


board = [[".","4",".",".",".",".",".",".","."],
         [".",".","4",".",".",".",".",".","."],
         [".",".",".","1",".",".","7",".","."],
         [".",".",".",".",".",".",".",".","."],
         [".",".",".","3",".",".",".","6","."],
         [".",".",".",".",".","6",".","9","."],
         [".",".",".",".","1",".",".",".","."],
         [".",".",".",".",".",".","2",".","."],
         [".",".",".","8",".",".",".",".","."]]


#数独
def solveSudoku(board):
    if(board is None or len(board)<9 or len(board[0])<9):
        return None;
    solve(board);

def solve(board):
    values = ['1','2','3','4','5','6','7','8','9'];
    for i in range(9):
        for j in range(9):
            if(board[i][j]=='.'):
                for val in values:
                    if(isvalid(board,i,j,val)):
                        board[i][j] = val;
                        if(solve(board)):
                            return True;
                        else:
                            board[i][j] = '.';
                return False;
    return True;

def isvalid(board,row,col,val):
    for i in range(9):
        if(i != col and board[row][i] == val):
            return False;
    
    for j in range(9):
        if(j != row and board[j][col] == val):
            return False;
    
    sub_box_r = row//3;
    sub_box_c = col//3;
    for i in range(3):
        for j in range(3):
            if(sub_box_r*3 + i != row and sub_box_c * 3 + j != col and board[sub_box_r*3 + 1][sub_box_c*3 + j] == val):
                return False;
    
    return True;


#规律查找问题
def countAndSay(n):
    if(n==1):
        return "1";
    if(n==2):
        return "11";
    
    temp = "11";
    for i in range(3,n+1):
        c = temp[0];
        count_c = 1;
        temp_res = "";
        for j in range(1,len(temp)):
            if(c != temp[j]):
                temp_res = temp_res + str(count_c) + c;
                c = temp[j];
                count_c = 1;
            else:
                count_c = count_c + 1;
        temp_res = temp_res + str(count_c) + c;
        temp = temp_res;
    return temp;


#统计所有能使得和为 target的组合情况
def combinationSum(candidates, target):
    candidates.sort();
    res = [];
    def combin(candidates,cur,curlist,k):
        if(cur == target):
            res.append(curlist);
            return ;
        if(cur > target):
            return ;
        for i in range(k,len(candidates)):
            curlist.append(candidates[i]);
            combin(candidates,cur+candidates[i],curlist.copy(),i); #传递curlist 不加copy 时，curlist 保留了上一次递归的结果
            curlist = curlist[:-1]
    combin(candidates,0,[],0);
    return res;

                

#每个元素只能使用一次
def combinationSum2(candidates,target):
    candidates.sort();
    res = [];
    def combin(candidates,cur,curlist,k):
        if(cur == target):
            if(curlist not in res):
                res.append(curlist);
            return ;
        if(cur > target):
            return ;
        for i in range(k,len(candidates)):
            curlist.append(candidates[i]);
            combin(candidates,cur+candidates[i],curlist.copy(),i+1); #传递curlist 不加copy 时，curlist 保留了上一次递归的结果
            curlist = curlist[:-1]
    combin(candidates,0,[],0);
    
    return res;



def findFirstMiss(nums):
    n = len(nums);
    for i in range(n):
        while(nums[i] > 0 and nums[i] <= n and nums[nums[i] - 1] != nums[i]):
            nums[i], nums[nums[i] - 1] = nums[nums[i]-1], nums[i];
        
    for i in range(n):
        if(nums[i] != i + 1):
            return i + 1;
        
    return n + 1;


#计算低洼面积
#****
#https://leetcode.com/problems/trapping-rain-water/submissions/
def trap(height):
    left = 0;
    right = len(height) - 1;
    ans = 0;
    left_max = 0;
    right_max = 0;
    while (left < right):
        if (height[left] < height[right]):
            if(height[left] >= left_max):
                left_max = height[left];
            else:
                ans = ans + (left_max - height[left]);
            left = left + 1;
        else:
            if(height[right] >= right_max):
                right_max = height[right];
            else:
                ans = ans + (right_max - height[right]);
            right = right - 1;

    return ans;


#两个数字字符串相加
def add_two_str(str1,str2):
    n1 = len(str1);
    n2 = len(str2);
    result = [];
    i = n1-1;
    j = n2-1;
    pre_val = 0;
    while(i >= 0 and j >= 0):
        temp = int(str1[i]) + int(str2[j]);
        temp = temp + pre_val;
        val = temp % 10;
        pre_val = int(temp/10);
        result.append(str(val));
        i = i - 1;
        j = j - 1;
    
    while(i >= 0):
        temp = int(str1[i]) + pre_val;
        val = temp % 10;
        pre_val = int(temp/10);
        result.append(str(val));
        i = i - 1;
    
    while(j >= 0):
        temp = int(str2[j]) + pre_val;
        val = temp % 10;
        pre_val = int(temp/10);
        result.append(str(val));
        j = j - 1;
    if(pre_val > 0):
        result.append(str(pre_val));
    result = result[::-1];
    return ''.join(result);

#两个数字字符串相乘
def strMutiple(str1,str2):
    n1 = len(str1);
    n2 = len(str2);
    r = None;
    for i in range(n1):
        val = int(str1[n1-1-i]);
        result = [];
        pre_val = 0;
        for j in range(n2):
            val_2 = int(str2[n2-1-j]);
            cur_val = val * val_2 + pre_val;
            pre_val = int(cur_val/10);
            v = cur_val % 10;
            result.append(str(v));
        
        if(pre_val > 0):
            result.append(str(pre_val));
        result = result[::-1];
        for k in range(i):
            result.append(str(0));
        
        result = ''.join(result);
        if(r is None):
            r = result;
        else:
            r = add_two_str(result,r);
    return r;


#数字字符串相乘
def multiply(num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        retVal = 0
        for j,ch in enumerate(num2[::-1]):
            sumVal, prevCarry = 0, 0 
            num = nums.index(ch)
            for i,ch2 in enumerate(num1[::-1]):
                num2 = nums.index(ch2)
                val = num * num2
                val += prevCarry
                carry = val / 10
                rem = val % 10
                sumVal += rem* (10**i)
                prevCarry = carry
            sumVal += (prevCarry * (10**(i+1)))
            retVal += (sumVal * (10**j))
        return str(retVal)



#简单正则匹配
def isMatch(s,p):
    #p == ""
    if(not p):
        return not s;
    
    if(not s):
        if(len(p) > 0):
            l = ''.join(list(set(list(p))));
            return False if l != "*" else True;
        else:
            return True;
    #p != ""
    if(p == "*"):
        return True;
    
    flag =  bool(s[0]) and p[0] in {s[0],"?"};
    if(p[0] == "*"):
        flag = True;
        
    if(len(p)>=1 and p[0]=="*"):
        return flag and isMatch(s,p[1:]) or isMatch(s[1:],p);
    else:
        return flag and isMatch(s[1:],p[1:]);
    


print(isMatch("babaaababaabababbbbbbaabaabbabababbaababbaaabbbaaab","***bba**a*bbba**aab**b"))


#跳棋游戏
def jump( nums):
    n = len(nums);
    if(n<=1):
        return 0;
    i =1;
    minstep = 1;
    maxdis = nums[0];  #the max index the next step can go
    while((maxdis+1)>i and (maxdis< (n-1))):  # when maxDis + 1 <=i, it means it can't go further, return 0
        j = maxdis+1; 
        while(i < j): # caculate the max distance the next step can go
            maxdis = max(nums[i] + i, maxdis);
            i = i + 1;
        minstep += 1;
    return minstep if maxdis>=n-1 else 0;




#print the whole permute
def permute(nums):
        res = [];
        n = len(nums);
        def backtrace(nums,curlist):
            if(len(curlist) == n):
                res.append(curlist);
                return ;
            else:
                for i in range(n):
                    if(nums[i] not in curlist):
                        curlist.append(nums[i]);
                        backtrace(nums,curlist.copy());
                        curlist.pop(-1);
        
        backtrace(nums,[]);
        return res;


#Given a collection of numbers that might contain duplicates, return all possible unique permutations. 
#slow 1190ms
def permuteUnique(nums):
        res = [];
        n = len(nums);
        visited = [False for i in range(n)];
        def backtrace(nums,curlist,visited):
            if(len(curlist) == n):
                res.append(curlist);
                return ;
            else:
                for i in range(n):
                    if(not visited[i]):
                        curlist.append(nums[i]);
                        visited[i] = True;
                        backtrace(nums,curlist.copy(),visited.copy());
                        curlist.pop(-1);
                        visited[i] = False;
        
        backtrace(nums,[],visited);
        
        distinct_res = []
        for l in res:
            if(l not in distinct_res):
                distinct_res.append(l);
                
        return distinct_res;
               

#quick
def permuteUnique(nums):
    res = [];
    n = len(nums);
    visited = [False for i in range(n)];
    def backtrace(nums,temp,visited):
        if(len(temp) == n):
            res.append(temp);
        else:
            for i in range(n):
                if( visited[i] or i > 0 and nums[i] == nums[i-1] and not visited[i-1]):
                    continue;
                temp.append(nums[i]);
                visited[i] = True;
                backtrace(nums, temp.copy(), visited);
                visited[i] = False;
                temp.pop(-1);
    
    backtrace(nums,[],visited);
    return res;

#矩阵旋转
def rotate(matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix);
        if(n == 1):
            return matrix;
        new_matrix = [];
        for i in range(n):
            row = [0 for k in range(n)];
            for j in range(n):
                row[j] = matrix[n-1-j][i];
            
            new_matrix.append(row);
            
        for i in range(n):
            for j in range(n):
                matrix[i][j] = new_matrix[i][j];
        

"""
*()
 * clockwise rotate
 * first reverse up to down, then swap the symmetry 
 * 1 2 3     7 8 9     7 4 1
 * 4 5 6  => 4 5 6  => 8 5 2
 * 7 8 9     1 2 3     9 6 3
*/
void rotate(vector<vector<int> > &matrix) {
    reverse(matrix.begin(), matrix.end());
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = i + 1; j < matrix[i].size(); ++j)
            swap(matrix[i][j], matrix[j][i]);
    }
}

*()
 * anticlockwise rotate
 * first reverse left to right, then swap the symmetry
 * 1 2 3     3 2 1     3 6 9
 * 4 5 6  => 6 5 4  => 2 5 8
 * 7 8 9     9 8 7     1 4 7
*/
void anti_rotate(vector<vector<int> > &matrix) {
    for (auto vi : matrix) reverse(vi.begin(), vi.end());
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = i + 1; j < matrix[i].size(); ++j)
            swap(matrix[i][j], matrix[j][i]);
    }
}
"""


def groupAnagrams(strs): 
    dic = {};
    for s in strs:
        k = tuple(sorted(s));
        if(k in dic):
            dic[k].append(s);
        else:
            dic[k] = [s];
    return list(dic.values());
    

#皇后问题 来自LeetCode解决方案
def solveNQueens(n):
    def could_place(row, col):
        return not (cols[col] + hill_diagonals[row - col] + dale_diagonals[row + col])

    def place_queen(row, col):
        queens.add((row, col))
        cols[col] = 1
        hill_diagonals[row - col] = 1
        dale_diagonals[row + col] = 1

    def remove_queen(row, col):
        queens.remove((row, col))
        cols[col] = 0
        hill_diagonals[row - col] = 0
        dale_diagonals[row + col] = 0

    def add_solution():
        solution = []
        for _, col in sorted(queens):
            solution.append('.' * col + 'Q' + '.' * (n - col - 1))
        output.append(solution)

    def backtrack(row = 0):
        for col in range(n):
            if could_place(row, col):
                place_queen(row, col)
                if row + 1 == n:
                    add_solution()
                else:
                    backtrack(row + 1)
                remove_queen(row, col)

    cols = [0] * n
    hill_diagonals = [0] * (2 * n - 1)
    dale_diagonals = [0] * (2 * n - 1)
    queens = set()
    output = []
    backtrack()
    return output




def solveNQueens(n):
    board = [['.']*n for i in range(n)];
    res = [];
    dfs(board, 0, res);
    return res;

def dfs(board, colIndex, res):
    if(colIndex == len(board)):
        res.append(construct(board));
        return;

    for i in range(len(board)):
        if(validate(board, i, colIndex)):
            board[i][colIndex] = 'Q';
            dfs(board, colIndex + 1, res);
            board[i][colIndex] = '.';


def validate(board,x,y):
    for i in range(len(board)):
        for j in range(y):
            if(board[i][j] == 'Q' and (x + j == y + i or x + y == i + j or x == i)):
                return False;
    return True;

def construct(board):
    res = [];
    for i in range(len(board)):
        s = ''.join(board[i]);
        res.append(s);
    return res;

res = solveNQueens(4)


#find the contiguous subarray which is the max sum
def maxSubArray(nums):
        for i in range(len(nums)-1):
            if(nums[i] > 0):
                nums[i + 1] = nums[i+1] + nums[i];
        return max(nums);

#螺旋序列
def spiralOrder(matrix):    
    n = len(matrix);
    if(n==0):
        return [];
    m = len(matrix[0]);

    steps = [m,n-1,m-1,n-2];
    x = 0;
    y = -1;
    dx = [0,1,0,-1];
    dy = [1,0,-1,0];
    res = [];
    i = 0;
    while(True):
        step = steps[i];
        if(step <= 0):
            break;
        steps[i] = steps[i] - 2;
        while(step):
            x = x + dx[i];
            y = y + dy[i];
            res.append(matrix[x][y]);
            step = step - 1;
        i = i + 1;
        i = i % 4;
    return res;



#my solution
def canJump(nums):
    max_dis = nums[0];
    n = len(nums);
    if(n==1):
        return True;

    i = 0;
    k = 1;
    while(i < n):
        j = max_dis + 1;
        while(k < j):
            max_dis = max(nums[k]+k,max_dis);
            if(max_dis >= n-1):
                return True;
            k = k + 1;
        i = i + 1;
    return False;
    

#leetcode solution
def canJump(nums):
    n = len(nums);
    lastpostion = n - 1;
    for i in range(n-1,-1,-1):
        if(nums[i] + i >= lastpostion):
            lastpostion = i;
    return lastpostion == 0;


#区间合并
def merge(intervals):
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def merge(intervals):
    intervals = sorted(intervals,key=lambda x:x[0]);
    n = len(intervals);
    i = 0;
    while(i < n-1):
        if(intervals[i][1] >= intervals[i+1][1]):
            intervals.pop(i+1);
            n = n - 1;
        elif(intervals[i][1] >= intervals[i+1][0] and intervals[i][1] <= intervals[i+1][1]):
            intervals[i][1] = intervals[i+1][1];
            intervals.pop(i+1);
            n = n - 1;
        else:
            i = i + 1;
    return intervals;


#获得第k个排列
def getPermutation(n, k):
    nums = list(range(1,n+1));
    nums = [str(s) for s in nums];
    res = [];

    def backtrace(nums,curlist):
        if(len(curlist) == n):
            res.append(''.join(curlist));
            return ;
        for i in range(0,len(nums)):
            if(nums[i] not in curlist):
                curlist.append(nums[i]);
                backtrace(nums,curlist.copy());
                curlist.pop(-1);
    backtrace(nums,[]);
    print(res);
    return res[k-1];



def getPermutation(n, k):
    candidate = [i for i in range(1, n + 1)]
    total = 1
    ans = []
    for i in range(1, n+1):
        total *= i
    while len(candidate) > 0:
        idx = (k -1)  // (total // n)
        ans.append(str(candidate[idx]))
        del candidate[idx]
        total = total // n
        n, k = n-1 , k - idx * total
    ans = ''.join(ans)
    return ans



def find(nums):
    res = 0;
    pre = -1;
    for i in range(len(nums)):
        if(nums[i] < 0):
            res = nums[i];
            pre = i;
            break;
    if(pre == -1):
        res = min(nums);
        nums.remove(res);
        res = res + min(nums);
        return res;
    
    for i in range(pre+1,len(nums)):
        if(nums[i] <= 0 and i - pre == 1):
            res = res + nums[i];
            pre = i;
        else:
            break;   
    second_pre = pre;
    for i in range(second_pre+1,len(nums)):
        if(nums[i] < 0):
            second_pre = i;
            res = res + nums[i];
            break;
    if(second_pre == pre):
        res = res + min(nums[pre+1:]);
        return res;
    else:
        for i in range(second_pre + 1,len(nums)):
            if(nums[i] <= 0 and i - second_pre == 1):
                res = res + nums[i];
                second_pre = i;
            else:
                return res;
    return res;



