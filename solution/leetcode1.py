#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
测试案例
Input: num = "123", target = 6
Output: ["1+2+3", "1*2*3"] 
    
Input: num = "105", target = 5
Output: ["1*0+5","10-5"]

Input: num = "00", target = 0
Output: ["0+0", "0-0", "0*0"]

Input: num = "3456237490", target = 9191
Output: []
"""

class Solution_1:
    def addOperators(self, num: 'str', target: 'int') -> 'List[str]':

        N = len(num)
        answers = []
        def recurse(index, prev_operand, current_operand, value, string):

            # Done processing all the digits in num
            if index == N:

                # If the final value == target expected AND
                # no operand is left unprocessed
                if value == target and current_operand == 0:
                    answers.append("".join(string[1:]))
                return

            # Extending the current operand by one digit
            current_operand = current_operand*10 + int(num[index])
            str_op = str(current_operand)

            # To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
            # valid operand. Hence this check
            if current_operand > 0:

                # NO OP recursion
                recurse(index + 1, prev_operand, current_operand, value, string)

            # ADDITION
            string.append('+'); string.append(str_op)
            recurse(index + 1, current_operand, 0, value + current_operand, string)
            string.pop();string.pop()

            # Can subtract or multiply only if there are some previous operands
            if string:

                # SUBTRACTION
                string.append('-'); string.append(str_op)
                recurse(index + 1, -current_operand, 0, value - current_operand, string)
                string.pop();string.pop()

                # MULTIPLICATION
                string.append('*'); string.append(str_op)
                recurse(index + 1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
                string.pop();string.pop()
        recurse(0, 0, 0, 0, [])    
        return answers


class Solution_2(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        res=[]
        self.target=target
        for i in range(1,len(num)+1):
            if i==1 or (i>1 and num[0]!='0'):#to avoid '00'
                self.dfs(num[i:],num[:i],int(num[:i]),int(num[:i]),res)
                print(num[i:])
                print(res);
        return res
        #last_val is the previous single value add to cur, 
        #in the case of adding '*' ,last_val need to be subtracted and added back after multiplication 
    def dfs(self,num,cur,cur_val,last_val,res):
        """
        num 当前数字字符串
        cur 当前表达式字符串
        cur_value 当前表达式数值
        last_val 最近一次运算数，若为乘法则x*y视为一个数
        res 目标表达式集合
        当表达式中出现乘法运算时，推到过程如下：设最近一次运算数是last_value,最近一次以前值为x，当前表达式值为cur，当前值为val
        cur = x + last_value
        now = x + last_value*val
        now = cur - last_value + last_value*val
        """
        if not num: #if all the nums have been accounted for
            if cur_val==self.target:
                res.append(cur)
            else:
                return 
        else: 
            for i in range(1,len(num)+1):
                val=num[:i]
                if i==1 or (i>1 and num[0]!='0'): #to avoid '00'
                    self.dfs(num[i:],cur+'+'+val,cur_val+int(val),int(val),res);
                    self.dfs(num[i:],cur+'-'+val,cur_val-int(val),-int(val),res);
                    self.dfs(num[i:],cur+'*'+val,cur_val-last_val+last_val*int(val),last_val*int(val),res)
                    

#longest substring
def lengthOfLongestSubstring(s):
        N = len(s);
        i = 0;
        substrlen = [];
        j = i;
        dic = {};
        while(i < N and j < N):
            c = s[j];
            if(c in dic):
                num = j-i;
                substrlen.append(num);
                i = i + 1;
                j = i;
                dic = {};
            else:
                dic[c] = c;
                j = j + 1;
        substrlen.append(len(dic));
        return max(substrlen);



#https://leetcode.com/problems/median-of-two-sorted-arrays/solution/
def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0



#get longest palindrome
def longestPalindrome(s):
        if len(s) <= 1:
            return s;
        start = end = 0;
        length = len(s);
        for i in range(length):
            max_len_1 = get_max_len(s, i, i + 1);#the even palindrome
            max_len_2 = get_max_len(s, i, i);# the odd palindrome
            max_len = max(max_len_1, max_len_2);
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start: end+1]
    
#from center to border
def get_max_len(s, left, right):
    length = len(s);
    i = 1;
    max_len = 0;
    while(left >= 0 and right < length and s[left] == s[right]):
        left = left - 1;
        right = right + 1;
    return right - left - 1;



#zigzag 
def readzigzag(strs,n):
    if(n==1):
        return strs;
    zigzag = [[] for i in range(n)];
    T = n + n - 2;
    for i in range(len(strs)//T):
        temp = strs[i*T:(i+1)*T];
        for j in range(T):
            if(j//n > 0):
                index = n - 2 - j%n;
                zigzag[index].append(temp[j]);
            else:
                zigzag[j%n].append(temp[j]);
    
    remain = strs[(len(strs)//T)*T:];
    for j in range(len(remain)):
        if(j//n > 0):
            index = n-2-j%n;
            zigzag[index].append(remain[j]);
        else:
            zigzag[j%n].append(remain[j]);
    
    result = "";
    for s in zigzag:
        result = result + ''.join(s);
    return result;


#zigzag solution from leetcode
def convert(s, numRows):
    if (numRows == 1):
        return s;
    rows = ["" for i in range(min(numRows,len(s)))];
    curRow = 0;
    goingDown = False;
    for c in s:
        rows[curRow] += c;
        if (curRow == 0 or curRow == numRows - 1):
            goingDown = not goingDown;
        curRow += 1 if goingDown else -1;

    ret = "";
    for row in rows:
        ret += row;
    return ret;


def revers(x):
    strx = str(x);
    strx = strx[::-1];
    strx = list(strx);
    if(strx[-1] == '-'):
        strx.insert(0,strx.pop(-1));
    strx = ''.join(strx);
    rx = int(strx);
    return rx;


#find interge in a string
def myAtoi(str: s) -> int: 
    flag = False;
    sign = '';
    res = '';
    nums = ".0123456789";
    for c in s:
        if(flag):
            if(c in nums):
                if(c == '.'):
                    if(len(res)==0):
                        res = "0" + c;
                    else:
                        res = res + c;
                else:
                    res = res + c;
            else:
                break;
        else:
            if(c == ' '):
                continue;
            else:
                if((c == '-' or c == '+') and len(res)==0):
                    sign = c;
                    flag = True;
                elif(c in nums):
                    if(c == '.'):
                        if(len(res)==0):
                            res = "0" + c;
                    else:
                        res = res + c;
                    flag = True;
                else:
                    break;

    #res = int(res);
    #res = -res if sign=='-' else res;
    if(len(res)==0):
        return 0;
    res = float(sign+res);
    res = int(res);
    if(res > pow(2,31) - 1):
        return pow(2,31) - 1;
    if(res < pow(-2,31)):
        return pow(-2,31);
    return res;



#paline number
def isPalindrome(x: int) -> bool:
    str_x = str(x);
    n = len(str_x);
    for i in range(n//2):
        if(str_x[i] != str_x[n-1-i]):
            return False;
    return True;



#regular express match
#s is the string needed to match
#p is the pattern
#there are some wrong in this programe
def isMatch(s: str, p: str) -> bool:
    if(len(s)==0):
        return True;
    #find the start point
    real_p = None;
    for i in range(len(p)):
        if(p[i] == s[0] or p[i]=='.'):
            real_p = p[i:];
            break;
    #begin match
    if(real_p is None):
        return False;
    
    n = len(s);
    i = 0;
    prechar = '';
    for j in range(len(real_p)):
        curp = real_p[j];
        flag = True;
        while(i < n and flag):
            print(str(i),real_p[j]);
            if(curp == '*'):
                if(real_p[j-1] == '.'):
                    #prechar = s[i];
                    i = i + 1;
                    if(j+1 < n):
                        isMatch(s[i+1:],real_p[j+1:])
                        
                    continue;
                elif(real_p[j-1] != s[i]):
                    flag = False;
                else:
                    i = i + 1;
                    continue;     
            elif(curp == '.'):
                #prechar = s[i];
                i = i + 1;
                break;
            else:
                if(curp != s[i]):
                    return False;
                else:
                    i = i + 1;
                    break;
        if(i==n):
            return True;
        
    if(i == n):
        return True;
    else:
        return False;

        
#regular express match
#****
def isMatch(text, pattern):
    if not pattern:
        return not text

    first_match = bool(text) and pattern[0] in {text[0], '.'}

    if len(pattern) >= 2 and pattern[1] == '*':
        return (isMatch(text, pattern[2:]) or
                first_match and isMatch(text[1:], pattern))
    else:
        return first_match and isMatch(text[1:], pattern[1:])


#Dynamic Programming
#Bottom-Up Variation
def isMatch(text, pattern):
    dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]

    dp[-1][-1] = True
    for i in range(len(text), -1, -1):
        for j in range(len(pattern) - 1, -1, -1):
            first_match = i < len(text) and pattern[j] in {text[i], '.'}
            if j+1 < len(pattern) and pattern[j+1] == '*':
                dp[i][j] = dp[i][j+2] or first_match and dp[i+1][j]
            else:
                dp[i][j] = first_match and dp[i+1][j+1]

    return dp[0][0]



#Container With Most Water
def maxArea(height):
    maxarea = 0;
    for i in range(len(height)):
        for j in range(i+1,len(height)):
            h = min(height[i],height[j]);
            w = abs(j - i);
            if(maxarea < h * w):
                maxarea = h * w;
    return maxarea;



def maxArea(height):
    #运用快速排序的思想
    #若以某点为中心点，分别向左右寻找，时间复杂度太高。
    maxarea = 0;
    l = 0;
    r = len(height)-1;
    while(l<r):
        area = min(height[l],height[r]) * (r-l);
        if(area > maxarea):
            maxarea = area;
        if(height[l] > height[r]):
            r = r - 1;
        else:
            l = l + 1;
    return maxarea;



#integer to Roman
def intToRoman(num):
    res = '';
    m_num = num//1000;
    remain = num % 1000;
    if(m_num>0):
        for i in range(m_num):
            res = res + 'M';
    #小于1000
    d_num = remain // 500;
    remain = remain % 500;
    
    c_num = remain // 100;
    remain = remain % 100;
    if(c_num == 4 and d_num > 0):
        res = res + 'CM';
    elif(c_num == 4 and d_num == 0):
        res = res + 'CD';
    else:
        if(d_num>0):
            for i in range(d_num):
                res = res + 'D';
        if(c_num>0):
            for i in range(c_num):
                res = res + 'C';
        
    
    #小于100
    l_num = remain // 50;
    remain = remain % 50;
    
    x_num = remain // 10;
    remain = remain % 10;
    
    if(x_num == 4 and l_num > 0):
        res = res + 'XC';
    elif(x_num == 4 and l_num == 0):
        res = res + 'XL';
    else:
        if(l_num>0):
            for i in range(l_num):
                res = res + 'L';
        if(x_num>0):
            for i in range(x_num):
                res = res + 'X';
    #小于10
    v_num = remain // 5;
    remain = remain % 5;
    
    i_num = remain;
    if(i_num == 4 and v_num>0):
        res = res + 'IX';
    elif(i_num == 4 and v_num == 0):
        res = res + 'IV';
    else:
        if(v_num>0):
            for i in range(v_num):
                res = res + 'V';
        if(i_num>0):
            for i in range(i_num):
                res = res + 'I';
    return res;

    
def intToRoman(num):
    M = ["", "M", "MM", "MMM"]; #1000-3000
    C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]; #100-900
    X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]; #10-90
    I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]; #1-9
    return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10];



def RomanToint(s):
    dic = {'M':1000,'D':500,'C':100,'L':50,'X':10,'V':5,'I':1,'O':0};
    i = 0;
    res = 0;
    while(i < len(s)):
        cur = s[i];
        if(i + 1 < len(s)):
            ne = s[i + 1];
        else:
            ne = 'O';
        if(dic[cur] < dic[ne]):
            res = res + dic[ne] - dic[cur];
            i = i + 2;
        else:
            res = res + dic[cur];
            i = i + 1;
    return res;



def longestCommonPrefix(strs):
    if(len(strs) == 0):
        return '';
    if(len(strs) == 1):
        return strs[0];
    n = min([len(s) for s in strs]);
    if(n==0):
        return '';
    k = 0;
    for i in range(n):
        for j in range(len(strs)-1):
            if(strs[j][i] == strs[j+1][i]):
                k = i;
                continue;
            else:
                return strs[0][:i];
    return strs[0][:k+1];
    


#time is complex
def threeSum(nums):
    res = {};
    for i in range(len(nums)):
        l = i - 1;
        r = i + 1;
        while(l>=0 and r < len(nums)):
            try:
                index = nums[r:].index(0 -(nums[i] + nums[l]));
                candidate = sorted([nums[i],nums[l],nums[r+index]]);
                candidate = tuple(candidate);
                if(candidate not in res):
                    res[candidate] = list(candidate);
                l = l - 1;
            except:
                l = l - 1;
                continue;
                                       
    return list(res.values());
        



def threeSum(nums):
    if(len(nums)< 3):
        return [];
    nums = sorted(nums);
    res = [];
    for i in range(len(nums)-2):
        if(nums[i] + nums[i+1] + nums[i+2] >0):
            break;
        if(nums[i] == 0):
            if(nums[i+1] == 0 and nums[i+2]==0):
                res.append([0,0,0]);
                return res;
            else:
                break;
        if(nums[i] > 0):
            break;
        
        l = i + 1;
        r = len(nums) - 1;
        while(r>l):
            s = nums[i] + nums[l] + nums[r];
            if(s == 0):
                if([nums[l],nums[i],nums[r]] not in res):
                    res.append([nums[l],nums[i],nums[r]]);
                l = l + 1;
                r = r - 1;
            elif(s < 0):
                l = l + 1;
            else:
                r = r - 1;
    return res;




def threeSumClosest(nums,target):
    if(len(nums)< 3):
        return '';
    if(len(nums)==3):
        return sum(nums);
    nums = sorted(nums);
    i = 0;
    closest = 1000000;
    closetarget = 0;
    while(i<len(nums)):
        l = i + 1;
        r = len(nums) - 1;
        while(r>l):
            s = nums[i] + nums[l] + nums[r] - target;
            if(s == 0):
                return target;
            elif(s < 0):
                if(-s < closest):
                    closest = -s;
                    closetarget = nums[i] + nums[l] + nums[r];
                    l = l + 1;
                else:
                    l = l + 1;
            else:
                if(s < closest):
                    closest = s;
                    closetarget = nums[i] + nums[l] + nums[r];
                    r = r - 1;
                else:
                    r = r - 1;
        nums.pop(i);
    return closetarget;


#手机打字组合
def letterCombinations(digits):
    letters = {0:' ',1:'',2:'abc',3:'def',4:'ghi',5:'jkl',6:'mno',7:'pqrs',8:'tuv',9:'wxyz'};
    res = None;
    for i in digits:
        k = int(i);
        letterstr = letters[k];
        if(res is None):
            res = list(letterstr);
        else:
            temp = [];
            for i in res:
                for l in letterstr:
                    temp.append(i+l);
            res = temp;
    return res;



def fourSum(nums,target):
    if(len(nums)< 4):
        return [];
    nums = sorted(nums);
    res = [];
    for j in range(len(nums)-3):
        if(nums[j] + nums[j+1] + nums[j+2] + nums[j+3] > target):
            break;
        subtarget = target - nums[j];
        for i in range(j+1,len(nums)-2):
            if(nums[j] + nums[i] + nums[i+1] + nums[i+2] > target):
                break;
            l = i + 1;
            r = len(nums) - 1;
            while(r>l):
                s = nums[i] + nums[l] + nums[r] - subtarget;
                if(s == 0):
                    if([nums[j],nums[i],nums[l],nums[r]] not in res):
                        res.append([nums[j],nums[i],nums[l],nums[r]]);
                    r = r - 1;
                elif(s < 0):
                    l = l + 1;
                else:
                    r = r - 1;
    return res;



