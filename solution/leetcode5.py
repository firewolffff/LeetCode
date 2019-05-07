#!/usr/bin/env python
# coding: utf-8


"""寻找有序数组（升序）旋转的旋转点索引 数组发生旋转后，其左边最小数大于等于右边最大数"""
def FindRotIndex(nums):
    left = 0;
    right = len(nums)- 1;
    while(left < right):
        mid = left + (right - left) // 2;
        if(nums[mid] == nums[right] and nums[mid] == nums[left]):
            left += 1
            continue;

        if(nums[mid] > nums[right]):
            left = mid + 1;
        else:
            right = mid;
    return left;

#计算最大矩形面积 木桶原理
def largestRectangleArea(heights):
    if(not heights):
        return 0;
    n = len(heights);
    if(n == 1):
        return heights[0];

    maxarea = 0;
    for i in range(n):
        if(i==n-1 or heights[i] > heights[i+1]):
            min_height = heights[i];
            for j in range(i,-1,-1):
                min_height = min(min_height,heights[j]);
                maxarea = max(maxarea,min_height*(i-j+1));

    return maxarea;



"""
We can use a monotone increasing stack to track width range and corresponding threshold height.
In such way, for w[i:j], the threshold height will be height[i] and width is j-i. 
When a new column index j+1 that height[j+1] < height[w[-1]] comes, 
the monotonicity breaks and we need to pop stack elements and update candidate size as sz = max(sz, height[i] * (j-i)) until height[j+1] ≥ height[w[-1]] 
and monotonicity holds again. And we need to insert a 0 at the end of height to update the size when column index iteration ends.
当高度单调递增时，面积持续增大；当出现高度下降时，则面积出现拐点，新的局部最大面积可能会产生。此时保存拐点之前的最大面积，待于后局部最大面积
做比较
"""
def largestRectangleArea2(heights):
    w = [-1];
    n = len(heights);
    sz = 0;
    #防止高度一直增加，没有机会算局部最大面积
    heights.append(0);
    for j in range(n+1):
        while heights[j] < heights[w[-1]]:
            height = heights[w.pop()]
            sz = max(sz, height * (j-1-w[-1]))
        w.append(j)
    return sz;


#计算最大矩形
def maximalRectangle(matrix):
    if(not matrix):
        return 0;
    n = len(matrix);
    m = len(matrix[0]);
    heights = [[0]*m for _ in range(n)];

    for i in range(n):
        for j in range(m):
            if(matrix[i][j] == '1'):
                heights[i][j] = heights[i-1][j] + 1;

    max_area = 0;
    for h in heights:
        area = largestRectangleArea(h);
        if(area > max_area):
            max_area = area;
    return max_area;


"""
left(i,j) = max(left(i-1,j), cur_left), cur_left can be determined from the current row
right(i,j) = min(right(i-1,j), cur_right), cur_right can be determined from the current row
height(i,j) = height(i-1,j) + 1, if matrix[i][j]=='1';
height(i,j) = 0, if matrix[i][j]=='0'
"""
int maximalRectangle(vector<vector<char> > &matrix) {
    if(matrix.empty()) return 0;
    const int m = matrix.size();
    const int n = matrix[0].size();
    int left[n], right[n], height[n];
    fill_n(left,n,0); fill_n(right,n,n); fill_n(height,n,0);
    int maxA = 0;
    for(int i=0; i<m; i++) {
        int cur_left=0, cur_right=n; 
        for(int j=0; j<n; j++) { // compute height (can do this from either side)
            if(matrix[i][j]=='1') height[j]++; 
            else height[j]=0;
        }
        for(int j=0; j<n; j++) { // compute left (from left to right)
            if(matrix[i][j]=='1') left[j]=max(left[j],cur_left);
            else {left[j]=0; cur_left=j+1;}
        }
        // compute right (from right to left)
        for(int j=n-1; j>=0; j--) {
            if(matrix[i][j]=='1') right[j]=min(right[j],cur_right);
            else {right[j]=n; cur_right=j;}    
        }
        // compute the area of rectangle (can do this from either side)
        for(int j=0; j<n; j++)
            maxA = max(maxA,(right[j]-left[j])*height[j]);
    }
    return maxA;
}


from collections import Counter

"""只考虑一种切分方法"""
def isScramble(s1,s2):
    print(s1,s2);
    if(len(s1) != len(s2)):
        return False;
    if(len(s1) == len(s2)==2):
        return Counter(s1)==Counter(s2);
    
    if(len(s1)==len(s2)==1 and s1==s2):
        return True;

    if(Counter(s1)==Counter(s2)):

        n = len(s1);
        sub_s11 = s1[:n//2];
        sub_s12 = s1[n//2:];

        sub_s21 = s2[:n//2];
        sub_s22 = s2[n//2:];
        return isScramble(sub_s11,sub_s21) and isScramble(sub_s12,sub_s22);
    else:
        return False;


"""考虑所有的切分方法"""
def isScramble(s1, s2):
    if(len(s1) != len(s2)):
        return False;

    if(len(s1) == len(s2)==2):
        return Counter(s1)==Counter(s2);
    if(len(s1)==len(s2)==1 and s1==s2):
        return True;

    if(Counter(s1)==Counter(s2)):
        n = len(s1);
        for i in range(1,n):
            if(isScramble(s1[:i],s2[:i]) and isScramble(s1[i:],s2[i:])):
                return True;
            if(isScramble(s1[:i],s2[n-i:]) and isScramble(s1[i:],s2[:n-i])):
                return True;
        return False;
    else:
        return False;

#雷格码 格雷码是二进制数字系统，其中两个连续值仅在一位上不同。
#以0作为第一个数
def grayCode(n):
    res = [];
    sequence = graySequence(n);
    for s in sequence:
        res.append(int(s,base=2));
    return res;
    
    
def graySequence(n):
    if(n==0):
        return ['0'];
    if(n==1):
        return ['0','1'];
    if(n==2):
        return ['00','01','11','10'];

    res = ['00','01','11','10'];
    m = 3;
    while(m<=n):
        pre = res[0];
        temp = [];
        for i in range(len(res)):
            if(res[i] == pre):
                temp.append(res[i]+'0');
                temp.append(res[i]+'1');
                pre = res[i];
            else:
                r = temp[-1][-1];
                temp.append(res[i]+r);
                next_c = '0' if r=='1' else '1';
                temp.append(res[i]+next_c);
                pre = res[i];
        res = temp;
        m = m + 1;
    return res;


#统计能组成1-27的数字组合数目
count = 0;
def splitNum(s):
    global count;
    
    if(len(s)==0):
        return 0;
    if(len(s) == 1):
        if(0 < int(s) < 10):
            count += 1;
    elif(len(s) == 2):
        if(9 < int(s) < 27):
            count += 1;
        if(0<int(s[0])<10 and 0<int(s[1])<10):
            count += 1;
    else:
        t = s[:2];
        if(t[0] == '0'):
            return count;
        else:
            if(int(t)<27):
                splitNum(s[2:]);
            splitNum(s[1:]);
    return count;


#数字编码 1-27范围内的所有数字
def numDecodings(s):
    l=len(s);
    dp=[1]*l;
    s=list(s);
    s=list(int(i) for i in s);
    if s[0]==0:
        return 0;
    if l==1:
        return 1;
    for i in range(1,l):
        if s[i]==0 and s[i-1]!=1 and s[i-1]!=2:
            return 0
    if s[0]==1 and s[1]!=0 or s[0]==2 and 1<=s[1]<=6:
        dp[1]=2
    else: 
        dp[1]=1
    for i in range(2,l):
        if s[i-1]==1 and s[i]!=0 or s[i-1]==2 and 1<=s[i]<=6:
            dp[i]=dp[i-1]+dp[i-2]
        elif s[i]==0:
            dp[i]=dp[i-2]
        else:
            dp[i]=dp[i-1]
    return dp[l-1]


#统计所有可能的IP地址划分
def restoreIpAddresses(s):
    N = len(s);
    res = [];
    combine = [];
    def backtrace(curlist,cur):
        if(len(curlist) > 4):
            return ;
        if(cur == N and len(curlist)==4):
            combine.append(curlist);
            return ;
        for i in [1,2,3]:
            curlist.append(i);
            backtrace(curlist.copy(),cur+i);
            curlist = curlist[:-1];
    backtrace([],0);
    for c in combine:
        r = "";
        pre = 0;
        for n in c:
            if((n==1) or (n>=2 and s[pre] != '0' and 10<= int(s[pre:n+pre]) <= 255)):
                r = r + s[pre:n+pre] + '.';
            else:
                break;
            pre = pre + n;
        if(len(r) == N + 4):
            res.append(r[:-1]);
    return res;
        


#统计所有可能的IP地址划分
#来自LeetCode 网友分享
def restoreIpAddresses(s):
    def dfs(s, k, c, r):   
        if not s:
            return

        if c == 3:
            if int(s) == 0 and len(s) == 1 or int(s) <= 255 and s[0] != '0':
                r += [k + s]
            return

        if s[0] == '0':
            dfs(s[1:], k + s[0] + '.', c + 1, r)
        else:

            dfs(s[1:], k + s[:1] + '.', c + 1, r)
            dfs(s[2:], k + s[:2] + '.', c + 1, r)
            if int(s[:3]) <= 255:
                dfs(s[3:], k + s[:3] + '.', c + 1, r)

    if not s:
        return []

    res = []
    dfs(s, '', 0, res)
    return res


#combine two number return the max value 
#input: 23 123
#output 23123. the whole combination are 23123,12323, 23123 is bigger
def combineNum(num1,num2):
    count1 = 0;
    count2 = 0;
    t1 = num1;
    t2 = num2;
    while(t1):
        count1 += 1;
        t1 = t1//10;
    
    while(t2):
        count2 += 1;
        t2 = t2//10;
    
    c1 = num1 * pow(10,count2) + num2;
    c2 = num2 * pow(10,count1) + num1;
    return True if c1>=c2 else False;
    #return max(c1,c2);

#find the combination which will get the max number
#input [23,12,456,78]
#output 784562312
def SearchMaxCombine(nums):
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if(not combineNum(nums[i],nums[j])):
                nums[i],nums[j] = nums[j],nums[i];
    
    res = "";
    for n in nums:
        res = res + str(n);
    return res;

#0-1 背包问题
#V 价值列表，W 重量列表， C背包总容量
#m[i,j] 表示 在面对第 i 件物品，且背包容量为  j 时所能获得的最大价值 
#j < w[i] 的情况，这时候背包容量不足以放下第 i 件物品，只能选择不拿
#则 m[i][j] = m[i-1][ j ]
#j >= w[i] 的情况，这时背包容量可以放下第 i 件物品，我们就要考虑拿这件物品是否能获取更大的价值。
#如果拿取，m[i][j]= MAX(m[i-1][j-w[i]] + v[i], m[i-1][j]

def knapasck(v,w,c):
    n = len(v);
    m = [[0]*(c+1) for _ in range(n+1)];
    v.insert(0,0);
    w.insert(0,0);
    
    for i in range(1,n+1):
        for j in range(1,c+1):
            m[i][j] = m[i-1][j];
            if(w[i] <= j):
                if(v[i] + m[i-1][j-w[i]] > m[i-1][j]):
                    m[i][j] = v[i] + m[i-1][j-w[i]];
    return m;

v = [8, 10, 6, 3, 7, 2];
w = [4, 6, 2, 2, 5, 1];
c = 12;
m = knapasck(v,w,c);

