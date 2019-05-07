#!/usr/bin/env python
# coding: utf-8

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


#remove nth node from end
def removeNthFromEnd( head: ListNode, n: int):
        N = 0;
        temp_head = head;
        next_node = temp_head.next;
        while(next_node is not None):
            next_node  = next_node.next;
            N = N + 1;
        N = N + 1;
        k = N - n;
        print(N);
        if(k<0):
            return None;
        #remove head node
        if(k==0):
            return head.next;
        
        #temp_head = head;
        next_node = head;
        while(k>=2 and next_node is not None):
            next_node = next_node.next;
            k = k - 1;
        #remove next_node  
        next_node.next = next_node.next.next if next_node.next is not None else None;
        return head;



#two pointer.first pointer is separated from second pointer by n nodes;
def removeNthFromEnd( head: ListNode, n: int):
    dummy = ListNode(0);
    dummy.next = head;
    first = dummy;
    second = dummy;
    # Advances first pointer so that the gap between first and second is n nodes apart
    for i in range(1,n + 1):
        first = first.next;
    
    # Move first to the end, maintaining the gap
    while (first is not None):
        first = first.next;
        second = second.next;
        
    second.next = second.next.next;
    return dummy.next;



#判断括号配对是否合法
def isValid( s):
    dic = {'(':1,')':1,'{':2,'}':2,'[':3,']':3};
    list_s = list(s);
    while(len(list_s)>0 and len(list_s)%2==0):
        i = 0;
        n = len(list_s);
        while(i < n-1):
            if(list_s[i] != list_s[i + 1]):
                if(dic[list_s[i]] == dic[list_s[i+1]]):
                    print(i,n);
                    list_s.pop(i);
                    list_s.pop(i);
                    break;
                else:
                    i = i + 1;
            else:
                i = i + 1;
        if(i == n-1):
            break;
    if(len(list_s)==0):
        return True;
    else:
        return False;


#产生包含N对合法的括号的所有组合
#左括号和右括号 位置存在奇偶对应关系
#***
def generateParenthesis(N):
        if N == 0: return ['']
        ans = []
        for c in range(N):
            for left in generateParenthesis(c):
                for right in generateParenthesis(N-1-c):
                    print('the pre ans %s' % ans);
                    ans.append('({}){}'.format(left, right))
        return ans


def generateParenthesis(N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)

        backtrack()
        return ans

#暴力遍历
def generateParenthesis( n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    ans.append("".join(A))
            else:
                A.append('(')
                generate(A)
                A.pop()
                A.append(')')
                generate(A)
                A.pop()

        def valid(A):
            bal = 0
            for c in A:
                if c == '(': bal += 1
                else: bal -= 1
                if bal < 0: return False
            return bal == 0

        ans = []
        generate()
        return ans


#合并两个链表
def mergeTwoLists(list1,list2):
    l1 = list1;
    l2 = list2;
    t = ListNode(0);
    t_next = t;
    while(l1 is not None and l2 is not None):
        if(l1.val < l2.val):
            t_next.next = ListNode(l1.val);
            l1 = l1.next;
        else:
            t_next.next = ListNode(l2.val);
            l2 = l2.next;
        t_next = t_next.next;
    
    while(l1 is not None):
        t_next.next = ListNode(l1.val);
        l1 = l1.next;
        t_next = t_next.next;
    
    while(l2 is not None):
        t_next.next = ListNode(l2.val);
        l2 = l2.next;
        t_next = t_next.next;
    
    return t.next;


#合并k个链表
def mergeKLists(lists):
    t = None;
    for li in lists:
        if(t is None):
            t = li;
        else:
            t = mergeTwoLists(t,li);
    return t;



#交换内容
#位置交换 需要变更节点的先驱和后驱
def swapPairs(head):
    point = head;
    while(point and point.next):
        p2 = point.next;
        point.val,p2.val = p2.val,point.val;
        point = p2.next;
    return head;



def reverseKGroup(head, k):
    temp_stack = [];
    h = head;
    reserve = ListNode(0);
    r_p = reserve;
    while(h is not None):
        t = k;
        while(t>0 and h is not None):
            temp_stack.append(h.val);
            h = h.next;
            t = t - 1;
        if(t==0):
            while(temp_stack):
                r_p.next = ListNode(temp_stack.pop(-1));
                r_p = r_p.next;
        else:
            while(temp_stack):
                r_p.next = ListNode(temp_stack.pop(0));
                r_p = r_p.next;
    return reserve.next;



def removeDuplicates(nums):
    i = 0;
    while(i < len(nums)):
        if(i + 1 < len(nums)):
            if(nums[i] == nums[i+1]):
                nums.pop(i+1);
            else:
                i = i + 1;
        else:
            break;
    return len(nums);




#计算除法
#a / b without using multiplication, division and mod operator.
# a/b = e^lna / e^lnb = e^(lna - lnb)
def divide(dividend: int, divisor: int) -> int:
        flag = False;
        if((dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0)):
            flag = True;
        if(dividend == 0):
            return 0;
        
        k = int(math.exp(math.log(abs(dividend)) - math.log(abs(divisor))));
        
        if(flag):
            if(k > pow(2,31)-1):
                return pow(2,31) -1;
            else:
                return k;
        else:
            if(k > pow(2,31)):
                return pow(2,31) -1;
            else:
                return -k;



def findSubstring(s,words):
    if(len(words)==0):
        return [];
    if(len(s) == 0):
        return [];
    
    w_len = len(words[0]);
    w_num = len(words);
    s_len = len(s);
    word_count = {};
    for word in words:
        if(word in word_count):
            word_count[word] = word_count[word] + 1;
        else:
            word_count[word] = 1;
    res = [];
    for i in range(s_len - w_len*w_num + 1):
        sub_s = s[i:i+w_len*w_num];
        dic = word_count.copy();
        flag = True;
        for j in range(0,len(sub_s),w_len):
            w = sub_s[j:j+w_len];
            if(w in dic):
                dic[w] = dic[w] - 1;
            else:
                dic[w] = -1;
        for k in dic.keys():
            if(dic[k] < 0):
                flag = False;
        
        if(flag):
            res.append(i);
    return res;


def nextPermutation(nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        k = len(nums)-2;
        while(k>=0 and nums[k+1] <= nums[k]):
            k = k - 1;
        
        if(k>=0):
            j = len(nums) - 1;
            while(j >= 0 and nums[j] <=nums[k]):
                j = j - 1;
            
            nums[k],nums[j] = nums[j],nums[k];
            
        nums[k+1:] = sorted(nums[k+1:]);


#最长有效括号串
def longestValidParentheses(s):
        maxans = 0;
        stack = [];
        stack.append(-1);
        for i in range(len(s)):
            if (s[i] == '('):
                stack.append(i);
            else:
                stack.pop();
                if(not stack):
                    stack.append(i);
                else:
                    maxans = max(maxans, i - stack[-1]);
        return maxans;
                    

#判断括号配对是否合法
def isValid(s):
    temp_stack = [];
    for c in s:
        if(c == '('):
            temp_stack.append('(');
        else:
            if(temp_stack):
                temp_stack.pop();
            else:
                return False;
    if(not temp_stack):
        return True;
    else:
        return False;


def longest(s):
    res = 0;
    for i in range(len(s)):
        for j in range(i,len(s)+1,2):
            if(isValid(s[i:j])):
                res = max(res,j-i);
    return res;



#动态规划 https://leetcode.com/problems/longest-valid-parentheses/solution/
#dp 记录第i 位置时有效括号的长度
def longestValidParentheses(s):
        res = 0;
        dp = [0 for i in range(len(s))];
        for i in range(1,len(s)):
            if(s[i] == ')'):
                if(s[i-1] == '('):
                    dp[i] = (dp[i-2] if i>=2 else 0) + 2;
                elif(i - dp[i-1] >0 and s[i-dp[i-1]-1] == '('):
                    dp[i] = dp[i-1] + (dp[i-dp[i-1]-2] if (i-dp[i-1])>=2 else 0) + 2;
                res = max(res,dp[i]);
        return res;

def longestValidParentheses(s):
        left = 0;
        right = 0;
        maxlength = 0;
        for i in range(len(s)):
            if (s[i] == '('):
                left = left + 1;
            else:
                right = right + 1;
            if (left == right):
                maxlength = max(maxlength, 2 * right);
            elif (right >= left):
                left = right = 0;
                
        left = right = 0;
        for i in range(len(s)-1,-1,-1):
            if (s[i] == '('):
                left = left + 1;
            else:
                right = right + 1;
                
            if (left == right):
                maxlength = max(maxlength, 2 * left);
            elif(left >= right):
                left = right = 0;
        return maxlength;
		
"""
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
"""
def search(A,target):
    lo=0;
    hi=n-1;
    # find the index of the smallest value using binary search.
    # Loop will terminate since mid < hi, and lo or hi will shrink by at least 1.
    # Proof by contradiction that mid < hi: if mid==hi, then lo==hi and loop would have been terminated.
    while(lo<hi):
        mid=(lo+hi)//2;
        if(A[mid]>A[hi]):
            lo=mid+1;
        else:
            hi=mid;
    # lo==hi is the index of the smallest value and also the number of places rotated.
    rot=lo;
    lo=0;
    hi=n-1;
    # The usual binary search and accounting for rotation.
    while(lo<=hi):
        mid=(lo+hi)/2;
        realmid=(mid+rot)%n;
        if(A[realmid]==target):
            return realmid;
        if(A[realmid]<target):
            lo=mid+1;
        else:
            hi=mid-1;
    return -1;






