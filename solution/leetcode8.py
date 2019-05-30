#!/usr/bin/env python
# coding: utf-8

#Ispalindrome[i,j]=True 表示 s[i:j] 是回文串
"""
dp[i][j] representing whether s[i...j] is a palindrome
dp[i][j] = true if s[i] == s[j] and s[i], s[j] is next to each other (i - j < 2),
or s[i] == s[j] and the subproblem s[i+1...j-1] is a valid palindrome (dp[i+1][j-1])
"""
def partition(s) :
    n = len(s);
    if(n == 0):
        return [];
    
    Ispalindrome = [[False] * n for _ in range(n)];
    for i in range(n-1,-1,-1):
        for j in range(i, n):
            Ispalindrome[i][j] = (s[i] == s[j]) and (j - i < 2 or Ispalindrome[i + 1][j - 1]);

    res = [];
    def backtrace(index, cur):
        if(index == n):
            res.append(cur.split());
        for k in range(index, n):
            if Ispalindrome[index][k]:
                backtrace(k + 1, cur + s[index:k+1] + ' ');

    backtrace(0, '');

    return res;


#求解将字符串S切割成所有字串都是回文串需要的最小切割次数
def minCut(s):
    n = len(s);
    if(n == 0):
        return [];

    cut = [n for _ in range(n)];
    cut.append(0);
    Ispalindrome = [[False] * n for _ in range(n)];
    for i in range(n-1,-1,-1):
        for j in range(i, n):
            Ispalindrome[i][j] = (s[i] == s[j]) and (j - i < 2 or Ispalindrome[i + 1][j - 1]);
            if(Ispalindrome[i][j]):
                cut[i] = min(cut[i],cut[j+1]+1);

    if(Ispalindrome[0][n-1]):
        return 0;
    print(cut);
    return cut[0]-1;


"""
class Node:
    def __init__(self,val,neighbors=[]):
        self.val = val;
        self.neighbors = neighbors;
"""
#拷贝无向图
import copy
del cloneGraph(node):
    cy = copy.deepcopy(node);
    return cy;


#拷贝无向图
dic = {};
def cloneGraph(node):
    if(not node):
        return None;
    neighbors = node.neighbors;
    copy = Node(node.val,[]);
    
    dic[copy.val] = copy;
    for n in neighbors:
        if(n.val in dic):
            copy.neighbors.append(dic[n.val]);
        else:
            copy.neighbors.append(cloneGraph(n));
    return copy;
        


def canCompleteCircuit(gas, cost):
    n = len(gas);
    if(not gas):
        return -1;
    if(sum(gas) < sum(cost)):
        return -1;

    start = 0;
    tank = 0;
    #owe 记录所有不足的消耗
    owe = 0;
    for i in range(n):
        cur = gas[i] - cost[i];
        #当前油箱没有油，并且从加油量大于从 i 站到 i + 1 站的耗油量
        if(cur >=0 and tank == 0):
            start = i;
            tank += cur;
        #加油量不足并且油箱的油不够，说明前一个起始点不满足情况，此时需要将油箱的油量置为为0，等待下一个起始点
        elif(cur < 0 and tank + cur < 0):
            owe += tank + cur;
            tank = 0;
        #加油后，剩余油量足够转移到下一站
        else:
            tank += cur;
    if(owe + tank >=0):
        return start;
    else:
        return -1;


print(canCompleteCircuit([5,5,1,3,4],[8,1,7,1,1]))

#分糖果
def candy( ratings):
    n = len(ratings);
    candys = [1 for _ in range(n)];

    left = [1 for _ in range(n)];
    right = [1 for _ in range(n)];

    for i in range(1,n):
        if(ratings[i]>ratings[i-1]):
            left[i] = left[i-1] + 1;

    for i in range(n-2,-1,-1):
        if(ratings[i] > ratings[i+1]):
            right[i] = right[i+1] + 1;

    for i in range(n):
        candys[i] = max(left[i],right[i]);

    return sum(candys);


#flag 用于标记candys 是否发生变化，若不在变化说明糖果的分配达到稳定状态
def candy(ratings):
    candys = [1 for _ in range(n)];
    flag = True;
    while(flag):
        flag = False;
        for i in range(n):
            if(i!= 0 and ratings[i] > ratings[i-1] and candys[i] < candys[i-1]):
                candys[i] = candys[i-1] + 1;
                flag = True;
            if(i != n-1 and ratings[i] > ratings[i+1] and candys[i] < candys[i+1]):
                candys[i] = candys[i+1] + 1;
                flag = True;
    
    return sum(candys);


#经典 数学方法 2(a+b+c) - (a+b+c+a+b) = c
#找出数字列表中只出现一次的数字，每个数字出现两次除了只出现一次的那个数字
def singleNumber(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return 2 * sum(set(nums)) - sum(nums);


def singleNumber(nums):
    dic = {};
    for n in nums:
        if(n in dic):
            del dic[n];
        else:
            dic[n] = n;
    return list(dic.keys())[0];
    


import copy
#拷贝随机列表
def copyRandomList(head): 
    return copy.deepcopy(head);



#https://leetcode.com/problems/copy-list-with-random-pointer/discuss/289594/Python3-solutions%3A-O(n)-Time-O(1)-space-and-O(n)-Time-O(n)-Space
from collections import defaultdict
def copyRandomList(head): 
    if head is None:
        return None

    dfdict = defaultdict(lambda: Node(0,None,None)) 
    dfdict[None] = None
    n = head
    while n:
        dfdict[n].val = n.val
        dfdict[n].next = dfdict[n.next]
        dfdict[n].random = dfdict[n.random]
        n = n.next

    return dfdict[head]

#https://leetcode.com/problems/copy-list-with-random-pointer/discuss/289594/Python3-solutions%3A-O(n)-Time-O(1)-space-and-O(n)-Time-O(n)-Space
# Inserting Clones into existing chain & removing them afterwards (inspired from @liaison intuition) - O(n) Time & O(1) Space
def copyRandomList(head):
        
    if head is None:
        return None

    #add clones to the chain
    n = head
    while n:
        clone = Node(n.val,None,None)
        next_node = n.next
        clone.next = next_node
        n.next = clone
        n = clone.next

    #update random pointers
    n = head
    while n:
        clone = n.next
        if n.random:
            clone.random = n.random.next
        n = clone.next

    #extract clones & restore original chain
    n = head
    res = Node(0,None,None)
    clone_head = res
    while n:
        next_original_node = n.next.next

        #extract clones
        clone = n.next
        clone_head.next = clone

        #restore original chain
        n.next = next_original_node

        # move forward
        n = next_original_node
        clone_head = clone_head.next


    return res.next

#判断是否能用wordDict 中的单词组成s
def wordBreak(s, wordDict):
    n = len(s);
    w_dict = {};
    for w in wordDict:
        if(w[0] in w_dict):
            w_dict[w[0]].append(w);
        else:
            w_dict[w[0]] = [w];
    print(w_dict)

    dp = [False for _ in range(n+1)];
    dp[0] = True;
    s = ' ' + s;
    for i in range(1,n+1):
        c = s[i];
        #以c 开头，并且第i-1 个位置存在匹配的单词
        if(c in w_dict and dp[i-1]):
            ws = w_dict[c];
            for w in ws:
                if(i + len(w) > n+1):
                    continue;

                if(s[i: i + len(w)] == w):
                    dp[i+len(w)-1] = True;
    print(dp);
    return dp[n];




print(wordBreak('leetcode',['leet','code']))



#超时
#给出所有使用wordDict 中的单词组成s 的可能情况
def wordBreak(s, wordDict):
        
    n = len(s);
    w_dict = {};
    for w in wordDict:
        if(w[0] in w_dict):
            w_dict[w[0]].append(w);
        else:
            w_dict[w[0]] = [w];
    res = [];
    def dfs(sub_s,curlist):

        if(''.join(curlist) == s):
            res.append(' '.join(curlist));
            curlist = [];
            return ;

        if(len(sub_s)==0):
            curlist = [];
            return ;

        if(sub_s[0] in w_dict):
            ws = w_dict[sub_s[0]];
            for w in ws:
                if(len(w) > len(sub_s)):
                    continue;

                if(sub_s[:len(w)] == w):
                    curlist.append(w);
                    dfs(sub_s[len(w):],curlist.copy());
                    curlist = curlist[:-1]

        else:
            return ;

    dfs(s,[]);
    return res;



print(wordBreak("catsanddog",["cat","cats","and","sand","dog"]))


#超时
#给出所有使用wordDict 中的单词组成s 的可能情况
#https://leetcode.com/problems/word-break-ii/discuss/293599/Python-Super-Easy-Super-Concise-DFS-52ms
def wordBreak(s, wordDict):
    def dfs(s):
        if not s: 
            return None;
        wlist = [];
        for word in wordDict:
            if word == s:
                wlist.append(word);
                continue ;
            if s.startswith(word):
                getback = dfs(s[len(word):]);
                for i in getback:
                    wlist.append(word+' '+i);
        return wlist;
    return dfs(s);


print(wordBreak("catsanddog",["cat","cats","and","sand","dog"]))

#判断链表中是否存在环
def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    dic = {};
    while(True):

        if(head is None):
            return False;
        if(head in dic):
            return True;
        else:
            dic[head] = head;

        head = head.next;


#如果存在环则slow 会追上 fast
def hasCycle(head):
    if (head == None or head.next == None):
        return False;
    
    slow = head;
    fast = head.next;
    while (slow != fast):
        if (fast == None or fast.next == None):
            return False;
        
        slow = slow.next;
        fast = fast.next.next;
    return true;



#找出环开始的节点
def detectCycle(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    i = 0;
    dic = {};
    while(head):
        if(head in dic):
            return head;
        dic[head] = i;
        i = i + 1;   
        head = head.next;

    return None;




#列表顺序重排 保证新的链表中 node[i] 与 node[i+1] 是原链表中的node[i]与node[n-1-i]. 下标从0开始
def reorderList(self, head):
    """
    :type head: ListNode
    :rtype: None Do not return anything, modify head in-place instead.
    """
    if(not head):
        return None;
    h = head;
    mid = self.getMid(h);

    l2 = mid.next;
    mid.next = None;

    l2 = self.reverse(l2);

    l1 = head;
    while l1 and l2:
        p,q = l1.next,l2.next;
        l1.next = l2;
        l2.next = p;
        l1,l2 = p,q;

    return head;

#fast 前进的速度是 slow 的两倍，当fast 到达终点时，slow到达中点。
def getMid(head):
    fast = slow = head;
    while(fast and fast.next and fast.next.next):
        fast = fast.next.next;
        slow = slow.next;
    return slow;

def reverse(head):
    newhead = None;
    while(head):
        p = head.next;
        head.next = newhead;
        newhead = head;
        head = p;
    return newhead;



#先遍历所有节点，然后前后同时进行移动
def reorderList(head):
    """
    :type head: ListNode
    :rtype: None Do not return anything, modify head in-place instead.
    """
    if(not head):
        return None;
    nodes = []
    cur = head

    while(cur):
        nodes.append(cur);
        cur = cur.next;

    insert = nodes.pop();
    cur = head;

    #一个从头往后，一个从尾往头，当相等时则到达链表的中间位置
    while(cur != insert and cur.next != insert):
        nxt = cur.next;
        cur.next = insert;
        insert.next = nxt;

        nodes[-1].next = None;
        insert = nodes.pop();
        cur = nxt;
        


#二叉树前序遍历
def preorderTraversal(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    res = [];
    def preorder(root):
        if(root):
            res.append(root.val);
        else:
            return ;
        preorder(root.left);
        preorder(root.right);

    preorder(root);
    return res;



#二叉树后序遍历 
def postorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    res = [];
    def postorder(root):
        if(root):
            if(root.left):
                postorder(root.left);

            if(root.right):
                postorder(root.right);

            res.append(root.val);
        else:
            return ;
    postorder(root);
    return res;
    



#二叉树前序遍历 非递归
def preorderTraversal(root):
    result = [];
    stack = [];
    p = root;
    while(stack or p is not None):
        if(p is not None):
            stack.append(p);
            result.append(p.val);  # Add before going to children
            p = p.left;
        else:
            node = stack.pop();
            p = node.right;   
    
    return result;


#二叉树中序遍历 非递归
def inorderTraversal( root):
    result = [];
    stack = [];
    p = root;
    while(stack or p is not None):
        if(p is not None):
            stack.append(p);
            p = p.left;
        else:
            node = stack.pop();
            result.append(node.val);  # Add after all left children
            p = node.right;   
    return result;



#二叉树后序遍历 非递归
def postorderTraversal(root):
    result = [];
    stack = [];
    p = root;
    while(stack or p is not None):
        if(p is not None):
            stack.append(p);
            result.insert(0,p.val);  #Reverse the process of preorder
            p = p.right;             # Reverse the process of preorder
        else:
            node = stack.pop();
            p = node.left;           #Reverse the process of preorder
    
    return result;

