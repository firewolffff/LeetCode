#!/usr/bin/env python
# coding: utf-8

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

"""
二叉树中序遍历，递归实现
"""
def inorderTraversal(root):
    res = [];
    def dfs(root):
        if(root):
            if(root.left):
                dfs(root.left);
            res.append(root.val);
            if(root.right):
                dfs(root.right);

    dfs(root);
    return res;


#二叉树的中序遍历 非递归实现
def inorderTraversal(root):
    res = [];
    stack = [];
    cur = root;
    while(cur or stack):
        while(cur):
            stack.append(cur);
            cur = cur.left;
        cur = stack.pop();
        res.append(cur.val);
        cur = cur.right;
    return res;


def generateTree(start,end):
    res = [];
    if(start > end):
        res.append(None);
        return res;
    
    if(start == end):
        res.append(TreeNode(start));
        return res;
    
    for i in range(start,end+1):
        left = generateTree(start,i-1);
        right generateTree(i+1,end);
        for lnode in left:
            for rnode in right:
                root = TreeNode(i);
                root.left = lnode;
                root.right = rnode;
                res.append(root);
    return res;




def gearateTrees(n):
    if(n == 0):
        return [];
    return genreateTree(1,n);

def insertnode(self,root,node):
    if(root is None):
        root = node;
        return root;
    if(node.val < root.val):
        if(root.left is None):
            root.left = node;
            return ;
        else:
            self.insertnode(root.left,node);

    elif(node.val > root.val):
        if(root.right is None):
            root.right = node;
            return ;
        else:
            self.insertnode(root.right,node);

#修正一个非法的二叉排序树
def recoverTree(root):
    """
    :type root: TreeNode
    :rtype: None Do not return anything, modify root in-place instead.
    """

    if not root:
        return

    prev = None
    first = None
    secont = None
    #first secont 起什么作用？
    #中序遍历
    def dfs(root):
        if not root:
            return 

        dfs(root.left)
        if prev and prev.val >= root.val:
            if not first:
                first = self.prev
            second = root
        prev = root
        dfs(root.right)

    dfs(root) 
    first.val, second.val = second.val, first.val

#G(n) = G(0)*G(n-1) + G(1)*G(n-2) + .... + G(n-1)*G(0)
#G(i) 表示序列长度为i时二叉排序树的数量
#G(0)=G(1) = 1;
def numTrees(n):
    if(n==0):
        return 0;
    if(n==1):
        return 1;

    G = [0 for _ in range(2)];
    G[0] = 1;
    G[1] = 1;


    for i in range(2,n+1):
        m = len(G);
        g = 0;
        for j in range(m):
            g = g + G[j] * G[m-1-j];
        G.append(g);
    return G[n];


#判断是否为合法的二叉查找树
def check(treenode,lower,upper):
    if(lower is not None and treenode.val <= lower):
        return False;
    if(upper is not None and treenode.val >= upper):
        return False;

    left = check(treenode.left,lower,treenode.val) if treenode.left else True;
    if(left):
        right = check(treenode.right,treenode.val,upper) if treenode.right else True;
        return right;
    else:
        return False;
        
#判断二叉树是否为对称树
def isSymmetric(root):
    if(not root):
        return True;

    def compare(r1,r2):
        if(r1 and r2 and r1.val == r2.val):
            if(r1.left is None and r2.right is None and r1.right is None and r2.left is None):
                return True;
            else:
                return compare(r1.left,r2.right) and compare(r1.right,r2.left);
        elif(r1 is None and r2 is None):
            return True;
        else:
            return False;

    return compare(root.left,root.right);
    

def isSymmetric(root):
    if(not root):
        return True;

    def compare(r1,r2):
        if(r1 is None and r2 is None):
            return True;
        if(r1 is None or r2 is None):
            return False;
        return r1.val == r2.val and compare(r1.left,r2.right) and compare(r1.right,r2.left);

    return compare(root.left,root.right);

#二叉树广度优先遍历
def levelOrder(root):
    res = [];
    if(not root):
        return res;
    res.append([root.val]);
    stack = [root.left,root.right]
    while(stack):
        n = len(stack);
        i = 0;
        temp = [];
        while(i < n):
            node = stack[0];
            if(node):
                temp.append(node.val);
                if(node.left):
                    stack.append(node.left);
                if(node.right):
                    stack.append(node.right);
            stack.pop(0);
            i = i + 1;
        if(temp):
            res.append(temp);
    return res;


#二叉树的最大深度
def maxDepth(root):
    if(not root):
        return 0;

    def dfs(node,depth):
        if(not (node.left or node.right)):
            return depth;
        else:
            if(node.left):
                dl = dfs(node.left,depth+1);
            else:
                dl = depth;
            if(node.right):
                dr = dfs(node.right,depth+1);
            else:
                dr = depth;

            return max(dl,dr);
    maxdepth = dfs(root,1);
    return maxdepth;


#二叉树最小深度
#注意单只树的情况
def minDepth(root):
    if(not root):
        return 0;

    ldepth = minDepth(root.left);
    rdepth = minDepth(root.right);

    return ldepth + rdepth + 1 if(ldepth == 0 or rdepth == 0) else min(ldepth,rdepth) + 1;



#preorder, inorder 前序遍历和中序遍历链表
def buildTree(preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if len(preorder) == 0:
        return None
    if len(preorder) == 1:
        tree = TreeNode(preorder[0])
        return tree
    else:
        val = preorder[0]
        tree = TreeNode(val)
        valIndex = inorder.index(val)
        inorderLeft = inorder[:valIndex]
        inorderRight = inorder[valIndex+1:]
        preorderLeft = preorder[1:1+len(inorderLeft)]
        preorderRight = preorder[1+len(inorderLeft):]
        tree.left = buildTree(preorderLeft,inorderLeft )
        tree.right = buildTree(preorderRight, inorderRight)
        return tree
            


# In[4]:


def buildTree(preorder, inorder):
    if(inorder):
        index = inorder.index(preorder.pop(0));
        root = TreeNode(inorder[index]);
        root.left = self.buildTree(preorder, inorder[0:index]); #建立左子树
        root.right = self.buildTree(preorder, inorder[index+1:]); #建立右子树
        return root;
    else:
        return None;


#将一个升序序列构造成平衡查找二叉树
def sortedArrayToBST(nums):
    if(not nums):
        return None;

    n = len(nums);
    rootindex = n//2;
    root = TreeNode(nums[rootindex]);
    left = nums[:rootindex];
    right = nums[rootindex+1:];

    if(len(left) == 1):
        root.left = TreeNode(left[0]);
    elif(len(left) > 1):
        root.left = sortedArrayToBST(left);


    if(len(right) == 1):
        root.right = TreeNode(right[0]);
    elif(len(right) > 1):
        root.right = sortedArrayToBST(right);

    return root;


#巧妙 来自LeetCode 解决方案
class Solution:

    def findMiddle(self, head):

        # The pointer used to disconnect the left half from the mid node.
        prevPtr = None
        slowPtr = head
        fastPtr = head

        # Iterate until fastPr doesn't reach the end of the linked list.
        while fastPtr and fastPtr.next:
            prevPtr = slowPtr
            slowPtr = slowPtr.next
            fastPtr = fastPtr.next.next

        # Handling the case when slowPtr was equal to head.
        if prevPtr:
            prevPtr.next = None

        return slowPtr


    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """

        # If the head doesn't exist, then the linked list is empty
        if not head:
            return None

        # Find the middle element for the list.
        mid = self.findMiddle(head)

        # The mid becomes the root of the BST.
        node = TreeNode(mid.val)

        # Base case when there is just one element in the linked list
        if head == mid:
            return node

        # Recursively form balanced BSTs using the left and right halves of the original list.
        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(mid.next)
        return node


#valus the number list,ascending
def convertListToBST(l, r):

        # Invalid case
        if l > r:
            return None

        # Middle element forms the root.
        mid = (l + r) // 2
        node = TreeNode(values[mid])

        # Base case for when there is only one element left in the array
        if l == r:
            return node

        # Recursively form BST on the two halves
        node.left = convertListToBST(l, mid - 1)
        node.right = convertListToBST(mid + 1, r)
        return node
    return convertListToBST(0, len(values) - 1)

#判断是否为平衡二叉树
def isBalanced(root):
    if(not root):
        return True;
    if(root.left is None and root.right is None):
        return True;

    def dfs(node,depth):
        if(not (node.left or node.right)):
            return depth;
        else:
            if(node.left):
                dl = dfs(node.left,depth+1);
            else:
                dl = depth;
            if(node.right):
                dr = dfs(node.right,depth+1);
            else:
                dr = depth;

            return max(dl,dr) if(abs(dl-dr)<=1) else -1;
    depth = dfs(root,1);
    if(depth == -1):
        return False;
    else:
        return True;


#存在一条 从根节点到叶节点的路径，使得路径上元素的和等于 sum_
#叶节点 的左右节点为None
def hasPathSum(root, sum_):
    if(not root):
        return False;
    if(root.val == sum_ and root.left is None and root.right is None):
        return True;
    else:
        l = False;
        if(root.left):
            l = hasPathSum(root.left,sum_-root.val);
        if(not l):
            if(root.right):
                return hasPathSum(root.right,sum_-root.val);
            else:
                return False;
        else:
            return True;
            


#找出所有从根节点到叶节点的路径，使得路径上元素的和等于 sum_
#叶节点 的左右节点为None
def pathSum(root, sum_):
        
    res = [];
    def dfs(node,curlist,cur):
        if(node):
            curlist.append(node.val);
            if(sum_ == cur + node.val and node.left is None and node.right is None):
                res.append(curlist);
                return ;
            else:
                dfs(node.left,curlist.copy(),cur+node.val);
                dfs(node.right,curlist.copy(),cur+node.val);
        else:
            return ;

    dfs(root,[],0);
    return res;


#将二叉树扁平化，转化为一颗单只树（只有右子树）
def flatten(root):
    """
    Do not return anything, modify root in-place instead.
    """
    if(not root):
        return ;

    l = root.left;
    root.left = None;
    right = root.right;
    if(l):
        if(l.right is None):
            l.right = right;
        else:
            lr = l;
            while(lr.right):
                lr = lr.right;
            lr.right = right;

        root.right = l;

    flatten(root.right);
        

pre = None;
def flatten(root):
    """
    Do not return anything, modify root in-place instead.
    """
    if(not root):
        return ;

    flatten(root.right);
    flatten(root.left);
    root.right = pre;
    root.left = None;
    pre = root;



class Node:
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

#将一个完美二叉树同层节点链接
def connect(root):
    r = root;
    while(r):
        temp = r;
        while(temp):
            #非叶节点连接
            if(temp.left is not None and temp.right is not None):
                temp.left.next = temp.right;
            #下一层兄弟节点连接
            if(temp.next is not None and temp.right is not None):
                temp.right.next = temp.next.left;

            temp = temp.next;
        r = r.left;

    return root;

#递归方案，将一个完美二叉树同层节点链接
def connect(root):
    if(not root):
        return root;
    
    def hepler(left,right):
        if(left is None or  right is None):
            return ;
        left.next = right;
        helper(left.left, left.right);
        helper(left.right, right.left);
        helper(right.left, right.right);
    
    helper(root.left,root.right);
    return root;

#将一个二叉树中同层节点链接 广度优先
def connect(root):
    if(not root):
        return root;
    stack = [];
    if(root.left):
        stack.append(root.left);
    if(root.right):
        stack.append(root.right);

    while(stack):
        n = len(stack);
        i = 1;
        pre = stack.pop(0);
        while(i < n):
            if(pre):
                if(pre.left):
                    stack.append(pre.left);
                if(pre.right):
                    stack.append(pre.right);
            next_ = stack.pop(0);
            pre.next = next_;
            pre = next_;
            i = i + 1;
        if(pre):
            if(pre.left):
                stack.append(pre.left);
            if(pre.right):
                stack.append(pre.right);

    return root;





