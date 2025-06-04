"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters if it is non-empty.
"""

def longestCommonPrefix(Lst): 
    
    prefix_list = [] #create list to store values of prefixes from string []
    count_list = [] # built to store counts of prefix
    
    for word in Lst: #loop/iterate through list items strings
        prefix_list.append(word[0:2])       #check index 0 and 1 of each list item
                                            #save index 0 and 1 of each list item to created list storage as string

    for prefix in prefix_list:
        # count_list = [[prefix, prefix_list.count(prefix)] for prefix in set(prefix_list)]
        count_list.append(prefix_list.count(prefix))
        max_count = max(count_list)
        index = count_list.index(max_count)
        if max_count < 2:
            return ""
        else:
            return prefix_list[index]
    
        # count_list = prefix_list.count(prefix)   #for prefixes stored in created list
                                                 #compare strings in stored list to find most common
                                                 #return the most common prefix
        
    #handle edge case
        #if no common prefix return empty string
        #reorder logic if needed 
    


# Lst = ["flower","flow","flight"]        
Lst = ["dog","racecar","car"]    
print(longestCommonPrefix(Lst))  

    
        
