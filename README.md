# CS Fundamentals

## Algorithms:
1. Prefix Sum:
    - On 2D array:
    ```
    prefix[i] = prefix[i - 1] + A[i];
    query(i, j) = prefix[j] - prefix[i - 1]; // sum from index i to index j
    ```
    - On 3D array:
    ```
    prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + A[i - 1][j - 1];
    query(u, v, i, j) = prefix[i + 1][j + 1] - prefix[u, j + 1] - prefix[u + 1, j] + prefix[u, v]; // sum all numbers in rectangle (u, v) to (i , j)
    ```
    - Prefix XOR: 
    ```
    prefixXor[i] = prefixXor[i - 1] ^ A[i];
    query(left, right) = prefixXor[right] ^ prefixXor[left - 1];
    ```
   - Practice:
       - https://leetcode.com/problems/running-sum-of-1d-array/
       - https://leetcode.com/problems/range-sum-query-immutable/
       - https://leetcode.com/problems/range-sum-query-2d-immutable/


2. Dutch Flag Partition: Given an array A and index i. Rearrange elements of A such that all elements less than A[i] appear first, followed by elements equal to A[i], then elements greater than A[i]
   ```
   int pivot = A[i];
   int boundary = 0;
   for (int i = 0; i < A.length; i++) {
       if (A[i] < pivot) {
            swap(A, i, boundary);
       }
       boundary++;
   }
   ```
   - Time Complexity: O(n) - where n is the length of the input array
   - Space Complexity: O(1)
   - Practice:
        - https://leetcode.com/problems/move-zeroes/
        - https://leetcode.com/problems/sort-colors/
        - https://leetcode.com/problems/sort-array-by-parity
        - https://leetcode.com/problems/sort-array-by-parity-ii/


3. Bit Manipulation:
   ```
   getBit(bitmask, position) = (bitmask >> position) & 1 // shift the bitmask at position to the end, then and with 1 to get that bit
   setBit(bitmask, position) = bitmask | (1 << position)
   flipBit(bitmask, position) = bitmask ^ (1 << position)
   turnOffRightMostBit(bitmask) = bitmask & (bitmask - 1)
   ```
   - Practice:
        - https://leetcode.com/problems/single-number/
        - https://leetcode.com/problems/power-of-two/
        - https://leetcode.com/problems/number-of-1-bits/
        - https://leetcode.com/problems/subsets/
    

4. Merge 2 sorted arrays:
   ```
    private List<Integer> merge(int[] A, int[] B) {
        List<Integer> merged = new ArrayList<>();
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            if (A[i] <= B[j]) {
                merged.add(A[i]);
            } else {
                merged.add(B[j]);
            }
        }
        while (i < A.length) {
            merged.add(A[i]);
        }
        while (j < B.length) {
            merged.add(B[j]);
        }
        
        return merged;
    }
   ```
   

5. Merge Sort:
    - If the list has length is 0 or 1 => already sorted, do nothing
    - If the list has > 1 elements, split it into 2 lists, sort each list
    - Merged 2 sorted sublists (use above solution)
    ```
    private int[] mergeSort(int[] arr) {
        if (arr.length < 2) {
            return arr;
        }
        int mid = arr.length / 2;
        int[] left = mergeSort(Arrays.copyOfRange(arr, 0, mid);
        int[] right = mergeSort(Arrays.copyOfRange(arr, mid + 1, arr.length));
        
        return merge(left, right);
    }
    ```
   - Time Complexity: O(nlogn) - where n is the length of input array
   - Space Complexity: O(n)
    

6. Quick Sort:
   - Select a pivot
   - Create 2 arrays to hold elements that are less than pivot and elements that are greater than or equals to pivot
   - Recursively sort these 2 arrays
   ```
   public void quickSort(int[] array) {
      helper(array, 0, array.length - 1);
   }
   
   private void helper(int[] array, int startIdx, int endIdx) {
        if (startIdx >= endIdx) {
            return;
        }
        int pivotIdx = startIdx;
        int leftIdx = startIdx + 1;
        int rightIdx = endIndx;
        while (leftIdx <= rightIdx) {
            if (array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx]) {
                swap(array, leftIdx, rightIdx);
            }
            if (array[leftIdx] <= array[pivotIdx]) {
                leftIdx++;
            }
            if (array[rightIdx] >= array[pivotIdx]) {
                rightIdx--;
            }
        }
        swap(array, rightIdx, pivotIdx);
        helper(array, startIdx, rightIdx - 1);
        helper(array, rightIdx + 1, endIdx);
   }
   ```
   - Time Complexity:
     - Worst case: O(n^2)
       - Array is sorted
       - Pivot is always the smallest or biggest number
     - Best case: O(nlogn)
       - Pivot is the median of the array
     - Average case: O(nlogn)
       - Randomly select pivot
   - Space Complexity: O(logn)
    

7. Quick Select: Find the K-th smallest element in an array in O(n)
   - Choose a pivot, then apply Dutch Flag Partition -> after this, we know exactly the index of pivot as it will stand in the correct position.
   - If the index of pivot == k, we find out the answer.
   - If pivot < k, find in the left part. Else, find in the right part.
   ```
   private void quickSelect(int[] array, int startIdx, int endIdx, int position) {
       while (true) {
           if (startIdx > endIdx) {
               throw new RuntimeException("Never arrive here");
           }
           int pivotIdx = startIdx;
           int leftIdx = startIdx + 1;
           int rightIdx = endIdx;
           while (leftIdx <= rightIdx) {
                if (array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx]) {
                    swap(array, leftIdx, rightIdx);
                } 
                if (array[leftIdx] <= array[pivotIdx]) {
                    leftIdx++;
                }
                if (array[rightIdx] >= array[pivotIdx]) {
                    rightIdx--;
                }
           }
           swap(array, rightIdx, pivotIdx);
           if (rightIdx == position) {
                return array[rightIdx];
           }
           if (rightIdx < position) {
                startIdx = rightIdx + 1;
           } else {
                endIdx = rightIdx - 1;
           }
       }
   }
   ```
   - Time Complexity:
     - Worst case: O(n^2)
     - Avg case: O(n) 
   - Space Complexity: O(1)
   - Practice:
     - https://leetcode.com/problems/kth-largest-element-in-an-array
     - https://leetcode.com/problems/k-closest-points-to-origin
     - https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix
     - https://leetcode.com/problems/the-k-strongest-values-in-an-array
     - https://leetcode.com/problems/find-kth-largest-xor-coordinate-value
    

8. Counting Sort:
    - Count the frequency of each element in the list then build the result
    ```
   private int[] countingSort(int[] A) {
      int[] res = new int[A.length];
      Map<Integer, Integer> frequency = new HashMap<>();
      int min = Integer.MAX_VALUE;
      int max = Integer.MIN_VALUE;
      for (int num : A) {
        frequency.put(num, frequency.getOrDefault(num, 0) + 1);
        min = Math.min(min, num);
        max = Math.max(max, num);
      }
      int idx = 0;
      for (int value = min; value <= max; value++) {
        for (int freq = 0; freq < frequency.get(value); freq++) {
            res[idx++] = value;
        }
      }
    
      return res;
   }
   ```
   - Time Complexity: O(n + max - min)
   - Space Complexity: O(n)
   - When to use: range is small enough: 0 <= A[i] <= 10^4 => O(n + 10^4) << O(nlogn)
   - When should not use:
        - A[i] is too big (-10^9 <= A[i] <= 10^9)
        - A[i] is not integer
   - Practice:
        - https://leetcode.com/problems/sort-array-by-increasing-frequency/


9. Depth First Search:
   - Visit graph as depth as possible, go back when there are no nodes available
   ```
   public void dfs(int currentNode, Map<Integer, List<Integer> adjList, boolean[] visited) {
       visited[currentNode] = true;
       for (int neighbour : adjList.getOrDefault(currentNode, new ArrayList<>()) {
            if (!visited[neighbour]) {
                dfs(neighbour);
            }
       }
   }
   ```
   - Time Complexity: O(n + m) - where n is the number of node and m is the number of edge
   - Space Complexity: O(n + m)
   - Practice:
        - https://leetcode.com/problems/number-of-provinces/
        - https://leetcode.com/problems/keys-and-rooms/
        - https://leetcode.com/problems/number-of-islands/
        - https://leetcode.com/problems/max-area-of-island/
        - https://leetcode.com/problems/sum-of-left-leaves/
        - https://leetcode.com/problems/minimum-depth-of-binary-tree/
        - https://leetcode.com/problems/diameter-of-binary-tree/
        - https://leetcode.com/problems/most-frequent-subtree-sum/
        - https://leetcode.com/problems/diameter-of-n-ary-tree/
        - https://leetcode.com/problems/longest-univalue-path/
        - https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
        - https://leetcode.com/problems/house-robber-iii/
    

10. Breadth First Search:
    - Use a queue to store the node being process
    - Use an array 'visited' to indicate the node was visited or not
    - Push source node 's' to the queue, set visited[s] = true, all other node 'v' set visited[v] = false
    - Loop until the queue is empty and in each iteration
        - Pop a vertex from the front of the queue
        - Iterate through all the edges going out of this vertex and if some of these edges go to vertices that are not already visited
            - Mark it as visited
            - Push to the queue
    ```
    public void bfs(int currentNode, Map<Integer, List<Integer> adjList, boolean[] visited) {
        Deque<Integer> queue = new ArrayDeque<>();
        queue.offerLast(currentNode);
        
        while (!queue.isEmpty()) {
            int curr = queue.pollFirst();
            visited[curr] = true;
            for (int neighbour : adjList.getOrDefault(curr, new ArrayList<>()) {
                queue.offerLast(neighbour);
            }
        }
    }
    ```
    - Time Complexity: O(n + m) - where n is the number of node and m is the number of edge
    - Space Complexity: O(n + m)
    - Note:
        - Change 'queue' to 'stack', the above implementation will become DFS.
        - If the problem needs to find the shortest path / minimum cost / minimize something -> can think about BFS
    - Practice:
        - https://leetcode.com/problems/n-ary-tree-level-order-traversal/
        - https://leetcode.com/problems/binary-tree-level-order-traversal/
        - https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
        - https://leetcode.com/problems/deepest-leaves-sum/
        - https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/
        - https://leetcode.com/problems/binary-tree-right-side-view/
        - https://leetcode.com/problems/find-bottom-left-tree-value/
        - https://leetcode.com/problems/find-largest-value-in-each-tree-row/
        - https://leetcode.com/problems/shortest-path-in-binary-matrix/
        - https://leetcode.com/problems/minimum-knight-moves/
        - https://leetcode.com/problems/word-ladder/
        - https://leetcode.com/problems/minimum-genetic-mutation/
        - https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/


12. Binary Search:
    
    12.1. Vanila Binary Search:
    - Prerequisite: The input array must be sorted
    - Idea:
        - Compare 'target' with the middle element of array
        - If 'target == array[mid]' => return true
        - If 'target < array[mid]' => target must belong the left part. Else, target must belong to the right part
    ```
    public boolean binarySearch(int[] array, int target) {
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int middle = left + (right - left) / 2; // prefer to use instead of (left + right) / 2, to avoid number overflow 
            if (target == array[middle]) {
                return true;
            }
            if (target < array[middle]) {
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        }
        return false;
    }
    ```
    - Time Complexity: O(logn)
    - Space Complexity: O(1)
    
    12.2. Find the smallest number that is greater than or equals to 'target'
    ```
    public int binarySearch(int[] array, int target) {
        int left = 0;
        int right = array.length - 1;
        int ans = Integer.MIN_VALUE;
        while (left < right) {
            int middle = left + (right - left) / 2;
            if (array[middle] >= target) {
                ans = array[middle];
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        }
    
        return ans;
    }
    ```
    - Practice:
        - https://leetcode.com/problems/peak-index-in-a-mountain-array/
        - https://leetcode.com/problems/search-in-rotated-sorted-array/


13. Trie (prefix tree):
    - Trie is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker
    ```
    public class Node {
        public Node[] children;
        public boolean isEnd;
    
        Node() {
            this.children = new Node[26]; // 26 alphabet characters
            this.isEnd = false;
        }
    
        public void set(char c, Node node) {
            this.children[c - 'a'] = node;
        }
    
        public Node get(char c) {
            return this.children[c - 'a'];
        }
    
        public boolean contains(char c) {
            return get(c) != null;
        }
    }
    
    public class Trie {
        Node root;
        
        Trie() {
            this.root = new Node();
        }
    
        public void insert(String word) {
            Node node = root;
            for (char c : word.toCharArray()) {
                if (!node.contains(c)) {
                    node.set(c, new Node());
                }
                node = node.get(c);
            }
            node.isEnd = true;
        }
    
        public boolean search(String word) {
            Node node = searchPrefix(word);
            return node != null && node.isEnd;
        }
    
        public boolean startsWith(String prefix) {
            return searchPrefix(prefix) != null;
        }
    
        private Node searchPrefix(String prefix) {
            Node node = root;
            for (char c : prefix.toCharArray()) {
                if (!node.contains(c)) {
                    return null;
                }
                node = node.get(c);
            }
            
            return node;
        }
    }
    ```
    - Time Complexity: O(n) - where n is the length of word in each operation (insert, search, startsWith)
    - Space Complexity: O(total number of char of all words)
    - Practice:
        - https://leetcode.com/problems/implement-trie-prefix-tree/
        - https://leetcode.com/problems/map-sum-pairs/
        - https://leetcode.com/problems/longest-word-in-dictionary/
        - https://leetcode.com/problems/search-suggestions-system/
        - https://leetcode.com/problems/design-search-autocomplete-system/
    

15. Suffix Trie:
    - Same as prefix trie but build from bottom to top
    - Practice:
        - https://leetcode.com/problems/stream-of-characters/
        - https://leetcode.com/problems/prefix-and-suffix-search/
        - https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/


16. Dijkstra:
    - Purpose: To find the shortest path in a weighted graph
    - Note: Dijkstra only works correctly in non-negative weight
    - Practice:
        - https://leetcode.com/problems/network-delay-time/
        - https://leetcode.com/problems/path-with-maximum-probability/
        - https://leetcode.com/problems/cheapest-flights-within-k-stops/
    

## System Design
1. Hashing:
   
    1.1. Hash Function: 
    - Any function that can be used to map data of arbitrary size to fixed-size values. Eg: SHA-256, SHA-512, MD5,...
    - A good hash function:
        - One-way algorithm
        - Efficiently computable
        - Randomly distributed
    
    1.2. Hashing:
    - The process of converting a given key into another fixed-size value using Hash function.
    
    1.3. Hashing vs Encoding vs Encryption
    
    | Hashing | Encoding | Encryption |
    | --- | --- | --- |
    | Converting a given key into another fixed-size value (eg: SHA256, MD5) | data is transformed from one form to other form (eg: BASE64, ASCII, MP4, MP3) | An special encoding technique, in which message is encoded by encryption algorithm that only authorized people can read the message (eg: RSA (Private key, public key), AES) |
    | Not reversible (1 way) | Reversible (2 ways) | Reversible (2 ways) |
    | key => value. multi purpose Data Integrity | Purpose: transform data into a form that is readable by external process | Purpose: transfer private data |
    
    1.4. Hash Table:
    - Map keys to values
    - Use hash functions to compute an index (hashcode) from key
    
    1.5. Load factor:
    - Measure how full the hash table
    - Eg: If hash table has 100 slots, 69 slots are fill => load factor = 69/100 = 69%
        - Load factor == 75% => double size of hash table (Java)

    1.6. Collision:
    - When two different keys but generate same hash value
    - Why: Output space is limited but input is unlimited
    - Avoid:
        - Chaining: each slot of the table is a linked list
        - Open addressing
    

2. Reliability:
    - The system should continue working correctly even in the face of adversity (hardware or software faults, and even human error)
    - How to achieve High Availability:
        - Monitoring
        - Fault tolerance:
            - Remove SPoF (single point of failure)
                - Server has to be stateless
                - Make sure each server can handle extra load
                - N + 2 rule: if your system needs N servers, make it N + 2 => 1 extra for upgrade or testing and 1 for failure
    

3. Scalability:
    - The ability of a system to continue working as user load and data grow
    - Common metrics:
        - Throughput: The number of queries or requests processed in a given period. Eg: query per second (qps), request per second (rps)
        - Latency: response time
  
  
4. Latency:
    - Mean: Average of all request's latency
    - Percentile: The response time threshold at which x% of request are faster than that particular threshold
    

5. Maintainability:
    - Operability: Make it easy for operations team to keep the system running smoothly
    - Simplicity: Make it easy for new engineers to understand the system
    - Evolvability: Meke it easy for engineers to make changes to the system in the future


6. CAP theorem:
    - CAP theorem states it is impossible for a distributed system to simultaneously provide more than two of these three guarantees: 
        - Consistency: All clients see the same data at the same time no matter which node they connect to
        - Availability: Any client which requests data gets a response even if some of the nodes are down
        - Partition tolerance: A partition indicates a communication break between two nodes. Partition tolerance means the system continues to operate despite network partitions
    

## Database
1. Sharding:
    - Storing a large database across multiple machines
    - Why sharding:
        - A single machine, or database server, can only store a limited amount of data
        - The database becomes bottleneck if the data volume becomes too large and too many users attempt to use the application to read or save information simultaneously
    - Benefits:
        - Reduce latency
        - Increase throughput
        - Improve availability
        - Scalable
    - How to shard?
        - Shard by key range. Eg: Shard A contains names that starts with A - I, shard B contains names that starts with J - S, shard C contains names that starts with T to Z
            - Cons: Data hotspots (the amount of names starts with A - I (shard A) usually larger than the amount of names starts with T - Z (shard C))
        - Shard by hash of key: shard key is produced by a hash function
    - However, sharding is not perfect. Why??
        - Data hotspots: Some of the shards become unbalance due to the uneven distribution of data
        - Application Complexity: Most database management systems do not have built-in sharding features. This means that database designer and software developers must manually split, distribute, and manage the database
        - Infra & operational cost
    

2. Replication:
    - Database replication can be used in many database management systems, usually with a master/slave relationship between the original (master) and the copies (slaves)
        - Master: Only write
        - Slave: Copy data from master, and support only read
    - Advantages:
        - Better performance: In the master-slave model, all writes and updates happen in master nodes; whereas, read operations are distributed across slave nodes. This model improves performance because it allows more queries to be processed in parallel.
        - Reliability: If one of your database servers is destroyed by a natural disaster, such as a typhoon or an earthquake, data is still preserved. You do not need to worry about data loss because data is replicated across multiple locations.
        - High availability: By replicating data across different locations, your website remains in operation even if a database is offline as you can access data stored in another database server.
    

3. How DBMS store data on hard disk?
    - Database is just a bunch of files on disk
    - Storage manager is the one that maintain the database files
    - It organizes file as a collection of pages
    - A page is a fixed-size block of data:
        - Contains records, meta data, indexes
        - MySQL: 16KB (default)
        - MS SQL, PostgreSQL: 8KB
    - In a page, all records belong to same table
    - When DBMS read data, it reads by page
    

4. Index:
    - A database index is a data structure that improves the speed of data retrieval operations on database table
    - Types:
        - Hash indexes
        - B+ Tree
        - SSTables (NoSQL)
    
    4.1. Hash Indexes:
    - Based on a Hash Table
        - Key: hash code of the indexed columns
        - Value: pointer to the corresponding row
        - Only useful for exact look up. Cannot support range query, unequal condition (>, < , >=, <=)
        - Eg: SELECT * FROM students WHERE name = 'Long';
        - DBMS calculates hash code of 'Long' by using hash function, and look for it in the hash table
    

## Security
