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


13. Dijkstra:
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